import json
import math
from collections import deque, defaultdict
from typing import Any, TypeAlias, List, Dict, Optional, Tuple
from abc import abstractmethod
import jsonpickle # Make sure this is imported
import numpy as np # Needed for standard deviation

# Assuming datamodel classes are available as in the competition environment
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# Type Alias for JSON data
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# Logger class (Defined before use)
class Logger:
    # (Keep the full Logger class implementation from previous correct responses)
    def __init__(self) -> None: self.logs = ""; self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None: self.logs += sep.join(map(str, objects)) + end
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # --- Full Compression Logic ---
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
            compressed_conversion_observations = {}
            if hasattr(state, 'observations') and state.observations is not None and hasattr(state.observations, 'conversionObservations') and state.observations.conversionObservations is not None:
                 for product, observation in state.observations.conversionObservations.items():
                     if observation is not None: compressed_conversion_observations[product]=[getattr(observation,'bidPrice',None),getattr(observation,'askPrice',None),getattr(observation,'transportFees',0.0),getattr(observation,'exportTariff',0.0),getattr(observation,'importTariff',0.0),getattr(observation,'sunlight',None),getattr(observation,'humidity',None)]
                     else: compressed_conversion_observations[product]=[None]*7
            plain_obs = state.observations.plainValueObservations if hasattr(state.observations, 'plainValueObservations') and state.observations is not None else {}
            compressed_observations = [plain_obs, compressed_conversion_observations]
            return [
                state.timestamp, trader_data, compress_listings(state.listings),
                compress_order_depths(state.order_depths), compress_trades(state.own_trades or {}),
                compress_trades(state.market_trades or {}), state.position or {}, compressed_observations,
            ]
        def compress_listings(listings: dict[Symbol, Listing]) -> list[list[Any]]:
            compressed = [];
            if listings:
                for listing in listings.values():
                     if listing: compressed.append([listing.symbol, listing.product, listing.denomination])
            return compressed
        def compress_order_depths(order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
            compressed = {};
            if order_depths:
                for symbol, order_depth in order_depths.items():
                    if order_depth: buy_orders_dict=order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}; sell_orders_dict=order_depth.sell_orders if isinstance(order_depth.sell_orders,dict) else {}; compressed[symbol]=[buy_orders_dict, sell_orders_dict]
                    else: compressed[symbol] = [{}, {}]
            return compressed
        def compress_trades(trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
            compressed = [];
            if trades:
                for arr in trades.values():
                     if arr:
                        for trade in arr:
                             if trade: compressed.append([trade.symbol,trade.price,trade.quantity,trade.buyer or "",trade.seller or "",trade.timestamp])
            return compressed
        def compress_orders(orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
            compressed = [];
            if orders:
                for arr in orders.values():
                    if arr:
                        for order in arr:
                            if order: compressed.append([order.symbol, order.price, order.quantity])
            return compressed
        def to_json(value: Any) -> str:
             try: return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
             except Exception: return json.dumps(value, separators=(",", ":"))
        def truncate(value: str, max_length: int) -> str:
            if not isinstance(value, str): value = str(value)
            if len(value) <= max_length: return value
            return value[:max_length - 3] + "..."
        # --- End Compression Helpers ---
        try:
            min_obs=Observation({}, {}); min_state=TradingState(state.timestamp,"",{},{},{},{},{},min_obs)
            base_value=[compress_state(min_state,""),compress_orders({}),conversions,"",""]; base_json=to_json(base_value); base_length=len(base_json)
            available_length=self.max_log_length-base_length-200;
            if available_length < 0: available_length=0
            max_item_length=available_length//3
            truncated_trader_data_state=truncate(state.traderData if state.traderData else "", max_item_length)
            truncated_trader_data_out=truncate(trader_data, max_item_length); truncated_logs=truncate(self.logs, max_item_length)
            log_entry=[compress_state(state, truncated_trader_data_state), compress_orders(orders), conversions, truncated_trader_data_out, truncated_logs]
            final_json_output = to_json(log_entry)
            if len(final_json_output)>self.max_log_length: final_json_output=final_json_output[:self.max_log_length-5]+"...]}"
            print(final_json_output)
        except Exception as e: print(json.dumps({"error": f"Logging failed: {e}", "timestamp": state.timestamp}))
        self.logs = ""

logger = Logger()

# --- Base Strategy Classes ---
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None: self.symbol = symbol; self.limit = limit; self.orders: List[Order] = []
    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        try: self.act(state)
        except Exception as e: logger.print(f"ERROR {self.symbol}: {e}"); self.orders = []
        return self.orders
    @abstractmethod
    def act(self, state: TradingState) -> None: raise NotImplementedError()
    def buy(self, price: float, quantity: float) -> None:
        if quantity > 0: self.orders.append(Order(self.symbol, int(round(price)), int(round(quantity))))
    def sell(self, price: float, quantity: float) -> None:
        if quantity > 0: self.orders.append(Order(self.symbol, int(round(price)), -int(round(quantity))))
    def save(self) -> JSON: return None
    def load(self, data: JSON) -> None: pass


class MarketMakingStrategy(Strategy):
    DEFAULT_EMA_ALPHA = 0.2
    def __init__(self, symbol: Symbol, limit: int, ema_alpha: float = DEFAULT_EMA_ALPHA) -> None: super().__init__(symbol, limit); self.ema_alpha=ema_alpha; self.ema_value:Optional[float]=None; self.last_mid_price:Optional[float]=None
    def get_current_midprice(self, state: TradingState) -> Optional[float]:
        ods=state.order_depths.get(self.symbol);
        if not ods: return None
        bids=ods.buy_orders; asks=ods.sell_orders;
        if not bids or not asks: return min(asks.keys()) if asks else (max(bids.keys()) if bids else None)
        best_ask=min(asks.keys()); best_bid=max(bids.keys()); best_ask_vol=abs(asks[best_ask]); best_bid_vol=bids[best_bid]
        if best_ask_vol+best_bid_vol==0: return (best_ask+best_bid)/2.0
        return (best_bid*best_ask_vol+best_ask*best_bid_vol)/(best_bid_vol+best_ask_vol)
    def get_fair_value(self, state: TradingState) -> Optional[float]:
        current_mid=self.get_current_midprice(state); self.last_mid_price=current_mid
        if current_mid is not None:
            if self.ema_value is None: self.ema_value=current_mid
            else: self.ema_value=self.ema_alpha*current_mid+(1-self.ema_alpha)*self.ema_value
        return self.ema_value
    def act(self, state: TradingState) -> None: # Default MM logic
        fair_value_float=self.get_fair_value(state);
        if fair_value_float is None: return
        fair_value=int(round(fair_value_float)); ods=state.order_depths[self.symbol];buy_orders=sorted(ods.buy_orders.items(),reverse=True);sell_orders=sorted(ods.sell_orders.items())
        pos=state.position.get(self.symbol,0); buy_cap=self.limit-pos; sell_cap=self.limit+pos
        pass_buy_p=fair_value-1; pass_sell_p=fair_value+1; aggr_buy_thr=fair_value; aggr_sell_thr=fair_value
        inv_frac=pos/self.limit
        if inv_frac>0.5: skew=int(round(inv_frac*2)); aggr_buy_thr-=skew; pass_buy_p-=skew
        elif inv_frac<-0.5: skew=int(round(abs(inv_frac)*2)); aggr_sell_thr+=skew; pass_sell_p+=skew
        taken_buy=0
        if sell_orders:
            for p,v_neg in sell_orders:
                if p<=aggr_buy_thr and buy_cap>0: v=abs(v_neg); vol=min(v,buy_cap); self.buy(p,vol); buy_cap-=vol; taken_buy+=vol
                else: break
        taken_sell=0
        if buy_orders:
            for p,v in buy_orders:
                if p>=aggr_sell_thr and sell_cap>0: vol=min(v,sell_cap); self.sell(p,vol); sell_cap-=vol; taken_sell+=vol
                else: break
        pass_size=5
        if buy_cap>0 and taken_buy==0: self.buy(pass_buy_p, min(buy_cap, pass_size))
        if sell_cap>0 and taken_sell==0: self.sell(pass_sell_p, min(sell_cap, pass_size))
    def save(self) -> JSON: return {"ema_value": self.ema_value, "last_mid_price": self.last_mid_price}
    def load(self, data: JSON) -> None:
        if isinstance(data,dict): self.ema_value=data.get("ema_value"); self.last_mid_price=data.get("last_mid_price")
        else: self.ema_value=None; self.last_mid_price=None

# --- Concrete Strategy Implementations ---

class RainforestResinStrategy(Strategy):
    # (Keep implementation as before)
    def __init__(self, symbol: Symbol, limit: int) -> None: super().__init__(symbol, limit); self.fair_value = 10000
    def act(self, state: TradingState) -> None:
        ods=state.order_depths.get(self.symbol,OrderDepth()); pos=state.position.get(self.symbol,0); buy_cap=self.limit-pos; sell_cap=self.limit+pos
        best_ask=min(ods.sell_orders.keys()) if ods.sell_orders else float('inf'); best_bid=max(ods.buy_orders.keys()) if ods.buy_orders else float('-inf')
        passive_size=5; taken=False
        if best_ask < self.fair_value and buy_cap > 0: vol=min(buy_cap,abs(ods.sell_orders[best_ask])); self.buy(best_ask,vol); buy_cap-=vol; taken=True
        if best_bid > self.fair_value and sell_cap > 0: vol=min(sell_cap,ods.buy_orders[best_bid]); self.sell(best_bid,vol); sell_cap-=vol; taken=True
        if not taken:
            if buy_cap>0: self.buy(self.fair_value-1,min(buy_cap,passive_size))
            if sell_cap>0: self.sell(self.fair_value+1,min(sell_cap,passive_size))
    def save(self)->JSON: return None
    def load(self,data:JSON)->None: pass

class KelpStrategy(MarketMakingStrategy):
    # (Keep implementation as before)
    def __init__(self, symbol: Symbol, limit: int) -> None: super().__init__(symbol, limit, ema_alpha=0.3)
    def act(self, state: TradingState) -> None: super().act(state)


class SquidInkStrategyV18_AdapSpread(MarketMakingStrategy):
    """MM Strategy for SQUID_INK adjusting spread based on volatility & inventory."""
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit, ema_alpha=0.25) # Moderate EMA
        self.vol_window_size = 30 # Increased window for smoother volatility
        self.price_history = deque(maxlen=self.vol_window_size)
        self.volatility: Optional[float] = None

        # Volatility/Spread Parameters (TUNABLE)
        self.base_passive_spread = 2.0 # Base half-spread
        self.volatility_spread_multiplier = 0.3 # Reduced sensitivity
        self.min_volatility = 1.5 # Minimum assumed volatility
        self.max_spread = 8.0 # Increased max spread slightly

        # Size Parameters
        self.passive_order_size_base = 4
        self.passive_order_size_high_vol = 2
        self.aggressive_order_size = 2
        self.high_vol_threshold_abs = 8.0 # Absolute vol level considered "high"

        # Inventory Skew
        self.aggr_skew_multiplier = 2.0
        self.passive_price_skew_multiplier = 0.75 # How much inventory skews passive price
        self.passive_spread_skew_multiplier = 0.25 # How much inventory *widens* spread

    # Override get_fair_value to update price history and calculate volatility
    def get_fair_value(self, state: TradingState) -> Optional[float]:
        current_mid = self.get_current_midprice(state)
        if current_mid is not None: self.price_history.append(current_mid)
        self.last_mid_price = current_mid
        _ = super().get_fair_value(state) # Update self.ema_value
        # Calculate volatility
        if len(self.price_history) >= self.vol_window_size // 2:
             self.volatility = max(np.std(list(self.price_history)), self.min_volatility)
        else: self.volatility = self.min_volatility
        return self.ema_value

    def act(self, state: TradingState) -> None:
        fair_value_float = self.get_fair_value(state) # Updates EMA and Volatility
        if fair_value_float is None or self.volatility is None: return

        fair_value = int(round(fair_value_float))
        volatility = self.volatility

        ods=state.order_depths.get(self.symbol,OrderDepth()); buy_orders=sorted(ods.buy_orders.items(),reverse=True); sell_orders=sorted(ods.sell_orders.items())
        pos=state.position.get(self.symbol,0); buy_cap=self.limit-pos; sell_cap=self.limit+pos
        best_ask = min(sell_orders, key=lambda x: x[0])[0] if sell_orders else float('inf')
        best_bid = max(buy_orders, key=lambda x: x[0])[0] if buy_orders else float('-inf')

        # --- Calculate Dynamic Spread based on Volatility ---
        target_half_spread = self.base_passive_spread + self.volatility * self.volatility_spread_multiplier
        target_half_spread = min(target_half_spread, self.max_spread / 2)
        target_half_spread = max(target_half_spread, 1.0) # Min half-spread of 1

        # --- Inventory Skew ---
        aggr_buy_thr=fair_value; aggr_sell_thr=fair_value;
        pass_buy_p=fair_value - target_half_spread; pass_sell_p=fair_value + target_half_spread;
        inv_frac=pos/self.limit
        # Skew aggression thresholds AND passive prices AND widen spread based on inventory
        if inv_frac > 0.2:
            skew_val = abs(inv_frac)
            aggr_skew = int(round(skew_val * self.aggr_skew_multiplier))
            pass_price_skew = int(round(skew_val * self.passive_price_skew_multiplier))
            spread_widening = int(round(skew_val * self.passive_spread_skew_multiplier))
            aggr_buy_thr -= aggr_skew
            pass_buy_p -= (pass_price_skew + spread_widening) # Lower buy price more
            pass_sell_p += spread_widening # Push sell price further away

        elif inv_frac < -0.2:
            skew_val = abs(inv_frac)
            aggr_skew = int(round(skew_val * self.aggr_skew_multiplier))
            pass_price_skew = int(round(skew_val * self.passive_price_skew_multiplier))
            spread_widening = int(round(skew_val * self.passive_spread_skew_multiplier))
            aggr_sell_thr += aggr_skew
            pass_sell_p += (pass_price_skew + spread_widening) # Raise sell price more
            pass_buy_p -= spread_widening # Push buy price further away

        pass_buy_p = int(round(pass_buy_p))
        pass_sell_p = int(round(pass_sell_p))

        # --- Aggressive Orders ---
        taken_buy=0
        if best_ask <= aggr_buy_thr and buy_cap > 0:
            vol=min(abs(ods.sell_orders[best_ask]),buy_cap,self.aggressive_order_size); self.buy(best_ask,vol); buy_cap-=vol; taken_buy+=vol
        taken_sell=0
        if best_bid >= aggr_sell_thr and sell_cap > 0:
            vol=min(ods.buy_orders[best_bid],sell_cap,self.aggressive_order_size); self.sell(best_bid,vol); sell_cap-=vol; taken_sell+=vol

        # --- Passive Orders ---
        passive_size = self.passive_order_size_high_vol if volatility > self.high_vol_threshold_abs else self.passive_order_size_base

        if buy_cap > 0 and taken_buy == 0:
            # Ensure passive buy is below best ask
            final_buy_price = min(pass_buy_p, best_ask - 1) if best_ask != float('inf') else pass_buy_p
            self.buy(final_buy_price, min(buy_cap, passive_size))
            # logger.print(f"Placing Passive BUY {self.symbol} {min(buy_cap, passive_size)}@{final_buy_price} (Vol:{volatility:.1f})")

        if sell_cap > 0 and taken_sell == 0:
            # Ensure passive sell is above best bid
            final_sell_price = max(pass_sell_p, best_bid + 1) if best_bid != float('-inf') else pass_sell_p
            self.sell(final_sell_price, min(sell_cap, passive_size))
            # logger.print(f"Placing Passive SELL {self.symbol} {min(sell_cap, passive_size)}@{final_sell_price} (Vol:{volatility:.1f})")

    # Save/Load includes volatility state
    def save(self) -> JSON:
        base_state = super().save()
        base_state["price_history"] = list(self.price_history)
        base_state["volatility"] = self.volatility
        return base_state
    def load(self, data: JSON) -> None:
        super().load(data)
        if isinstance(data, dict):
             history=data.get("price_history",[]); self.price_history=deque(history,maxlen=self.vol_window_size); self.volatility=data.get("volatility", None)
             if self.volatility is None and len(self.price_history)>=self.vol_window_size//2: self.volatility=max(np.std(list(self.price_history)),self.min_volatility)
        else: self.price_history=deque(maxlen=self.vol_window_size); self.volatility=None

# --- Main Trader Class ---
class Trader:
    # (Updated to use SquidInkStrategyV18_AdapSpread)
    def __init__(self) -> None:
        self.limits = { "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50 }
        self.strategy_map = {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidInkStrategyV18_AdapSpread, # Use the Adaptive Spread strategy
        }
        self.strategies: Dict[Symbol, Strategy] = {}
        for symbol, limit in self.limits.items():
            if symbol in self.strategy_map: self.strategies[symbol] = self.strategy_map[symbol](symbol, limit)
            else: logger.print(f"WARNING: No strategy defined for {symbol}")
        self.initial_load_done = False
    # Keep load_trader_data and save_trader_data as before
    def load_trader_data(self, traderData: str):
        if not traderData: return
        try:
            saved_states = jsonpickle.decode(traderData);
            if not isinstance(saved_states, dict): saved_states = {}
            for symbol, strategy in self.strategies.items(): strategy.load(saved_states.get(symbol, None))
        except Exception as e: logger.print(f"ERROR decoding traderData: {e}"); [s.load(None) for s in self.strategies.values()]
    def save_trader_data(self) -> str:
        states_to_save = {symbol: strategy.save() for symbol, strategy in self.strategies.items()}
        try: return jsonpickle.encode(states_to_save, unpicklable=False)
        except Exception as e: logger.print(f"ERROR encoding traderData: {e}"); return ""
    # Keep run method as before
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        if not self.initial_load_done or not state.traderData: self.load_trader_data(state.traderData); self.initial_load_done = True
        all_orders: Dict[Symbol, List[Order]] = {symbol: [] for symbol in self.limits}; conversions = 0
        for symbol, strategy in self.strategies.items():py
        if symbol in state.order_depths: all_orders[symbol] = strategy.run(state)
        traderData_output = self.save_trader_data()
        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output