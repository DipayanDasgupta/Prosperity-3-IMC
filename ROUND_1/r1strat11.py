import json
import math
from collections import deque, defaultdict
from typing import Any, TypeAlias, List, Dict, Optional, Tuple
from abc import abstractmethod
import jsonpickle # Make sure this is imported
import numpy as np # Needed for calculations

# Assuming datamodel classes are available as in the competition environment
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# Type Alias for JSON data
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# Logger class (Ensure this definition is complete and correct *before* instantiation)
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
            return [state.timestamp, trader_data, compress_listings(state.listings), compress_order_depths(state.order_depths), compress_trades(state.own_trades or {}), compress_trades(state.market_trades or {}), state.position or {}, compressed_observations]
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
    # (Keep implementation as before)
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
    # (Keep implementation as before)
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


class SquidInkStrategyV16_MACD(Strategy): # Not based on MMStrategy
    """Trades SQUID_INK based on EMA Crossover filtered by MACD Histogram."""
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        # EMA Crossover Params
        self.ema_short_period = 10
        self.ema_long_period = 25
        self.ema_short: Optional[float] = None
        self.ema_long: Optional[float] = None
        self.prev_ema_short: Optional[float] = None
        self.prev_ema_long: Optional[float] = None

        # MACD Params
        self.macd_fast_period = 12
        self.macd_slow_period = 26
        self.macd_signal_period = 9
        self.macd_ema_fast: Optional[float] = None
        self.macd_ema_slow: Optional[float] = None
        self.macd_line: Optional[float] = None
        self.signal_line: Optional[float] = None
        self.macd_histogram: Optional[float] = None
        self.prev_macd_histogram: Optional[float] = None # To detect histogram zero cross

        # Trade Params
        self.trade_size = 6 # Moderate size per signal

    def _calculate_ema(self, current_price: float, prev_ema: Optional[float], period: int) -> float:
        """Helper to calculate EMA."""
        if prev_ema is None: return current_price # Initialize
        alpha = 2 / (period + 1)
        return alpha * current_price + (1 - alpha) * prev_ema

    def update_indicators(self, state: TradingState) -> Optional[float]:
        """Calculates EMAs and MACD based on current mid-price."""
        ods = state.order_depths.get(self.symbol)
        if not ods: return None
        bids = ods.buy_orders; asks = ods.sell_orders
        if not bids or not asks: return None # Need both sides for mid-price
        best_ask=min(asks.keys()); best_bid=max(bids.keys()); best_ask_vol=abs(asks[best_ask]); best_bid_vol=bids[best_bid]
        if best_ask_vol+best_bid_vol==0: current_mid = (best_ask+best_bid)/2.0
        else: current_mid = (best_bid*best_ask_vol+best_ask*best_bid_vol)/(best_bid_vol+best_ask_vol)

        # Store previous EMA values
        self.prev_ema_short = self.ema_short
        self.prev_ema_long = self.ema_long

        # Update EMAs for Crossover
        self.ema_short = self._calculate_ema(current_mid, self.ema_short, self.ema_short_period)
        self.ema_long = self._calculate_ema(current_mid, self.ema_long, self.ema_long_period)

        # Update EMAs for MACD
        self.macd_ema_fast = self._calculate_ema(current_mid, self.macd_ema_fast, self.macd_fast_period)
        self.macd_ema_slow = self._calculate_ema(current_mid, self.macd_ema_slow, self.macd_slow_period)

        # Calculate MACD Line
        if self.macd_ema_fast is not None and self.macd_ema_slow is not None:
            self.macd_line = self.macd_ema_fast - self.macd_ema_slow
        else:
            self.macd_line = None

        # Calculate Signal Line
        if self.macd_line is not None:
            self.signal_line = self._calculate_ema(self.macd_line, self.signal_line, self.macd_signal_period)
        else:
            self.signal_line = None

        # Calculate MACD Histogram
        self.prev_macd_histogram = self.macd_histogram # Store previous histogram value
        if self.macd_line is not None and self.signal_line is not None:
            self.macd_histogram = self.macd_line - self.signal_line
        else:
            self.macd_histogram = None

        return current_mid # Return for potential use

    def act(self, state: TradingState) -> None:
        current_mid = self.update_indicators(state)
        if current_mid is None or self.ema_short is None or self.ema_long is None or \
           self.prev_ema_short is None or self.prev_ema_long is None or \
           self.macd_histogram is None or self.prev_macd_histogram is None:
            # logger.print("SQUID MACD: Not enough data yet")
            return # Need all indicators calculated

        pos = state.position.get(self.symbol, 0)
        buy_cap = self.limit - pos
        sell_cap = self.limit + pos
        order_depth = state.order_depths[self.symbol]
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else float('-inf')

        # --- Entry Signals ---
        ema_crossed_up = self.ema_short > self.ema_long and self.prev_ema_short <= self.prev_ema_long
        ema_crossed_down = self.ema_short < self.ema_long and self.prev_ema_short >= self.prev_ema_long
        macd_hist_positive = self.macd_histogram > 0
        macd_hist_crossed_up = self.macd_histogram > 0 and self.prev_macd_histogram <= 0

        # Enter Long: EMA cross up AND histogram is positive (or just crossed up)
        if ema_crossed_up and macd_hist_positive and pos <= 0 and buy_cap > 0: # Allow entering or flipping from short
             trade_vol = min(buy_cap, abs(pos) + self.trade_size) # Buy enough to flatten short + enter long
             self.buy(best_ask, trade_vol) # Aggressive entry at ask
             logger.print(f"SQUID MACD ENTRY LONG: {trade_vol}@{best_ask}")


        # Enter Short: EMA cross down AND histogram is negative (or just crossed down)
        elif ema_crossed_down and not macd_hist_positive and pos >= 0 and sell_cap > 0: # Allow entering or flipping from long
            trade_vol = min(sell_cap, pos + self.trade_size) # Sell enough to flatten long + enter short
            self.sell(best_bid, trade_vol) # Aggressive entry at bid
            logger.print(f"SQUID MACD ENTRY SHORT: {trade_vol}@{best_bid}")


        # --- Exit Signals ---
        # Exit Long: EMA cross down OR MACD histogram turns negative
        elif pos > 0 and (ema_crossed_down or (self.macd_histogram < 0 and self.prev_macd_histogram >= 0)):
             self.sell(best_bid, min(pos, sell_cap)) # Aggressively flatten
             logger.print(f"SQUID MACD EXIT LONG: {min(pos, sell_cap)}@{best_bid}")


        # Exit Short: EMA cross up OR MACD histogram turns positive
        elif pos < 0 and (ema_crossed_up or (self.macd_histogram > 0 and self.prev_macd_histogram <= 0)):
             self.buy(best_ask, min(abs(pos), buy_cap)) # Aggressively flatten
             logger.print(f"SQUID MACD EXIT SHORT: {min(abs(pos), buy_cap)}@{best_ask}")


    # Save/Load all necessary indicator states
    def save(self) -> JSON:
        return {
            "ema_short": self.ema_short, "ema_long": self.ema_long,
            "prev_ema_short": self.prev_ema_short, "prev_ema_long": self.prev_ema_long,
            "macd_ema_fast": self.macd_ema_fast, "macd_ema_slow": self.macd_ema_slow,
            "macd_line": self.macd_line, "signal_line": self.signal_line,
            "macd_histogram": self.macd_histogram, "prev_macd_histogram": self.prev_macd_histogram
        }

    def load(self, data: JSON) -> None:
        if isinstance(data, dict):
            self.ema_short = data.get("ema_short"); self.ema_long = data.get("ema_long")
            self.prev_ema_short = data.get("prev_ema_short"); self.prev_ema_long = data.get("prev_ema_long")
            self.macd_ema_fast = data.get("macd_ema_fast"); self.macd_ema_slow = data.get("macd_ema_slow")
            self.macd_line = data.get("macd_line"); self.signal_line = data.get("signal_line")
            self.macd_histogram = data.get("macd_histogram"); self.prev_macd_histogram = data.get("prev_macd_histogram")
        else: # Reset state if data is invalid
            self.ema_short=None; self.ema_long=None; self.prev_ema_short=None; self.prev_ema_long=None
            self.macd_ema_fast=None; self.macd_ema_slow=None; self.macd_line=None; self.signal_line=None
            self.macd_histogram=None; self.prev_macd_histogram=None


# --- Main Trader Class ---
class Trader:
    # (Updated to use SquidInkStrategyV16_MACD)
    def __init__(self) -> None:
        self.limits = { "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50 }
        self.strategy_map = {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidInkStrategyV16_MACD, # Use the MACD strategy
        }
        self.strategies: Dict[Symbol, Strategy] = {}
        for symbol, limit in self.limits.items():
            if symbol in self.strategy_map: self.strategies[symbol] = self.strategy_map[symbol](symbol, limit)
            else: logger.print(f"WARNING: No strategy defined for {symbol}")
        self.initial_load_done = False

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

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        if not self.initial_load_done or not state.traderData: self.load_trader_data(state.traderData); self.initial_load_done = True
        all_orders: Dict[Symbol, List[Order]] = {symbol: [] for symbol in self.limits}; conversions = 0
        for symbol, strategy in self.strategies.items():
            if symbol in state.order_depths: all_orders[symbol] = strategy.run(state)
        traderData_output = self.save_trader_data()
        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output