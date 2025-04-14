import json
import math
from collections import deque, defaultdict
from typing import Any, TypeAlias, List, Dict, Optional, Tuple
from abc import abstractmethod
import jsonpickle # Make sure this is imported

# Assuming datamodel classes are available as in the competition environment
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# Type Alias for JSON data
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# Logger class (Ensure this definition is complete and correct *before* instantiation)
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        # Appends logs internally; flushing handles printing to stdout
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # --- Full Compression Logic ---
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
            compressed_conversion_observations = {}
            # Check if observations and conversionObservations exist and are not None
            if hasattr(state, 'observations') and state.observations is not None and \
               hasattr(state.observations, 'conversionObservations') and state.observations.conversionObservations is not None:
                 for product, observation in state.observations.conversionObservations.items():
                     # Handle potential missing attributes in nested observation object
                     if observation is not None:
                         compressed_conversion_observations[product] = [
                             getattr(observation, 'bidPrice', None), getattr(observation, 'askPrice', None),
                             getattr(observation, 'transportFees', 0.0), getattr(observation, 'exportTariff', 0.0),
                             getattr(observation, 'importTariff', 0.0), getattr(observation, 'sunlight', None),
                             getattr(observation, 'humidity', None),
                         ]
                     else:
                         compressed_conversion_observations[product] = [None] * 7 # Placeholder if observation object itself is None
            # Handle potentially missing plainValueObservations
            plain_obs = state.observations.plainValueObservations if hasattr(state.observations, 'plainValueObservations') and state.observations is not None else {}

            compressed_observations = [plain_obs, compressed_conversion_observations]

            return [
                state.timestamp,
                trader_data, # Already truncated before calling compress_state
                compress_listings(state.listings),
                compress_order_depths(state.order_depths),
                compress_trades(state.own_trades or {}), # Pass empty dict if None
                compress_trades(state.market_trades or {}), # Pass empty dict if None
                state.position or {}, # Pass empty dict if None
                compressed_observations,
            ]

        def compress_listings(listings: dict[Symbol, Listing]) -> list[list[Any]]:
            compressed = []
            if listings:
                for listing in listings.values():
                     if listing: # Check if listing object is not None
                        compressed.append([listing.symbol, listing.product, listing.denomination])
            return compressed

        def compress_order_depths(order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
            compressed = {}
            if order_depths:
                for symbol, order_depth in order_depths.items():
                    if order_depth: # Check if order_depth object is not None
                        buy_orders_dict = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
                        sell_orders_dict = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
                        compressed[symbol] = [buy_orders_dict, sell_orders_dict]
                    else:
                         compressed[symbol] = [{}, {}] # Empty if order_depth is None
            return compressed

        def compress_trades(trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
            compressed = []
            if trades:
                for arr in trades.values():
                     if arr: # Check if list of trades is not None or empty
                        for trade in arr:
                             if trade: # Check if trade object is not None
                                compressed.append([
                                    trade.symbol, trade.price, trade.quantity,
                                    trade.buyer or "", trade.seller or "", trade.timestamp,
                                ])
            return compressed

        def compress_orders(orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
            compressed = []
            if orders:
                for arr in orders.values():
                    if arr:
                        for order in arr:
                            if order: # Check if order object is not None
                                compressed.append([order.symbol, order.price, order.quantity])
            return compressed

        def to_json(value: Any) -> str:
             try:
                 # Use ProsperityEncoder first if available in the environment
                 return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
             except NameError:
                 # Fallback to default json encoder
                 return json.dumps(value, separators=(",", ":"))
             except Exception as e_json:
                  # Log the specific error during JSON encoding for debugging
                  # Avoid printing directly here if it interferes with platform communication
                  # logger.print(f"JSON encoding error: {e_json}") # Log internally if needed
                  return "[]" # Return valid empty JSON on error


        def truncate(value: str, max_length: int) -> str:
            if not isinstance(value, str): value = str(value) # Ensure string type
            if len(value) <= max_length: return value
            return value[:max_length - 3] + "..."
        # --- End Compression Helpers ---

        # Main flushing logic
        try:
            # Create a minimal, valid state for base length calculation
            min_obs = Observation({}, {})
            min_state = TradingState(
                timestamp=state.timestamp, traderData="", listings={}, order_depths={},
                own_trades={}, market_trades={}, position={}, observations=min_obs
            )
            compressed_state_base = compress_state(min_state, "") # Compress with empty data
            compressed_orders_base = compress_orders({})
            base_value = [compressed_state_base, compressed_orders_base, conversions, "", ""]
            base_json = to_json(base_value) # Encode this minimal structure
            base_length = len(base_json)

            # Calculate max length for dynamic fields, ensure it's non-negative
            available_length = self.max_log_length - base_length
            available_length -= 200 # Reduce buffer slightly, ensure it fits
            if available_length < 0: available_length = 0
            max_item_length = available_length // 3

            # Truncate dynamic data *before* compression
            truncated_trader_data_state = truncate(state.traderData if state.traderData else "", max_item_length)
            truncated_trader_data_out = truncate(trader_data, max_item_length)
            truncated_logs = truncate(self.logs, max_item_length)

            # Prepare log entry with truncated data used in compression where needed
            log_entry = [
                compress_state(state, truncated_trader_data_state), # Pass truncated state.traderData here
                compress_orders(orders),
                conversions,
                truncated_trader_data_out, # Pass truncated outgoing trader_data
                truncated_logs,            # Pass truncated logs
            ]

            final_json_output = to_json(log_entry)
            # Ensure final output doesn't exceed limit (rare case, but possible if compression adds unexpected length)
            if len(final_json_output) > self.max_log_length:
                 # Extremely basic truncation if somehow still too long
                 final_json_output = final_json_output[:self.max_log_length-5] + "...]}"

            print(final_json_output) # Print the final JSON string

        except Exception as e:
            # Fallback logging if any error occurs during flush
            print(json.dumps({"error": f"Logging failed: {e}", "timestamp": state.timestamp}))

        self.logs = "" # Clear internal logs after flushing

# Instantiate the logger *after* the class definition
logger = Logger()

# --- Base Strategy Classes ---
class Strategy:
    # (Keep implementation as in the previous response)
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: List[Order] = []

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        try:
            self.act(state)
        except Exception as e:
            logger.print(f"ERROR running strategy for {self.symbol} at ts {state.timestamp}: {e}")
            # import traceback
            # logger.print(traceback.format_exc())
            self.orders = []
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
    # (Keep implementation as in the previous response, including EMA logic and save/load)
    DEFAULT_EMA_ALPHA = 0.2

    def __init__(self, symbol: Symbol, limit: int, ema_alpha: float = DEFAULT_EMA_ALPHA) -> None:
        super().__init__(symbol, limit)
        self.ema_alpha = ema_alpha
        self.ema_value: Optional[float] = None
        self.last_mid_price: Optional[float] = None

    def get_current_midprice(self, state: TradingState) -> Optional[float]:
        order_depth = state.order_depths.get(self.symbol);
        if not order_depth: return None
        bids = order_depth.buy_orders; asks = order_depth.sell_orders
        if not bids or not asks: return min(asks.keys()) if asks else (max(bids.keys()) if bids else None)
        best_ask=min(asks.keys()); best_bid=max(bids.keys()); best_ask_vol=abs(asks[best_ask]); best_bid_vol=bids[best_bid]
        if best_ask_vol + best_bid_vol == 0: return (best_ask + best_bid) / 2.0
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_fair_value(self, state: TradingState) -> Optional[float]:
        current_mid = self.get_current_midprice(state); self.last_mid_price = current_mid
        if current_mid is not None:
            if self.ema_value is None: self.ema_value = current_mid
            else: self.ema_value = self.ema_alpha * current_mid + (1 - self.ema_alpha) * self.ema_value
        return self.ema_value

    def act(self, state: TradingState) -> None: # Default MM logic
        fair_value_float = self.get_fair_value(state);
        if fair_value_float is None: return
        fair_value = int(round(fair_value_float)); ods=state.order_depths[self.symbol];buy_orders=sorted(ods.buy_orders.items(),reverse=True);sell_orders=sorted(ods.sell_orders.items())
        pos=state.position.get(self.symbol,0); buy_cap=self.limit-pos; sell_cap=self.limit+pos
        pass_buy_p=fair_value-1; pass_sell_p=fair_value+1; aggr_buy_thr=fair_value; aggr_sell_thr=fair_value
        inv_frac=pos/self.limit
        if inv_frac > 0.5: skew=int(round(inv_frac*2)); aggr_buy_thr-=skew; pass_buy_p-=skew
        elif inv_frac < -0.5: skew=int(round(abs(inv_frac)*2)); aggr_sell_thr+=skew; pass_sell_p+=skew
        taken_buy=0
        if sell_orders:
            for p,v_neg in sell_orders:
                if p <= aggr_buy_thr and buy_cap > 0: v=abs(v_neg); vol=min(v,buy_cap); self.buy(p,vol); buy_cap-=vol; taken_buy+=vol
                else: break
        taken_sell=0
        if buy_orders:
            for p,v in buy_orders:
                if p >= aggr_sell_thr and sell_cap > 0: vol=min(v,sell_cap); self.sell(p,vol); sell_cap-=vol; taken_sell+=vol
                else: break
        pass_size=5
        if buy_cap > 0 and taken_buy == 0: self.buy(pass_buy_p, min(buy_cap, pass_size))
        if sell_cap > 0 and taken_sell == 0: self.sell(pass_sell_p, min(sell_cap, pass_size))

    def save(self) -> JSON: return {"ema_value": self.ema_value, "last_mid_price": self.last_mid_price}
    def load(self, data: JSON) -> None:
        if isinstance(data,dict): self.ema_value=data.get("ema_value"); self.last_mid_price=data.get("last_mid_price")
        else: self.ema_value=None; self.last_mid_price=None


# --- Concrete Strategy Implementations ---

class RainforestResinStrategy(Strategy):
    # (Keep implementation as in the previous response)
    def __init__(self, symbol: Symbol, limit: int) -> None: super().__init__(symbol, limit); self.fair_value = 10000
    def act(self, state: TradingState) -> None:
        ods=state.order_depths.get(self.symbol,OrderDepth()); pos=state.position.get(self.symbol,0); buy_cap=self.limit-pos; sell_cap=self.limit+pos
        best_ask=min(ods.sell_orders.keys()) if ods.sell_orders else float('inf'); best_bid=max(ods.buy_orders.keys()) if ods.buy_orders else float('-inf')
        taken=False
        if best_ask < self.fair_value and buy_cap > 0: vol=min(buy_cap,abs(ods.sell_orders[best_ask])); self.buy(best_ask,vol); buy_cap-=vol; taken=True
        if best_bid > self.fair_value and sell_cap > 0: vol=min(sell_cap,ods.buy_orders[best_bid]); self.sell(best_bid,vol); sell_cap-=vol; taken=True
        if not taken:
            if buy_cap>0: self.buy(self.fair_value-1,min(buy_cap,5))
            if sell_cap>0: self.sell(self.fair_value+1,min(sell_cap,5))
    def save(self) -> JSON: return None
    def load(self, data: JSON) -> None: pass

class KelpStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None: super().__init__(symbol, limit, ema_alpha=0.3)

class SquidInkStrategyV5(MarketMakingStrategy):
    """More cautious: Wider spread(3), stronger deviation(5), trend filter, smaller size(aggr:2, pass:2)."""
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit, ema_alpha=0.1) # Slow EMA
        self.trend_ema_alpha_fast = 0.4
        self.trend_ema_fast: Optional[float] = None
        self.deviation_threshold = 5.0  # Increased further
        self.passive_spread = 3         # Keep wider spread
        self.aggr_spread_adj = 2
        self.passive_order_size = 2     # Reduced further
        self.aggressive_order_size = 2  # Reduced aggressive takes

    def get_fair_value(self, state: TradingState) -> Optional[float]: # Overridden to update fast EMA too
        current_mid = self.get_current_midprice(state)
        self.last_mid_price = current_mid
        if current_mid is not None:
            if self.ema_value is None: self.ema_value = current_mid
            else: self.ema_value = self.ema_alpha * current_mid + (1 - self.ema_alpha) * self.ema_value
            if self.trend_ema_fast is None: self.trend_ema_fast = current_mid
            else: self.trend_ema_fast = self.trend_ema_alpha_fast * current_mid + (1 - self.trend_ema_alpha_fast) * self.trend_ema_fast
        return self.ema_value # Return slow EMA

    def act(self, state: TradingState) -> None:
        fair_value_slow_ema = self.get_fair_value(state)
        if fair_value_slow_ema is None or self.trend_ema_fast is None or self.last_mid_price is None: return

        fair_value = int(round(fair_value_slow_ema))
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        buy_capacity = self.limit - position
        sell_capacity = self.limit + position

        aggr_buy_thresh=fair_value; aggr_sell_thresh=fair_value; passive_buy_price=fair_value-self.passive_spread; passive_sell_price=fair_value+self.passive_spread
        inv_frac=position/self.limit
        if inv_frac > 0.4: skew=int(round(inv_frac*self.aggr_spread_adj*1.5)); aggr_buy_thresh-=skew; passive_buy_price-=skew
        elif inv_frac < -0.4: skew=int(round(abs(inv_frac)*self.aggr_spread_adj*1.5)); aggr_sell_thresh+=skew; passive_sell_price+=skew

        trend = 0
        trend_diff_threshold = 0.5 # How much fast EMA needs to differ from slow EMA to confirm trend
        if self.trend_ema_fast > fair_value_slow_ema + trend_diff_threshold: trend = 1
        elif self.trend_ema_fast < fair_value_slow_ema - trend_diff_threshold: trend = -1

        # --- Aggressive Orders (Reduced Size) ---
        taken_buy = 0
        if sell_orders:
            best_ask, vol_neg = sell_orders[0]
            if best_ask <= aggr_buy_thresh and buy_capacity > 0:
                vol=min(abs(vol_neg), buy_capacity, self.aggressive_order_size); self.buy(best_ask, vol); buy_cap-=vol; taken_buy+=vol

        taken_sell = 0
        if buy_orders:
            best_bid, vol_pos = buy_orders[0]
            if best_bid >= aggr_sell_thresh and sell_capacity > 0:
                vol=min(vol_pos, sell_capacity, self.aggressive_order_size); self.sell(best_bid, vol); sell_cap-=vol; taken_sell+=vol

        # --- Passive Orders (Stricter Conditions & Reduced Size) ---
        deviation = abs(self.last_mid_price - fair_value_slow_ema)
        place_passive_buy = (buy_capacity > 0 and taken_buy == 0 and deviation >= self.deviation_threshold and trend >= 0)
        place_passive_sell = (sell_capacity > 0 and taken_sell == 0 and deviation >= self.deviation_threshold and trend <= 0)

        # Additional check: Don't place passive buy if it's above the fast EMA (fighting momentum)
        # Don't place passive sell if it's below the fast EMA
        if place_passive_buy and passive_buy_price >= self.trend_ema_fast + 1: # +1 buffer
             place_passive_buy = False
             # logger.print(f"Blocking Passive BUY {self.symbol} due to fast EMA ({self.trend_ema_fast:.1f})")

        if place_passive_sell and passive_sell_price <= self.trend_ema_fast - 1: # -1 buffer
             place_passive_sell = False
             # logger.print(f"Blocking Passive SELL {self.symbol} due to fast EMA ({self.trend_ema_fast:.1f})")


        if place_passive_buy: self.buy(passive_buy_price, min(buy_capacity, self.passive_order_size))
        if place_passive_sell: self.sell(passive_sell_price, min(sell_capacity, self.passive_order_size))

    # Save/Load includes fast EMA state
    def save(self) -> JSON: return {"ema_value":self.ema_value, "last_mid_price":self.last_mid_price, "trend_ema_fast":self.trend_ema_fast}
    def load(self, data: JSON) -> None:
        super().load(data); self.trend_ema_fast = data.get("trend_ema_fast", None) if isinstance(data, dict) else None

# --- Main Trader Class ---
class Trader:
    # (Keep implementation as in the previous response, ensuring it uses SquidInkStrategyV5)
    def __init__(self) -> None:
        self.limits = { "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50 }
        self.strategy_map = {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidInkStrategyV5, # Use the latest cautious version
        }
        self.strategies: Dict[Symbol, Strategy] = {}
        for symbol, limit in self.limits.items():
            if symbol in self.strategy_map:
                self.strategies[symbol] = self.strategy_map[symbol](symbol, limit)
            else: logger.print(f"WARNING: No strategy defined for symbol {symbol}")
        self.initial_load_done = False

    def load_trader_data(self, traderData: str):
        if not traderData: return
        try:
            saved_states = jsonpickle.decode(traderData) # Make sure jsonpickle is imported
            if not isinstance(saved_states, dict): saved_states = {}
            for symbol, strategy in self.strategies.items():
                strategy.load(saved_states.get(symbol, None)) # Pass None if symbol not in saved_states
        except Exception as e:
            logger.print(f"ERROR decoding traderData: {e}")
            for strategy in self.strategies.values(): strategy.load(None)

    def save_trader_data(self) -> str:
        states_to_save = {symbol: strategy.save() for symbol, strategy in self.strategies.items()}
        try:
            return jsonpickle.encode(states_to_save, unpicklable=False) # Ensure jsonpickle is imported
        except Exception as e:
            logger.print(f"ERROR encoding traderData: {e}")
            return ""

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        if not self.initial_load_done or not state.traderData:
             self.load_trader_data(state.traderData)
             self.initial_load_done = True
        all_orders: Dict[Symbol, List[Order]] = {symbol: [] for symbol in self.limits} # Initialize for all potential symbols
        conversions = 0
        for symbol, strategy in self.strategies.items():
            if symbol in state.order_depths: # Check if data exists
                all_orders[symbol] = strategy.run(state)
        traderData_output = self.save_trader_data()
        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output