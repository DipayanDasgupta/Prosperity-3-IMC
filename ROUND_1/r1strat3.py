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

# Logger class (Ensure this definition is complete and correct)
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # --- Full Compression Logic (from your original template) ---
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
            # Modifications to handle potentially missing conversion observation fields
            compressed_conversion_observations = {}
            if hasattr(state.observations, 'conversionObservations') and state.observations.conversionObservations:
                 for product, observation in state.observations.conversionObservations.items():
                     # Provide defaults (e.g., 0 or None) if attributes might be missing
                     compressed_conversion_observations[product] = [
                         getattr(observation, 'bidPrice', None),
                         getattr(observation, 'askPrice', None),
                         getattr(observation, 'transportFees', 0.0), # Assuming 0 if missing
                         getattr(observation, 'exportTariff', 0.0),  # Assuming 0 if missing
                         getattr(observation, 'importTariff', 0.0),  # Assuming 0 if missing
                         getattr(observation, 'sunlight', None),      # Keep None if missing
                         getattr(observation, 'humidity', None),      # Keep None if missing
                     ]
            compressed_observations = [
                state.observations.plainValueObservations if hasattr(state.observations, 'plainValueObservations') else {},
                compressed_conversion_observations
            ]

            return [
                state.timestamp,
                trader_data,
                compress_listings(state.listings),
                compress_order_depths(state.order_depths),
                compress_trades(state.own_trades),
                compress_trades(state.market_trades),
                state.position,
                compressed_observations,
            ]

        def compress_listings(listings: dict[Symbol, Listing]) -> list[list[Any]]:
            compressed = []
            if listings:
                for listing in listings.values():
                    compressed.append([listing.symbol, listing.product, listing.denomination])
            return compressed

        def compress_order_depths(order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
            compressed = {}
            if order_depths:
                for symbol, order_depth in order_depths.items():
                    # Ensure buy_orders and sell_orders are dicts
                    buy_orders_dict = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
                    sell_orders_dict = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
                    compressed[symbol] = [buy_orders_dict, sell_orders_dict]
            return compressed

        def compress_trades(trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
            compressed = []
            if trades:
                for arr in trades.values():
                     if arr: # Check if list is not empty
                        for trade in arr:
                            compressed.append([
                                trade.symbol,
                                trade.price,
                                trade.quantity,
                                trade.buyer or "", # Use "" if None
                                trade.seller or "", # Use "" if None
                                trade.timestamp,
                            ])
            return compressed

        def compress_orders(orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
            compressed = []
            if orders:
                for arr in orders.values():
                    if arr:
                        for order in arr:
                            compressed.append([order.symbol, order.price, order.quantity])
            return compressed

        def to_json(value: Any) -> str:
             try:
                 # ProsperityEncoder should handle the specific data types if needed
                 return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
             except NameError:
                  # Fallback to default JSON encoder if ProsperityEncoder isn't available
                 return json.dumps(value, separators=(",", ":"))
             except Exception as e_json:
                  print(f"JSON encoding error: {e_json}")
                  return "[]" # Return valid empty JSON on error


        def truncate(value: str, max_length: int) -> str:
            if not isinstance(value, str): value = str(value)
            if len(value) <= max_length: return value
            return value[:max_length - 3] + "..."
        # --- End Compression Helpers ---

        try:
            # Calculate base length
            # Create a minimal state for accurate base length calculation
            min_state = TradingState(
                timestamp=state.timestamp, traderData="", listings={}, order_depths={},
                own_trades={}, market_trades={}, position={},
                observations=Observation({}, {}) # Empty observations
            )
            compressed_state_base = compress_state(min_state, "")
            compressed_orders_base = compress_orders({})
            base_value = [compressed_state_base, compressed_orders_base, conversions, "", ""]
            base_json = to_json(base_value)
            base_length = len(base_json)

            # Calculate max length for dynamic fields
            available_length = self.max_log_length - base_length
            # Subtract some buffer for JSON overhead (brackets, commas, quotes)
            available_length -= 200 # Increased buffer
            if available_length < 0: available_length = 0
            max_item_length = available_length // 3 # Divide remaining space

            # Prepare log entry with truncated data
            log_entry = [
                compress_state(state, truncate(state.traderData if state.traderData else "", max_item_length)),
                compress_orders(orders),
                conversions,
                truncate(trader_data, max_item_length),
                truncate(self.logs, max_item_length),
            ]

            print(to_json(log_entry))

        except Exception as e:
            print(f"Error during logging/compression: {e}")
            # Fallback logging
            print(json.dumps({"error": "Logging failed", "timestamp": state.timestamp}))

        self.logs = "" # Clear logs

# Instantiate the logger *after* the class definition
logger = Logger()

# --- Base Strategy Classes ---
class Strategy:
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
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def buy(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        self.orders.append(Order(self.symbol, int(round(price)), int(round(quantity))))

    def sell(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        self.orders.append(Order(self.symbol, int(round(price)), -int(round(quantity))))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class MarketMakingStrategy(Strategy):
    DEFAULT_WINDOW_SIZE = 10
    DEFAULT_EMA_ALPHA = 0.2

    def __init__(self, symbol: Symbol, limit: int, ema_alpha: float = DEFAULT_EMA_ALPHA) -> None:
        super().__init__(symbol, limit)
        self.ema_alpha = ema_alpha
        self.ema_value: Optional[float] = None
        self.last_mid_price: Optional[float] = None
        # Window no longer used in core logic, removed from state
        # self.window = deque(maxlen=self.DEFAULT_WINDOW_SIZE)


    def get_current_midprice(self, state: TradingState) -> Optional[float]:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return None
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        if not bids or not asks:
            if asks: return min(asks.keys())
            if bids: return max(bids.keys())
            return None

        best_ask = min(asks.keys())
        best_bid = max(bids.keys())
        best_ask_vol = abs(asks[best_ask])
        best_bid_vol = bids[best_bid]

        if best_ask_vol + best_bid_vol == 0: return (best_ask + best_bid) / 2.0
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_fair_value(self, state: TradingState) -> Optional[float]:
        current_mid = self.get_current_midprice(state)
        self.last_mid_price = current_mid
        if current_mid is not None:
            if self.ema_value is None: self.ema_value = current_mid
            else: self.ema_value = self.ema_alpha * current_mid + (1 - self.ema_alpha) * self.ema_value
        return self.ema_value

    def act(self, state: TradingState) -> None:
        # Default MM logic (used by Kelp unless overridden)
        fair_value_float = self.get_fair_value(state)
        if fair_value_float is None: return
        fair_value = int(round(fair_value_float))

        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        buy_capacity = self.limit - position
        sell_capacity = self.limit + position

        passive_buy_price = fair_value - 1
        passive_sell_price = fair_value + 1
        aggression_buy_threshold = fair_value
        aggression_sell_threshold = fair_value

        # Inventory skewing
        inventory_fraction = position / self.limit
        if inventory_fraction > 0.5:
             skew = int(round(inventory_fraction * 2)) # Skew more aggressively
             aggression_buy_threshold -= skew
             passive_buy_price -= skew
        elif inventory_fraction < -0.5:
             skew = int(round(abs(inventory_fraction) * 2)) # Skew more aggressively
             aggression_sell_threshold += skew
             passive_sell_price += skew

        # Aggressive orders
        taken_buy_volume = 0
        if sell_orders:
            for price, volume_neg in sell_orders:
                if price <= aggression_buy_threshold and buy_capacity > 0:
                    volume = abs(volume_neg); vol_to_take = min(volume, buy_capacity)
                    self.buy(price, vol_to_take); buy_capacity -= vol_to_take; taken_buy_volume += vol_to_take
                else: break

        taken_sell_volume = 0
        if buy_orders:
            for price, volume in buy_orders:
                if price >= aggression_sell_threshold and sell_capacity > 0:
                    vol_to_take = min(volume, sell_capacity)
                    self.sell(price, vol_to_take); sell_capacity -= vol_to_take; taken_sell_volume += vol_to_take
                else: break

        # Passive orders
        passive_order_size = 5
        if buy_capacity > 0 and taken_buy_volume == 0:
            self.buy(passive_buy_price, min(buy_capacity, passive_order_size))
        if sell_capacity > 0 and taken_sell_volume == 0:
            self.sell(passive_sell_price, min(sell_capacity, passive_order_size))

    # Simplified save/load just for EMA state
    def save(self) -> JSON:
        return {"ema_value": self.ema_value, "last_mid_price": self.last_mid_price}

    def load(self, data: JSON) -> None:
        if isinstance(data, dict):
            self.ema_value = data.get("ema_value", None)
            self.last_mid_price = data.get("last_mid_price", None)
        else: # Reset if data format is wrong
            self.ema_value = None
            self.last_mid_price = None

# --- Concrete Strategy Implementations ---

class RainforestResinStrategy(Strategy): # Simple fixed-price strategy
    def __init__(self, symbol: Symbol, limit: int) -> None:
         super().__init__(symbol, limit)
         self.fixed_fair_value = 10000

    def act(self, state: TradingState) -> None:
        """Simple strategy: buy below fair, sell above. More cautious passive placement."""
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        position = state.position.get(self.symbol, 0)
        buy_capacity = self.limit - position
        sell_capacity = self.limit + position

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else float('-inf')

        # Aggressive take if price crosses fair value
        taken_aggressively = False
        if best_ask < self.fixed_fair_value and buy_capacity > 0:
             vol_to_take = min(buy_capacity, abs(order_depth.sell_orders[best_ask]))
             self.buy(best_ask, vol_to_take)
             buy_capacity -= vol_to_take
             taken_aggressively = True

        if best_bid > self.fixed_fair_value and sell_capacity > 0:
             vol_to_take = min(sell_capacity, order_depth.buy_orders[best_bid])
             self.sell(best_bid, vol_to_take)
             sell_capacity -= vol_to_take
             taken_aggressively = True

        # Reduced passive quoting if not taken aggressively
        passive_size = 5
        # Only place passive orders if NOT taken aggressively this round
        if not taken_aggressively:
            if buy_capacity > 0:
                 self.buy(self.fixed_fair_value - 1, min(buy_capacity, passive_size))
            if sell_capacity > 0:
                 self.sell(self.fixed_fair_value + 1, min(sell_capacity, passive_size))

    # No state needed
    def save(self) -> JSON: return None
    def load(self, data: JSON) -> None: pass

class KelpStrategy(MarketMakingStrategy):
    # Uses default EMA Market Making from parent
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit, ema_alpha=0.3) # Slightly faster EMA


class SquidInkStrategyV4(MarketMakingStrategy):
    """Even more cautious: Wider spread, stronger deviation, trend filter, smaller size."""
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit, ema_alpha=0.1) # Slow EMA for stable reference
        self.deviation_threshold = 4.0  # Increased threshold
        self.passive_spread = 3         # Wider spread
        self.aggr_spread_adj = 2        # Wider skew for aggression based on inventory
        self.passive_order_size = 3     # Smaller size
        self.trend_ema_alpha_fast = 0.4 # Faster EMA to detect short-term trend
        self.trend_ema_fast: Optional[float] = None

    # Override get_fair_value to also calculate fast EMA for trend
    def get_fair_value(self, state: TradingState) -> Optional[float]:
        current_mid = self.get_current_midprice(state)
        self.last_mid_price = current_mid

        if current_mid is not None:
            # Update slow EMA (self.ema_value)
            if self.ema_value is None: self.ema_value = current_mid
            else: self.ema_value = self.ema_alpha * current_mid + (1 - self.ema_alpha) * self.ema_value

            # Update fast EMA (self.trend_ema_fast)
            if self.trend_ema_fast is None: self.trend_ema_fast = current_mid
            else: self.trend_ema_fast = self.trend_ema_alpha_fast * current_mid + (1 - self.trend_ema_alpha_fast) * self.trend_ema_fast

        return self.ema_value # Return the slow EMA as the 'fair value' reference

    def act(self, state: TradingState) -> None:
        fair_value_slow_ema = self.get_fair_value(state) # This also updates fast EMA and last_mid_price
        if fair_value_slow_ema is None or self.trend_ema_fast is None: return

        fair_value = int(round(fair_value_slow_ema))

        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        buy_capacity = self.limit - position
        sell_capacity = self.limit + position

        # --- Price Levels & Aggression ---
        aggression_buy_threshold = fair_value
        aggression_sell_threshold = fair_value
        passive_buy_price = fair_value - self.passive_spread
        passive_sell_price = fair_value + self.passive_spread

        # Stronger inventory skew
        inventory_fraction = position / self.limit
        if inventory_fraction > 0.4: # Start skewing earlier
             skew = int(round(inventory_fraction * self.aggr_spread_adj * 1.5)) # Apply stronger skew
             aggression_buy_threshold -= skew
             passive_buy_price -= skew
             # logger.print(f"SQUID_INK Long Skew: {skew}")
        elif inventory_fraction < -0.4:
             skew = int(round(abs(inventory_fraction) * self.aggr_spread_adj * 1.5))
             aggression_sell_threshold += skew
             passive_sell_price += skew
             # logger.print(f"SQUID_INK Short Skew: {skew}")

        # --- Trend Filter ---
        # Trend is up if fast EMA is above slow EMA, down if below
        trend = 0
        if self.trend_ema_fast > fair_value_slow_ema + 0.5: trend = 1   # Uptrend
        elif self.trend_ema_fast < fair_value_slow_ema - 0.5: trend = -1 # Downtrend

        # --- Aggressive Orders ---
        # Still take clear edges, respecting skewed thresholds
        taken_buy_volume = 0
        if sell_orders:
            best_ask = sell_orders[0][0]
            if best_ask <= aggression_buy_threshold and buy_capacity > 0:
                volume = abs(sell_orders[0][1]); vol_to_take = min(volume, buy_capacity, 3) # Limit aggressive take size too
                self.buy(best_ask, vol_to_take); buy_capacity -= vol_to_take; taken_buy_volume += vol_to_take

        taken_sell_volume = 0
        if buy_orders:
            best_bid = buy_orders[0][0]
            if best_bid >= aggression_sell_threshold and sell_capacity > 0:
                volume = buy_orders[0][1]; vol_to_take = min(volume, sell_capacity, 3) # Limit aggressive take size too
                self.sell(best_bid, vol_to_take); sell_capacity -= vol_to_take; taken_sell_volume += vol_to_take


        # --- Passive Orders (Stricter Conditions) ---
        # Place only if:
        # 1. Didn't take aggressive on that side
        # 2. Price deviates enough from *slow* EMA (mean reversion signal)
        # 3. Don't place against the short-term trend (fast EMA vs slow EMA)
        deviation = abs(self.last_mid_price - fair_value_slow_ema) if self.last_mid_price is not None else 0.0
        place_passive_buy = (buy_capacity > 0 and taken_buy_volume == 0 and
                             deviation >= self.deviation_threshold and trend >= 0) # Allow buy in neutral or uptrend
        place_passive_sell = (sell_capacity > 0 and taken_sell_volume == 0 and
                              deviation >= self.deviation_threshold and trend <= 0) # Allow sell in neutral or downtrend

        if place_passive_buy:
            self.buy(passive_buy_price, min(buy_capacity, self.passive_order_size))
            # logger.print(f"Placing Passive BUY {self.symbol} (Dev: {deviation:.1f}, Trend: {trend})")

        if place_passive_sell:
            self.sell(passive_sell_price, min(sell_capacity, self.passive_order_size))
            # logger.print(f"Placing Passive SELL {self.symbol} (Dev: {deviation:.1f}, Trend: {trend})")

    # Save both slow and fast EMA states
    def save(self) -> JSON:
        return {
            "ema_value": self.ema_value,
            "last_mid_price": self.last_mid_price,
            "trend_ema_fast": self.trend_ema_fast
        }

    def load(self, data: JSON) -> None:
        super().load(data) # Load ema_value and last_mid_price using parent
        if isinstance(data, dict):
             self.trend_ema_fast = data.get("trend_ema_fast", None)
        else:
             self.trend_ema_fast = None


# --- Main Trader Class ---
class Trader:
    def __init__(self) -> None:
        self.limits = { "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50 }
        self.strategy_map = {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidInkStrategyV4, # Use the latest cautious version
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
                if symbol in saved_states and saved_states[symbol] is not None:
                    try: strategy.load(saved_states[symbol])
                    except Exception as e: logger.print(f"ERROR loading state for {symbol}: {e}")
                else: strategy.load(None)
        except Exception as e:
            logger.print(f"ERROR decoding traderData: {e}")
            for strategy in self.strategies.values(): strategy.load(None)

    def save_trader_data(self) -> str:
        states_to_save = {}
        for symbol, strategy in self.strategies.items():
             try: states_to_save[symbol] = strategy.save()
             except Exception as e: logger.print(f"Error saving state for {symbol}: {e}"); states_to_save[symbol] = None
        try:
            return jsonpickle.encode(states_to_save, unpicklable=False) # Ensure jsonpickle is imported
        except Exception as e:
            logger.print(f"ERROR encoding traderData: {e}")
            return ""

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        if not self.initial_load_done or not state.traderData:
             self.load_trader_data(state.traderData)
             self.initial_load_done = True

        all_orders: Dict[Symbol, List[Order]] = {}
        conversions = 0
        for symbol, strategy in self.strategies.items():
            if symbol in state.order_depths:
                all_orders[symbol] = strategy.run(state)
            else:
                all_orders[symbol] = []

        traderData_output = self.save_trader_data()
        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output