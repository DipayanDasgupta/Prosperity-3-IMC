import json
import math
from collections import deque, defaultdict # Added defaultdict
from typing import Any, TypeAlias, List, Dict, Optional, Tuple
from abc import abstractmethod

# Assuming datamodel classes are available as in the competition environment
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# Type Alias for JSON data
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# Logger class (as provided) - Ensure it's included in your final file
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        # Simple print to stdout for local testing/backtesting redirection
        # Platform logger truncates, but for local, full logs are fine.
        print(sep.join(map(str, objects)), end=end)
        # Store logs internally if needed
        # self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Simplified compression for flushing (actual compression logic removed for clarity)
        # You should use the full compression logic from your previous working version here
        # for platform compatibility. This is a placeholder.
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
             # Basic representation, replace with your full compression
             compressed_observations = [state.observations.plainValueObservations, {}] # Placeholder
             return [state.timestamp, trader_data, {}, {}, {}, {}, state.position, compressed_observations]
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
                 return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
             except NameError:
                 return json.dumps(value, separators=(",", ":"))
        def truncate(value: str, max_length: int) -> str:
            if not isinstance(value, str): value = str(value)
            if len(value) <= max_length: return value
            return value[:max_length - 3] + "..."

        log_entry = [
            compress_state(state, truncate(state.traderData if state.traderData else "", 1000)), # Example truncation
            compress_orders(orders),
            conversions,
            truncate(trader_data, 1000), # Example truncation
            truncate(self.logs, 1000),    # Example truncation
        ]
        print(to_json(log_entry))
        self.logs = "" # Clear logs

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
            # logger.print(traceback.format_exc()) # Uncomment for detailed debug
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
    DEFAULT_EMA_ALPHA = 0.2 # Default smoothing factor

    def __init__(self, symbol: Symbol, limit: int, ema_alpha: float = DEFAULT_EMA_ALPHA) -> None:
        super().__init__(symbol, limit)
        self.window = deque(maxlen=self.DEFAULT_WINDOW_SIZE)
        self.ema_alpha = ema_alpha
        self.ema_value: Optional[float] = None
        self.last_mid_price: Optional[float] = None # Store the raw mid-price calculation

    def get_current_midprice(self, state: TradingState) -> Optional[float]:
        """Calculates the current midprice (weighted or simple)."""
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth: return None

        bids = order_depth.buy_orders
        asks = order_depth.sell_orders

        if not bids or not asks:
            if asks: return min(asks.keys())
            if bids: return max(bids.keys())
            return None # No price if book is empty

        best_ask = min(asks.keys())
        best_bid = max(bids.keys())
        best_ask_vol = abs(asks[best_ask])
        best_bid_vol = bids[best_bid]

        if best_ask_vol + best_bid_vol == 0:
            return (best_ask + best_bid) / 2.0
        else:
            return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_fair_value(self, state: TradingState) -> Optional[float]:
        """Calculates the smoothed fair value using EMA."""
        current_mid = self.get_current_midprice(state)
        self.last_mid_price = current_mid # Store raw midprice

        if current_mid is not None:
            if self.ema_value is None:
                self.ema_value = current_mid # Initialize EMA
            else:
                self.ema_value = self.ema_alpha * current_mid + (1 - self.ema_alpha) * self.ema_value
        # If current_mid is None, we just return the existing ema_value without updating it

        return self.ema_value

    def act(self, state: TradingState) -> None:
        """Default Market Making Logic (can be overridden by subclasses)."""
        fair_value_float = self.get_fair_value(state)
        if fair_value_float is None: return

        fair_value = int(round(fair_value_float))

        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        buy_capacity = self.limit - position
        sell_capacity = self.limit + position

        # Default price levels
        passive_buy_price = fair_value - 1
        passive_sell_price = fair_value + 1
        aggression_buy_threshold = fair_value
        aggression_sell_threshold = fair_value

        # Inventory skewing adjustment
        if position > self.limit * 0.5:
            aggression_buy_threshold -= 1
            passive_buy_price -= 1
        elif position < -self.limit * 0.5:
            aggression_sell_threshold += 1
            passive_sell_price += 1

        # Aggressive orders
        taken_buy_volume = 0
        if sell_orders:
            for price, volume_neg in sell_orders:
                if price <= aggression_buy_threshold and buy_capacity > 0:
                    volume = abs(volume_neg)
                    vol_to_take = min(volume, buy_capacity)
                    self.buy(price, vol_to_take)
                    buy_capacity -= vol_to_take
                    taken_buy_volume += vol_to_take
                else: break

        taken_sell_volume = 0
        if buy_orders:
            for price, volume in buy_orders:
                if price >= aggression_sell_threshold and sell_capacity > 0:
                    vol_to_take = min(volume, sell_capacity)
                    self.sell(price, vol_to_take)
                    sell_capacity -= vol_to_take
                    taken_sell_volume += vol_to_take
                else: break

        # Passive orders
        passive_order_size = 5
        if buy_capacity > 0 and taken_buy_volume == 0: # Only place if didn't take aggressively
            self.buy(passive_buy_price, min(buy_capacity, passive_order_size))

        if sell_capacity > 0 and taken_sell_volume == 0: # Only place if didn't take aggressively
            self.sell(passive_sell_price, min(sell_capacity, passive_order_size))

    def save(self) -> JSON:
        return {"ema_value": self.ema_value, "last_mid_price": self.last_mid_price}

    def load(self, data: JSON) -> None:
        if isinstance(data, dict):
            self.ema_value = data.get("ema_value", None)
            self.last_mid_price = data.get("last_mid_price", None)
        else:
            self.ema_value = None
            self.last_mid_price = None


# --- Concrete Strategy Implementations ---

class RainforestResinStrategy(Strategy): # Doesn't need market making base
    def __init__(self, symbol: Symbol, limit: int) -> None:
         super().__init__(symbol, limit)
         self.fixed_fair_value = 10000

    def act(self, state: TradingState) -> None:
        """Simple strategy: buy below fair, sell above."""
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        position = state.position.get(self.symbol, 0)
        buy_capacity = self.limit - position
        sell_capacity = self.limit + position

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else float('-inf')

        # Aggressive take
        if best_ask < self.fixed_fair_value and buy_capacity > 0:
             vol_to_take = min(buy_capacity, abs(order_depth.sell_orders[best_ask]))
             self.buy(best_ask, vol_to_take)
             buy_capacity -= vol_to_take # Reduce capacity *within* this tick

        if best_bid > self.fixed_fair_value and sell_capacity > 0:
             vol_to_take = min(sell_capacity, order_depth.buy_orders[best_bid])
             self.sell(best_bid, vol_to_take)
             sell_capacity -= vol_to_take # Reduce capacity *within* this tick

        # Passive orders
        passive_size = 5 # Example size
        if buy_capacity > 0:
             self.buy(self.fixed_fair_value - 1, min(buy_capacity, passive_size))
        if sell_capacity > 0:
             self.sell(self.fixed_fair_value + 1, min(sell_capacity, passive_size))

    # No state needed to save/load for this simple version
    def save(self) -> JSON: return None
    def load(self, data: JSON) -> None: pass


class KelpStrategy(MarketMakingStrategy):
    # Uses the default EMA-based fair value and market making logic
    pass


class SquidInkStrategy(MarketMakingStrategy):
    # Override act method for specific Squid Ink logic
    def act(self, state: TradingState) -> None:
        fair_value_float = self.get_fair_value(state) # Gets smoothed EMA value
        if fair_value_float is None: return

        fair_value = int(round(fair_value_float))
        current_mid_price = self.last_mid_price # Get the raw midprice from the last calculation

        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        buy_capacity = self.limit - position
        sell_capacity = self.limit + position

        # Price levels based on smoothed fair value
        passive_buy_price = fair_value - 1
        passive_sell_price = fair_value + 1
        aggression_buy_threshold = fair_value
        aggression_sell_threshold = fair_value

        # Inventory skewing
        if position > self.limit * 0.5:
            aggression_buy_threshold -= 1
            passive_buy_price -= 1
        elif position < -self.limit * 0.5:
            aggression_sell_threshold += 1
            passive_sell_price += 1

        # Trend indication (simple: compare current mid to EMA)
        trend = 0 # 0 = neutral/no trend, 1 = up, -1 = down
        if current_mid_price is not None and self.ema_value is not None:
            if current_mid_price > self.ema_value + 0.5 : # Threshold to detect trend
                trend = 1
            elif current_mid_price < self.ema_value - 0.5:
                trend = -1

        # Aggressive orders (always try to take edge)
        taken_buy_volume = 0
        if sell_orders:
            best_ask = sell_orders[0][0]
            if best_ask <= aggression_buy_threshold and buy_capacity > 0:
                volume = abs(sell_orders[0][1])
                vol_to_take = min(volume, buy_capacity)
                self.buy(best_ask, vol_to_take)
                buy_capacity -= vol_to_take
                taken_buy_volume += vol_to_take

        taken_sell_volume = 0
        if buy_orders:
            best_bid = buy_orders[0][0]
            if best_bid >= aggression_sell_threshold and sell_capacity > 0:
                 volume = buy_orders[0][1]
                 vol_to_take = min(volume, sell_capacity)
                 self.sell(best_bid, vol_to_take)
                 sell_capacity -= vol_to_take
                 taken_sell_volume += vol_to_take

        # Passive orders - Conditioned by trend
        passive_order_size = 5

        # Only place passive buy if NOT in clear downtrend AND capacity allows
        if trend >= 0 and buy_capacity > 0 and taken_buy_volume == 0 :
            self.buy(passive_buy_price, min(buy_capacity, passive_order_size))
            # logger.print(f"  Placing Passive BUY {self.symbol} (Trend {trend})")

        # Only place passive sell if NOT in clear uptrend AND capacity allows
        if trend <= 0 and sell_capacity > 0 and taken_sell_volume == 0 :
            self.sell(passive_sell_price, min(sell_capacity, passive_order_size))
            # logger.print(f"  Placing Passive SELL {self.symbol} (Trend {trend})")


# --- Main Trader Class ---
class Trader:
    def __init__(self) -> None:
        """Initialize the Trader class with strategies for each product."""
        #logger.print("Initializing Trader...") # Reduced verbosity

        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
        }

        self.strategy_map = {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidInkStrategy,
        }

        self.strategies: Dict[Symbol, Strategy] = {}
        for symbol, limit in self.limits.items():
            if symbol in self.strategy_map:
                self.strategies[symbol] = self.strategy_map[symbol](symbol, limit)
            else:
                logger.print(f"WARNING: No strategy defined for symbol {symbol}")

        self.initial_load_done = False

    def load_trader_data(self, traderData: str):
        if not traderData: return
        try:
            saved_states = jsonpickle.decode(traderData)
            if not isinstance(saved_states, dict): saved_states = {}
            # logger.print(f"Loading traderData...")
            for symbol, strategy in self.strategies.items():
                if symbol in saved_states and saved_states[symbol] is not None:
                    try:
                        strategy.load(saved_states[symbol])
                    except Exception as e:
                        logger.print(f"ERROR loading state for {symbol}: {e}")
                else:
                    strategy.load(None) # Ensure reset if no data
        except Exception as e:
            logger.print(f"ERROR decoding traderData string: {e}")
            for strategy in self.strategies.values(): strategy.load(None)

    def save_trader_data(self) -> str:
        states_to_save = {}
        for symbol, strategy in self.strategies.items():
             try:
                 states_to_save[symbol] = strategy.save()
             except Exception as e:
                  logger.print(f"Error saving state for {symbol}: {e}")
                  states_to_save[symbol] = None
        try:
            return jsonpickle.encode(states_to_save, unpicklable=False)
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

        # Ensure the logger object instance is used
        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output