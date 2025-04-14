import json
import math
from collections import deque, defaultdict # Added defaultdict here just in case, though not strictly needed by the final code logic shown
from typing import Any, TypeAlias, List, Dict, Optional, Tuple # Added Tuple here
from abc import abstractmethod # Required for abstract Strategy class

# Assuming datamodel classes are available as in the competition environment
# Removed duplicate imports
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# Type Alias for JSON data
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# Logger class (as provided in your template)
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        # Simple print to stdout for local testing/backtesting redirection
        # The platform logger truncates, but for local testing, full logs are fine.
        print(sep.join(map(str, objects)), end=end)
        # Store logs internally if needed, though the provided flush method handles printing
        # self.logs += sep.join(map(str, objects)) + end # Optional: Keep internal log

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # This method is primarily for the competition platform's structured logging.
        # When running locally or with a basic backtester, you might just print logs directly.
        # However, we keep the structure for compatibility.

        # --- Helper methods for compression ---
        # (Copied from user's provided Logger class in the prompt)
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
            # Simplified observation compression for this example
            compressed_observations = [state.observations.plainValueObservations, {}]
            if hasattr(state.observations, 'conversionObservations'):
                 # Basic check if conversionObservations exists
                 conversion_observations_compressed = {}
                 for product, observation in state.observations.conversionObservations.items():
                    # Attempt to access attributes, provide defaults if missing
                    conversion_observations_compressed[product] = [
                        getattr(observation, 'bidPrice', None),
                        getattr(observation, 'askPrice', None),
                        getattr(observation, 'transportFees', None),
                        getattr(observation, 'exportTariff', None),
                        getattr(observation, 'importTariff', None),
                        getattr(observation, 'sunlight', None), # Check exact names if needed
                        getattr(observation, 'humidity', None), # Check exact names if needed
                    ]
                 compressed_observations = [state.observations.plainValueObservations, conversion_observations_compressed]


            return [
                state.timestamp,
                trader_data,
                compress_listings(state.listings),
                compress_order_depths(state.order_depths),
                compress_trades(state.own_trades),
                compress_trades(state.market_trades),
                state.position,
                compressed_observations, # Use compressed observations
            ]

        def compress_listings(listings: dict[Symbol, Listing]) -> list[list[Any]]:
            compressed = []
            for listing in listings.values():
                compressed.append([listing.symbol, listing.product, listing.denomination])
            return compressed

        def compress_order_depths(order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
            compressed = {}
            for symbol, order_depth in order_depths.items():
                compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
            return compressed

        def compress_trades(trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
            compressed = []
            if trades: # Check if trades is not None or empty
                for arr in trades.values():
                    for trade in arr:
                        compressed.append([
                            trade.symbol,
                            trade.price,
                            trade.quantity,
                            trade.buyer,
                            trade.seller,
                            trade.timestamp,
                        ])
            return compressed

        def compress_orders(orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
            compressed = []
            if orders: # Check if orders is not None or empty
                for arr in orders.values():
                     if arr: # Check if list of orders is not empty
                        for order in arr:
                            compressed.append([order.symbol, order.price, order.quantity])
            return compressed

        def to_json(value: Any) -> str:
             # Attempt to use ProsperityEncoder if available, otherwise default json
             try:
                 return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
             except NameError: # Handle if ProsperityEncoder wasn't defined/imported
                 return json.dumps(value, separators=(",", ":"))


        def truncate(value: str, max_length: int) -> str:
            if not isinstance(value, str): # Ensure value is a string
                value = str(value)
            if len(value) <= max_length:
                return value
            return value[:max_length - 3] + "..."

        # --- Flushing logic ---
        try:
            # Calculate base length carefully
            compressed_state_empty_trader = compress_state(state, "")
            compressed_orders_empty = compress_orders({})
            base_value = [
                compressed_state_empty_trader,
                compressed_orders_empty,
                conversions,
                "",
                "",
            ]
            base_json = to_json(base_value)
            base_length = len(base_json)

            # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
            # Added buffer for safety margin
            max_item_length = max(0, (self.max_log_length - base_length - 50) // 3)


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
            # Fallback or simplified logging if compression fails
            print(json.dumps({"error": "Logging failed", "timestamp": state.timestamp}))


        self.logs = "" # Clear logs after flushing

logger = Logger()


# --- Base Strategy Classes ---
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: List[Order] = [] # Orders generated in this step

    def run(self, state: TradingState) -> list[Order]:
        self.orders = [] # Clear orders at the start of each run
        try:
            self.act(state)
        except Exception as e:
            logger.print(f"ERROR running strategy for {self.symbol}: {e}")
            # Optionally add more detailed traceback logging
            # import traceback
            # logger.print(traceback.format_exc())
            self.orders = [] # Ensure no orders are placed if an error occurs
        return self.orders

    @abstractmethod
    def act(self, state: TradingState) -> None:
        # This method should contain the core logic for the strategy
        raise NotImplementedError()

    # Helper methods to create orders
    def buy(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        # Ensure integer price and quantity as per datamodel
        self.orders.append(Order(self.symbol, int(round(price)), int(round(quantity))))

    def sell(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        # Ensure integer price and quantity, negative quantity for sells
        self.orders.append(Order(self.symbol, int(round(price)), -int(round(quantity))))

    # Methods for state persistence via traderData
    def save(self) -> JSON:
        # Return data to be saved in traderData
        return None # Default: no state saved

    def load(self, data: JSON) -> None:
        # Load data from traderData
        pass # Default: no state loaded


class MarketMakingStrategy(Strategy):
    DEFAULT_WINDOW_SIZE = 10 # Define default window size

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        # Use a deque for efficient window management
        self.window = deque(maxlen=self.DEFAULT_WINDOW_SIZE)
        self.last_mid_price: Optional[float] = None
        self.ema_alpha = 0.2 # Alpha for EMA smoothing of fair value
        self.ema_value: Optional[float] = None

    def get_fair_value(self, state: TradingState) -> Optional[float]:
        """Calculates the fair value estimate based on weighted mid-price."""
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            #logger.print(f"DEBUG {self.symbol}: No order depth found, returning EMA {self.ema_value}")
            return self.ema_value # Return smoothed value if no current book data

        bids = order_depth.buy_orders
        asks = order_depth.sell_orders

        if not bids or not asks:
             # If book is one-sided, use the best available price for mid calc proxy
             if asks:
                 current_mid = min(asks.keys())
                 #logger.print(f"DEBUG {self.symbol}: One sided (ask), mid proxy {current_mid}")
             elif bids:
                 current_mid = max(bids.keys())
                 #logger.print(f"DEBUG {self.symbol}: One sided (bid), mid proxy {current_mid}")

             else: # No orders at all
                  #logger.print(f"DEBUG {self.symbol}: No orders, returning EMA {self.ema_value}")
                  return self.ema_value # Fallback to smoothed value or None if unavailable
        else:
            best_ask = min(asks.keys())
            best_bid = max(bids.keys())

            # Handle potential empty dictionaries after filtering (shouldn't happen if check above passes)
            if not asks or not bids:
                 return self.ema_value

            best_ask_vol = abs(asks[best_ask])
            best_bid_vol = bids[best_bid]

            if best_ask_vol + best_bid_vol == 0: # Avoid division by zero
                current_mid = (best_ask + best_bid) / 2.0
                #logger.print(f"DEBUG {self.symbol}: Zero volume at best bid/ask, using midpoint {current_mid}")

            else:
                # Weighted average: Price * Opposite Volume / Total Volume
                current_mid = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
                #logger.print(f"DEBUG {self.symbol}: Weighted mid {current_mid} (bid {best_bid}@{best_bid_vol}, ask {best_ask}@{best_ask_vol})")


        # Update EMA
        if current_mid is not None:
            if self.ema_value is None:
                self.ema_value = current_mid
                #logger.print(f"DEBUG {self.symbol}: Initialized EMA to {self.ema_value}")
            else:
                self.ema_value = self.ema_alpha * current_mid + (1 - self.ema_alpha) * self.ema_value
                #logger.print(f"DEBUG {self.symbol}: Updated EMA to {self.ema_value}")


        self.last_mid_price = current_mid # Store un-smoothed mid for reference if needed
        return self.ema_value # Return the smoothed value


    def act(self, state: TradingState) -> None:
        """Core market making logic: place aggressive and passive orders."""
        fair_value_float = self.get_fair_value(state)
        if fair_value_float is None:
            # logger.print(f"Warning: Could not calculate fair value for {self.symbol} at ts {state.timestamp}")
            return # Skip acting if no fair value

        fair_value = int(round(fair_value_float)) # Ensure integer for comparisons/orders
        # logger.print(f"INFO {self.symbol}: Fair Value Estimate: {fair_value}")


        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items()) # Sell orders dict uses negative quantity

        position = state.position.get(self.symbol, 0)
        buy_capacity = self.limit - position
        sell_capacity = self.limit + position
        # logger.print(f"INFO {self.symbol}: Position: {position}, Buy Capacity: {buy_capacity}, Sell Capacity: {sell_capacity}")


        orders_to_place: List[Order] = []
        placed_buy_aggr = False
        placed_sell_aggr = False

        # --- Define Price Levels ---
        # More conservative spread, can be adjusted
        passive_buy_price = fair_value - 1
        passive_sell_price = fair_value + 1
        aggression_buy_threshold = fair_value - 0 # Buy if ask is AT or below fair value
        aggression_sell_threshold = fair_value + 0 # Sell if bid is AT or above fair value

        # Adjust aggression based on inventory risk
        if position > self.limit * 0.5: # If significantly long, be less willing to buy
            aggression_buy_threshold = fair_value - 1
            passive_buy_price = fair_value - 2
            # logger.print(f"INFO {self.symbol}: Heavy Long Pos -> Reducing buy aggression (Threshold: {aggression_buy_threshold}, Passive: {passive_buy_price})")

        elif position < -self.limit * 0.5: # If significantly short, be less willing to sell
            aggression_sell_threshold = fair_value + 1
            passive_sell_price = fair_value + 2
            # logger.print(f"INFO {self.symbol}: Heavy Short Pos -> Reducing sell aggression (Threshold: {aggression_sell_threshold}, Passive: {passive_sell_price})")


        # --- Aggressive Orders (Taking Liquidity) ---
        # Buy available asks below our threshold
        taken_buy_volume = 0
        if sell_orders:
            for price, volume_neg in sell_orders:
                if price <= aggression_buy_threshold and buy_capacity > 0:
                    volume = abs(volume_neg)
                    vol_to_take = min(volume, buy_capacity)
                    self.buy(price, vol_to_take) # Use helper to create order
                    buy_capacity -= vol_to_take
                    taken_buy_volume += vol_to_take
                    placed_buy_aggr = True
                    # logger.print(f"  Aggressive BUY {self.symbol}: {vol_to_take}@{price}")
                else:
                    # Stop iterating asks if price is too high
                    break

        # Sell available bids above our threshold
        taken_sell_volume = 0
        if buy_orders:
            for price, volume in buy_orders:
                if price >= aggression_sell_threshold and sell_capacity > 0:
                    vol_to_take = min(volume, sell_capacity)
                    self.sell(price, vol_to_take) # Use helper to create order
                    sell_capacity -= vol_to_take
                    taken_sell_volume += vol_to_take
                    placed_sell_aggr = True
                    # logger.print(f"  Aggressive SELL {self.symbol}: {vol_to_take}@{price}")

                else:
                     # Stop iterating bids if price is too low
                    break


        # --- Passive Orders (Providing Liquidity) ---
        # Determine passive order size (example: small base size)
        passive_order_size = 5

        # Place passive buy if capacity remains and didn't just place an aggressive buy order
        # (We check placed_buy_aggr; alternative: check if taken_buy_volume > 0)
        if buy_capacity > 0 and not placed_buy_aggr:
            passive_buy_volume = min(buy_capacity, passive_order_size)
            if passive_buy_volume > 0:
               self.buy(passive_buy_price, passive_buy_volume)
               # logger.print(f"  Passive BUY {self.symbol}: {passive_buy_volume}@{passive_buy_price}")

        # Place passive sell if capacity remains and didn't just place an aggressive sell order
        if sell_capacity > 0 and not placed_sell_aggr: # Alternative: check if taken_sell_volume > 0
            passive_sell_volume = min(sell_capacity, passive_order_size)
            if passive_sell_volume > 0:
                self.sell(passive_sell_price, passive_sell_volume)
                # logger.print(f"  Passive SELL {self.symbol}: {passive_sell_volume}@{passive_sell_price}")


    # Override save/load to include EMA state
    def save(self) -> JSON:
        # Save the EMA value and last mid price along with the window
        return {
            "window": list(self.window), # Convert deque to list for JSON
            "ema_value": self.ema_value,
            "last_mid_price": self.last_mid_price
        }

    def load(self, data: JSON) -> None:
        # Load the state, handling potential missing keys or old formats
        if isinstance(data, dict):
            # Load window, ensuring it's a deque with the correct maxlen
            self.window = deque(data.get("window", []), maxlen=self.DEFAULT_WINDOW_SIZE)
            self.ema_value = data.get("ema_value", None)
            self.last_mid_price = data.get("last_mid_price", None)
        else:
            # Fallback for older data format or invalid data: reset state
            self.window = deque(maxlen=self.DEFAULT_WINDOW_SIZE)
            self.ema_value = None
            self.last_mid_price = None
        #logger.print(f"DEBUG {self.symbol}: Loaded state - EMA: {self.ema_value}, Window: {list(self.window)}")


# --- Concrete Strategy Implementations ---

class RainforestResinStrategy(MarketMakingStrategy):
    # Resin is stable, use a fixed fair value and skip EMA
    def __init__(self, symbol: Symbol, limit: int) -> None:
         super().__init__(symbol, limit)
         self.fixed_fair_value = 10000

    def get_fair_value(self, state: TradingState) -> Optional[float]:
         # Override to return fixed value, ignore EMA calculation from parent
         self.last_mid_price = float(self.fixed_fair_value) # Store for consistency
         self.ema_value = float(self.fixed_fair_value)      # Store for consistency
         return float(self.fixed_fair_value)

    # Override save/load because we don't need EMA state for this specific strategy
    def save(self) -> JSON:
        # Only save the window state if needed by parent logic (e.g., liquidation checks)
        # If not needed, can return None or {}
        return {"window": list(self.window)} # Still save window for potential parent logic use

    def load(self, data: JSON) -> None:
         # Load only the window state
         if isinstance(data, dict):
             self.window = deque(data.get("window", []), maxlen=self.DEFAULT_WINDOW_SIZE)
         else:
             # Fallback for older data format or invalid data: reset state
             self.window = deque(maxlen=self.DEFAULT_WINDOW_SIZE)


class KelpStrategy(MarketMakingStrategy):
    # Inherits fair value calculation (weighted midpoint + EMA) from MarketMakingStrategy
    # No override needed unless KELP needs very specific logic beyond the base MM strategy
    pass

class SquidInkStrategy(MarketMakingStrategy):
    # Start by inheriting the standard fair value calculation (weighted midpoint + EMA)
    # Future: Could override get_fair_value to incorporate specific patterns if identified
    pass


# --- Main Trader Class ---
class Trader:
    def __init__(self) -> None:
        """
        Initialize the Trader class with strategies for each product.
        """
        logger.print("Initializing Trader...")

        # Define position limits for each product for Round 1
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50, # Added limit for Squid Ink
        }

        # Map symbols to their strategy classes
        self.strategy_map = {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidInkStrategy, # Added Squid Ink strategy
        }

        # Instantiate strategies
        self.strategies: Dict[Symbol, Strategy] = {}
        for symbol, limit in self.limits.items():
            if symbol in self.strategy_map:
                self.strategies[symbol] = self.strategy_map[symbol](symbol, limit)
                # logger.print(f"  - Strategy for {symbol} initialized (Limit: {limit})") # Verbose logging
            else:
                logger.print(f"  - WARNING: No strategy defined for symbol {symbol}")

        # Load initial state from traderData (if any) in the first run
        self.initial_load_done = False


    def load_trader_data(self, traderData: str):
        """Load strategy states from the traderData string."""
        if not traderData:
            # logger.print("No traderData found to load.") # Less verbose
            return

        try:
            # Use jsonpickle for potentially complex objects saved by strategies
            saved_states = jsonpickle.decode(traderData)
            if not isinstance(saved_states, dict):
                 # logger.print("TraderData is not a dictionary, cannot load states.")
                 saved_states = {} # Use empty dict if decode fails or is wrong type

            # logger.print(f"Loading traderData...") # Less verbose
            for symbol, strategy in self.strategies.items():
                if symbol in saved_states:
                    try:
                        strategy.load(saved_states[symbol])
                        # logger.print(f"  - Loaded state for {symbol}") # Verbose
                    except Exception as e:
                        logger.print(f"  - ERROR loading state for {symbol}: {e}")
                # else:
                    # logger.print(f"  - No saved state found for {symbol}") # Verbose

        except Exception as e:
            logger.print(f"ERROR decoding traderData string: {e}. Strategies will use default state.")
            # Reset all strategy states if decoding fails
            for strategy in self.strategies.values():
                 strategy.load(None) # Ensure strategies reset their state


    def save_trader_data(self) -> str:
        """Save strategy states into the traderData string."""
        states_to_save = {}
        for symbol, strategy in self.strategies.items():
             try:
                 states_to_save[symbol] = strategy.save()
             except Exception as e:
                  logger.print(f"Error saving state for {symbol}: {e}")
                  states_to_save[symbol] = None # Save None if error occurs


        try:
            # Use jsonpickle for potentially complex objects (like deques)
            # Use unpicklable=False for simpler JSON output if complex objects aren't strictly needed
            traderData_string = jsonpickle.encode(states_to_save, unpicklable=False)
            return traderData_string
        except Exception as e:
            logger.print(f"ERROR encoding traderData: {e}")
            return "" # Return empty string if encoding fails


    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """
        Main method called by the platform/backtester for each time step.
        """
        # Load trader data initially or if it seems empty/reset
        # Resetting initial_load_done if traderData is empty allows reloading if state gets lost
        if not self.initial_load_done or not state.traderData:
             self.load_trader_data(state.traderData)
             self.initial_load_done = True # Mark as done after first attempt


        # Initialize results
        all_orders: Dict[Symbol, List[Order]] = {}
        conversions = 0 # No conversions implemented in this strategy

        # Run strategy for each product
        for symbol, strategy in self.strategies.items():
            # Check if market data exists for this symbol in the current state
            if symbol in state.order_depths:
                # logger.print(f"Running strategy for {symbol}...") # Verbose
                all_orders[symbol] = strategy.run(state) # Run returns the list of orders
            else:
                # logger.print(f"No order depth data for {symbol} at timestamp {state.timestamp}") # Verbose
                all_orders[symbol] = [] # Ensure empty list if no data for this symbol


        # Save state for the next iteration
        traderData_output = self.save_trader_data()

        # Log and return results
        # The logger.flush call handles the actual printing for the platform
        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output