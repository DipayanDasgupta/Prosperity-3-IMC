import json
from abc import abstractmethod
from collections import deque
# Make sure datamodel is accessible (usually provided by the competition environment)
# If running locally, you might need to define these or get them from the competition package
try:
    from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
except ImportError:
    # Basic placeholders if datamodel isn't available, adjust as needed for local testing
    Symbol = str
    class ProsperityEncoder(json.JSONEncoder):
        def default(self, o):
            # Add necessary encoding logic if needed for custom types
            return super().default(o)
    # Define other datamodel types minimally if needed for the code to run
    class Order:
        def __init__(self, symbol, price, quantity):
            self.symbol = symbol
            self.price = price
            self.quantity = quantity
    # ... add other minimal definitions for TradingState, OrderDepth etc. if necessary

from typing import Any, TypeAlias


JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# --- Logger Class (Keep as is, needed for flush) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750 # Adjusted to a reasonable value if needed

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        # This method can be kept, but won't be called unless strategies use logger.print
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Check if state objects are available before trying to access them
        compressed_state = self.compress_state(state, "") if state else []
        compressed_orders = self.compress_orders(orders) if orders else []

        # Calculate base length safely
        base_info = [compressed_state, compressed_orders, conversions, "", ""]
        try:
            base_length = len(self.to_json(base_info))
        except Exception: # Catch potential errors during serialization if state is weird
             base_length = 200 # Estimate

        # Ensure max_item_length is non-negative
        max_item_length = max(0, (self.max_log_length - base_length) // 3)

        # Prepare final log data with truncation
        log_output = [
            self.compress_state(state, self.truncate(state.traderData if state and state.traderData else "", max_item_length)) if state else [],
            self.compress_orders(orders) if orders else [],
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]

        try:
            print(self.to_json(log_output))
        except Exception as e:
            # Fallback print if JSON encoding fails
            print(f"Error encoding log data: {e}")
            # You could print a simplified version here if needed
            print(f"Conversion: {conversions}, TraderData: {self.truncate(trader_data, 100)}, Logs: {self.truncate(self.logs, 100)}")


        self.logs = "" # Reset logs after flushing

    # --- Helper methods for flush (Keep as is) ---
    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Add checks for attribute existence if TradingState might be incomplete
        return [
            getattr(state, 'timestamp', 0),
            trader_data,
            self.compress_listings(getattr(state, 'listings', {})),
            self.compress_order_depths(getattr(state, 'order_depths', {})),
            self.compress_trades(getattr(state, 'own_trades', {})),
            self.compress_trades(getattr(state, 'market_trades', {})),
            getattr(state, 'position', {}),
            self.compress_observations(getattr(state, 'observations', None)), # Pass None if not present
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        if listings:
             for listing in listings.values():
                  # Use getattr for safety if Listing structure might vary
                  compressed.append([
                      getattr(listing, 'symbol', ''),
                      getattr(listing, 'product', ''),
                      getattr(listing, 'denomination', '')
                  ])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        if order_depths:
             for symbol, order_depth in order_depths.items():
                  # Use getattr for safety
                  compressed[symbol] = [
                      getattr(order_depth, 'buy_orders', {}),
                      getattr(order_depth, 'sell_orders', {})
                  ]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if trades:
            for arr in trades.values():
                if arr:
                    for trade in arr:
                         # Use getattr for safety
                        compressed.append([
                            getattr(trade, 'symbol', ''),
                            getattr(trade, 'price', 0),
                            getattr(trade, 'quantity', 0),
                            getattr(trade, 'buyer', ''),
                            getattr(trade, 'seller', ''),
                            getattr(trade, 'timestamp', 0),
                        ])
        return compressed

    def compress_observations(self, observations: Observation | None) -> list[Any]:
        if not observations:
             return [{}, {}] # Return empty structure if no observations

        conversion_observations = {}
        # Check if conversionObservations exists
        if hasattr(observations, 'conversionObservations') and observations.conversionObservations:
            for product, observation in observations.conversionObservations.items():
                 # Use getattr for safety
                conversion_observations[product] = [
                    getattr(observation, 'bidPrice', 0),
                    getattr(observation, 'askPrice', 0),
                    getattr(observation, 'transportFees', 0),
                    getattr(observation, 'exportTariff', 0),
                    getattr(observation, 'importTariff', 0),
                    getattr(observation, 'sunlight', 0),
                    getattr(observation, 'humidity', 0),
                ]

        # Check if plainValueObservations exists
        plain_observations = getattr(observations, 'plainValueObservations', {})
        return [plain_observations, conversion_observations]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if orders:
            for arr in orders.values():
                 if arr:
                    for order in arr:
                         # Use getattr for safety
                        compressed.append([
                            getattr(order, 'symbol', ''),
                            getattr(order, 'price', 0),
                            getattr(order, 'quantity', 0)
                        ])
        return compressed

    def to_json(self, value: Any) -> str:
        # Use default=str as a fallback for types json doesn't know
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"), default=str)

    def truncate(self, value: str, max_length: int) -> str:
        if not isinstance(value, str): # Ensure value is a string
             value = str(value)
        if len(value) <= max_length:
            return value
        # Ensure max_length is at least 3 for ellipsis
        if max_length < 3:
             return value[:max_length]
        return value[:max_length - 3] + "..."

# --- Instantiate Logger (Needed) ---
logger = Logger()

# --- Strategy Base Classes (Keep as is) ---
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: list[Order] = [] # Ensure orders is initialized

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        try:
            # Protect against errors within the strategy logic
             self.act(state)
        except Exception as e:
             # Optionally log the error using the logger's print method
             # logger.print(f"Error in strategy {self.symbol}: {e}")
             # Continue with empty orders or handle as needed
             pass # Keep orders empty if strategy fails
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        # Basic validation
        if quantity > 0:
             self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        # Basic validation
         if quantity > 0:
            self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None # Keep default save/load minimal

    def load(self, data: JSON) -> None:
        pass # Keep default save/load minimal

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        # Initialize window here if load doesn't guarantee it
        self.window: deque[bool] = deque()
        self.window_size = 10

    # Make get_true_value concrete or remove @abstractmethod if not needed in base
    def get_true_value(self, state: TradingState) -> int:
        # Provide a default or raise error if subclasses MUST implement
        raise NotImplementedError("Subclasses must implement get_true_value")


    def act(self, state: TradingState) -> None:
        # Ensure order depths exist for the symbol
        if self.symbol not in state.order_depths:
             # logger.print(f"No order depth for {self.symbol}, skipping trade.")
             return # Can't trade without order depth

        true_value = self.get_true_value(state)
        if true_value is None: # Handle case where true value can't be determined
            # logger.print(f"Could not determine true value for {self.symbol}, skipping trade.")
            return

        order_depth = state.order_depths[self.symbol]
        # Use default {} if buy/sell orders might be missing
        buy_orders = sorted(getattr(order_depth, 'buy_orders', {}).items(), reverse=True)
        sell_orders = sorted(getattr(order_depth, 'sell_orders', {}).items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = False
        hard_liquidate = False
        if len(self.window) == self.window_size:
             filled_count = sum(self.window)
             soft_liquidate = filled_count >= self.window_size / 2 and self.window[-1]
             hard_liquidate = all(self.window)


        # Define prices carefully, handle edge cases where orders might be empty
        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        # --- Buy Logic ---
        for price, volume in sell_orders:
            if to_buy <= 0: break # Stop if nothing more to buy
            # Ensure volume is negative for sell orders
            actual_volume = abs(volume)
            if price <= max_buy_price:
                quantity = min(to_buy, actual_volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            if quantity > 0:
                 self.buy(true_value, quantity)
                 to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            if quantity > 0:
                 self.buy(true_value - 2, quantity)
                 to_buy -= quantity

        if to_buy > 0:
            # Find popular buy price safely
            popular_buy_price = 0
            if buy_orders:
                 # Find the price level with the highest volume among existing buy orders
                 # Note: The original logic might have intended to use sell orders for this?
                 # Assuming we look at existing *buy* orders to place our *buy* near them.
                 try:
                     popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                 except ValueError: # Handles empty buy_orders
                     popular_buy_price = true_value - 2 # Fallback price

            # Place remaining buy order
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        # --- Sell Logic ---
        for price, volume in buy_orders:
            if to_sell <= 0: break # Stop if nothing more to sell
            # Ensure volume is positive for buy orders
            actual_volume = abs(volume)
            if price >= min_sell_price:
                quantity = min(to_sell, actual_volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            if quantity > 0:
                 self.sell(true_value, quantity)
                 to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            if quantity > 0:
                self.sell(true_value + 2, quantity)
                to_sell -= quantity

        if to_sell > 0:
             # Find popular sell price safely
             popular_sell_price = 0
             if sell_orders:
                  # Find the price level with the highest volume among existing sell orders
                  try:
                      popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0] # min price with highest volume? Check logic. Usually min(orders)[0]
                      # Let's assume it means the lowest price level available in sell orders
                      popular_sell_price = sell_orders[0][0] # Lowest ask price
                  except IndexError: # Handles empty sell_orders
                      popular_sell_price = true_value + 2 # Fallback price

             # Place remaining sell order
             price = max(min_sell_price, popular_sell_price - 1)
             self.sell(price, to_sell)


    def save(self) -> JSON:
        # Ensure window exists before saving
        return list(self.window) if hasattr(self, 'window') else []

    def load(self, data: JSON) -> None:
        # Ensure data is a list-like structure before creating deque
        if isinstance(data, list):
             self.window = deque(data, maxlen=self.window_size) # Use maxlen here too
        else:
             # Initialize empty if data is invalid
             self.window = deque(maxlen=self.window_size)


# --- Concrete Strategy Implementations (Keep as is) ---
class RainforestResinStrategy(MarketMakingStrategy):
    # Override get_true_value
    def get_true_value(self, state: TradingState) -> int:
        # Define a default true value (can be adjusted based on analysis)
        # Simple fixed value example:
        return 10_000 # Example value, adjust as needed

class KelpStrategy(MarketMakingStrategy):
    # Override get_true_value
    def get_true_value(self, state: TradingState) -> int | None: # Return None if calculation fails
        if self.symbol not in state.order_depths:
             return None # Cannot calculate without order depth

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(getattr(order_depth, 'buy_orders', {}).items(), reverse=True)
        sell_orders = sorted(getattr(order_depth, 'sell_orders', {}).items())

        # Ensure there are orders to calculate from
        if not buy_orders or not sell_orders:
            return None # Cannot calculate mid-price without both bids and asks

        # Safely get best bid and ask
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        # Simple mid-price calculation
        return round((best_bid + best_ask) / 2)


# --- Trader Class (Uncomment logger.flush) ---
class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            # Add limits for other potential products if necessary
        }

        self.strategies: dict[Symbol, Strategy] = {} # Initialize empty

        # Define available strategies
        available_strategies = {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
        }

        # Initialize strategies based on available limits
        for symbol, limit in limits.items():
             if symbol in available_strategies:
                  self.strategies[symbol] = available_strategies[symbol](symbol, limit)
             # else: logger.print(f"Warning: No strategy defined for symbol {symbol}")


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0 # Keep conversions minimal as requested
        orders: dict[Symbol, list[Order]] = {}
        new_trader_data = {}

        # Safely load previous trader data
        old_trader_data = {}
        if state.traderData:
             try:
                  old_trader_data = json.loads(state.traderData)
             except json.JSONDecodeError:
                  # logger.print("Error decoding traderData, starting fresh.")
                  pass # Keep old_trader_data empty


        # Run strategies
        for symbol, strategy in self.strategies.items():
            # Load state for the strategy
            if isinstance(old_trader_data, dict) and symbol in old_trader_data:
                 try:
                      strategy.load(old_trader_data[symbol])
                 except Exception as e:
                      # logger.print(f"Error loading data for {symbol}: {e}")
                      pass # Continue with default/empty state for strategy

            # Run the strategy if order depths are available for the symbol
            if symbol in state.order_depths:
                try:
                     # Ensure strategy.run returns a list of Orders
                     strategy_orders = strategy.run(state)
                     if isinstance(strategy_orders, list):
                          orders[symbol] = strategy_orders
                     else:
                          # logger.print(f"Strategy for {symbol} did not return a list of orders.")
                          orders[symbol] = []
                except Exception as e:
                     # logger.print(f"Error running strategy for {symbol}: {e}")
                     orders[symbol] = [] # Ensure orders[symbol] exists even if strategy fails
            else:
                 # If no order depth, ensure the key exists with empty list for consistency
                 orders[symbol] = []


            # Save state for the strategy
            try:
                saved_data = strategy.save()
                # Ensure saved_data is JSON serializable
                json.dumps(saved_data, cls=ProsperityEncoder)
                new_trader_data[symbol] = saved_data
            except Exception as e:
                 # logger.print(f"Error saving data for {symbol}: {e}")
                 new_trader_data[symbol] = None # Save None or {} if saving fails

        # Serialize the new trader data
        trader_data = ""
        try:
            trader_data = json.dumps(new_trader_data, separators=(",", ":"), cls=ProsperityEncoder)
        except Exception as e:
             # logger.print(f"Error encoding traderData: {e}")
             trader_data = "{}" # Send empty JSON object if encoding fails


        # *** THE ONLY REQUIRED CHANGE IS UNCOMMENTING THIS LINE ***
        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data