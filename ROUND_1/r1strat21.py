import json
from abc import abstractmethod
from collections import deque
# Ensure necessary imports from datamodel
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias, List, Dict, Optional, Tuple # Added List, Dict, Optional, Tuple
import statistics # Ensure statistics is imported

# Type Alias for JSON data
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Product: # Define Product names for clarity
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

class Logger:
    # Using the simpler logger from the user's provided code
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
         # Basic length check to avoid excessive memory use if logging goes wrong
        if len(self.logs) < self.max_log_length * 2:
            self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Using the simpler compression logic from the user's provided code
        # This might be less robust than the detailed one but matches the input script.
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        available_length = self.max_log_length - base_length - 100 # Keep small buffer
        max_item_length = max(100, available_length // 3) # Ensure minimum length

        # Safely truncate values
        trader_data_in = state.traderData if isinstance(state.traderData, str) else ""
        trader_data_out = trader_data if isinstance(trader_data, str) else ""
        logs_out = self.logs if isinstance(self.logs, str) else ""


        print(self.to_json([
            self.compress_state(state, self.truncate(trader_data_in, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data_out, max_item_length),
            self.truncate(logs_out, max_item_length),
        ]))

        self.logs = "" # Reset logs

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
         # Simplified compression from user code
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings or {}), # Add default empty dict
            self.compress_order_depths(state.order_depths or {}),
            self.compress_trades(state.own_trades or {}),
            self.compress_trades(state.market_trades or {}),
            state.position or {},
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            if listing: # Check if listing is not None
                compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            if order_depth: # Check if order_depth is not None
                 # Ensure buy/sell orders are dicts, default to empty if not
                buy_orders = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
                sell_orders = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
                compressed[symbol] = [buy_orders, sell_orders]
            else:
                 compressed[symbol] = [{},{}] # Default for None order_depth
        return compressed

    def compress_trades(self, trades: Optional[dict[Symbol, list[Trade]]]) -> list[list[Any]]: # Handle Optional
        compressed = []
        if trades: # Check trades is not None
            for arr in trades.values():
                if arr: # Check list is not None or empty
                    for trade in arr:
                        if trade: # Check trade is not None
                             compressed.append([
                                trade.symbol,
                                trade.price,
                                trade.quantity,
                                trade.buyer or "", # Handle None buyer/seller
                                trade.seller or "",
                                trade.timestamp,
                            ])
        return compressed

    def compress_observations(self, observations: Optional[Observation]) -> list[Any]: # Handle Optional
        if not observations:
             return [{}, {}] # Return default if observations is None

        conversion_observations = {}
        # Check if conversionObservations exists and is a dict
        if hasattr(observations, 'conversionObservations') and isinstance(observations.conversionObservations, dict):
             for product, observation in observations.conversionObservations.items():
                 if observation: # Check observation is not None
                     conversion_observations[product] = [
                        observation.bidPrice, observation.askPrice, observation.transportFees,
                        observation.exportTariff, observation.importTariff, observation.sunlight,
                        observation.humidity,
                    ]
        # Check if plainValueObservations exists and is a dict
        plain_obs = observations.plainValueObservations if hasattr(observations, 'plainValueObservations') and isinstance(observations.plainValueObservations, dict) else {}
        return [plain_obs, conversion_observations]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if orders: # Check orders is not None
            for arr in orders.values():
                if arr: # Check list is not None or empty
                     for order in arr:
                        if order: # Check order is not None
                            compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
         # Use ProsperityEncoder if available, otherwise standard json
        cls_to_use = ProsperityEncoder if "ProsperityEncoder" in globals() else None
        try:
            return json.dumps(value, cls=cls_to_use, separators=(",", ":"))
        except Exception:
             # Fallback if encoding fails
             return json.dumps(value, separators=(",", ":"))


    def truncate(self, value: Any, max_length: int) -> str: # Accept Any
        # Safely convert to string
        s_value = ""
        if isinstance(value, str):
             s_value = value
        else:
             try: s_value = str(value)
             except Exception: s_value = "" # Default to empty if str() fails

        if len(s_value) <= max_length:
            return s_value
        return s_value[:max_length - 3] + "..."


logger = Logger()

# --- Base Strategy Class ---
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: List[Order] = [] # Initialize orders list

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = [] # Reset orders for this run
        try:
            self.act(state)
        except Exception as e:
             logger.print(f"ERROR in {self.symbol} strategy: {e}")
             # Optional: Add fallback logic like trying to flatten position
             self.orders = [] # Clear any potentially bad orders on error
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        # Ensure quantity is positive for buy orders
        q = int(round(quantity))
        if q > 0:
            self.orders.append(Order(self.symbol, int(round(price)), q))

    def sell(self, price: int, quantity: int) -> None:
        # Ensure quantity is positive for sell orders (will be negated in Order)
        q = int(round(quantity))
        if q > 0:
            self.orders.append(Order(self.symbol, int(round(price)), -q))

    # Keep original simple save/load
    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

# --- Market Making Strategy (For RESIN/KELP - Logic kept but unused) ---
class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window: deque[bool] = deque(maxlen=10) # Correct type hint and maxlen
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int: # Keep original signature
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        # Check if symbol exists in order depths first
        if self.symbol not in state.order_depths:
             return

        order_depth = state.order_depths[self.symbol]
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
             return # Need valid book data

        # --- Original MM Logic (will not be called for RESIN/KELP in the modified Trader) ---
        true_value = self.get_true_value(state)
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) >= self.limit * 0.95)
        soft_liquidate = len(self.window) == self.window.maxlen and sum(self.window) >= self.window.maxlen / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window.maxlen and sum(self.window) >= self.window.maxlen * 0.8

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < -self.limit * 0.5 else true_value

        # Aggressive Buys
        temp_to_buy = to_buy # Use temp var to avoid modifying original `to_buy` needed later
        for price, volume in sell_orders:
            if temp_to_buy <= 0: break
            if price <= max_buy_price:
                quantity = min(temp_to_buy, abs(volume))
                if quantity > 0: self.buy(price, quantity); temp_to_buy -= quantity
            else: break

        # Liquidation Buys
        liq_buy_size = max(1, self.limit // 4)
        if position < 0:
            if hard_liquidate and to_buy > 0:
                quantity = min(abs(position), to_buy, liq_buy_size)
                if quantity > 0: self.buy(true_value, quantity) # Note: Modifies self.orders, but to_buy not updated here
            elif soft_liquidate and to_buy > 0:
                 quantity = min(abs(position), to_buy, liq_buy_size)
                 if quantity > 0: self.buy(true_value - 1, quantity) # Note: Modifies self.orders

        # Passive Buys
        if to_buy > 0:
            if buy_orders:
                 popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                 price = min(max_buy_price, popular_buy_price + 1)
                 self.buy(price, to_buy) # Order for remaining capacity
            else:
                 self.buy(max_buy_price, to_buy)

        # Aggressive Sells
        temp_to_sell = to_sell # Use temp var
        for price, volume in buy_orders:
             if temp_to_sell <= 0: break
             if price >= min_sell_price:
                 quantity = min(temp_to_sell, abs(volume))
                 if quantity > 0: self.sell(price, quantity); temp_to_sell -= quantity
             else: break

        # Liquidation Sells
        liq_sell_size = max(1, self.limit // 4)
        if position > 0:
            if hard_liquidate and to_sell > 0:
                 quantity = min(position, to_sell, liq_sell_size)
                 if quantity > 0: self.sell(true_value, quantity) # Note: Modifies self.orders
            elif soft_liquidate and to_sell > 0:
                 quantity = min(position, to_sell, liq_sell_size)
                 if quantity > 0: self.sell(true_value + 1, quantity) # Note: Modifies self.orders

        # Passive Sells
        if to_sell > 0:
            if sell_orders:
                popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
                price = max(min_sell_price, popular_sell_price - 1)
                self.sell(price, to_sell) # Order for remaining capacity
            else:
                 self.sell(min_sell_price, to_sell)

    def save(self) -> JSON:
        # Ensure window state is saved correctly
        return {"window": list(self.window)} # Save as dict

    def load(self, data: JSON) -> None:
        # Load window state safely
        if isinstance(data, dict) and "window" in data and isinstance(data["window"], list):
            # Ensure maxlen is set correctly
            self.window = deque(data["window"], maxlen=self.window_size)
        else:
             # Initialize correctly if data is missing or wrong format
            self.window = deque(maxlen=self.window_size)

# --- Ink Mean Reversion Strategy (SQUID_INK - This will be used) ---
class InkMeanReversionTrader(Strategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        # Using window size from original user code for this class
        self.prices: deque[float] = deque(maxlen=300)
        self.window_size = 300 # Match maxlen

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return # Need book data

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        mid_price = (best_bid + best_ask) / 2.0
        self.prices.append(mid_price)

        # Require window to be full
        if len(self.prices) < self.window_size:
            return

        try:
            mean = statistics.mean(self.prices)
            # Use population stdev if full, sample stdev otherwise
            std_dev = statistics.pstdev(self.prices) if len(self.prices) == self.window_size else statistics.stdev(self.prices)
        except statistics.StatisticsError:
            return # Not enough data or all points are the same

        if std_dev < 0.1: return # Avoid trading if stdev is too low

        position = state.position.get(self.symbol, 0)
        buy_cap = self.limit - position
        sell_cap = self.limit + position

        # Define Bands
        upper_band_2 = mean + 2.0 * std_dev
        upper_band_1 = mean + 1.0 * std_dev
        lower_band_1 = mean - 1.0 * std_dev
        lower_band_2 = mean - 2.0 * std_dev
        exit_threshold = 0.25 # Fixed exit threshold from original code

        # Uses full remaining capacity, as per original code for this class
        # === BUY ZONE ===
        if mid_price < lower_band_2 and buy_cap > 0:
            self.buy(best_ask, buy_cap) # Aggressive buy, full capacity

        elif mid_price < lower_band_1 and buy_cap > 0:
            self.buy(best_bid, buy_cap) # Passive buy, full capacity

        # === SELL ZONE ===
        elif mid_price > upper_band_2 and sell_cap > 0:
            self.sell(best_bid, sell_cap) # Aggressive sell, full capacity

        elif mid_price > upper_band_1 and sell_cap > 0:
            self.sell(best_ask, sell_cap) # Passive sell, full capacity

        # === EXIT ZONE ===
        elif abs(mid_price - mean) <= exit_threshold:
            if position > 0 and sell_cap > 0:
                self.sell(best_bid, position) # Exit long completely
            elif position < 0 and buy_cap > 0:
                self.buy(best_ask, abs(position)) # Exit short completely


    # Save/Load needs to handle the deque correctly
    def save(self) -> JSON:
        return {"prices": list(self.prices)} # Save as dict

    def load(self, data: JSON) -> None:
         # Load prices deque safely
        if isinstance(data, dict) and "prices" in data and isinstance(data["prices"], list):
            # Ensure maxlen is set correctly
            self.prices = deque(data["prices"], maxlen=self.window_size)
        else:
             # Initialize correctly if data is missing or wrong format
            self.prices = deque(maxlen=self.window_size)


# --- Concrete Strategy Implementations (Unused but class definitions kept) ---
class RainforestResinStrategy(MarketMakingStrategy):
     def __init__(self, symbol: Symbol, limit: int): # Add init
         super().__init__(symbol, limit)
     def get_true_value(self, state: TradingState) -> int:
        return 10000

class KelpStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int): # Add init
         super().__init__(symbol, limit)
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
             logger.print(f"Warning: Empty book for {self.symbol} in get_true_value")
             return 0 # Needs better handling

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
             logger.print(f"Warning: Empty buy or sell orders for {self.symbol} in get_true_value")
             return 0 # Needs better handling

        # Ensure keys exist before accessing - this part seems fragile in original
        # Let's add a check, although a better true value method (like EMA) is recommended
        try:
             popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
             popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
             return round((popular_buy_price + popular_sell_price) / 2)
        except ValueError: # Handles empty sequence for max/min
             logger.print(f"Error finding popular prices for {self.symbol}")
             return 0 # Needs better handling

# --- Main Trader Class (Modified for SQUID_INK only) ---
class Trader:
    def __init__(self) -> None:
        limits = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
        }

        self.strategies: Dict[Symbol, Strategy] = {} # Initialize as empty dict

        # --- ONLY INSTANTIATE SQUID_INK STRATEGY ---
        # Ensure the correct class name InkMeanReversionTrader is used
        self.strategies[Product.SQUID_INK] = InkMeanReversionTrader(
            Product.SQUID_INK, limits[Product.SQUID_INK]
        )
        # --- Instantiation for RESIN and KELP commented out ---
        # resin_strategy_class = RainforestResinStrategy
        # kelp_strategy_class = KelpStrategy
        # if resin_strategy_class: # Check added for safety, though class exists
        #      self.strategies[Product.RAINFOREST_RESIN] = resin_strategy_class(
        #          Product.RAINFOREST_RESIN, limits[Product.RAINFOREST_RESIN]
        #      )
        # if kelp_strategy_class: # Check added for safety
        #      self.strategies[Product.KELP] = kelp_strategy_class(
        #          Product.KELP, limits[Product.KELP]
        #      )
        # ----------------------------------------------------

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        # Initialize orders dict to ensure all symbols are present in the output
        orders: Dict[Symbol, list[Order]] = {
             Product.RAINFOREST_RESIN: [],
             Product.KELP: [],
             Product.SQUID_INK: []
        }

        # Load trader data (only Ink state will be present/relevant)
        # Using simple json load/dump as per the original r1strat20 structure
        old_trader_data = {}
        if state.traderData:
             try:
                 # Check if traderData is already a dict (can happen in some environments)
                 if isinstance(state.traderData, dict):
                     old_trader_data = state.traderData
                 elif isinstance(state.traderData, str):
                      old_trader_data = json.loads(state.traderData)
             except (json.JSONDecodeError, TypeError) as e:
                 logger.print(f"Error decoding traderData: {e}, data: {state.traderData[:100]}")
                 old_trader_data = {} # Reset if decode fails

        new_trader_data = {}

        # --- ONLY RUN SQUID_INK STRATEGY ---
        symbol_to_run = Product.SQUID_INK
        if symbol_to_run in self.strategies:
            strategy = self.strategies[symbol_to_run]

            # Load state for Ink strategy if it exists and is valid
            if symbol_to_run in old_trader_data and isinstance(old_trader_data.get(symbol_to_run), (dict, list)): # Check type
                try:
                    strategy.load(old_trader_data[symbol_to_run])
                except Exception as e:
                     logger.print(f"Error loading state for {symbol_to_run}: {e}")

            # Run strategy only if order depth exists for this symbol
            if symbol_to_run in state.order_depths:
                 # Get orders from the strategy run method
                 orders[symbol_to_run] = strategy.run(state) # strategy.run() returns the list of orders

            # Save state for Ink strategy
            saved_state = strategy.save()
            if saved_state is not None: # Only save if state is actually returned
                 new_trader_data[symbol_to_run] = saved_state
        # ------------------------------------

        # Convert new_trader_data to string
        trader_data = ""
        if new_trader_data: # Only dump if not empty
             try:
                 # Ensure ProsperityEncoder is used if available
                 cls_to_use = ProsperityEncoder if "ProsperityEncoder" in globals() else None
                 trader_data = json.dumps(new_trader_data, cls=cls_to_use, separators=(",", ":"))
             except Exception as e:
                 logger.print(f"Error encoding traderData: {e}")
                 trader_data = "" # Reset on error

        # Ensure logger uses the orders dict which contains empty lists for non-traded symbols
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data