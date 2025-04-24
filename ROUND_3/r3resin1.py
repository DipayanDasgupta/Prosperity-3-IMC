#!/usr/bin/env python3
import json
import jsonpickle
import math
import random
import statistics
from abc import abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional

# -------------------------------
# Minimal Data Model Definitions
# (Copied from your Round 2 sample - ensure these match the competition exactly)
# -------------------------------
class Symbol(str):
    pass

class Listing:
    def __init__(self, symbol: Symbol, product: str, denomination: str) -> None:
        self.symbol = symbol
        self.product = product
        self.denomination = denomination

class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __repr__(self):
        return f"Order(symbol={self.symbol}, price={self.price}, quantity={self.quantity})"

class OrderDepth:
    # Using dict[int, int] for simplicity if ProsperityEncoder handles it
    def __init__(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]) -> None:
        # Ensure orders are sorted numerically for reliable best bid/ask finding later
        # It's often safer to work with sorted lists/tuples, but dicts are used here.
        self.buy_orders = buy_orders
        self.sell_orders = sell_orders

class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: str, seller: str, timestamp: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

class Observation:
     # Adjusted based on typical Prosperity structures if needed
    def __init__(self, plainValueObservations: Dict = None, conversionObservations: Dict = None) -> None:
        self.plainValueObservations = plainValueObservations if plainValueObservations is not None else {}
        self.conversionObservations = conversionObservations if conversionObservations is not None else {}


class TradingState:
    def __init__(self,
                 timestamp: int,
                 traderData: str,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Symbol, int],
                 observations: Observation) -> None:
        self.timestamp = timestamp
        self.traderData = traderData
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

class ProsperityEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, deque):
             return list(o) # Serialize deques as lists
        if hasattr(o, "__dict__"):
            return o.__dict__
        return super().default(o)

# ---------------------------
# Logger (Using R2 version logic)
# ---------------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Re-implementing compression similar to R2 sample
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            ""
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        if max_item_length < 0: max_item_length = 0 # Prevent negative length

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData if state.traderData is not None else "", max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data if trader_data is not None else "", max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Simplified compression based on R2 example format
         return [
            state.timestamp,
            trader_data, # Use truncated traderData passed in
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            # R2 format example [buy_orders, sell_orders]
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        # R2 format example compresses trades into a flat list
        for arr in trades.values():
            for trade in arr:
                 compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    # R2 sample might not have buyer/seller if they are always fixed strings
                    getattr(trade, 'buyer', ''),
                    getattr(trade, 'seller', ''),
                    trade.timestamp,
                ])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
         # R2 format example [plainValueObservations, conversionObservations]
         # Assuming conversionObservations structure needs specific fields if present
        compressed_conv_obs = {}
        if hasattr(observations, 'conversionObservations'):
             for product, observation in observations.conversionObservations.items():
                  # Example structure, adjust if needed
                  compressed_conv_obs[product] = [
                     getattr(observation, 'bidPrice', None),
                     getattr(observation, 'askPrice', None),
                     getattr(observation, 'transportFees', None),
                     getattr(observation, 'exportTariff', None),
                     getattr(observation, 'importTariff', None),
                     getattr(observation, 'sunlight', None),
                     getattr(observation, 'humidity', None),
                 ]

        plain_obs = getattr(observations, 'plainValueObservations', {})
        return [plain_obs, compressed_conv_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[:max_length - 3] + "..."

logger = Logger()


# -------------------------------
# Strategy Base Classes (Using R2 versions)
# -------------------------------
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: List[Order] = [] # Initialize orders list

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = [] # Clear orders before acting
        self.act(state)
        return self.orders

    # Renaming buy/sell to avoid conflict with potential keywords if used differently
    def place_buy_order(self, price: int, quantity: int) -> None:
        if quantity > 0:
            self.orders.append(Order(self.symbol, price, quantity))
            # logger.print(f"MM Placing BUY: {quantity}@{price}")

    def place_sell_order(self, price: int, quantity: int) -> None:
         if quantity > 0:
            self.orders.append(Order(self.symbol, price, -quantity))
            # logger.print(f"MM Placing SELL: {quantity}@{price}")

    # Using JSON type hint from R2 sample
    def save(self) -> Any: # Changed from JSON to Any for compatibility
        return None

    def load(self, data: Any) -> None: # Changed from JSON to Any
        pass


class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        # Using deque for window as in R2 example
        self.window = deque(maxlen=10) # Define maxlen directly
        self.window_size = 10 # Keep for logic reference

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int: # Made instance method
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
             logger.print(f"No order depth for {self.symbol}, skipping MM.")
             return

        true_value = self.get_true_value(state)
        # logger.print(f"MM True Value for {self.symbol}: {true_value}")

        order_depth = state.order_depths[self.symbol]
        # Ensure keys exist before sorting
        buy_orders_dict = order_depth.buy_orders if order_depth.buy_orders else {}
        sell_orders_dict = order_depth.sell_orders if order_depth.sell_orders else {}

        # Sort buy orders descending by price, sell orders ascending by price
        buy_orders = sorted(buy_orders_dict.items(), reverse=True)
        sell_orders = sorted(sell_orders_dict.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position
        # logger.print(f"MM Position: {position}, to_buy: {to_buy}, to_sell: {to_sell}")


        # --- Inventory Management (copied from R2 sample) ---
        self.window.append(abs(position) >= self.limit * 0.9) # Use a threshold slightly below limit
        # if len(self.window) > self.window_size: # deque handles maxlen
        #     self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 # Removed self.window[-1] check?
        hard_liquidate = len(self.window) == self.window_size and all(self.window)
        # logger.print(f"MM soft_liquidate: {soft_liquidate}, hard_liquidate: {hard_liquidate}")


        # --- Define Target Buy/Sell Prices ---
        # Adjust aggressiveness based on position (skewing)
        max_buy_price = true_value - 1 # Default: bid below fair value
        min_sell_price = true_value + 1 # Default: ask above fair value

        # Skew prices aggressively if inventory is high
        if position > self.limit * 0.6:
            max_buy_price -= 1 # Less willing to buy more
            min_sell_price -= 1 # More willing to sell
        elif position < -self.limit * 0.6:
            max_buy_price += 1 # More willing to buy
            min_sell_price += 1 # Less willing to sell

        # logger.print(f"MM Prices: max_buy={max_buy_price}, min_sell={min_sell_price}")


        # --- Take Available Liquidity (Hit Bids / Lift Offers) ---
        taken_buy_qty = 0
        if sell_orders:
            best_ask_price, best_ask_vol = sell_orders[0] # Assuming sorted ascending
            if to_buy > 0 and best_ask_price <= max_buy_price:
                quantity_to_take = min(to_buy, abs(best_ask_vol)) # abs() as sell vols are negative
                # logger.print(f"MM Taking Ask: {quantity_to_take}@{best_ask_price} (limit {to_buy}, available {abs(best_ask_vol)})")
                self.place_buy_order(best_ask_price, quantity_to_take)
                to_buy -= quantity_to_take
                taken_buy_qty = quantity_to_take # Track taken quantity

        taken_sell_qty = 0
        if buy_orders:
            best_bid_price, best_bid_vol = buy_orders[0] # Assuming sorted descending
            if to_sell > 0 and best_bid_price >= min_sell_price:
                quantity_to_take = min(to_sell, best_bid_vol)
                # logger.print(f"MM Taking Bid: {quantity_to_take}@{best_bid_price} (limit {to_sell}, available {best_bid_vol})")
                self.place_sell_order(best_bid_price, quantity_to_take)
                to_sell -= quantity_to_take
                taken_sell_qty = quantity_to_take # Track taken quantity

        # --- Provide Liquidity (Place Limit Orders) ---
        final_position_estimate = position + taken_buy_qty - taken_sell_qty

        # Place Buy Orders
        if to_buy > 0:
            target_buy_price = max_buy_price # Default place order 1 tick below FV (adjusted for skew)
            # R2 logic: Optionally adjust based on most popular price level if needed
            # if buy_orders:
            #     popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            #     target_buy_price = min(max_buy_price, popular_buy_price + 1)

            # Liquidation Logic Overrides (Place closer/at FV if desperate)
            if hard_liquidate and final_position_estimate < 0 : # Need to buy back desperately
                 target_buy_price = true_value
            elif soft_liquidate and final_position_estimate < 0: # Need to buy back
                 target_buy_price = max(target_buy_price, true_value - 2) # Move closer, R2 used -2

            self.place_buy_order(target_buy_price, to_buy)

        # Place Sell Orders
        if to_sell > 0:
            target_sell_price = min_sell_price # Default place order 1 tick above FV (adjusted for skew)
            # R2 logic: Optionally adjust based on most popular price level
            # if sell_orders:
            #     popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            #     target_sell_price = max(min_sell_price, popular_sell_price - 1)

             # Liquidation Logic Overrides
            if hard_liquidate and final_position_estimate > 0: # Need to sell desperately
                 target_sell_price = true_value
            elif soft_liquidate and final_position_estimate > 0: # Need to sell
                 target_sell_price = min(target_sell_price, true_value + 2) # Move closer, R2 used +2

            self.place_sell_order(target_sell_price, to_sell)


    # Save/load deque state as list
    def save(self) -> Any: # Changed from JSON to Any
        return list(self.window)

    def load(self, data: Any) -> None: # Changed from JSON to Any
        if isinstance(data, list):
             self.window = deque(data, maxlen=self.window_size)


# -----------------------------------------
# Specific Strategy for RAINFOREST_RESIN
# -----------------------------------------
class RainforestResinStrategy(MarketMakingStrategy):
    # Inherits MarketMakingStrategy, just need to define true value
    def get_true_value(self, state: TradingState) -> int:
        # Simplest approach: Fixed fair value based on observed stability
        return 10000


# -------------------------------
# Trader Class (Simplified for Resin Only)
# -------------------------------
class Trader:
    def __init__(self) -> None:
        self.limits = {
            "RAINFOREST_RESIN": 50, # Set limit for Resin
            # REMOVE other products
        }
        self.strategies: Dict[Symbol, Strategy] = {}

        # Instantiate ONLY the Rainforest strategy
        if "RAINFOREST_RESIN" in self.limits:
            self.strategies["RAINFOREST_RESIN"] = RainforestResinStrategy(
                "RAINFOREST_RESIN",
                self.limits["RAINFOREST_RESIN"]
            )
        # REMOVE other strategy instantiations

        self.trader_data_cache = {}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0 # Default conversions for single product trading
        trader_data_string = "" # Initialize empty trader data string

        # Load trader data safely
        try:
            # Only load data if it exists and is not empty
            self.trader_data_cache = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            self.trader_data_cache = {}
            logger.print("Error decoding traderData, resetting.")

        all_orders: Dict[Symbol, List[Order]] = {}
        new_trader_data_part: Dict[Symbol, Any] = {}

        # Loop through defined strategies (only Resin)
        for symbol, strategy in self.strategies.items():
            strategy_data = self.trader_data_cache.get(symbol)
            if strategy_data is not None:
                try:
                    strategy.load(strategy_data)
                except Exception as e:
                    logger.print(f"Error loading state for {symbol}: {e}")

            if symbol in state.order_depths:
                try:
                    # logger.print(f"\nRunning strategy for {symbol} at {state.timestamp}")
                    symbol_orders = strategy.run(state)
                    all_orders[symbol] = symbol_orders
                    # logger.print(f"Orders for {symbol}: {symbol_orders}")
                except Exception as e:
                    logger.print(f"Error running strategy for {symbol}: {e}")
                    all_orders[symbol] = []
            else:
                 logger.print(f"No order depth found for {symbol}, skipping strategy run.")

            try:
                saved_data = strategy.save()
                if saved_data is not None: # Only add if there's something to save
                     new_trader_data_part[symbol] = saved_data
            except Exception as e:
                logger.print(f"Error saving state for {symbol}: {e}")


        # Serialize the updated trader data *only if* there's data to serialize
        if new_trader_data_part:
            trader_data_string = json.dumps(new_trader_data_part, cls=ProsperityEncoder, separators=(",", ":"))

        logger.flush(state, all_orders, conversions, trader_data_string)
        return all_orders, conversions, trader_data_string

# Keep main block for potential platform checks
if __name__ == "__main__":
    pass