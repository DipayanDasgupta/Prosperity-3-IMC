#!/usr/bin/env python3
import json
import random
from abc import abstractmethod
from collections import deque
from typing import Any, Dict, List

# -------------------------------
# Minimal Data Model Definitions
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
    def __init__(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]) -> None:
        # Buy orders: price -> volume (volume positive)
        # Sell orders: price -> volume (volume stored as negative numbers)
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
    def __init__(self, plainValueObservations: List[Any], conversionObservations: Dict[str, List[Any]]) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations

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
        if hasattr(o, "__dict__"):
            return o.__dict__
        return super().default(o)

# ---------------------------
# Logger and Base Strategy Classes
# ---------------------------

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            ""
        ]))
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for trade_list in trades.values():
            for trade in trade_list:
                compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = observation
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for order_list in orders.values():
            for order in order_list:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[:max_length - 3] + "..."

logger = Logger()

class Strategy:
    def __init__(self, symbol: Symbol, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> Any:
        return None

    def load(self, data: Any) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        order_depth = state.order_depths[self.symbol]

        # Sort orders by price for consistency
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Tracking extreme positions
        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()
        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        # Buy side: capture sell orders below our threshold
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 1, quantity)
            to_buy -= quantity

        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        # Sell side: sell into buy orders above threshold
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 1, quantity)
            to_sell -= quantity

        if to_sell > 0 and sell_orders:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> Any:
        return list(self.window)

    def load(self, data: Any) -> None:
        self.window = deque(data)

# -------------------------------
# Custom Strategy: TrendVolatilityStrategy
# -------------------------------

class TrendVolatilityStrategy(MarketMakingStrategy):
    """
    Uses historical analysis parameters (median, trend, volatility) combined with live order book data
    to estimate a true value for the asset.
    
    For most assets the true value is computed as the average of:
      - The historical (analysis) based value: median price + trend
      - The current market midprice (average of best bid and ask)
      
    However, for SQUID_INK—because it has been loss‐making—we use a different weighting: we put
    70% weight on the historical value and only 30% on the current market midprice.
    This makes the price estimate more robust (less sensitive to noisy, adverse price movements).
    """
    def __init__(self, symbol: Symbol, limit: int, analysis_params: Dict[str, Dict]) -> None:
        super().__init__(symbol, limit)
        self.analysis_params = analysis_params

    def get_true_value(self, state: TradingState) -> int:
        params = self.analysis_params.get(self.symbol, {})
        base_value = params.get("median_price", 10000)
        trend = params.get("trend_slope", 0)
        # Historical analysis value
        analysis_value = base_value + trend
        order_depth = state.order_depths.get(self.symbol)
        if order_depth:
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else analysis_value
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else analysis_value
            market_mid = (best_bid + best_ask) // 2
        else:
            market_mid = analysis_value

        # For SQUID_INK, use a more conservative combination
        if self.symbol == "SQUID_INK":
            true_value = (0.7 * analysis_value + 0.3 * market_mid)
        else:
            true_value = (analysis_value + market_mid) / 2
        return round(true_value)

# -------------------------------
# Trader Class Setup
# -------------------------------

class Trader:
    def __init__(self) -> None:
        # Set position limits for all assets
        limits = {
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
            "PICNIC_BASKET1": 50,
            "PICNIC_BASKET2": 50,
            "DJEMBES": 50,
            "KELP": 50,
            "CROISSANTS": 50,
            "JAMS": 50,
        }
        # Example analysis parameters (update these based on your full data analysis)
        analysis_params = {
            "RAINFOREST_RESIN": {"median_price": 10000, "trend_slope": -0.1, "volatility": 6},
            "SQUID_INK":       {"median_price": 1836,  "trend_slope": -10,  "volatility": 20},
            "PICNIC_BASKET1":   {"median_price": 58710, "trend_slope": -100, "volatility": 300},
            "PICNIC_BASKET2":   {"median_price": 30254, "trend_slope": -50,  "volatility": 150},
            "DJEMBES":         {"median_price": 13409, "trend_slope": -20,  "volatility": 40},
            "KELP":            {"median_price": 2034,  "trend_slope": 11,   "volatility": 11},
            "CROISSANTS":      {"median_price": 4275,  "trend_slope": 0,    "volatility": 8},
            "JAMS":            {"median_price": 6542,  "trend_slope": 0,    "volatility": 10},
        }
        # Create strategy instances for each asset.
        self.strategies = {
            symbol: TrendVolatilityStrategy(symbol, limits[symbol], analysis_params)
            for symbol in limits.keys()
        }

    def run(self, state: TradingState) -> (Dict[Symbol, List[Order]], int, str):
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))
            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)
            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

# -------------------------------
# Simple Backtesting Engine
# -------------------------------

def simulate_trading():
    trader = Trader()
    trader_data = ""
    positions = {
        "RAINFOREST_RESIN": 0,
        "SQUID_INK": 0,
        "PICNIC_BASKET1": 0,
        "PICNIC_BASKET2": 0,
        "DJEMBES": 0,
        "KELP": 0,
        "CROISSANTS": 0,
        "JAMS": 0,
    }

    # Create listings for each asset
    listings = {
        symbol: Listing(symbol, symbol.title(), "USD")
        for symbol in positions.keys()
    }

    # Simulate 5 ticks
    for tick in range(5):
        timestamp = tick * 100
        order_depths = {}
        for symbol in positions.keys():
            if symbol == "RAINFOREST_RESIN":
                bid_price = 10000 - random.randint(0, 10)
                ask_price = 10000 + random.randint(0, 10)
            elif symbol == "SQUID_INK":
                bid_price = 1836 - random.randint(0, 5)
                ask_price = 1836 + random.randint(0, 5)
            elif symbol == "PICNIC_BASKET1":
                bid_price = 58710 - random.randint(0, 20)
                ask_price = 58710 + random.randint(0, 20)
            elif symbol == "PICNIC_BASKET2":
                bid_price = 30254 - random.randint(0, 15)
                ask_price = 30254 + random.randint(0, 15)
            elif symbol == "DJEMBES":
                bid_price = 13409 - random.randint(0, 15)
                ask_price = 13409 + random.randint(0, 15)
            elif symbol == "KELP":
                bid_price = 2034 - random.randint(0, 5)
                ask_price = 2034 + random.randint(0, 5)
            elif symbol == "CROISSANTS":
                bid_price = 4275 - random.randint(0, 3)
                ask_price = 4275 + random.randint(0, 3)
            elif symbol == "JAMS":
                bid_price = 6542 - random.randint(0, 4)
                ask_price = 6542 + random.randint(0, 4)
            order_depths[symbol] = OrderDepth(
                buy_orders={bid_price: random.randint(5, 15)},
                sell_orders={ask_price: -random.randint(5, 15)}
            )

        own_trades = {symbol: [] for symbol in positions.keys()}
        market_trades = {symbol: [] for symbol in positions.keys()}
        observations = Observation([], {})

        state = TradingState(timestamp, trader_data, listings, order_depths, own_trades, market_trades, positions.copy(), observations)

        orders, conversions, new_trader_data = trader.run(state)

        # Assume immediate execution
        for symbol, order_list in orders.items():
            for order in order_list:
                positions[symbol] += order.quantity
                print(f"Tick {timestamp}: Executed order -> {order}")
        trader_data = new_trader_data
        print(f"Tick {timestamp}: Updated Positions -> {positions}\n")

if __name__ == "__main__":
    simulate_trading()
