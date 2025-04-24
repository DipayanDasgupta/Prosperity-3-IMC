import json
import random
import math
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
        self.buy_orders = buy_orders  # price -> volume (positive)
        self.sell_orders = sell_orders  # price -> volume (negative)

class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: str, seller: str, timestamp: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

class Observation:
    def __init__(self, plainValueObservations: List[Any], conversionObservations: Dict[str, Any]) -> None:
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
# Logger Class
# ---------------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str=" ", end: str="\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions, "", ""
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        log_msg = self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length)
        ])
        print(log_msg)
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
            self.compress_observations(state.observations)
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, od in order_depths.items():
            compressed[symbol] = [od.buy_orders, od.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for trade_list in trades.values():
            for trade in trade_list:
                compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, obs: Observation) -> List[Any]:
        return [obs.plainValueObservations, obs.conversionObservations]

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

# ---------------------------
# Base Strategy Class
# ---------------------------
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
        return list(self.window) if hasattr(self, "window") else None

    def load(self, data: Any) -> None:
        if data is not None and hasattr(self, "window"):
            self.window = deque(data, maxlen=self.window_size)

# -------------------------------
# Dynamic Statistical Market Maker Strategy for Underlyings (Default)
# -------------------------------
class DynamicStatisticalMarketMakerStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, window_size: int = 20, k: float = 1.0, alpha: float = 0.5, order_size: int = 5) -> None:
        super().__init__(symbol, limit)
        self.window = deque(maxlen=window_size)
        self.window_size = window_size
        self.k = k
        self.alpha = alpha
        self.order_size = order_size

    def get_current_mid(self, state: TradingState) -> int:
        od = state.order_depths[self.symbol]
        if od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            return (best_bid + best_ask) // 2
        if self.window:
            return self.window[-1]
        return 10000

    def act(self, state: TradingState) -> None:
        current_mid = self.get_current_mid(state)
        self.window.append(current_mid)
        if len(self.window) < self.window_size:
            return
        mu = sum(self.window) / len(self.window)
        variance = sum((p - mu) ** 2 for p in self.window) / len(self.window)
        sigma = math.sqrt(variance)
        position = state.position.get(self.symbol, 0)
        adjusted_fair = mu - self.alpha * position
        target_bid = int(adjusted_fair - self.k * sigma)
        target_ask = int(adjusted_fair + self.k * sigma)
        capacity_buy = self.limit - position
        capacity_sell = self.limit + position
        if target_bid < current_mid and capacity_buy > 0:
            qty = min(self.order_size, capacity_buy)
            self.buy(target_bid, qty)
        if target_ask > current_mid and capacity_sell > 0:
            qty = min(self.order_size, capacity_sell)
            self.sell(target_ask, qty)

# -------------------------------
# Dynamic Option Market Maker Strategy (Updated with Time Decay)
# -------------------------------
class DynamicOptionMarketMakerStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, window_size: int = 20, k: float = 0.5, alpha: float = 0.3, order_size: int = 5, base_premium: int = 100, decay_rate: int = 2) -> None:
        super().__init__(symbol, limit)
        self.window = deque(maxlen=window_size)
        self.window_size = window_size
        self.k = k          # volatility multiplier for options
        self.alpha = alpha  # inventory adjustment factor
        self.order_size = order_size
        self.base_premium = base_premium
        self.decay_rate = decay_rate

    def get_underlying_mid(self, state: TradingState) -> int:
        underlying = "VOLCANIC_ROCK"
        od = state.order_depths.get(underlying)
        if od and od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            return (best_bid + best_ask) // 2
        return 10000

    def act(self, state: TradingState) -> None:
        underlying_mid = self.get_underlying_mid(state)
        self.window.append(underlying_mid)
        if len(self.window) < self.window_size:
            return
        mu = sum(self.window) / len(self.window)
        variance = sum((p - mu) ** 2 for p in self.window) / len(self.window)
        sigma = math.sqrt(variance)
        try:
            strike = int(self.symbol.split("_")[-1])
        except Exception:
            strike = underlying_mid
        intrinsic = max(underlying_mid - strike, 0)
        # Introduce time decay: assume each tick decreases the premium by (timestamp/100 * decay_rate)
        time_decay = (state.timestamp / 100) * self.decay_rate
        fair_value = intrinsic + self.base_premium - time_decay
        position = state.position.get(self.symbol, 0)
        adjusted_fair = fair_value - self.alpha * position
        target_bid = int(adjusted_fair - self.k * sigma)
        target_ask = int(adjusted_fair + self.k * sigma)
        capacity_buy = self.limit - position
        capacity_sell = self.limit + position
        if target_bid < adjusted_fair and capacity_buy > 0:
            qty = min(self.order_size, capacity_buy)
            self.buy(target_bid, qty)
        if target_ask > adjusted_fair and capacity_sell > 0:
            qty = min(self.order_size, capacity_sell)
            self.sell(target_ask, qty)

# -------------------------------
# Dedicated Strategy for KELP
# -------------------------------
class DynamicKelpStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, window_size: int = 15, k: float = 1.2, alpha: float = 0.6, order_size: int = 4) -> None:
        super().__init__(symbol, limit)
        self.window = deque(maxlen=window_size)
        self.window_size = window_size
        self.k = k
        self.alpha = alpha
        self.order_size = order_size

    def get_current_mid(self, state: TradingState) -> int:
        od = state.order_depths[self.symbol]
        if od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            return (best_bid + best_ask) // 2
        if self.window:
            return self.window[-1]
        return 2030

    def act(self, state: TradingState) -> None:
        current_mid = self.get_current_mid(state)
        self.window.append(current_mid)
        if len(self.window) < self.window_size:
            return
        mu = sum(self.window) / len(self.window)
        sigma = math.sqrt(sum((p - mu)**2 for p in self.window) / len(self.window))
        position = state.position.get(self.symbol, 0)
        adjusted_fair = mu - self.alpha * position
        target_bid = int(adjusted_fair - self.k * sigma)
        target_ask = int(adjusted_fair + self.k * sigma)
        capacity_buy = self.limit - position
        capacity_sell = self.limit + position
        if target_bid < current_mid and capacity_buy > 0:
            qty = min(self.order_size, capacity_buy)
            self.buy(target_bid, qty)
        if target_ask > current_mid and capacity_sell > 0:
            qty = min(self.order_size, capacity_sell)
            self.sell(target_ask, qty)

# -------------------------------
# Dedicated Strategy for PICNIC_BASKET2
# -------------------------------
class DynamicPicnicBasket2Strategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, window_size: int = 25, k: float = 0.8, alpha: float = 0.4, order_size: int = 5) -> None:
        super().__init__(symbol, limit)
        self.window = deque(maxlen=window_size)
        self.window_size = window_size
        self.k = k
        self.alpha = alpha
        self.order_size = order_size

    def get_current_mid(self, state: TradingState) -> int:
        od = state.order_depths[self.symbol]
        if od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            return (best_bid + best_ask) // 2
        if self.window:
            return self.window[-1]
        return 30000  # a baseline for picnic basket 2

    def act(self, state: TradingState) -> None:
        current_mid = self.get_current_mid(state)
        self.window.append(current_mid)
        if len(self.window) < self.window_size:
            return
        mu = sum(self.window) / len(self.window)
        sigma = math.sqrt(sum((p - mu)**2 for p in self.window) / len(self.window))
        position = state.position.get(self.symbol, 0)
        adjusted_fair = mu - self.alpha * position
        target_bid = int(adjusted_fair - self.k * sigma)
        target_ask = int(adjusted_fair + self.k * sigma)
        capacity_buy = self.limit - position
        capacity_sell = self.limit + position
        if target_bid < current_mid and capacity_buy > 0:
            qty = min(self.order_size, capacity_buy)
            self.buy(target_bid, qty)
        if target_ask > current_mid and capacity_sell > 0:
            qty = min(self.order_size, capacity_sell)
            self.sell(target_ask, qty)

# -------------------------------
# Trader Class Managing All Instruments
# -------------------------------
class Trader:
    def __init__(self) -> None:
        underlying_symbols = [
            "VOLCANIC_ROCK", "CROISSANTS", "DJEMBES", "JAMS",
            "KELP", "PICNIC_BASKET1", "PICNIC_BASKET2",
            "RAINFOREST_RESIN", "SQUID_INK"
        ]
        option_symbols = [
            "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500", "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750"
        ]
        limits = {symbol: 50 for symbol in underlying_symbols + option_symbols}
        self.strategies: Dict[Symbol, Strategy] = {}
        for symbol in underlying_symbols:
            if symbol == "KELP":
                self.strategies[symbol] = DynamicKelpStrategy(symbol, limits[symbol])
            elif symbol == "PICNIC_BASKET2":
                self.strategies[symbol] = DynamicPicnicBasket2Strategy(symbol, limits[symbol])
            else:
                self.strategies[symbol] = DynamicStatisticalMarketMakerStrategy(symbol, limits[symbol])
        for symbol in option_symbols:
            self.strategies[symbol] = DynamicOptionMarketMakerStrategy(symbol, limits[symbol])

    def run(self, state: TradingState) -> (Dict[Symbol, List[Order]], int, str):
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        orders: Dict[Symbol, List[Order]] = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol))
            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)
            new_trader_data[symbol] = strategy.save()
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

# -------------------------------
# Simulation Engine
# -------------------------------
def simulate_trading():
    trader = Trader()
    trader_data = ""
    instruments = list(trader.strategies.keys())
    positions = {symbol: 0 for symbol in instruments}
    listings = {symbol: Listing(symbol, symbol.replace("_", " ").title(), "USD") for symbol in instruments}
    base_prices = {
        "VOLCANIC_ROCK": 10000,
        "CROISSANTS": 4300,
        "DJEMBES": 13400,
        "JAMS": 6600,
        "KELP": 2030,
        "PICNIC_BASKET1": 59000,
        "PICNIC_BASKET2": 30400,
        "RAINFOREST_RESIN": 10000,
        "SQUID_INK": 2000
    }
    for tick in range(50):
        timestamp = tick * 100
        order_depths: Dict[Symbol, OrderDepth] = {}
        # Simulate underlyings
        for symbol, base in base_prices.items():
            od = OrderDepth(
                buy_orders={
                    base - random.randint(0, 5): random.randint(5, 15),
                    (base - 5) - random.randint(0, 5): random.randint(5, 10)
                },
                sell_orders={
                    base + random.randint(0, 5): -random.randint(5, 15),
                    (base + 5) + random.randint(0, 5): -random.randint(5, 10)
                }
            )
            order_depths[symbol] = od
        # Simulate options based on "VOLCANIC_ROCK" base
        underlying_base = base_prices["VOLCANIC_ROCK"]
        for option in [s for s in instruments if s.startswith("VOLCANIC_ROCK_VOUCHER")]:
            try:
                strike = int(option.split("_")[-1])
            except:
                strike = underlying_base
            intrinsic = max(underlying_base - strike, 0)
            # Base fair value: intrinsic plus premium (100) then time decay will be applied in strategy
            fair_value = intrinsic + 100
            od = OrderDepth(
                buy_orders={fair_value - random.randint(1, 3): random.randint(3, 10)},
                sell_orders={fair_value + random.randint(1, 3): -random.randint(3, 10)}
            )
            order_depths[option] = od
        own_trades = {symbol: [] for symbol in instruments}
        market_trades = {symbol: [] for symbol in instruments}
        observations = Observation([], {})
        state = TradingState(timestamp, trader_data, listings, order_depths, own_trades, market_trades, positions.copy(), observations)
        orders, conversions, new_trader_data = trader.run(state)
        for symbol, order_list in orders.items():
            for order in order_list:
                positions[symbol] += order.quantity
                print(f"Tick {timestamp}: {symbol} executed order -> {order}")
        trader_data = new_trader_data
        print(f"Tick {timestamp}: Updated Positions -> {positions}\n")

if __name__ == "__main__":
    simulate_trading()
