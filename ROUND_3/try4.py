#!/usr/bin/env python3
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
        if isinstance(o, deque):
            return list(o)
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

# -------------------------------
# Base Strategy Class
# -------------------------------
class Strategy:
    def __init__(self, symbol: Symbol, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: List[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        if quantity > 0:
            self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        if quantity > 0:
            self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> Any:
        return None

    def load(self, data: Any) -> None:
        pass

# -------------------------------
# Rainforest Resin Strategy (from your code)
# -------------------------------
class RainforestResinStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque(maxlen=10)
        self.window_size = 10

    def get_true_value(self, state: TradingState) -> int:
        return 10000

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            logger.print(f"No order depth for {self.symbol}, skipping.")
            return

        true_value = self.get_true_value(state)
        order_depth = state.order_depths[self.symbol]
        buy_orders_dict = order_depth.buy_orders if order_depth.buy_orders else {}
        sell_orders_dict = order_depth.sell_orders if order_depth.sell_orders else {}
        buy_orders = sorted(buy_orders_dict.items(), reverse=True)
        sell_orders = sorted(sell_orders_dict.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) >= self.limit * 0.9)
        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1
        min_sell_price = true_value + 1

        if position > self.limit * 0.6:
            max_buy_price -= 1
            min_sell_price -= 1
        elif position < -self.limit * 0.6:
            max_buy_price += 1
            min_sell_price += 1

        taken_buy_qty = 0
        if sell_orders:
            best_ask_price, best_ask_vol = sell_orders[0]
            if to_buy > 0 and best_ask_price <= max_buy_price:
                quantity_to_take = min(to_buy, abs(best_ask_vol))
                self.buy(best_ask_price, quantity_to_take)
                to_buy -= quantity_to_take
                taken_buy_qty = quantity_to_take

        taken_sell_qty = 0
        if buy_orders:
            best_bid_price, best_bid_vol = buy_orders[0]
            if to_sell > 0 and best_bid_price >= min_sell_price:
                quantity_to_take = min(to_sell, best_bid_vol)
                self.sell(best_bid_price, quantity_to_take)
                to_sell -= quantity_to_take
                taken_sell_qty = quantity_to_take

        final_position_estimate = position + taken_buy_qty - taken_sell_qty

        if to_buy > 0:
            target_buy_price = max_buy_price
            if hard_liquidate and final_position_estimate < 0:
                target_buy_price = true_value
            elif soft_liquidate and final_position_estimate < 0:
                target_buy_price = max(target_buy_price, true_value - 2)
            self.buy(target_buy_price, to_buy)

        if to_sell > 0:
            target_sell_price = min_sell_price
            if hard_liquidate and final_position_estimate > 0:
                target_sell_price = true_value
            elif soft_liquidate and final_position_estimate > 0:
                target_sell_price = min(target_sell_price, true_value + 2)
            self.sell(target_sell_price, to_sell)

    def save(self) -> Any:
        return list(self.window)

    def load(self, data: Any) -> None:
        if isinstance(data, list):
            self.window = deque(data, maxlen=self.window_size)

# -------------------------------
# Strategies from First Code
# -------------------------------
class ImprovedDynamicStatisticalMarketMakerStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int,
                 ema_alpha: float = 0.2, vol_alpha: float = 0.2,
                 k: float = 1.0, alpha: float = 0.5, order_size: int = 5,
                 min_spread: int = 20, vol_threshold: float = 50.0) -> None:
        super().__init__(symbol, limit)
        self.ema = None
        self.ema_vol = None
        self.ema_alpha = ema_alpha
        self.vol_alpha = vol_alpha
        self.k = k
        self.alpha = alpha
        self.order_size = order_size
        self.min_spread = min_spread
        self.vol_threshold = vol_threshold

    def get_current_mid(self, state: TradingState) -> int:
        od = state.order_depths[self.symbol]
        if od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            return (best_bid + best_ask) // 2
        if self.ema is not None:
            return int(self.ema)
        return 10000

    def act(self, state: TradingState) -> None:
        current_mid = self.get_current_mid(state)
        if self.ema is None:
            self.ema = current_mid
            self.ema_vol = 0.0
        else:
            prev_ema = self.ema
            self.ema = self.update_ema(current_mid, self.ema, self.ema_alpha)
            deviation = abs(current_mid - prev_ema)
            self.ema_vol = self.update_ema_vol(deviation, self.ema_vol, self.vol_alpha)

        position = state.position.get(self.symbol, 0)
        adjusted_fair = self.ema - self.alpha * position
        spread = self.k * self.ema_vol
        if spread < self.min_spread:
            spread = self.min_spread

        target_bid = int(adjusted_fair - spread)
        target_ask = int(adjusted_fair + spread)

        effective_order_size = self.order_size
        if self.ema_vol > self.vol_threshold:
            effective_order_size = max(1, self.order_size // 2)

        capacity_buy = self.limit - position
        capacity_sell = self.limit + position

        if target_bid < current_mid and capacity_buy > 0:
            qty = min(effective_order_size, capacity_buy)
            self.buy(target_bid, qty)
        if target_ask > current_mid and capacity_sell > 0:
            qty = min(effective_order_size, capacity_sell)
            self.sell(target_ask, qty)

    def update_ema(self, current_value: float, previous_ema: float, alpha: float) -> float:
        return alpha * current_value + (1 - alpha) * previous_ema

    def update_ema_vol(self, current_deviation: float, previous_ema_vol: float, alpha: float) -> float:
        return alpha * current_deviation + (1 - alpha) * previous_ema_vol

    def save(self) -> Any:
        if hasattr(self, "ema"):
            return {
                "ema": self.ema,
                "ema_vol": self.ema_vol
            }
        return None

    def load(self, data: Any) -> None:
        if data is not None and "ema" in data:
            self.ema = data.get("ema")
            self.ema_vol = data.get("ema_vol")

class ImprovedDynamicOptionMarketMakerStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int,
                 ema_alpha: float = 0.2, vol_alpha: float = 0.2,
                 k: float = 0.5, alpha: float = 0.3, order_size: int = 5,
                 base_premium: int = 100, decay_rate: int = 2,
                 min_spread: int = 15, vol_threshold: float = 40.0) -> None:
        super().__init__(symbol, limit)
        self.ema = None
        self.ema_vol = None
        self.ema_alpha = ema_alpha
        self.vol_alpha = vol_alpha
        self.k = k
        self.alpha = alpha
        self.order_size = order_size
        self.base_premium = base_premium
        self.decay_rate = decay_rate
        self.min_spread = min_spread
        self.vol_threshold = vol_threshold

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
        if self.ema is None:
            self.ema = underlying_mid
            self.ema_vol = 0.0
        else:
            prev_ema = self.ema
            self.ema = self.update_ema(underlying_mid, self.ema, self.ema_alpha)
            deviation = abs(underlying_mid - prev_ema)
            self.ema_vol = self.update_ema_vol(deviation, self.ema_vol, self.vol_alpha)

        try:
            strike = int(self.symbol.split("_")[-1])
        except Exception:
            strike = underlying_mid

        intrinsic = max(underlying_mid - strike, 0)
        time_decay = (state.timestamp / 100) * self.decay_rate
        fair_value = intrinsic + self.base_premium - time_decay

        position = state.position.get(self.symbol, 0)
        adjusted_fair = fair_value - self.alpha * position
        spread = self.k * self.ema_vol
        if spread < self.min_spread:
            spread = self.min_spread

        target_bid = int(adjusted_fair - spread)
        target_ask = int(adjusted_fair + spread)

        effective_order_size = self.order_size
        if self.ema_vol > self.vol_threshold:
            effective_order_size = max(1, self.order_size // 2)

        capacity_buy = self.limit - position
        capacity_sell = self.limit + position

        if target_bid < adjusted_fair and capacity_buy > 0:
            qty = min(effective_order_size, capacity_buy)
            self.buy(target_bid, qty)
        if target_ask > adjusted_fair and capacity_sell > 0:
            qty = min(effective_order_size, capacity_sell)
            self.sell(target_ask, qty)

    def update_ema(self, current_value: float, previous_ema: float, alpha: float) -> float:
        return alpha * current_value + (1 - alpha) * previous_ema

    def update_ema_vol(self, current_deviation: float, previous_ema_vol: float, alpha: float) -> float:
        return alpha * current_deviation + (1 - alpha) * previous_ema_vol

    def save(self) -> Any:
        if hasattr(self, "ema"):
            return {
                "ema": self.ema,
                "ema_vol": self.ema_vol
            }
        return None

    def load(self, data: Any) -> None:
        if data is not None and "ema" in data:
            self.ema = data.get("ema")
            self.ema_vol = data.get("ema_vol")

class DynamicKelpStrategy(ImprovedDynamicStatisticalMarketMakerStrategy):
    def __init__(self, symbol: Symbol, limit: int,
                 ema_alpha: float = 0.2, vol_alpha: float = 0.2,
                 k: float = 1.2, alpha: float = 0.6, order_size: int = 4,
                 min_spread: int = 20, vol_threshold: float = 50.0) -> None:
        super().__init__(symbol, limit, ema_alpha, vol_alpha, k, alpha, order_size, min_spread, vol_threshold)

class DynamicPicnicBasket2Strategy(ImprovedDynamicStatisticalMarketMakerStrategy):
    def __init__(self, symbol: Symbol, limit: int,
                 ema_alpha: float = 0.2, vol_alpha: float = 0.2,
                 k: float = 0.8, alpha: float = 0.4, order_size: int = 5,
                 min_spread: int = 30, vol_threshold: float = 60.0) -> None:
        super().__init__(symbol, limit, ema_alpha, vol_alpha, k, alpha, order_size, min_spread, vol_threshold)

# -------------------------------
# Merged Trader Class
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
        all_symbols = underlying_symbols + option_symbols
        limits = {symbol: 50 for symbol in all_symbols}
        self.strategies: Dict[Symbol, Strategy] = {}

        for symbol in underlying_symbols:
            if symbol == "RAINFOREST_RESIN":
                self.strategies[symbol] = RainforestResinStrategy(symbol, limits[symbol])
            elif symbol == "KELP":
                self.strategies[symbol] = DynamicKelpStrategy(symbol, limits[symbol])
            elif symbol == "PICNIC_BASKET2":
                self.strategies[symbol] = DynamicPicnicBasket2Strategy(symbol, limits[symbol])
            else:
                self.strategies[symbol] = ImprovedDynamicStatisticalMarketMakerStrategy(symbol, limits[symbol])
        for symbol in option_symbols:
            self.strategies[symbol] = ImprovedDynamicOptionMarketMakerStrategy(symbol, limits[symbol])

    def run(self, state: TradingState) -> (Dict[Symbol, List[Order]], int, str):
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        orders: Dict[Symbol, List[Order]] = {}
        
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                try:
                    strategy.load(old_trader_data.get(symbol))
                except Exception as e:
                    logger.print(f"Error loading state for {symbol}: {e}")
            if symbol in state.order_depths:
                try:
                    orders[symbol] = strategy.run(state)
                except Exception as e:
                    logger.print(f"Error running strategy for {symbol}: {e}")
                    orders[symbol] = []
            saved_data = strategy.save()
            if saved_data is not None:
                new_trader_data[symbol] = saved_data
        
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

# -------------------------------
# Simulation Engine (for testing)
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
        underlying_base = base_prices["VOLCANIC_ROCK"]
        for option in [s for s in instruments if s.startswith("VOLCANIC_ROCK_VOUCHER")]:
            try:
                strike = int(option.split("_")[-1])
            except:
                strike = underlying_base
            intrinsic = max(underlying_base - strike, 0)
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