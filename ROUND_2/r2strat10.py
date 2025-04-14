#!/usr/bin/env python3
import json
import jsonpickle
import math
import random
import statistics  # Added for mean-reversion strategy
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
# Logger
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

# -------------------------------
# Strategy Base Classes
# -------------------------------

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

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()
        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

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
# Strategy from r2strat4.py
# -------------------------------

class TrendVolatilityStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int, analysis_params: Dict[str, Dict]) -> None:
        super().__init__(symbol, limit)
        self.analysis_params = analysis_params

    def get_true_value(self, state: TradingState) -> int:
        params = self.analysis_params.get(self.symbol, {})
        base_value = params.get("median_price", 10000)
        trend = params.get("trend_slope", 0)
        analysis_value = base_value + trend
        order_depth = state.order_depths.get(self.symbol)
        if order_depth:
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else analysis_value
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else analysis_value
            market_mid = (best_bid + best_ask) // 2
        else:
            market_mid = analysis_value
        true_value = (analysis_value + market_mid) / 2
        return round(true_value)

# -------------------------------
# Mean Reversion Strategy from strat15K.py for KELP (and optionally SQUID_INK)
# -------------------------------

class InkMeanReversionTrader(Strategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.prices = deque(maxlen=300)
        self.window = 300

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return

        # Get Best Bid/Ask
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        if not buy_orders or not sell_orders:
            return

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        mid_price = (best_bid + best_ask) / 2
        self.prices.append(mid_price)

        if len(self.prices) < self.window:
            return

        mean = statistics.mean(self.prices)
        std_dev = statistics.stdev(self.prices)
        position = state.position.get(self.symbol, 0)

        # === BUY LOGIC ===
        if mid_price < mean - 2 * std_dev:
            # Instabuy at best ask
            self.buy(best_ask, self.limit - position)

        elif mean - 1.7 * std_dev < mid_price < mean - 0.8 * std_dev and position < self.limit:
            distance = mean - mid_price
            factor = (distance - 0.8 * std_dev) / (0.9 * std_dev)
            factor = min(max(factor, 0), 1)
            volume = int(factor * self.limit)
            self.buy(best_bid, volume)

        # === SELL LOGIC ===
        elif mid_price > mean + 2 * std_dev:
            # Insta-sell at best bid
            self.sell(best_bid, self.limit + position)

        elif mean + 0.8 * std_dev < mid_price < mean + 1.7 * std_dev and position > -self.limit:
            distance = mid_price - mean
            factor = (distance - 0.8 * std_dev) / (0.9 * std_dev)
            factor = min(max(factor, 0), 1)
            volume = int(factor * self.limit)
            self.sell(best_ask, volume)

        # === EXIT ZONE ===
        elif abs(mid_price - mean) <= 0.25:
            if position > 0:
                self.sell(best_ask, position)
            elif position < 0:
                self.buy(best_bid, -position)

    def save(self) -> Any:
        return list(self.prices)

    def load(self, data: Any) -> None:
        self.prices = deque(data, maxlen=300)

# -------------------------------
# Strategy from r2strat5.py for SQUID_INK and PICNIC_BASKET2
# -------------------------------

class AdvancedMarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, params: Dict) -> None:
        super().__init__(symbol, limit)
        self.params = params

    def dynamic_fair_value(self, state: TradingState, traderObject: Dict) -> float:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return self.params["fair_value"]
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        current_mid = (best_ask + best_bid) / 2
        last_price = traderObject.get(self.symbol + "_last_price", current_mid)
        returns = (current_mid - last_price) / last_price if last_price != 0 else 0
        adjustment = returns * self.params.get("reversion_beta", 0)
        new_fair_value = current_mid + (current_mid * adjustment)
        traderObject[self.symbol + "_last_price"] = current_mid
        return new_fair_value

    def compute_risk_factor(self, position: int) -> float:
        soft_limit = self.params.get("soft_position_limit", 40)
        factor = max(0.2, (soft_limit - abs(position)) / soft_limit)
        return factor

    def take_best_orders(self, state: TradingState, fair_value: float, orders: List[Order], position: int) -> tuple[int, int]:
        order_depth = state.order_depths[self.symbol]
        buy_order_volume = 0
        sell_order_volume = 0
        risk_factor = self.compute_risk_factor(position)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_qty = -order_depth.sell_orders[best_ask]
            prevent_adverse = self.params.get("prevent_adverse", False)
            adverse_volume = self.params.get("adverse_volume", 0)
            if (not prevent_adverse or best_ask_qty <= adverse_volume) and best_ask <= fair_value - self.params["take_width"]:
                quantity = min(math.floor(best_ask_qty * risk_factor), self.limit - position)
                if quantity > 0:
                    orders.append(Order(self.symbol, best_ask, quantity))
                    buy_order_volume += quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_qty = order_depth.buy_orders[best_bid]
            prevent_adverse = self.params.get("prevent_adverse", False)
            adverse_volume = self.params.get("adverse_volume", 0)
            if (not prevent_adverse or best_bid_qty <= adverse_volume) and best_bid >= fair_value + self.params["take_width"]:
                quantity = min(math.floor(best_bid_qty * risk_factor), self.limit + position)
                if quantity > 0:
                    orders.append(Order(self.symbol, best_bid, -quantity))
                    sell_order_volume += quantity
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, state: TradingState, fair_value: float, orders: List[Order], position: int, buy_order_volume: int, sell_order_volume: int) -> tuple[int, int]:
        order_depth = state.order_depths[self.symbol]
        position_after = position + buy_order_volume - sell_order_volume
        fair_bid = round(fair_value - self.params["clear_width"])
        fair_ask = round(fair_value + self.params["clear_width"])
        buy_qty = self.limit - (position + buy_order_volume)
        sell_qty = self.limit + (position - sell_order_volume)
        risk_factor = self.compute_risk_factor(position)
        if position_after > 0:
            clear_vol = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_ask)
            clear_vol = min(clear_vol, position_after)
            send_qty = min(math.floor(sell_qty * risk_factor), clear_vol)
            if send_qty > 0:
                orders.append(Order(self.symbol, fair_ask, -abs(send_qty)))
                sell_order_volume += abs(send_qty)
        if position_after < 0:
            clear_vol = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_bid)
            clear_vol = min(clear_vol, abs(position_after))
            send_qty = min(math.floor(buy_qty * risk_factor), clear_vol)
            if send_qty > 0:
                orders.append(Order(self.symbol, fair_bid, abs(send_qty)))
                buy_order_volume += abs(send_qty)
        return buy_order_volume, sell_order_volume

    def make_orders(self, state: TradingState, fair_value: float, position: int, buy_order_volume: int, sell_order_volume: int) -> tuple[List[Order], int, int]:
        orders = []
        order_depth = state.order_depths[self.symbol]
        disregard_edge = self.params["disregard_edge"]
        join_edge = self.params["join_edge"]
        default_edge = self.params["default_edge"]
        soft_position_limit = self.params.get("soft_position_limit", 40)
        asks_above = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        ask = round(fair_value + default_edge)
        if asks_above:
            best_ask = min(asks_above)
            if abs(best_ask - fair_value) <= join_edge:
                ask = best_ask
            else:
                ask = best_ask - 1
        bid = round(fair_value - default_edge)
        if bids_below:
            best_bid = max(bids_below)
            if abs(fair_value - best_bid) <= join_edge:
                bid = best_bid
            else:
                bid = best_bid + 1
        if position > soft_position_limit:
            ask -= 1
        elif position < -soft_position_limit:
            bid += 1
        risk_factor = self.compute_risk_factor(position)
        buy_qty = math.floor((self.limit - (position + buy_order_volume)) * risk_factor)
        if buy_qty > 0:
            orders.append(Order(self.symbol, round(bid), buy_qty))
        sell_qty = math.floor((self.limit + (position - sell_order_volume)) * risk_factor)
        if sell_qty > 0:
            orders.append(Order(self.symbol, round(ask), -sell_qty))
        return orders, buy_order_volume, sell_order_volume

    def act(self, state: TradingState) -> None:
        traderObject = state.traderData.get(self.symbol, {}) if isinstance(state.traderData, dict) else {}
        fair_value = self.dynamic_fair_value(state, traderObject) if self.symbol == "SQUID_INK" else self.params["fair_value"]
        position = state.position.get(self.symbol, 0)
        buy_order_volume, sell_order_volume = self.take_best_orders(state, fair_value, self.orders, position)
        buy_order_volume, sell_order_volume = self.clear_position_order(state, fair_value, self.orders, position, buy_order_volume, sell_order_volume)
        make_orders, buy_order_volume, sell_order_volume = self.make_orders(state, fair_value, position, buy_order_volume, sell_order_volume)
        self.orders.extend(make_orders)

    def save(self) -> Any:
        return {}

    def load(self, data: Any) -> None:
        pass

# -------------------------------
# Trader Class
# -------------------------------

class Trader:
    def __init__(self) -> None:
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
            "PICNIC_BASKET1": 50,
            "PICNIC_BASKET2": 100,
            "DJEMBES": 50,
            "KELP": 50,
            "CROISSANTS": 50,
            "JAMS": 50,
        }
        self.analysis_params = {
            "RAINFOREST_RESIN": {"median_price": 10000, "trend_slope": -0.1, "volatility": 6},
            "PICNIC_BASKET1":   {"median_price": 58710, "trend_slope": -100, "volatility": 300},
            "DJEMBES":         {"median_price": 13409, "trend_slope": -20,  "volatility": 40},
            "KELP":            {"median_price": 2034,  "trend_slope": 11,   "volatility": 11},
            "CROISSANTS":      {"median_price": 4275,  "trend_slope": 0,    "volatility": 8},
            "JAMS":            {"median_price": 6542,  "trend_slope": 0,    "volatility": 10},
        }
        self.params = {
            "SQUID_INK": {
                "fair_value": 60,
                "take_width": 1,
                "clear_width": 0.5,
                "prevent_adverse": True,
                "adverse_volume": 10,
                "reversion_beta": -0.2,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 40,
            },
            "PICNIC_BASKET2": {
                "fair_value": 100,
                "take_width": 1,
                "clear_width": 0,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 80,
            },
        }
        self.strategies = {}
        for symbol in self.limits:
            if symbol in ["KELP"]:  # Use InkMeanReversionTrader for KELP
                self.strategies[symbol] = InkMeanReversionTrader(symbol, self.limits[symbol])
            elif symbol in ["SQUID_INK", "PICNIC_BASKET2"]:
                self.strategies[symbol] = AdvancedMarketMakingStrategy(symbol, self.limits[symbol], self.params[symbol])
            else:
                self.strategies[symbol] = TrendVolatilityStrategy(symbol, self.limits[symbol], self.analysis_params)

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        conversions = 1
        old_trader_data = json.loads(state.traderData) if state.traderData else {}
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
    positions = {symbol: 0 for symbol in trader.limits}
    listings = {symbol: Listing(symbol, symbol.title(), "USD") for symbol in positions}

    for tick in range(5):
        timestamp = tick * 100
        order_depths = {}
        for symbol in positions:
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

        own_trades = {symbol: [] for symbol in positions}
        market_trades = {symbol: [] for symbol in positions}
        observations = Observation([], {})

        state = TradingState(timestamp, trader_data, listings, order_depths, own_trades, market_trades, positions.copy(), observations)

        orders, conversions, new_trader_data = trader.run(state)

        for symbol, order_list in orders.items():
            for order in order_list:
                positions[symbol] += order.quantity
                print(f"Tick {timestamp}: Executed order -> {order}")
        trader_data = new_trader_data
        print(f"Tick {timestamp}: Updated Positions -> {positions}\n")

if __name__ == "__main__":
    simulate_trading()