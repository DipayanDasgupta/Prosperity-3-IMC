import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias
import statistics

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# Logger class (unchanged from original)
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
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

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
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

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]
        return [observations.plainValueObservations, conversion_observations]

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

# Base Strategy class (unchanged)
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

# MarketMakingStrategy (unchanged)
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
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
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
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)

# InkMeanReversionTrader (unchanged)
class InkMeanReversionTrader(Strategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.prices = deque(maxlen=300)
        self.window = 300

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return

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

        if mid_price < mean - 1 * std_dev and position < self.limit:
            self.buy(best_bid, self.limit - position)
        elif mid_price > mean + 1 * std_dev and position > -self.limit:
            self.sell(best_ask, self.limit + position)
        elif abs(mid_price - mean) <= 0.25:
            if position > 0:
                self.sell(best_bid, position)
            elif position < 0:
                self.buy(best_ask, -position)

# Basket1Trader (unchanged)
class Basket1Trader(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)

    def act(self, state: TradingState) -> None:
        required_symbols = ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]
        for sym in required_symbols:
            if sym not in state.order_depths:
                return

        basket_od = state.order_depths["PICNIC_BASKET1"]
        croissants_od = state.order_depths["CROISSANTS"]
        jams_od = state.order_depths["JAMS"]
        djembe_od = state.order_depths["DJEMBES"]

        basket_bids = sorted(basket_od.buy_orders.items(), reverse=True)
        basket_asks = sorted(basket_od.sell_orders.items())
        if not basket_bids or not basket_asks:
            return
        basket_best_bid = basket_bids[0][0]
        basket_best_ask = basket_asks[0][0]
        basket_mid = (basket_best_bid + basket_best_ask) / 2

        def get_mid(order_depth):
            bids = sorted(order_depth.buy_orders.items(), reverse=True)
            asks = sorted(order_depth.sell_orders.items())
            if not bids or not asks:
                return None
            return (bids[0][0] + asks[0][0]) / 2

        mid_croissants = get_mid(croissants_od)
        mid_jams = get_mid(jams_od)
        mid_djembe = get_mid(djembe_od)
        if mid_croissants is None or mid_jams is None or mid_djembe is None:
            return

        composite_value = 6 * mid_croissants + 3 * mid_jams + mid_djembe
        fair_value = composite_value
        deviation = basket_mid - fair_value
        pos = state.position.get("PICNIC_BASKET1", 0)
        threshold = 1

        if deviation < -threshold and pos < self.limit:
            quantity = self.limit - pos
            self.buy(basket_best_bid, quantity)
        elif deviation > threshold and pos > -self.limit:
            quantity = self.limit + pos
            self.sell(basket_best_ask, quantity)
        elif abs(deviation) < threshold and pos != 0:
            if pos > 0:
                self.sell(basket_best_ask, pos)
            elif pos < 0:
                self.buy(basket_best_bid, -pos)

# Basket2Trader (unchanged)
class Basket2Trader(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)

    def act(self, state: TradingState) -> None:
        required_symbols = ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]
        for sym in required_symbols:
            if sym not in state.order_depths:
                return

        basket_od = state.order_depths["PICNIC_BASKET2"]
        croissants_od = state.order_depths["CROISSANTS"]
        jams_od = state.order_depths["JAMS"]

        basket_bids = sorted(basket_od.buy_orders.items(), reverse=True)
        basket_asks = sorted(basket_od.sell_orders.items())
        if not basket_bids or not basket_asks:
            return
        basket_best_bid = basket_bids[0][0]
        basket_best_ask = basket_asks[0][0]
        basket_mid = (basket_best_bid + basket_best_ask) / 2

        def get_mid(order_depth):
            bids = sorted(order_depth.buy_orders.items(), reverse=True)
            asks = sorted(order_depth.sell_orders.items())
            if not bids or not asks:
                return None
            return (bids[0][0] + asks[0][0]) / 2

        mid_croissants = get_mid(croissants_od)
        mid_jams = get_mid(jams_od)
        if mid_croissants is None or mid_jams is None:
            return

        composite_value = 4 * mid_croissants + 2 * mid_jams
        fair_value = composite_value - 12
        deviation = basket_mid - fair_value
        pos = state.position.get("PICNIC_BASKET2", 0)
        threshold = 1

        if deviation < -threshold and pos < self.limit:
            quantity = min(self.limit - pos, 10)
            self.buy(basket_best_bid, quantity)
        elif deviation > threshold and pos > -self.limit:
            quantity = min(self.limit + pos, 10)
            self.sell(basket_best_ask, quantity)
        elif abs(deviation) < threshold and pos != 0:
            if pos > 0:
                self.sell(basket_best_ask, pos)
            elif pos < 0:
                self.buy(basket_best_bid, -pos)

# JamMarketMaker (unchanged)
class JamMarketMaker(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.base_order_size = 10
        self.delta = 1

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return

        order_depth = state.order_depths[self.symbol]
        bids = sorted(order_depth.buy_orders.items(), reverse=True)
        asks = sorted(order_depth.sell_orders.items())
        if not bids or not asks:
            return

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        buy_price = mid_price - self.delta
        sell_price = mid_price + self.delta
        pos = state.position.get(self.symbol, 0)

        allowed_buy = self.limit - pos
        allowed_sell = self.limit + pos
        order_size_buy = min(self.base_order_size, allowed_buy) if allowed_buy > 0 else 0
        order_size_sell = min(self.base_order_size, allowed_sell) if allowed_sell > 0 else 0

        if order_size_buy > 0:
            self.buy(int(buy_price), order_size_buy)
        if order_size_sell > 0:
            self.sell(int(sell_price), order_size_sell)

# CroissantMarketMaker (unchanged)
class CroissantMarketMaker(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.base_order_size = 25
        self.delta = 1

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return

        order_depth = state.order_depths[self.symbol]
        bids = sorted(order_depth.buy_orders.items(), reverse=True)
        asks = sorted(order_depth.sell_orders.items())
        if not bids or not asks:
            return

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        buy_price = mid_price - self.delta
        sell_price = mid_price + self.delta
        pos = state.position.get(self.symbol, 0)

        allowed_buy = self.limit - pos
        allowed_sell = self.limit + pos
        order_size_buy = min(self.base_order_size, allowed_buy) if allowed_buy > 0 else 0
        order_size_sell = min(self.base_order_size, allowed_sell) if allowed_sell > 0 else 0

        if order_size_buy > 0:
            self.buy(int(buy_price), order_size_buy)
        if order_size_sell > 0:
            self.sell(int(sell_price), order_size_sell)

# RainforestResinStrategy (unchanged)
class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000

# KelpStrategy (unchanged)
class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return round((popular_buy_price + popular_sell_price) / 2)

# New TrendVolatilityStrategy for DJEMBES (adapted from r2strat10.py with improvements)
class TrendVolatilityStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int, analysis_params: dict) -> None:
        super().__init__(symbol, limit)
        self.analysis_params = analysis_params
        self.prices = deque(maxlen=100)  # Added for mean-reversion component
        self.volatility_window = 50  # Window for volatility calculation

    def get_true_value(self, state: TradingState) -> int:
        params = self.analysis_params.get(self.symbol, {})
        base_value = params.get("median_price", 10000)
        trend = params.get("trend_slope", 0)
        volatility = params.get("volatility", 10)

        # Calculate market mid-price
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return base_value

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else base_value
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else base_value
        market_mid = (best_bid + best_ask) // 2

        # Add price to history for mean-reversion
        self.prices.append(market_mid)

        # Calculate analysis-based value with trend
        analysis_value = base_value + trend

        # Blend analysis value with market mid, dampened by volatility
        true_value = (analysis_value + market_mid) / 2

        # Add mean-reversion adjustment if enough data
        if len(self.prices) >= self.volatility_window:
            mean = statistics.mean(self.prices)
            std_dev = statistics.stdev(self.prices)
            if market_mid < mean - 1.5 * std_dev:
                true_value += 2  # Nudge up if significantly below mean
            elif market_mid > mean + 1.5 * std_dev:
                true_value -= 2  # Nudge down if significantly above mean

        return round(true_value)

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Dynamic position sizing based on confidence
        volatility = self.analysis_params.get(self.symbol, {}).get("volatility", 10)
        confidence = max(0.3, 1 - (volatility / 50))  # Scale confidence inversely with volatility
        to_buy = int(to_buy * confidence)
        to_sell = int(to_sell * confidence)

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
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
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
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return {"window": list(self.window), "prices": list(self.prices)}

    def load(self, data: JSON) -> None:
        self.window = deque(data.get("window", []))
        self.prices = deque(data.get("prices", []), maxlen=100)

# Trader class (modified to include DJEMBES)
class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 50,  # Added DJEMBES with the same limit as in r2strat10.py
        }

        # Analysis parameters for DJEMBES (taken from r2strat10.py)
        analysis_params = {
            "DJEMBES": {"median_price": 13409, "trend_slope": -10, "volatility": 30},  # Reduced trend_slope and volatility for stability
        }

        self.strategies = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": InkMeanReversionTrader,
            "PICNIC_BASKET1": Basket1Trader,
            "PICNIC_BASKET2": Basket2Trader,
            "CROISSANTS": CroissantMarketMaker,
            "JAMS": JamMarketMaker,
            "DJEMBES": lambda symbol, limit: TrendVolatilityStrategy(symbol, limit, analysis_params),  # Added DJEMBES with TrendVolatilityStrategy
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
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