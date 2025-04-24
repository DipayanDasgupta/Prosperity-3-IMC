import json
from abc import abstractmethod
from typing import Any, TypeAlias
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

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
        seen = set()
        for arr in trades.values():
            for trade in arr:
                trade_tuple = (
                    trade.symbol,
                    int(trade.price),
                    int(trade.quantity),
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                )
                if trade_tuple not in seen:
                    seen.add(trade_tuple)
                    compressed.append(
                        [
                            trade.symbol,
                            int(trade.price),
                            int(trade.quantity),
                            trade.buyer,
                            trade.seller,
                            trade.timestamp,
                        ]
                    )
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
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, int(order.price), int(order.quantity)])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: list[Order] = []
        self.conversions: int = 0

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0
        self.act(state)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        if quantity > 0:
            self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        if quantity > 0:
            self.orders.append(Order(self.symbol, price, -quantity))

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        if not buy_orders or not sell_orders:
            return 0.0  # Return 0 if no valid orders
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return (popular_buy_price + popular_sell_price) / 2

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class PicnicBasketStrategy(Strategy):
    def act(self, state: TradingState) -> None:
        required_symbols = ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"]
        if any(symbol not in state.order_depths for symbol in required_symbols):
            logger.print("Missing order depth for required symbols")
            return

        CROISSANTS = self.get_mid_price(state, "CROISSANTS")
        JAMS = self.get_mid_price(state, "JAMS")
        DJEMBES = self.get_mid_price(state, "DJEMBES")
        PICNIC_BASKET1 = self.get_mid_price(state, "PICNIC_BASKET1")

        if any(price == 0.0 for price in [CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1]):
            logger.print("Invalid mid price for one or more symbols")
            return

        diff = PICNIC_BASKET1 - 6 * CROISSANTS - 3 * JAMS - DJEMBES
        long_threshold, short_threshold = (10, 80)  # Specific to CROISSANTS

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if diff < long_threshold and order_depth.sell_orders:
            price = min(order_depth.sell_orders.keys())
            to_buy = self.limit - position
            if to_buy > 0:
                available_volume = abs(order_depth.sell_orders.get(price, 0))
                quantity = min(to_buy, available_volume)
                self.buy(price, quantity)
                logger.print(f"Going long on {self.symbol}: BUY {quantity} @ {price}")
        elif diff > short_threshold and order_depth.buy_orders:
            price = max(order_depth.buy_orders.keys())
            to_sell = self.limit + position
            if to_sell > 0:
                available_volume = order_depth.buy_orders.get(price, 0)
                quantity = min(to_sell, available_volume)
                self.sell(price, quantity)
                logger.print(f"Going short on {self.symbol}: SELL {quantity} @ {price}")

class Trader:
    def __init__(self) -> None:
        self.strategies: dict[Symbol, Strategy] = {
            "CROISSANTS": PicnicBasketStrategy("CROISSANTS", 250)
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        symbol = "CROISSANTS"
        strategy = self.strategies[symbol]

        if symbol in old_trader_data:
            strategy.load(old_trader_data[symbol])

        if symbol in state.order_depths:
            strategy_orders, strategy_conversions = strategy.run(state)
            orders[symbol] = strategy_orders
            conversions += strategy_conversions

        new_trader_data[symbol] = strategy.save()
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data