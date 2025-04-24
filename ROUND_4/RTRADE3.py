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

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0
        self.act(state)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        if buy_orders and sell_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            return (popular_buy_price + popular_sell_price) / 2
        elif sell_orders:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            return popular_sell_price
        elif buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            return popular_buy_price
        else:
            return 0

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class VolcanicMarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.order_cap = 10
        self.min_spread = 0.5
        self.offset = 0.2
        self.max_loss = -10000

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            logger.print(f"No order depth for {self.symbol}")
            return
        depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        if not depth.buy_orders or not depth.sell_orders:
            logger.print(f"No valid bids/asks for {self.symbol}")
            return
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2.0
        logger.print(f"{self.symbol}: bid={best_bid}, ask={best_ask}, spread={spread}, position={position}")
        realized_pnl = sum(
            (trade.price - mid_price) * trade.quantity
            for trade in state.own_trades.get(self.symbol, [])
        )
        if realized_pnl < self.max_loss:
            if position > 0:
                qty = min(position, depth.buy_orders.get(best_bid, 0), self.order_cap)
                if qty > 0:
                    self.orders.append(Order(self.symbol, best_bid, -qty))
                    logger.print(f"Liquidate {self.symbol}: SELL {qty} @ {best_bid}")
            elif position < 0:
                qty = min(-position, abs(depth.sell_orders.get(best_ask, 0)), self.order_cap)
                if qty > 0:
                    self.orders.append(Order(self.symbol, best_ask, qty))
                    logger.print(f"Liquidate {self.symbol}: BUY {qty} @ {best_ask}")
        else:
            if spread >= self.min_spread:
                if position < self.limit:
                    qty = min(self.order_cap, self.limit - position, abs(depth.sell_orders.get(best_ask, 0)))
                    if qty > 0:
                        buy_price = int(best_bid + self.offset)
                        self.orders.append(Order(self.symbol, buy_price, qty))
                        logger.print(f"Market-make {self.symbol}: BUY {qty} @ {buy_price}")
                if position > -self.limit:
                    qty = min(self.order_cap, self.limit + position, depth.buy_orders.get(best_bid, 0))
                    if qty > 0:
                        sell_price = int(best_ask - self.offset)
                        self.orders.append(Order(self.symbol, sell_price, -qty))
                        logger.print(f"Market-make {self.symbol}: SELL {qty} @ {sell_price}")

class Trader:
    def __init__(self) -> None:
        limits = {
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }
        self.strategies: dict[Symbol, Strategy] = {
            symbol: VolcanicMarketMakingStrategy(symbol, limits[symbol])
            for symbol in limits
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        for symbol, strategy in self.strategies.items():
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