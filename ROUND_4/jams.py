from typing import Dict, List, Tuple
import json
import math

# Supporting classes
class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

class Order:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

class TradingState:
    def __init__(self, timestamp: int, listings: Dict[str, object], order_depths: Dict[str, OrderDepth],
                 own_trades: Dict[str, List[object]], market_trades: Dict[str, List[object]], position: Dict[str, int],
                 observations: object, traderData: str = ""):
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        self.traderData = traderData

class ProsperityEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

# Logger class
class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def flush(self, state: TradingState, orders: dict[str, list[Order]], conversions: int, trader_data: str):
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
            self.truncate(self.logs, max_item_length)
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str):
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

    def compress_listings(self, listings: dict[str, object]):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[str, OrderDepth]):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[str, list[object]]):
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in trades.values() for t in arr]

    def compress_observations(self, obs: object):
        co = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
              for p, o in obs.conversionObservations.items()}
        return [obs.plainValueObservations, co]

    def compress_orders(self, orders: dict[str, list[Order]]):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int):
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

# Product class
class Product:
    JAMS = "JAMS"

# Parameters for JAMS
PARAMS = {
    Product.JAMS: {
        "fair_value": 15,
        "take_width": 0.5,
        "clear_width": 0.2,
        "disregard_edge": 0.5,
        "join_edge": 1,
        "default_edge": 1,
        "soft_position_limit": 175
    }
}

class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.position_limits = {Product.JAMS: 350}

    def take_best_orders(self, product: str, fair_value: float, take_width: float, orders: List[Order],
                        order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int):
        position_limit = self.position_limits[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: float, orders: List[Order],
                            order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, position: int,
                    buy_order_volume: int, sell_order_volume: int):
        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, math.floor(bid), buy_quantity))
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, math.ceil(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float,
                   position: int):
        orders = []
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position, 0, 0
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: float,
                    position: int, buy_order_volume: int, sell_order_volume: int):
        orders = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(self, product: str, order_depth: OrderDepth, fair_value: float, position: int,
                    buy_order_volume: int, sell_order_volume: int, disregard_edge: float, join_edge: float,
                    default_edge: float, manage_position: bool = False, soft_position_limit: int = 0):
        orders = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        ask = round(fair_value + default_edge)
        bid = round(fair_value - default_edge)

        # Adjust ask price
        if best_ask_above_fair:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair - 1  # Place just below the best ask
            else:
                ask = best_ask_above_fair

        # Adjust bid price
        if best_bid_below_fair:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair + 1  # Place just above the best bid
            else:
                bid = best_bid_below_fair

        # Adjust for position management
        if manage_position:
            if position > soft_position_limit:
                ask -= 1  # Tighten ask to reduce position
            elif position < -soft_position_limit:
                bid += 1  # Tighten bid to reduce position

        # Place market-making orders
        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def jams_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.JAMS):
            position = state.position.get(Product.JAMS, 0)
            order_depth = state.order_depths[Product.JAMS]
            fair_value = self.params[Product.JAMS]["fair_value"]
            jams_take, bo, so = self.take_orders(
                Product.JAMS, order_depth, fair_value,
                self.params[Product.JAMS]["take_width"],
                position
            )
            jams_clear, bo, so = self.clear_orders(
                Product.JAMS, order_depth, fair_value,
                self.params[Product.JAMS]["clear_width"],
                position, bo, so
            )
            jams_make, bo, so = self.make_orders(
                Product.JAMS, order_depth, fair_value, position, bo, so,
                self.params[Product.JAMS]["disregard_edge"],
                self.params[Product.JAMS]["join_edge"],
                self.params[Product.JAMS]["default_edge"],
                True,
                self.params[Product.JAMS]["soft_position_limit"]
            )
            orders = jams_take + jams_clear + jams_make
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData and state.traderData != "":
            try:
                traderObject = json.loads(state.traderData)
            except Exception:
                traderObject = {}

        result = {Product.JAMS: []}
        conversions = 0

        if Product.JAMS in state.order_depths:
            jams_orders = self.jams_strategy(state)
            result[Product.JAMS].extend(jams_orders)

        final_result = {k: v for k, v in result.items() if v}
        traderData = json.dumps(traderObject)
        logger.flush(state, final_result, conversions, traderData)
        return final_result, conversions, traderData