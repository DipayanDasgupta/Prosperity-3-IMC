from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import json, jsonpickle, math, numpy as np

# Logger class (unchanged, required for flushing)
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
        return json.dumps(value, separators=(",", ":"))

    def truncate(self, value: str, max_length: int):
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

# Product class (only PICNIC_BASKET2 and components)
class Product:
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"

# Parameters for PICNIC_BASKET2
PARAMS = {
    Product.PICNIC_BASKET2: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 2,
        "synthetic_weight": 0.03,
        "volatility_window_size": 10,
        "adverse_volatility": 0.1,
    },
}

PICNIC2_WEIGHTS = {Product.CROISSANTS: 4, Product.JAMS: 2}

# Trader class (only PICNIC_BASKET2 logic)
class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.position_limits = {
            Product.PICNIC_BASKET2: 100,
        }

    def _filtered_mid(self, product: str, order_depth: OrderDepth) -> float | None:
        """Return volume-filtered midpoint to mitigate tiny orders."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        threshold = self.params.get(product, {}).get("adverse_volume", 0)
        valid_asks = [p for p, v in order_depth.sell_orders.items() if abs(v) >= threshold]
        valid_bids = [p for p, v in order_depth.buy_orders.items() if abs(v) >= threshold]
        filt_ask = min(valid_asks) if valid_asks else None
        filt_bid = max(valid_bids) if valid_bids else None
        if filt_ask is not None and filt_bid is not None:
            return (filt_ask + filt_bid) / 2
        return (best_ask + best_bid) / 2

    def basket2_fair_value(
        self,
        basket_depth: OrderDepth,
        cro_depth: OrderDepth,
        jam_depth: OrderDepth,
        position: int,
        traderObject: dict,
    ) -> float | None:
        """Synthetic fair value for PICNIC_BASKET2 using croissants & jams."""
        mid = self._filtered_mid(Product.PICNIC_BASKET2, basket_depth)
        cro_mid = self._filtered_mid(Product.CROISSANTS, cro_depth)
        jam_mid = self._filtered_mid(Product.JAMS, jam_depth)
        if mid is None:
            return None
        if cro_mid is None or jam_mid is None:
            return mid
        synthetic_mid = (
            cro_mid * PICNIC2_WEIGHTS[Product.CROISSANTS]
            + jam_mid * PICNIC2_WEIGHTS[Product.JAMS]
        )
        weight = self.params[Product.PICNIC_BASKET2]["synthetic_weight"]
        if position:
            ratio = abs(position) / self.position_limits[Product.PICNIC_BASKET2]
            weight *= math.exp(-ratio)
        return (1 - weight) * mid + weight * synthetic_mid

    def take_best_orders(self, product: str, fair_value: float, take_width: float, orders: List[Order],
                        order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int,
                        prevent_adverse: bool = False, adverse_volume: int = 0):
        position_limit = self.position_limits[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
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
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order],
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
                   position: int, prevent_adverse: bool = False, adverse_volume: int = 0):
        orders = []
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position, 0, 0,
            prevent_adverse, adverse_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: int,
                    position: int, buy_order_volume: int, sell_order_volume: int):
        orders = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(self, product, order_depth: OrderDepth, fair_value: float, position: int,
                    buy_order_volume: int, sell_order_volume: int, disregard_edge: float, join_edge: float,
                    default_edge: float):
        orders = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        ask = round(fair_value + default_edge)
        if best_ask_above_fair:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair + 1
            else:
                ask = best_ask_above_fair
        bid = round(fair_value - default_edge)
        if best_bid_below_fair:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1
        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}

        result = {Product.PICNIC_BASKET2: []}
        conversions = 0

        # PICNIC_BASKET2 synthetic-value strategy
        if Product.PICNIC_BASKET2 in self.params and all(p in state.order_depths for p in [Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS]):
            basket2_position = state.position.get(Product.PICNIC_BASKET2, 0)
            basket2_fair_value = self.basket2_fair_value(
                state.order_depths[Product.PICNIC_BASKET2],
                state.order_depths[Product.CROISSANTS],
                state.order_depths[Product.JAMS],
                basket2_position,
                traderObject
            )
            if basket2_fair_value is not None:
                b2_take, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.PICNIC_BASKET2,
                    state.order_depths[Product.PICNIC_BASKET2],
                    basket2_fair_value,
                    self.params[Product.PICNIC_BASKET2]["take_width"],
                    basket2_position,
                    self.params[Product.PICNIC_BASKET2]["prevent_adverse"],
                    self.params[Product.PICNIC_BASKET2]["adverse_volume"],
                )
                b2_clear, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.PICNIC_BASKET2,
                    state.order_depths[Product.PICNIC_BASKET2],
                    basket2_fair_value,
                    self.params[Product.PICNIC_BASKET2]["clear_width"],
                    basket2_position,
                    buy_order_volume,
                    sell_order_volume,
                )
                b2_make, _, _ = self.make_orders(
                    Product.PICNIC_BASKET2,
                    state.order_depths[Product.PICNIC_BASKET2],
                    basket2_fair_value,
                    basket2_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.PICNIC_BASKET2]["disregard_edge"],
                    self.params[Product.PICNIC_BASKET2]["join_edge"],
                    self.params[Product.PICNIC_BASKET2]["default_edge"],
                )
                result[Product.PICNIC_BASKET2].extend(b2_take + b2_clear + b2_make)

        final_result = {k: v for k, v in result.items() if v}
        traderData = jsonpickle.encode(traderObject, unpicklable=False)
        logger.flush(state, final_result, conversions, traderData)
        return final_result, conversions, traderData