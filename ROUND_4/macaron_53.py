import json
from typing import Any, Dict, List
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Symbol, Observation
import jsonpickle, numpy as np, statistics
from collections import deque

# ------------------- Logger Class -------------------
class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def flush(self, state: TradingState, orders: dict[str, List[Order]], conversions: int, trader_data: str):
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

    def compress_trades(self, trades: dict[str, List[object]]):
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, obs: object):
        co = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
              for p, o in obs.conversionObservations.items()}
        return [obs.plainValueObservations, co]

    def compress_orders(self, orders: dict[str, List[Order]]):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int):
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.position_limit = 75
        self.sunlight_history = []

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""
        product = "MAGNIFICENT_MACARONS"

        if product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)

            # Get current CSI
            conv_obs = state.observations.conversionObservations[product]
            current_csi = conv_obs.sunlightIndex

            # Track CSI history
            self.sunlight_history.append(current_csi)
            if len(self.sunlight_history) > 3:
                self.sunlight_history.pop(0)

            product_orders = []

            if current_csi <= 46 and len(self.sunlight_history) == 3:
                # Trend-based strategy for CSI < 46
                prev_prev, prev, curr = self.sunlight_history
                if curr < prev < prev_prev:
                    # Buy condition
                    buy_qty = self.position_limit - position
                    if buy_qty > 0 and order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        product_orders.append(Order(product, best_ask, buy_qty))

                elif curr < 43 and curr > prev > prev_prev:
                    # Sell condition
                    sell_qty = self.position_limit + position
                    if sell_qty > 0 and order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        product_orders.append(Order(product, best_bid, -sell_qty))

            elif current_csi > 46:
                # Market making strategy for CSI > 46
                best_bids = order_depth.buy_orders.keys()
                best_asks = order_depth.sell_orders.keys()
                if best_bids and best_asks:
                    best_bid = max(best_bids)
                    best_ask = min(best_asks)
                    spread = best_ask - best_bid

                    if spread >= 2:  # ensure there's a spread to capture
                        fair_value = (best_ask + best_bid) / 2
                        edge = 1

                        # Place a buy order inside the spread
                        buy_price = best_bid + edge
                        buy_qty = min(10, self.position_limit - position)
                        if buy_qty > 0:
                            product_orders.append(Order(product, buy_price, buy_qty))

                        # Place a sell order inside the spread
                        sell_price = best_ask - edge
                        sell_qty = min(10, self.position_limit + position)
                        if sell_qty > 0:
                            product_orders.append(Order(product, sell_price, -sell_qty))

            if product_orders:
                orders[product] = product_orders

        trader_data = jsonpickle.encode({"sunlight_history": self.sunlight_history})
        final_result = {k: v for k, v in orders.items() if v}
        logger.flush(state, final_result, conversions, trader_data)
        return orders, conversions, trader_data
