from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import math

class Trader:
    def __init__(self):
        self.position_limits = {"PEARLS": 20}

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        product = "PEARLS"

        if product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            position_limit = self.position_limits.get(product, 20)
            current_position = state.position.get(product, 0)

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if best_bid is not None and best_ask is not None:
                mid_point = (best_bid + best_ask) / 2.0
                target_buy_price = math.floor(mid_point - 1)
                target_sell_price = math.ceil(mid_point + 1)

                available_buy_capacity = position_limit - current_position
                if available_buy_capacity > 0:
                    buy_order_size = 5
                    place_buy_volume = min(buy_order_size, available_buy_capacity)
                    orders.append(Order(product, target_buy_price, place_buy_volume))

                available_sell_capacity = position_limit + current_position
                if available_sell_capacity > 0:
                    sell_order_size = 5
                    place_sell_volume = min(sell_order_size, available_sell_capacity)
                    orders.append(Order(product, target_sell_price, -place_sell_volume))

            if orders:
                result[product] = orders

        return result, conversions, traderData