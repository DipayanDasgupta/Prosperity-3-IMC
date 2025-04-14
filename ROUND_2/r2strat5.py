import json
import jsonpickle
import math
from datamodel import Order, TradingState, OrderDepth
from typing import List

# Parameters for each product. Adjust these based on your historical analysis.
PARAMS = {
    "RAINFOREST_RESIN": {
        "fair_value": 100,   # preset fair value for a stable product
        "take_width": 1,
        "clear_width": 0.5,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 40,  # internal risk limit
    },
    "KELP": {
        "fair_value": 50,
        "take_width": 1,
        "clear_width": 0.5,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 40,
    },
    "SQUID_INK": {  
        # For volatile products, use dynamic fair value
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
    "PICNIC_BASKET1": {
        "fair_value": 150,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 50,
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

# Exchange-imposed hard limits.
LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

    def dynamic_fair_value(self, product: str, order_depth: OrderDepth, traderObject: dict) -> float:
        # Only for volatile products like SQUID_INK.
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return self.params[product]["fair_value"]
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        current_mid = (best_ask + best_bid) / 2
        last_price = traderObject.get(product + "_last_price", current_mid)
        returns = (current_mid - last_price) / last_price if last_price != 0 else 0
        adjustment = returns * self.params[product].get("reversion_beta", 0)
        new_fair_value = current_mid + (current_mid * adjustment)
        traderObject[product + "_last_price"] = current_mid
        return new_fair_value

    def compute_risk_factor(self, product: str, position: int) -> float:
        # The closer current position is to the soft limit, the smaller the factor.
        soft_limit = self.params[product].get("soft_position_limit", 40)
        # When position is zero, risk_factor=1; when near soft_limit, it diminishes (minimum 0.2 to always allow some orders).
        factor = max(0.2, (soft_limit - abs(position)) / soft_limit)
        return factor

    def take_best_orders(self, product: str, fair_value: float, take_width: float, 
                         orders: List[Order], order_depth: OrderDepth, position: int, 
                         buy_order_volume: int, sell_order_volume: int, prevent_adverse: bool=False, 
                         adverse_volume: int=0) -> (int, int):
        position_limit = LIMITS[product]
        risk_factor = self.compute_risk_factor(product, position)
        # Take liquidity on the buy side
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_qty = -order_depth.sell_orders[best_ask]
            # Only take if the best ask is favorably below fair value by the take_width.
            if (not prevent_adverse or best_ask_qty <= adverse_volume) and best_ask <= fair_value - take_width:
                # Scale quantity based on risk factor.
                quantity = min(math.floor(best_ask_qty * risk_factor), position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
        # Take liquidity on the sell side
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_qty = order_depth.buy_orders[best_bid]
            if (not prevent_adverse or best_bid_qty <= adverse_volume) and best_bid >= fair_value + take_width:
                quantity = min(math.floor(best_bid_qty * risk_factor), position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, 
                    position: int, buy_order_volume: int, sell_order_volume: int) -> (int, int):
        position_limit = LIMITS[product]
        risk_factor = self.compute_risk_factor(product, position)
        # Scale down orders based on risk factor.
        buy_qty = math.floor((position_limit - (position + buy_order_volume)) * risk_factor)
        if buy_qty > 0:
            orders.append(Order(product, round(bid), buy_qty))
        sell_qty = math.floor((position_limit + (position - sell_order_volume)) * risk_factor)
        if sell_qty > 0:
            orders.append(Order(product, round(ask), -sell_qty))
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: int, 
                             orders: List[Order], order_depth: OrderDepth, position: int, 
                             buy_order_volume: int, sell_order_volume: int) -> (int, int):
        position_after = position + buy_order_volume - sell_order_volume
        fair_bid = round(fair_value - width)
        fair_ask = round(fair_value + width)
        buy_qty = LIMITS[product] - (position + buy_order_volume)
        sell_qty = LIMITS[product] + (position - sell_order_volume)
        risk_factor = self.compute_risk_factor(product, position)
        if position_after > 0:
            clear_vol = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_ask)
            clear_vol = min(clear_vol, position_after)
            send_qty = min(math.floor(sell_qty * risk_factor), clear_vol)
            if send_qty > 0:
                orders.append(Order(product, fair_ask, -abs(send_qty)))
                sell_order_volume += abs(send_qty)
        if position_after < 0:
            clear_vol = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_bid)
            clear_vol = min(clear_vol, abs(position_after))
            send_qty = min(math.floor(buy_qty * risk_factor), clear_vol)
            if send_qty > 0:
                orders.append(Order(product, fair_bid, abs(send_qty)))
                buy_order_volume += abs(send_qty)
        return buy_order_volume, sell_order_volume

    def make_orders(self, product: str, order_depth: OrderDepth, fair_value: float, 
                    position: int, buy_order_volume: int, sell_order_volume: int, 
                    disregard_edge: float, join_edge: float, default_edge: float, 
                    manage_position: bool=False, soft_position_limit: int=0) -> (List[Order], int, int):
        orders: List[Order] = []
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
        
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(product, orders, bid, ask, position, buy_order_volume, sell_order_volume)
        return orders, buy_order_volume, sell_order_volume

    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, 
                    take_width: float, position: int, prevent_adverse: bool=False, 
                    adverse_volume: int=0) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position, 
            buy_order_volume, sell_order_volume, prevent_adverse, adverse_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, 
                     clear_width: int, position: int, buy_order_volume: int, 
                     sell_order_volume: int) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position, 
            buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}

        # Process each product defined in our parameters.
        for product in self.params:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                od = state.order_depths[product]
                params = self.params[product]
                # For volatile products, update fair value dynamically.
                if product == "SQUID_INK":
                    fair_value = self.dynamic_fair_value(product, od, traderObject)
                else:
                    fair_value = params["fair_value"]
                # Take liquidity orders.
                take_list, b_vol, s_vol = self.take_orders(
                    product, od, fair_value, params["take_width"], position,
                    params.get("prevent_adverse", False), params.get("adverse_volume", 0)
                )
                # Clear orders to limit position risk.
                clear_list, b_vol, s_vol = self.clear_orders(
                    product, od, fair_value, params["clear_width"], position, b_vol, s_vol
                )
                # Market making orders.
                make_list, _, _ = self.make_orders(
                    product, od, fair_value, position, b_vol, s_vol,
                    params["disregard_edge"], params["join_edge"], params["default_edge"],
                    manage_position=True, soft_position_limit=params.get("soft_position_limit", 0)
                )
                result[product] = take_list + clear_list + make_list

        # Composite products (baskets) can be added later if desired.
        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData
