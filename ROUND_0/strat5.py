from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import math

class Trader:
    def __init__(self):
        self.kelp_prices = []  # For KELP price history
        self.kelp_vwap = []    # For KELP VWAP history
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.kelp_price_history = []  # To store mid-prices for moving average

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Adjusted to handle edge cases with default values
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value], default=fair_value + 1)
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value], default=fair_value - 1)

        # Take opportunities more aggressively
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value:  # Tightened condition to capture more trades
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value:  # Tightened condition
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -quantity))
                    sell_order_volume += quantity

        # Clear positions more aggressively to avoid holding large positions
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, 1
        )

        # Market making with tighter spreads
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", fair_value - 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", fair_value + 1, -sell_quantity))

        return orders

    def kelp_fair_value(self, order_depth: OrderDepth) -> float:
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return 5000  # Fallback value (adjust based on sample data)

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 10]  # Lowered volume filter
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 10]
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid
        mid_price = (mm_ask + mm_bid) / 2

        # Use a weighted moving average for fair value
        self.kelp_price_history.append(mid_price)
        if len(self.kelp_price_history) > 5:  # Short window for responsiveness
            self.kelp_price_history.pop(0)
        
        weights = [1, 2, 3, 4, 5][:len(self.kelp_price_history)]  # Higher weight to recent prices
        weighted_sum = sum(p * w for p, w in zip(self.kelp_price_history, weights))
        total_weight = sum(weights)
        fair_value = weighted_sum / total_weight if total_weight > 0 else mid_price

        return fair_value

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, width: float, take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            fair_value = self.kelp_fair_value(order_depth)

            # Take opportunities with tighter thresholds
            if best_ask <= fair_value - take_width:
                ask_amount = -order_depth.sell_orders[best_ask]
                quantity = min(ask_amount, position_limit - position, 25)  # Increased cap to 25
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity

            if best_bid >= fair_value + take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                quantity = min(bid_amount, position_limit + position, 25)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -quantity))
                    sell_order_volume += quantity

            # Clear positions
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 1
            )

            # Market making with tighter spreads
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP", round(fair_value - width), buy_quantity))

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", round(fair_value + width), -sell_quantity))

        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int) -> tuple:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # More aggressive position clearing
        if position_after_take > 10:  # Start unwinding if position exceeds 10
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

        if position_after_take < -10:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def run(self, state: TradingState) -> tuple:
        result = {}
        resin_fair_value = 10000  # Adjusted to a more realistic value (based on AMETHYSTS analogy)
        resin_width = 1  # Tighter spread for more trades
        kelp_make_width = 1.5  # Tighter spread
        kelp_take_width = 0.5  # More aggressive taking
        timespan = 5  # Shorter window for responsiveness

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position.get("RAINFOREST_RESIN", 0)
            resin_orders = self.resin_orders(
                state.order_depths["RAINFOREST_RESIN"], resin_fair_value, resin_width, resin_position, self.position_limits["RAINFOREST_RESIN"]
            )
            result["RAINFOREST_RESIN"] = resin_orders

        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"], timespan, kelp_make_width, kelp_take_width, kelp_position, self.position_limits["KELP"]
            )
            result["KELP"] = kelp_orders

        traderData = jsonpickle.encode({"kelp_prices": self.kelp_prices, "kelp_vwap": self.kelp_vwap, "kelp_price_history": self.kelp_price_history})
        conversions = 0  # No conversions in tutorial round

        return result, conversions, traderData