from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import math

class Trader:
    def __init__(self):
        self.kelp_prices = []  # For KELP price history
        self.kelp_vwap = []    # For KELP VWAP history
        self.kelp_price_history = []  # For moving average
        self.resin_trade_prices = []  # To track recent trade prices for RAINFOREST_RESIN
        self.resin_price_history = []  # For longer-term fair value average
        self.kelp_fair_value_history = []  # For longer-term fair value average
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}

    def resin_fair_value(self, state: TradingState) -> int:
        # Base fair value
        base_fair_value = 10000
        
        # Incorporate order book mid-price
        order_depth = state.order_depths.get("RAINFOREST_RESIN", OrderDepth())
        mid_price = base_fair_value
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

        # Adjust based on recent trades
        trade_adjustment = 0
        if "RAINFOREST_RESIN" in state.market_trades and state.market_trades["RAINFOREST_RESIN"]:
            recent_trade = state.market_trades["RAINFOREST_RESIN"][-1]
            self.resin_trade_prices.append(recent_trade.price)
            if len(self.resin_trade_prices) > 15:
                self.resin_trade_prices.pop(0)
            avg_trade_price = sum(self.resin_trade_prices) / len(self.resin_trade_prices)
            trade_adjustment = (avg_trade_price - mid_price) * 0.4  # Increased weight

        # Combine mid-price, base, and trade adjustment
        fair_value = base_fair_value * 0.3 + mid_price * 0.4 + trade_adjustment
        self.resin_price_history.append(fair_value)
        if len(self.resin_price_history) > 50:  # Long window for trend detection
            self.resin_price_history.pop(0)

        return round(fair_value)

    def get_resin_dynamic_widths(self, order_depth: OrderDepth, fair_value: int) -> tuple:
        """Dynamic widths for RAINFOREST_RESIN with trend adjustment."""
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return 0.8, 0.1
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid
        
        base_make_width = 0.8  # Tighter base width
        base_take_width = 0.1  # More aggressive taking
        if spread > 5:
            make_width = base_make_width * 0.8
            take_width = base_take_width * 0.8
        elif spread < 2:
            make_width = base_make_width * 1.2
            take_width = base_take_width * 1.2
        else:
            make_width = base_make_width
            take_width = base_take_width

        # Adjust take width based on trend
        if len(self.resin_price_history) >= 50:
            long_term_avg = sum(self.resin_price_history[-50:]) / 50
            if fair_value > long_term_avg:  # Uptrend
                take_width *= 0.9  # More aggressive buying
            elif fair_value < long_term_avg:  # Downtrend
                take_width *= 0.9  # More aggressive selling

        return make_width, take_width

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        make_width, take_width = self.get_resin_dynamic_widths(order_depth, fair_value)

        # Determine trend for quantity adjustment
        trend_factor = 1.0
        if len(self.resin_price_history) >= 50:
            long_term_avg = sum(self.resin_price_history[-50:]) / 50
            if fair_value > long_term_avg:
                trend_factor = 1.2  # Increase quantities in uptrend
            elif fair_value < long_term_avg:
                trend_factor = 1.2  # Increase quantities in downtrend

        # Take opportunities
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position, 40)  # Increased cap
                quantity = min(quantity * trend_factor, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, int(quantity)))
                    buy_order_volume += int(quantity)

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position, 40)
                quantity = min(quantity * trend_factor, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -int(quantity)))
                    sell_order_volume += int(quantity)

        # Clear positions and take profits
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, make_width
        )

        # Market making
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            buy_price = round(fair_value - make_width)
            orders.append(Order("RAINFOREST_RESIN", buy_price, int(buy_quantity)))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            sell_price = round(fair_value + make_width)
            orders.append(Order("RAINFOREST_RESIN", sell_price, -int(sell_quantity)))

        return orders

    def kelp_fair_value(self, order_depth: OrderDepth, state: TradingState) -> int:
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return 5000

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 10]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 10]
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid
        mid_price = (mm_ask + mm_bid) / 2

        # Incorporate recent market trades
        trade_adjustment = 0
        if "KELP" in state.market_trades and state.market_trades["KELP"]:
            recent_trade = state.market_trades["KELP"][-1]
            trade_adjustment = (recent_trade.price - mid_price) * 0.25  # Increased adjustment

        # Use a weighted moving average with a longer window
        self.kelp_price_history.append(mid_price)
        if len(self.kelp_price_history) > 10:  # Increased window
            self.kelp_price_history.pop(0)
        
        weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:len(self.kelp_price_history)]
        weighted_sum = sum(p * w for p, w in zip(self.kelp_price_history, weights))
        total_weight = sum(weights)
        fair_value = (weighted_sum / total_weight if total_weight > 0 else mid_price) + trade_adjustment

        fair_value = round(fair_value)
        self.kelp_fair_value_history.append(fair_value)
        if len(self.kelp_fair_value_history) > 50:
            self.kelp_fair_value_history.pop(0)

        return fair_value

    def get_dynamic_widths(self, order_depth: OrderDepth, fair_value: int) -> tuple:
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return 1.0, 0.1
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid
        
        base_make_width = 1.0  # Tighter base width
        base_take_width = 0.1  # More aggressive taking
        if spread > 5:
            make_width = base_make_width * 0.8
            take_width = base_take_width * 0.8
        elif spread < 2:
            make_width = base_make_width * 1.2
            take_width = base_take_width * 1.2
        else:
            make_width = base_make_width
            take_width = base_take_width

        # Adjust take width based on trend
        if len(self.kelp_fair_value_history) >= 50:
            long_term_avg = sum(self.kelp_fair_value_history[-50:]) / 50
            if fair_value > long_term_avg:
                take_width *= 0.9
            elif fair_value < long_term_avg:
                take_width *= 0.9

        return make_width, take_width

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, position: int, position_limit: int, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            fair_value = self.kelp_fair_value(order_depth, state)
            make_width, take_width = self.get_dynamic_widths(order_depth, fair_value)

            # Determine trend for quantity adjustment
            trend_factor = 1.0
            if len(self.kelp_fair_value_history) >= 50:
                long_term_avg = sum(self.kelp_fair_value_history[-50:]) / 50
                if fair_value > long_term_avg:
                    trend_factor = 1.2
                elif fair_value < long_term_avg:
                    trend_factor = 1.2

            # Take opportunities
            if best_ask <= fair_value - take_width:
                ask_amount = -order_depth.sell_orders[best_ask]
                quantity = min(ask_amount, position_limit - position, 40)
                quantity = min(quantity * trend_factor, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, int(quantity)))
                    buy_order_volume += int(quantity)

            if best_bid >= fair_value + take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                quantity = min(bid_amount, position_limit + position, 40)
                quantity = min(quantity * trend_factor, position_limit + position)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -int(quantity)))
                    sell_order_volume += int(quantity)

            # Clear positions and take profits
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, make_width
            )

            # Market making
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                buy_price = round(fair_value - make_width)
                orders.append(Order("KELP", buy_price, int(buy_quantity)))

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                sell_price = round(fair_value + make_width)
                orders.append(Order("KELP", sell_price, -int(sell_quantity)))

        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: float) -> tuple:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        threshold = 30  # Increased to allow larger positions
        stop_loss_threshold = 30  # Tighter stop-loss
        profit_take_threshold = 20  # Take profits if price moves favorably

        # Clear positions if over threshold
        if position_after_take > threshold:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

        if position_after_take < -threshold:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)

        # Stop-loss: close position if price moves against us
        if position_after_take > 0 and fair_for_ask < fair_value - stop_loss_threshold:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid > fair_value + stop_loss_threshold:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)

        # Profit-taking: close position if price moves favorably
        if position_after_take > 0 and fair_for_ask > fair_value + profit_take_threshold:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid < fair_value - profit_take_threshold:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def run(self, state: TradingState) -> tuple:
        result = {}
        timespan = 10  # Adjusted for KELP fair value calculation

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_fair_value = self.resin_fair_value(state)
            resin_position = state.position.get("RAINFOREST_RESIN", 0)
            resin_orders = self.resin_orders(
                state.order_depths["RAINFOREST_RESIN"], resin_fair_value, resin_position, self.position_limits["RAINFOREST_RESIN"]
            )
            result["RAINFOREST_RESIN"] = resin_orders

        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"], timespan, kelp_position, self.position_limits["KELP"], state
            )
            result["KELP"] = kelp_orders

        traderData = jsonpickle.encode({
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "kelp_price_history": self.kelp_price_history,
            "resin_trade_prices": self.resin_trade_prices,
            "resin_price_history": self.resin_price_history,
            "kelp_fair_value_history": self.kelp_fair_value_history
        })
        conversions = 0  # No conversions in tutorial round

        return result, conversions, traderData