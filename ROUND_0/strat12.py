#Linear Regression Based Model 1
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import math
import numpy as np

# Define product constants
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"

# Define parameters
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "base_fair_value": 10000,
        "trade_adjustment_weight": 0.5,
        "take_width": 0.4,  # Wider to take more profitable trades
        "alpha_weight": 0.3,  # Weight for linear regression alpha
        "window_size": 10,  # Window for linear regression
    },
    Product.KELP: {
        "take_width": 0.4,
        "alpha_weight": 0.5,  # Higher weight for KELP since it relies more on prediction
        "window_size": 10,
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_price_history = []
        self.resin_trade_prices = []
        self.resin_price_history = []
        self.kelp_mid_prices = []  # For linear regression
        self.resin_mid_prices = []  # For linear regression
        self.position_limits = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50}

    def linear_regression_alpha(self, mid_prices: List[float], window_size: int) -> float:
        if len(mid_prices) < window_size:
            return 0.0

        # Use the last window_size mid-prices
        y = np.array(mid_prices[-window_size:])
        x = np.arange(window_size).reshape(-1, 1)

        # Add a column of ones for the intercept
        X = np.hstack([np.ones((window_size, 1)), x])

        # Linear regression: (X^T X)^(-1) X^T y
        try:
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            return 0.0  # If matrix is singular, return 0 alpha

        # Predict the next mid-price (at x = window_size)
        next_x = np.array([1, window_size])
        predicted_price = next_x @ beta

        # Current mid-price
        current_price = mid_prices[-1]

        # Alpha is the predicted price change
        alpha = predicted_price - current_price
        return alpha

    def resin_fair_value(self, state: TradingState) -> int:
        base_fair_value = self.params[Product.RAINFOREST_RESIN]["base_fair_value"]
        order_depth = state.order_depths.get(Product.RAINFOREST_RESIN, OrderDepth())
        mid_price = base_fair_value
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

        # Store mid-price for linear regression
        self.resin_mid_prices.append(mid_price)
        if len(self.resin_mid_prices) > 50:
            self.resin_mid_prices.pop(0)

        # Compute alpha using linear regression
        alpha = self.linear_regression_alpha(self.resin_mid_prices, self.params[Product.RAINFOREST_RESIN]["window_size"])

        trade_adjustment = 0
        if Product.RAINFOREST_RESIN in state.market_trades and state.market_trades[Product.RAINFOREST_RESIN]:
            recent_trade = state.market_trades[Product.RAINFOREST_RESIN][-1]
            self.resin_trade_prices.append(recent_trade.price)
            if len(self.resin_trade_prices) > 10:
                self.resin_trade_prices.pop(0)
            avg_trade_price = sum(self.resin_trade_prices) / len(self.resin_trade_prices)
            trade_adjustment = (avg_trade_price - mid_price) * self.params[Product.RAINFOREST_RESIN]["trade_adjustment_weight"]

        # Combine base, mid-price, trade adjustment, and alpha
        fair_value = (base_fair_value * 0.2 + mid_price * 0.4 + trade_adjustment +
                      self.params[Product.RAINFOREST_RESIN]["alpha_weight"] * alpha)
        self.resin_price_history.append(fair_value)
        if len(self.resin_price_history) > 50:
            self.resin_price_history.pop(0)

        return round(fair_value)

    def kelp_fair_value(self, order_depth: OrderDepth) -> int:
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return 5000

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        # Store mid-price for linear regression
        self.kelp_mid_prices.append(mid_price)
        if len(self.kelp_mid_prices) > 50:
            self.kelp_mid_prices.pop(0)

        # Compute alpha using linear regression
        alpha = self.linear_regression_alpha(self.kelp_mid_prices, self.params[Product.KELP]["window_size"])

        # Fair value is mid-price adjusted by alpha
        fair_value = mid_price + self.params[Product.KELP]["alpha_weight"] * alpha
        self.kelp_price_history.append(fair_value)
        if len(self.kelp_price_history) > 50:
            self.kelp_price_history.pop(0)

        return round(fair_value)

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        position_limit = self.position_limits[product]

        # Use alpha to determine trend
        alpha = 0.0
        if product == Product.RAINFOREST_RESIN and len(self.resin_mid_prices) >= self.params[product]["window_size"]:
            alpha = self.linear_regression_alpha(self.resin_mid_prices, self.params[product]["window_size"])
        elif product == Product.KELP and len(self.kelp_mid_prices) >= self.params[product]["window_size"]:
            alpha = self.linear_regression_alpha(self.kelp_mid_prices, self.params[product]["window_size"])

        trend_factor = 1.0
        if product == Product.RAINFOREST_RESIN and len(self.resin_price_history) >= 50:
            long_term_avg = sum(self.resin_price_history[-50:]) / 50
            if fair_value > long_term_avg:
                trend_factor = 0.6  # More aggressive buying
            elif fair_value < long_term_avg:
                trend_factor = 0.6  # More aggressive selling
        elif product == Product.KELP and len(self.kelp_price_history) >= 50:
            long_term_avg = sum(self.kelp_price_history[-50:]) / 50
            if fair_value > long_term_avg:
                trend_factor = 0.6
            elif fair_value < long_term_avg:
                trend_factor = 0.6

        # Amplify trend factor if alpha agrees with the trend
        if alpha > 0 and fair_value > long_term_avg:
            trend_factor *= 0.8
        elif alpha < 0 and fair_value < long_term_avg:
            trend_factor *= 0.8

        adjusted_take_width = take_width * trend_factor

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - adjusted_take_width:
                quantity = min(best_ask_amount, position_limit - position)
                quantity = min(quantity * (1.5 if trend_factor < 1 else 1.0), position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, int(quantity)))
                    buy_order_volume += int(quantity)
                    order_depth.sell_orders[best_ask] += int(quantity)
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + adjusted_take_width:
                quantity = min(best_bid_amount, position_limit + position)
                quantity = min(quantity * (1.5 if trend_factor < 1 else 1.0), position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -int(quantity)))
                    sell_order_volume += int(quantity)
                    order_depth.buy_orders[best_bid] -= int(quantity)
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return orders, buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value)
        fair_for_ask = round(fair_value)

        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)

        threshold = 15  # Lowered for more aggressive clearing
        stop_loss_threshold = 20  # More aggressive stop-loss
        profit_take_threshold = 10  # More aggressive profit-taking

        if position_after_take > threshold:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < -threshold:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        if position_after_take > 0 and fair_for_ask < fair_value - stop_loss_threshold:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid > fair_value + stop_loss_threshold:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        if position_after_take > 0 and fair_for_ask > fair_value + profit_take_threshold:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid < fair_value - profit_take_threshold:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Take orders
        take_orders, buy_order_volume, sell_order_volume = self.take_best_orders(
            Product.RAINFOREST_RESIN,
            fair_value,
            self.params[Product.RAINFOREST_RESIN]["take_width"],
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        # Clear positions and take profits
        buy_order_volume, sell_order_volume = self.clear_position_order(
            Product.RAINFOREST_RESIN,
            fair_value,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        # Market making
        alpha = self.linear_regression_alpha(self.resin_mid_prices, self.params[Product.RAINFOREST_RESIN]["window_size"])
        bid = round(fair_value - 2 + alpha * 0.5)  # Skew bid/ask based on alpha
        ask = round(fair_value + 2 + alpha * 0.5)

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.RAINFOREST_RESIN, bid, int(buy_quantity)))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.RAINFOREST_RESIN, ask, -int(sell_quantity)))

        return orders

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        fair_value = self.kelp_fair_value(order_depth)

        # Take orders
        take_orders, buy_order_volume, sell_order_volume = self.take_best_orders(
            Product.KELP,
            fair_value,
            self.params[Product.KELP]["take_width"],
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        # Clear positions and take profits
        buy_order_volume, sell_order_volume = self.clear_position_order(
            Product.KELP,
            fair_value,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        # Market making
        alpha = self.linear_regression_alpha(self.kelp_mid_prices, self.params[Product.KELP]["window_size"])
        bid = round(fair_value - 2 + alpha * 0.5)
        ask = round(fair_value + 2 + alpha * 0.5)

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.KELP, bid, int(buy_quantity)))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.KELP, ask, -int(sell_quantity)))

        return orders

    def run(self, state: TradingState):
        result = {}

        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_fair_value = self.resin_fair_value(state)
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_orders = self.resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                resin_fair_value,
                resin_position,
                self.position_limits[Product.RAINFOREST_RESIN]
            )
            result[Product.RAINFOREST_RESIN] = resin_orders

        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_orders = self.kelp_orders(
                state.order_depths[Product.KELP],
                10,
                kelp_position,
                self.position_limits[Product.KELP]
            )
            result[Product.KELP] = kelp_orders

        traderData = jsonpickle.encode({
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "kelp_price_history": self.kelp_price_history,
            "resin_trade_prices": self.resin_trade_prices,
            "resin_price_history": self.resin_price_history,
            "kelp_mid_prices": self.kelp_mid_prices,
            "resin_mid_prices": self.resin_mid_prices,
        })
        conversions = 0

        return result, conversions, traderData