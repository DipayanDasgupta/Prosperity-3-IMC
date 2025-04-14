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
        self.resin_price_history = []  # For RAINFOREST_RESIN moving average
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.kelp_ema = None  # For KELP EMA
        self.resin_momentum = 0  # For RAINFOREST_RESIN momentum
        self.kelp_momentum = 0  # For KELP momentum

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average (EMA) for given prices."""
        if not prices:
            return None
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def calculate_momentum(self, prices: List[float], window: int = 5) -> float:
        """Calculate momentum as the slope of recent prices."""
        if len(prices) < window:
            return 0
        recent_prices = prices[-window:]
        # Simple linear regression slope
        x = list(range(window))
        y = recent_prices
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope

    def resin_fair_value(self, state: TradingState) -> int:
        """Calculate fair value for RAINFOREST_RESIN with dynamic adjustments."""
        order_depth = state.order_depths.get("RAINFOREST_RESIN", OrderDepth())
        
        # Base fair value from order book
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
        else:
            mid_price = 10000  # Fallback

        # Update price history for moving average
        self.resin_price_history.append(mid_price)
        if len(self.resin_price_history) > 20:  # Longer window for stability
            self.resin_price_history.pop(0)

        # Calculate moving average
        ma = sum(self.resin_price_history) / len(self.resin_price_history)

        # Adjust based on recent trades
        trade_adjustment = 0
        if "RAINFOREST_RESIN" in state.market_trades and state.market_trades["RAINFOREST_RESIN"]:
            recent_trade = state.market_trades["RAINFOREST_RESIN"][-1]
            self.resin_trade_prices.append(recent_trade.price)
            if len(self.resin_trade_prices) > 10:
                self.resin_trade_prices.pop(0)
            avg_trade_price = sum(self.resin_trade_prices) / len(self.resin_trade_prices)
            trade_adjustment = (avg_trade_price - mid_price) * 0.2

        # Calculate momentum
        self.resin_momentum = self.calculate_momentum(self.resin_price_history)

        # Combine mid-price, moving average, trade adjustment, and momentum
        fair_value = (mid_price * 0.4 + ma * 0.4 + trade_adjustment) + self.resin_momentum * 100
        return round(fair_value)

    def get_resin_dynamic_widths(self, order_depth: OrderDepth) -> tuple:
        """Calculate dynamic widths for RAINFOREST_RESIN based on volatility."""
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return 1.5, 0.3  # Default values
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid
        
        base_make_width = 1.5
        base_take_width = 0.3
        if spread > 5:  # High volatility
            make_width = base_make_width * 0.8
            take_width = base_take_width * 0.8
        elif spread < 2:  # Low volatility
            make_width = base_make_width * 1.2
            take_width = base_take_width * 1.2
        else:
            make_width = base_make_width
            take_width = base_take_width

        # Adjust widths based on momentum
        if self.resin_momentum > 0:  # Uptrend
            make_width *= 0.9  # Tighter spread to capture trend
            take_width *= 1.1  # More aggressive taking
        elif self.resin_momentum < 0:  # Downtrend
            make_width *= 1.1  # Wider spread to avoid losses
            take_width *= 0.9  # Less aggressive taking

        return make_width, take_width

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        make_width, take_width = self.get_resin_dynamic_widths(order_depth)

        # Take opportunities
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position, 40)  # Increased cap
                if self.resin_momentum > 0:  # Uptrend, be more aggressive
                    quantity = min(quantity * 1.2, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, int(quantity)))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position, 40)
                if self.resin_momentum < 0:  # Downtrend, be more aggressive
                    quantity = min(quantity * 1.2, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -int(quantity)))
                    sell_order_volume += quantity

        # Clear positions with stop-loss
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, make_width
        )

        # Market making
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            buy_price = round(fair_value - make_width)
            if self.resin_momentum > 0:  # Uptrend, increase buy quantity
                buy_quantity = min(buy_quantity * 1.2, position_limit - (position + buy_order_volume))
            orders.append(Order("RAINFOREST_RESIN", buy_price, int(buy_quantity)))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            sell_price = round(fair_value + make_width)
            if self.resin_momentum < 0:  # Downtrend, increase sell quantity
                sell_quantity = min(sell_quantity * 1.2, position_limit + (position - sell_order_volume))
            orders.append(Order("RAINFOREST_RESIN", sell_price, -int(sell_quantity)))

        return orders

    def kelp_fair_value(self, order_depth: OrderDepth, state: TradingState) -> int:
        """Calculate fair value for KELP using EMA and momentum."""
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return 5000

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 10]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 10]
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid
        mid_price = (mm_ask + mm_bid) / 2

        # Update price history
        self.kelp_price_history.append(mid_price)
        if len(self.kelp_price_history) > 10:  # Shorter window for EMA
            self.kelp_price_history.pop(0)

        # Calculate EMA
        ema = self.calculate_ema(self.kelp_price_history, 5)
        if ema is None:
            ema = mid_price

        # Incorporate recent market trades
        trade_adjustment = 0
        if "KELP" in state.market_trades and state.market_trades["KELP"]:
            recent_trade = state.market_trades["KELP"][-1]
            trade_adjustment = (recent_trade.price - mid_price) * 0.3  # Increased adjustment

        # Calculate momentum
        self.kelp_momentum = self.calculate_momentum(self.kelp_price_history)

        # Combine EMA, mid-price, trade adjustment, and momentum
        fair_value = (ema * 0.5 + mid_price * 0.3 + trade_adjustment) + self.kelp_momentum * 50
        return round(fair_value)

    def get_dynamic_widths(self, order_depth: OrderDepth) -> tuple:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return 1.5, 0.3
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid
        
        base_make_width = 1.5
        base_take_width = 0.3
        if spread > 5:
            make_width = base_make_width * 0.8
            take_width = base_take_width * 0.8
        elif spread < 2:
            make_width = base_make_width * 1.2
            take_width = base_take_width * 1.2
        else:
            make_width = base_make_width
            take_width = base_take_width

        # Adjust widths based on momentum
        if self.kelp_momentum > 0:  # Uptrend
            make_width *= 0.9
            take_width *= 1.1
        elif self.kelp_momentum < 0:  # Downtrend
            make_width *= 1.1
            take_width *= 0.9

        return make_width, take_width

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, position: int, position_limit: int, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders and order_depth.buy_orders:
            fair_value = self.kelp_fair_value(order_depth, state)
            make_width, take_width = self.get_dynamic_widths(order_depth)

            # Take opportunities
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position, 40)  # Increased cap
                if self.kelp_momentum > 0:  # Uptrend
                    quantity = min(quantity * 1.2, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, int(quantity)))
                    buy_order_volume += quantity

            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position, 40)
                if self.kelp_momentum < 0:  # Downtrend
                    quantity = min(quantity * 1.2, position_limit + position)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -int(quantity)))
                    sell_order_volume += quantity

            # Clear positions
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, make_width
            )

            # Market making
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                buy_price = round(fair_value - make_width)
                if self.kelp_momentum > 0:  # Uptrend
                    buy_quantity = min(buy_quantity * 1.2, position_limit - (position + buy_order_volume))
                orders.append(Order("KELP", buy_price, int(buy_quantity)))

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                sell_price = round(fair_value + make_width)
                if self.kelp_momentum < 0:  # Downtrend
                    sell_quantity = min(sell_quantity * 1.2, position_limit + (position - sell_order_volume))
                orders.append(Order("KELP", sell_price, -int(sell_quantity)))

        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: float) -> tuple:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # Increased threshold to allow larger positions
        threshold = 30
        stop_loss_threshold = 50  # Stop-loss if price moves against position

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

        # Stop-loss mechanism
        if product == "RAINFOREST_RESIN":
            momentum = self.resin_momentum
        else:
            momentum = self.kelp_momentum

        if position_after_take > 0 and fair_for_ask < fair_value - stop_loss_threshold:
            # Price dropped significantly, close position
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid > fair_value + stop_loss_threshold:
            # Price rose significantly, close position
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def run(self, state: TradingState) -> tuple:
        result = {}
        timespan = 3  # Keep for compatibility

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
            "resin_price_history": self.resin_price_history
        })
        conversions = 0  # No conversions in tutorial round

        return result, conversions, traderData