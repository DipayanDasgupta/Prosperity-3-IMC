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
        self.resin_ema_short = None  # For RAINFOREST_RESIN short EMA
        self.resin_ema_long = None  # For RAINFOREST_RESIN long EMA
        self.kelp_ema_short = None  # For KELP short EMA
        self.kelp_ema_long = None  # For KELP long EMA
        self.resin_momentum = 0  # For RAINFOREST_RESIN momentum
        self.kelp_momentum = 0  # For KELP momentum
        self.resin_volatility = 0  # For RAINFOREST_RESIN volatility
        self.kelp_volatility = 0  # For KELP volatility

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average (EMA) for given prices."""
        if not prices:
            return None
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def calculate_momentum(self, prices: List[float], window: int = 10) -> float:
        """Calculate momentum as the slope of recent prices over a longer window."""
        if len(prices) < window:
            return 0
        recent_prices = prices[-window:]
        x = list(range(window))
        y = recent_prices
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope

    def calculate_volatility(self, prices: List[float], window: int = 20) -> float:
        """Calculate volatility as the standard deviation of price changes."""
        if len(prices) < window:
            return 0
        recent_prices = prices[-window:]
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        if not price_changes:
            return 0
        mean_change = sum(price_changes) / len(price_changes)
        variance = sum((change - mean_change) ** 2 for change in price_changes) / len(price_changes)
        return math.sqrt(variance)

    def resin_fair_value(self, state: TradingState) -> int:
        """Calculate fair value for RAINFOREST_RESIN with balanced adjustments."""
        order_depth = state.order_depths.get("RAINFOREST_RESIN", OrderDepth())
        
        # Base fair value from order book
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
        else:
            mid_price = 10000  # Fallback

        # Update price history
        self.resin_price_history.append(mid_price)
        if len(self.resin_price_history) > 50:  # Longer window for stability
            self.resin_price_history.pop(0)

        # Calculate EMAs for trend confirmation
        self.resin_ema_short = self.calculate_ema(self.resin_price_history, 10)
        self.resin_ema_long = self.calculate_ema(self.resin_price_history, 20)
        if self.resin_ema_short is None or self.resin_ema_long is None:
            self.resin_ema_short = mid_price
            self.resin_ema_long = mid_price

        # Adjust based on recent trades
        trade_adjustment = 0
        if "RAINFOREST_RESIN" in state.market_trades and state.market_trades["RAINFOREST_RESIN"]:
            recent_trade = state.market_trades["RAINFOREST_RESIN"][-1]
            self.resin_trade_prices.append(recent_trade.price)
            if len(self.resin_trade_prices) > 10:
                self.resin_trade_prices.pop(0)
            avg_trade_price = sum(self.resin_trade_prices) / len(self.resin_trade_prices)
            trade_adjustment = (avg_trade_price - mid_price) * 0.1  # Reduced weight

        # Calculate momentum and volatility
        self.resin_momentum = self.calculate_momentum(self.resin_price_history)
        self.resin_volatility = self.calculate_volatility(self.resin_price_history)

        # Combine mid-price, EMA, trade adjustment, and momentum
        fair_value = (mid_price * 0.5 + self.resin_ema_long * 0.4 + trade_adjustment) + self.resin_momentum * 20  # Reduced momentum impact
        return round(fair_value)

    def get_resin_dynamic_widths(self, order_depth: OrderDepth) -> tuple:
        """Calculate dynamic widths for RAINFOREST_RESIN with moderated adjustments."""
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return 1.5, 0.3
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid
        
        base_make_width = 1.5
        base_take_width = 0.3
        if spread > 5:
            make_width = base_make_width * 0.9
            take_width = base_take_width * 0.9
        elif spread < 2:
            make_width = base_make_width * 1.1
            take_width = base_take_width * 1.1
        else:
            make_width = base_make_width
            take_width = base_take_width

        # Adjust widths based on momentum and trend confirmation
        trend_up = self.resin_ema_short > self.resin_ema_long
        trend_down = self.resin_ema_short < self.resin_ema_long
        if trend_up and self.resin_momentum > 0:
            make_width *= 0.95  # Slightly tighter spread
            take_width *= 1.05  # Slightly more aggressive taking
        elif trend_down and self.resin_momentum < 0:
            make_width *= 1.05  # Slightly wider spread
            take_width *= 0.95  # Less aggressive taking

        # Adjust for volatility
        if self.resin_volatility > 10:  # High volatility threshold
            make_width *= 1.2  # Widen spread to reduce trading
            take_width *= 0.8  # Less aggressive taking

        return make_width, take_width

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        make_width, take_width = self.get_resin_dynamic_widths(order_depth)

        # Pause aggressive trading in high volatility
        if self.resin_volatility > 15:  # Very high volatility
            return orders

        # Take opportunities
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position, 30)
                trend_up = self.resin_ema_short > self.resin_ema_long
                if trend_up and self.resin_momentum > 0:
                    quantity = min(quantity * 1.1, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, int(quantity)))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position, 30)
                trend_down = self.resin_ema_short < self.resin_ema_long
                if trend_down and self.resin_momentum < 0:
                    quantity = min(quantity * 1.1, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -int(quantity)))
                    sell_order_volume += quantity

        # Clear positions
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, make_width
        )

        # Market making
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            buy_price = round(fair_value - make_width)
            trend_up = self.resin_ema_short > self.resin_ema_long
            if trend_up and self.resin_momentum > 0:
                buy_quantity = min(buy_quantity * 1.1, position_limit - (position + buy_order_volume))
            orders.append(Order("RAINFOREST_RESIN", buy_price, int(buy_quantity)))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            sell_price = round(fair_value + make_width)
            trend_down = self.resin_ema_short < self.resin_ema_long
            if trend_down and self.resin_momentum < 0:
                sell_quantity = min(sell_quantity * 1.1, position_limit + (position - sell_order_volume))
            orders.append(Order("RAINFOREST_RESIN", sell_price, -int(sell_quantity)))

        return orders

    def kelp_fair_value(self, order_depth: OrderDepth, state: TradingState) -> int:
        """Calculate fair value for KELP with smoother EMA."""
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
        if len(self.kelp_price_history) > 30:  # Longer window for stability
            self.kelp_price_history.pop(0)

        # Calculate EMAs
        self.kelp_ema_short = self.calculate_ema(self.kelp_price_history, 10)
        self.kelp_ema_long = self.calculate_ema(self.kelp_price_history, 20)
        if self.kelp_ema_short is None or self.kelp_ema_long is None:
            self.kelp_ema_short = mid_price
            self.kelp_ema_long = mid_price

        # Incorporate recent market trades
        trade_adjustment = 0
        if "KELP" in state.market_trades and state.market_trades["KELP"]:
            recent_trade = state.market_trades["KELP"][-1]
            trade_adjustment = (recent_trade.price - mid_price) * 0.1  # Reduced weight

        # Calculate momentum and volatility
        self.kelp_momentum = self.calculate_momentum(self.kelp_price_history)
        self.kelp_volatility = self.calculate_volatility(self.kelp_price_history)

        # Combine EMA, mid-price, trade adjustment, and momentum
        fair_value = (self.kelp_ema_long * 0.5 + mid_price * 0.4 + trade_adjustment) + self.kelp_momentum * 10  # Reduced momentum impact
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
            make_width = base_make_width * 0.9
            take_width = base_take_width * 0.9
        elif spread < 2:
            make_width = base_make_width * 1.1
            take_width = base_take_width * 1.1
        else:
            make_width = base_make_width
            take_width = base_take_width

        # Adjust widths based on momentum and trend confirmation
        trend_up = self.kelp_ema_short > self.kelp_ema_long
        trend_down = self.kelp_ema_short < self.kelp_ema_long
        if trend_up and self.kelp_momentum > 0:
            make_width *= 0.95
            take_width *= 1.05
        elif trend_down and self.kelp_momentum < 0:
            make_width *= 1.05
            take_width *= 0.95

        # Adjust for volatility
        if self.kelp_volatility > 10:
            make_width *= 1.2
            take_width *= 0.8

        return make_width, take_width

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, position: int, position_limit: int, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders and order_depth.buy_orders:
            fair_value = self.kelp_fair_value(order_depth, state)
            make_width, take_width = self.get_dynamic_widths(order_depth)

            # Pause aggressive trading in high volatility
            if self.kelp_volatility > 15:
                return orders

            # Take opportunities
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position, 30)
                trend_up = self.kelp_ema_short > self.kelp_ema_long
                if trend_up and self.kelp_momentum > 0:
                    quantity = min(quantity * 1.1, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, int(quantity)))
                    buy_order_volume += quantity

            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position, 30)
                trend_down = self.kelp_ema_short < self.kelp_ema_long
                if trend_down and self.kelp_momentum < 0:
                    quantity = min(quantity * 1.1, position_limit + position)
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
                trend_up = self.kelp_ema_short > self.kelp_ema_long
                if trend_up and self.kelp_momentum > 0:
                    buy_quantity = min(buy_quantity * 1.1, position_limit - (position + buy_order_volume))
                orders.append(Order("KELP", buy_price, int(buy_quantity)))

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                sell_price = round(fair_value + make_width)
                trend_down = self.kelp_ema_short < self.kelp_ema_long
                if trend_down and self.kelp_momentum < 0:
                    sell_quantity = min(sell_quantity * 1.1, position_limit + (position - sell_order_volume))
                orders.append(Order("KELP", sell_price, -int(sell_quantity)))

        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: float) -> tuple:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # Position clearing threshold
        threshold = 25  # Balanced threshold
        stop_loss_threshold = 30  # Stop-loss if price moves against position

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
            volatility = self.resin_volatility
        else:
            momentum = self.kelp_momentum
            volatility = self.kelp_volatility

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
        timespan = 10  # For KELP EMA

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