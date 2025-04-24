from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np

class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

PARAMS = {
    Product.MAGNIFICENT_MACARONS: {
        "fast_ema_period": 5,
        "slow_ema_period": 20,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "atr_period": 14,
        "take_width_base": 1.0,  # Reduced from 1.5
        "min_edge_base": 2.0,
        "position_limit": 75,
        "position_clear_threshold": 0.5,
        "trailing_stop_factor": 0.02,
        "min_profit_factor": 0.03,  # Reduced from 0.05
        "min_volume_threshold": 50,  # Reduced from 75
        "history_window": 20,
        "order_size": 10,
    }
}

class Trader:
    def __init__(self, params=PARAMS):
        self.params = params
        self.price_history = {Product.MAGNIFICENT_MACARONS: []}
        self.fast_ema = {Product.MAGNIFICENT_MACARONS: None}
        self.slow_ema = {Product.MAGNIFICENT_MACARONS: None}
        self.rsi = {Product.MAGNIFICENT_MACARONS: None}
        self.atr = {Product.MAGNIFICENT_MACARONS: None}
        self.position = {Product.MAGNIFICENT_MACARONS: 0}
        self.pnl = 0
        self.fast_macd_ema = {Product.MAGNIFICENT_MACARONS: None}
        self.slow_macd_ema = {Product.MAGNIFICENT_MACARONS: None}
        self.signal_ema = {Product.MAGNIFICENT_MACARONS: None}
        self.macd_line = {Product.MAGNIFICENT_MACARONS: None}
        self.prev_macd_line = {Product.MAGNIFICENT_MACARONS: None}
        self.prev_signal_ema = {Product.MAGNIFICENT_MACARONS: None}
        # Faster MACD parameters
        self.fast_macd_period = 6   # Reduced from 12
        self.slow_macd_period = 13  # Reduced from 26
        self.signal_period = 4      # Reduced from 9
        self.alpha_fast_macd = 2 / (self.fast_macd_period + 1)
        self.alpha_slow_macd = 2 / (self.slow_macd_period + 1)
        self.alpha_signal = 2 / (self.signal_period + 1)

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None

    def calculate_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return None
        ema = prices[0]
        alpha = 2 / (period + 1)
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def calculate_rsi(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return None
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
            else:
                losses.append(-change)
        avg_gain = np.mean(gains[-period:]) if gains else 0
        avg_loss = np.mean(losses[-period:]) if losses else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return None
        trs = []
        for i in range(1, len(prices)):
            high = max(prices[i], prices[i - 1])
            low = min(prices[i], prices[i - 1])
            tr = high - low
            trs.append(tr)
        return np.mean(trs[-period:])

    def take_orders(self, product: str, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders = []
        position = state.position.get(product, 0)
        pos_limit = self.params[product]["position_limit"]
        fast_ema = self.fast_ema[product]
        slow_ema = self.slow_ema[product]
        macd_line = self.macd_line[product]
        signal_ema = self.signal_ema[product]
        prev_macd_line = self.prev_macd_line[product]
        prev_signal_ema = self.prev_signal_ema[product]
        atr = self.atr[product]
        if fast_ema is None or slow_ema is None or macd_line is None or signal_ema is None or atr is None:
            return orders

        min_profit = atr * self.params[product]["min_profit_factor"]
        take_width = atr * self.params[product]["take_width_base"]
        fair_value = (fast_ema + slow_ema) / 2  # Adjusted fair value

        # Detect MACD crossover
        macd_buy_signal = prev_macd_line is not None and prev_signal_ema is not None and prev_macd_line <= prev_signal_ema and macd_line > signal_ema
        macd_sell_signal = prev_macd_line is not None and prev_signal_ema is not None and prev_macd_line >= prev_signal_ema and macd_line < signal_ema

        # Buy conditions: EMA uptrend OR MACD buy signal
        if (fast_ema > slow_ema) or macd_buy_signal:
            for price, qty in sorted(order_depth.sell_orders.items(), key=lambda x: x[0]):
                if price < fair_value - take_width and (fair_value - price) > min_profit and position + qty <= pos_limit:
                    trade_qty = min(qty, pos_limit - position)
                    orders.append(Order(product, price, trade_qty))
                    position += trade_qty

        # Sell conditions: EMA downtrend OR MACD sell signal
        if (fast_ema < slow_ema) or macd_sell_signal:
            for price, qty in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True):
                if price > fair_value + take_width and (price - fair_value) > min_profit and position - qty >= -pos_limit:
                    trade_qty = min(qty, pos_limit + position)
                    orders.append(Order(product, price, -trade_qty))
                    position -= trade_qty

        return orders

    def make_orders(self, product: str, state: TradingState) -> List[Order]:
        orders = []
        position = state.position.get(product, 0)
        pos_limit = self.params[product]["position_limit"]
        fast_ema = self.fast_ema[product]
        slow_ema = self.slow_ema[product]
        atr = self.atr[product]
        if fast_ema is None or slow_ema is None or atr is None:
            return orders

        min_edge = atr * self.params[product]["min_edge_base"]
        order_size = self.params[product]["order_size"]

        if fast_ema > slow_ema:
            # Uptrend: Place sell orders
            sell_price = int(fast_ema + min_edge)
            sell_qty = min(order_size, pos_limit + position)
            if sell_qty > 0:
                orders.append(Order(product, sell_price, -sell_qty))
        elif fast_ema < slow_ema:
            # Downtrend: Place buy orders
            buy_price = int(fast_ema - min_edge)
            buy_qty = min(order_size, pos_limit - position)
            if buy_qty > 0:
                orders.append(Order(product, buy_price, buy_qty))
        return orders

    def clear_position(self, product: str, order_depth: OrderDepth, state: TradingState) -> List[Order]:
        orders = []
        position = state.position.get(product, 0)
        pos_limit = self.params[product]["position_limit"]
        threshold = self.params[product]["position_clear_threshold"]
        mid_price = self.get_mid_price(order_depth) or 0
        fast_ema = self.fast_ema[product] or mid_price
        atr = self.atr[product] or 0

        # Clear if position exceeds threshold or trailing stop is hit
        if abs(position) > pos_limit * threshold:
            if position > 0:
                best_bid = max(order_depth.buy_orders.keys(), default=mid_price)
                qty = min(position, order_depth.buy_orders.get(best_bid, 0))
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
            elif position < 0:
                best_ask = min(order_depth.sell_orders.keys(), default=mid_price)
                qty = min(-position, order_depth.sell_orders.get(best_ask, 0))
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
        # Trailing stop logic
        elif position > 0 and mid_price < fast_ema - atr * self.params[product]["trailing_stop_factor"]:
            best_bid = max(order_depth.buy_orders.keys(), default=mid_price)
            qty = min(position, order_depth.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
        elif position < 0 and mid_price > fast_ema + atr * self.params[product]["trailing_stop_factor"]:
            best_ask = min(order_depth.sell_orders.keys(), default=mid_price)
            qty = min(-position, order_depth.sell_orders.get(best_ask, 0))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0

        if Product.MAGNIFICENT_MACARONS in state.order_depths:
            order_depth = state.order_depths[Product.MAGNIFICENT_MACARONS]
            mid_price = self.get_mid_price(order_depth)
            if mid_price:
                self.price_history[Product.MAGNIFICENT_MACARONS].append(mid_price)
                if len(self.price_history[Product.MAGNIFICENT_MACARONS]) > self.params[Product.MAGNIFICENT_MACARONS]["history_window"]:
                    self.price_history[Product.MAGNIFICENT_MACARONS].pop(0)

                prices = self.price_history[Product.MAGNIFICENT_MACARONS]
                self.fast_ema[Product.MAGNIFICENT_MACARONS] = self.calculate_ema(prices, self.params[Product.MAGNIFICENT_MACARONS]["fast_ema_period"])
                self.slow_ema[Product.MAGNIFICENT_MACARONS] = self.calculate_ema(prices, self.params[Product.MAGNIFICENT_MACARONS]["slow_ema_period"])
                self.rsi[Product.MAGNIFICENT_MACARONS] = self.calculate_rsi(prices, self.params[Product.MAGNIFICENT_MACARONS]["rsi_period"])
                self.atr[Product.MAGNIFICENT_MACARONS] = self.calculate_atr(prices, self.params[Product.MAGNIFICENT_MACARONS]["atr_period"])

                # Update MACD
                if self.fast_macd_ema[Product.MAGNIFICENT_MACARONS] is None:
                    self.fast_macd_ema[Product.MAGNIFICENT_MACARONS] = mid_price
                    self.slow_macd_ema[Product.MAGNIFICENT_MACARONS] = mid_price
                    self.signal_ema[Product.MAGNIFICENT_MACARONS] = 0
                    self.macd_line[Product.MAGNIFICENT_MACARONS] = 0
                else:
                    self.fast_macd_ema[Product.MAGNIFICENT_MACARONS] = (
                        self.alpha_fast_macd * mid_price + (1 - self.alpha_fast_macd) * self.fast_macd_ema[Product.MAGNIFICENT_MACARONS]
                    )
                    self.slow_macd_ema[Product.MAGNIFICENT_MACARONS] = (
                        self.alpha_slow_macd * mid_price + (1 - self.alpha_slow_macd) * self.slow_macd_ema[Product.MAGNIFICENT_MACARONS]
                    )
                    macd_line = self.fast_macd_ema[Product.MAGNIFICENT_MACARONS] - self.slow_macd_ema[Product.MAGNIFICENT_MACARONS]
                    self.prev_macd_line[Product.MAGNIFICENT_MACARONS] = self.macd_line[Product.MAGNIFICENT_MACARONS]
                    self.prev_signal_ema[Product.MAGNIFICENT_MACARONS] = self.signal_ema[Product.MAGNIFICENT_MACARONS]
                    self.macd_line[Product.MAGNIFICENT_MACARONS] = macd_line
                    self.signal_ema[Product.MAGNIFICENT_MACARONS] = (
                        self.alpha_signal * macd_line + (1 - self.alpha_signal) * self.signal_ema[Product.MAGNIFICENT_MACARONS]
                    )

                # Ensure sufficient liquidity
                total_volume = sum(order_depth.buy_orders.values()) + sum(abs(v) for v in order_depth.sell_orders.values())
                if total_volume > self.params[Product.MAGNIFICENT_MACARONS]["min_volume_threshold"]:
                    take_orders = self.take_orders(Product.MAGNIFICENT_MACARONS, order_depth, state)
                    make_orders = self.make_orders(Product.MAGNIFICENT_MACARONS, state)
                    clear_orders = self.clear_position(Product.MAGNIFICENT_MACARONS, order_depth, state)
                    result[Product.MAGNIFICENT_MACARONS] = take_orders + make_orders + clear_orders
                    self.position[Product.MAGNIFICENT_MACARONS] = state.position.get(Product.MAGNIFICENT_MACARONS, 0)

        trader_data = jsonpickle.encode({
            "price_history": self.price_history,
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "rsi": self.rsi,
            "atr": self.atr,
            "fast_macd_ema": self.fast_macd_ema,
            "slow_macd_ema": self.slow_macd_ema,
            "signal_ema": self.signal_ema,
            "macd_line": self.macd_line,
            "prev_macd_line": self.prev_macd_line,
            "prev_signal_ema": self.prev_signal_ema,
            "pnl": self.pnl
        })
        return result, conversions, trader_data
