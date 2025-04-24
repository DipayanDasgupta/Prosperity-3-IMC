import json
import math
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any

# Minimal data models
class Order:
    def __init__(self, symbol: str, price: float, quantity: int):
        self.symbol = symbol
        self.price = int(price) if price == int(price) else price
        self.quantity = quantity

class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[float, int] = {}
        self.sell_orders: Dict[float, int] = {}

class TradingState:
    def __init__(self, timestamp: int, order_depths: Dict[str, OrderDepth], position: Dict[str, int], traderData: str):
        self.timestamp = timestamp
        self.order_depths = order_depths
        self.position = position
        self.traderData = traderData
        self.listings = {symbol: {"symbol": symbol} for symbol in order_depths.keys()}
        self.own_trades = {}
        self.market_trades = {}
        self.observations = {}

class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
    STRIKES = [9500, 9750, 10000, 10250, 10500]

# Define VOUCHER_SYMBOLS after the Product class
Product.VOUCHER_SYMBOLS = [f"{Product.VOUCHER_PREFIX}{K}" for K in Product.STRIKES]

# Logger class (simplified)
class Logger:
    def __init__(self):
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end
        print(*objects, sep=sep, end=end)

    def flush(self, state: TradingState, orders: Dict[str, List[Order]], conversions: int, trader_data: str):
        print(f"State: {state.timestamp}, Orders: {[(o.symbol, o.price, o.quantity) for k, v in orders.items() for o in v]}, Conversions: {conversions}, TraderData: {trader_data}")
        self.logs = ""

logger = Logger()

# Strategy base class
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders = []

    def buy(self, price: float, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: float, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def run(self, state: TradingState) -> Tuple[List[Order], str]:
        self.orders = []
        self.act(state)
        return self.orders, ""

    def act(self, state: TradingState) -> None:
        pass

# ParabolaFitIVStrategy
class ParabolaFitIVStrategy:
    def __init__(self, voucher: str, strike: int, adaptive: bool = False, absolute: bool = False):
        self.voucher = voucher
        self.strike = strike
        self.adaptive = adaptive
        self.absolute = absolute
        self.expiry_day = 7
        self.ticks_per_day = 1000
        self.window = 500
        self.position_limit = 200
        self.start_ts = None
        self.history = deque(maxlen=self.window)
        self.iv_cache = {}
        self.a = self.b = self.c = None

    def norm_cdf(self, x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def bs_price(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0 or S <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * self.norm_cdf(d2)

    def implied_vol(self, S, K, T, price, tol=1e-4, max_iter=50):
        key = (round(S, 1), round(K, 1), round(T, 5), round(price, 1))
        if key in self.iv_cache:
            return self.iv_cache[key]
        low, high = 1e-6, 5.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            val = self.bs_price(S, K, T, mid) - price
            if abs(val) < tol:
                self.iv_cache[key] = mid
                return mid
            if val > 0:
                high = mid
            else:
                low = mid
        return None

    def update_fit(self):
        m_vals = [m for m, v in self.history]
        v_vals = [v for m, v in self.history]
        self.a, self.b, self.c = np.polyfit(m_vals, v_vals, 2)

    def fitted_iv(self, m):
        return self.a * m ** 2 + self.b * m + self.c

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        orders = {}
        ts = state.timestamp
        if self.start_ts is None:
            self.start_ts = ts

        depth = state.order_depths.get(self.voucher, OrderDepth())
        rock_depth = state.order_depths.get(Product.VOLCANIC_ROCK, OrderDepth())
        if not depth.sell_orders or not depth.buy_orders:
            return {}, 0, ""
        if not rock_depth.sell_orders or not rock_depth.buy_orders:
            return {}, 0, ""

        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        rock_bid = max(rock_depth.buy_orders.keys())
        rock_ask = min(rock_depth.sell_orders.keys())
        spot_price = (rock_bid + rock_ask) / 2

        TTE = max(0.1, self.expiry_day - (ts - self.start_ts) / self.ticks_per_day)
        T = TTE / 365

        if self.absolute:
            iv_guess = 1.1
            fair_value = self.bs_price(spot_price, self.strike, T, iv_guess)
            mispricing = mid_price - fair_value

            position = state.position.get(self.voucher, 0)
            result = []

            if mispricing > 2 and position < self.position_limit:
                qty = min(20, self.position_limit - position)
                result.append(Order(self.voucher, best_ask, qty))
            elif mispricing < -2 and position > -self.position_limit:
                qty = min(20, self.position_limit + position)
                result.append(Order(self.voucher, best_bid, -qty))

            orders[self.voucher] = result
            return orders, 0, ""

        m_t = math.log(self.strike / spot_price) / math.sqrt(TTE)
        v_t = self.implied_vol(spot_price, self.strike, T, mid_price)
        if v_t is None or v_t < 0.5:
            return {}, 0, ""

        self.history.append((m_t, v_t))
        if len(self.history) < self.window:
            return {}, 0, ""

        self.update_fit()
        current_fit = self.fitted_iv(m_t)
        position = state.position.get(self.voucher, 0)
        result = []

        if v_t < current_fit - 0.019 and position < self.position_limit:
            qty = min(30, self.position_limit - position)
            result.append(Order(self.voucher, best_ask, qty))
        elif v_t > current_fit + 0.013 and position > -self.position_limit:
            qty = min(30, self.position_limit + position)
            result.append(Order(self.voucher, best_bid, -qty))

        orders[self.voucher] = result
        return orders, 0, ""

# VolcanicMarketMakingStrategy
class VolcanicMarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        self.order_cap = 10
        self.min_spread = 0.5
        self.offset = 0.2
        self.max_loss = -10000

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            logger.print(f"No order depth for {self.symbol}")
            return
        depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        if not depth.buy_orders or not depth.sell_orders:
            logger.print(f"No valid bids/asks for {self.symbol}")
            return
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2.0
        logger.print(f"{self.symbol}: bid={best_bid}, ask={best_ask}, spread={spread}, position={position}")
        realized_pnl = sum(
            (trade.price - mid_price) * trade.quantity
            for trade in state.own_trades.get(self.symbol, [])
        )
        if realized_pnl < self.max_loss:
            if position > 0:
                qty = min(position, depth.buy_orders.get(best_bid, 0), self.order_cap)
                if qty > 0:
                    self.orders.append(Order(self.symbol, best_bid, -qty))
                    logger.print(f"Liquidate {self.symbol}: SELL {qty} @ {best_bid}")
            elif position < 0:
                qty = min(-position, abs(depth.sell_orders.get(best_ask, 0)), self.order_cap)
                if qty > 0:
                    self.orders.append(Order(self.symbol, best_ask, qty))
                    logger.print(f"Liquidate {self.symbol}: BUY {qty} @ {best_ask}")
        else:
            if spread >= self.min_spread:
                if position < self.limit:
                    qty = min(self.order_cap, self.limit - position, abs(depth.sell_orders.get(best_ask, 0)))
                    if qty > 0:
                        buy_price = int(best_bid + self.offset)
                        self.orders.append(Order(self.symbol, buy_price, qty))
                        logger.print(f"Market-make {self.symbol}: BUY {qty} @ {buy_price}")
                if position > -self.limit:
                    qty = min(self.order_cap, self.limit + position, depth.buy_orders.get(best_bid, 0))
                    if qty > 0:
                        sell_price = int(best_ask - self.offset)
                        self.orders.append(Order(self.symbol, sell_price, -qty))
                        logger.print(f"Market-make {self.symbol}: SELL {qty} @ {sell_price}")

# Trader class (simplified for vouchers)
class Trader:
    def __init__(self):
        self.position_limits = {symbol: 200 for symbol in Product.VOUCHER_SYMBOLS}
        self.voucher_strategies = [
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_9500", 9500, absolute=True),
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_10000", 10000),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_9750", 200),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_10250", 200),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_10500", 200),
        ]

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {symbol: [] for symbol in state.listings.keys()}
        conversions = 0
        trader_data = ""

        if Product.VOLCANIC_ROCK in state.order_depths:
            for strategy in self.voucher_strategies:
                if isinstance(strategy, ParabolaFitIVStrategy):
                    orders, _, _ = strategy.run(state)
                    for symbol, order_list in orders.items():
                        result[symbol].extend(order_list)
                elif isinstance(strategy, VolcanicMarketMakingStrategy):
                    orders, _ = strategy.run(state)
                    result[strategy.symbol].extend(orders)

        final_result = {k: v for k, v in result.items() if v}
        logger.flush(state, final_result, conversions, trader_data)
        return final_result, conversions, trader_data

# Sample execution
def main():
    # Create sample order depths
    order_depths = {
        Product.VOLCANIC_ROCK: OrderDepth(),
        "VOLCANIC_ROCK_VOUCHER_9500": OrderDepth(),
        "VOLCANIC_ROCK_VOUCHER_9750": OrderDepth(),
        "VOLCANIC_ROCK_VOUCHER_10000": OrderDepth(),
        "VOLCANIC_ROCK_VOUCHER_10250": OrderDepth(),
        "VOLCANIC_ROCK_VOUCHER_10500": OrderDepth(),
    }
    # Populate order depths with sample data
    order_depths[Product.VOLCANIC_ROCK].buy_orders = {10000: 50}
    order_depths[Product.VOLCANIC_ROCK].sell_orders = {10050: -50}
    for symbol in Product.VOUCHER_SYMBOLS:
        order_depths[symbol].buy_orders = {100: 100}
        order_depths[symbol].sell_orders = {110: -100}

    # Create trading state
    state = TradingState(
        timestamp=1000,
        order_depths=order_depths,
        position={},
        traderData=""
    )

    # Initialize trader and run
    trader = Trader()
    result, conversions, trader_data = trader.run(state)
    
    # Print results
    print("Final Orders:")
    for symbol, orders in result.items():
        for order in orders:
            print(f"{symbol}: {'BUY' if order.quantity > 0 else 'SELL'} {abs(order.quantity)} @ {order.price}")

if __name__ == "__main__":
    main()