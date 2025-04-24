import json
from typing import Any, TypeAlias, List, Dict, Tuple, Optional
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
import jsonpickle
import math
import numpy as np
from enum import IntEnum
import statistics
from collections import deque
from math import log, sqrt, erf

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str):
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

    def compress_trades(self, trades: dict[str, list[object]]):
        compressed = []
        seen = set()
        for arr in trades.values():
            for trade in arr:
                trade_tuple = (
                    trade.symbol,
                    int(trade.price),
                    int(trade.quantity),
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                )
                if trade_tuple not in seen:
                    seen.add(trade_tuple)
                    compressed.append([
                        trade.symbol,
                        int(trade.price),
                        int(trade.quantity),
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ])
        return compressed

    def compress_observations(self, obs: object):
        co = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
              for p, o in obs.conversionObservations.items()}
        return [obs.plainValueObservations, co]

    def compress_orders(self, orders: dict[str, list[Order]]):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int):
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    ARTIFICAL1 = "ARTIFICAL1"
    ARTIFICAL2 = "ARTIFICAL2"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
    STRIKES = [9500, 9750, 10000, 10250, 10500]
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
Product.VOUCHER_SYMBOLS = [f"{Product.VOUCHER_PREFIX}{K}" for K in Product.STRIKES]

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000, "take_width": 1, "clear_width": 0, "disregard_edge": 1,
        "join_edge": 2, "default_edge": 4, "soft_position_limit": 25
    },
    Product.KELP: {
        "take_width": 1.5, "clear_width": 0.5, "reversion_beta": -1, 
        "disregard_edge": 0, "join_edge": 4, "default_edge": 1,
        "strategy": "volume_weighted_mid_price"
    },
    Product.SQUID_INK: {
        "take_width": 2, "clear_width": 1, "prevent_adverse": False, "adverse_volume": 15,
        "reversion_beta": -0.228, "disregard_edge": 2, "join_edge": 0, "default_edge": 1,
        "spike_lb": 3, "spike_ub": 5.6, "offset": 2, "reversion_window": 55,
        "reversion_weight": 0.12
    },
    Product.CROISSANTS: {
        "fair_value": 10, "take_width": 0.5, "clear_width": 0.2, "disregard_edge": 0.5,
        "join_edge": 1, "default_edge": 1, "soft_position_limit": 125
    },
    Product.JAMS: {
        "sma_short": 10, "sma_long": 50, "min_edge": 1, "volume_limit": 300,
        "order_volume": 50, "clear_threshold": 2, "buy_volume_reduction_bearish": 0,
        "sell_volume_increase_bearish": 1, "target_position_bearish": -300,
    },
    Product.DJEMBES: {
        "fair_value": 20, 
        "take_width": 0.5, 
        "clear_width": 0.2, 
        "disregard_edge": 0.5,
        "join_edge": 1, 
        "default_edge": 1, 
        "soft_position_limit": 30
    },
    Product.SPREAD1: {
        "default_spread_mean": 48.777856, 
        "default_spread_std": 85.119723,
        "spread_window": 55, 
        "zscore_threshold": 4, 
        "target_position": 60
    },
    Product.PICNIC_BASKET2: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 2,
        "synthetic_weight": 0.03,
        "volatility_window_size": 10,
        "adverse_volatility": 0.1,
    },
    Product.VOLCANIC_ROCK: {
        "window_size": 50,
        "zscore_threshold": 1.9
    }
}

PICNIC1_WEIGHTS = {Product.DJEMBES: 1, Product.CROISSANTS: 6, Product.JAMS: 3}
PICNIC2_WEIGHTS = {Product.CROISSANTS: 4, Product.JAMS: 2}

class Signal(IntEnum):
    NEUTRAL = 0
    SHORT = 1
    LONG = 2

class InitialStrategy:
    def __init__(self, symbol: Symbol, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders = []

    def buy(self, price: float, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: float, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def run(self, state: TradingState) -> Tuple[List[Order], str]:
        self.orders = []
        self.act(state)
        return self.orders, ""

    def act(self, state: TradingState) -> None:
        """Override this method to implement strategy-specific logic"""
        pass

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class SignalStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.signal = Signal.NEUTRAL

    @abstractmethod
    def get_signal(self, state: TradingState) -> Signal | None:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        new_signal = self.get_signal(state)
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                self.buy(self.get_buy_price(order_depth), -position)
            elif position > 0:
                self.sell(self.get_sell_price(order_depth), position)
        elif self.signal == Signal.SHORT:
            self.sell(self.get_sell_price(order_depth), self.limit + position)
        elif self.signal == Signal.LONG:
            self.buy(self.get_buy_price(order_depth), self.limit - position)

    def get_buy_price(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders.keys())

    def get_sell_price(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders.keys())

    def save(self) -> JSON:
        return self.signal.value

    def load(self, data: JSON) -> None:
        self.signal = Signal(data)

class ParabolaFitIVStrategy:
    def __init__(self, voucher: str, strike: int, adaptive: bool = False, absolute: bool = False):
        self.voucher = voucher
        self.strike = strike
        self.adaptive = adaptive
        self.absolute = absolute
        self.expiry_day = 3
        self.ticks_per_day = 1000
        self.window = 500
        self.position_limit = 200
        self.start_ts = None
        self.history = deque(maxlen=self.window)
        self.iv_cache = {}
        self.a = self.b = self.c = None

    def norm_cdf(self, x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    def bs_price(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0 or S <= 0:
            return max(S - K, 0)
        d1 = (log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
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

            if mispricing > 1 and position < self.position_limit:
                qty = min(20, self.position_limit - position)
                result.append(Order(self.voucher, best_ask, qty))
            elif mispricing < -1 and position > -self.position_limit:
                qty = min(20, self.position_limit + position)
                result.append(Order(self.voucher, best_bid, -qty))

            orders[self.voucher] = result
            return orders, 0, ""

        m_t = log(self.strike / spot_price) / sqrt(TTE)
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
            qty = min(35, self.position_limit - position)
            result.append(Order(self.voucher, best_ask, qty))
        elif v_t > current_fit + 0.013 and position > -self.position_limit:
            qty = min(40, self.position_limit + position)
            result.append(Order(self.voucher, best_bid, -qty))

        orders[self.voucher] = result
        return orders, 0, ""

class VolcanicMarketMakingStrategy(InitialStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
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

class BasketSignal(InitialStrategy):
    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.sell_orders.keys())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.buy_orders.keys())
        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position
        self.sell(price, to_sell)

class PicnicBasketStrategy(BasketSignal):
    def act(self, state: TradingState) -> None:
        if any(symbol not in state.order_depths for symbol in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"]):
            return
        CROISSANTS = self.get_mid_price(state, "CROISSANTS")
        JAMS = self.get_mid_price(state, "JAMS")
        DJEMBES = self.get_mid_price(state, "DJEMBES")
        PICNIC_BASKET1 = self.get_mid_price(state, "PICNIC_BASKET1")
        diff = PICNIC_BASKET1 - 6 * CROISSANTS - 3 * JAMS - DJEMBES
        long_threshold, short_threshold = {
            "CROISSANTS": (10, 80),
            "JAMS": (10, 80),
            "DJEMBES": (10, 80),
            "PICNIC_BASKET1": (10, 80),
        }[self.symbol]
        if diff < long_threshold:
            self.go_long(state)
        elif diff > short_threshold:
            self.go_short(state)

class RockPabloStrategy(InitialStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
    
    def act(self, state: TradingState) -> None:
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]
        bot_bought = False
        bot_sold = False
        
        trades = state.market_trades.get(self.symbol, [])
        trades = [t for t in trades if t.timestamp == state.timestamp - 100]
        
        for trade in trades:
            if trade.buyer == "Caesar":
                bot_sold = True
            elif trade.seller == "Pablo":
                bot_bought = True
                
        # If bot bought, go LONG
        if bot_bought and len(order_depth.sell_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            volume = min(self.limit - position, -order_depth.sell_orders[best_ask])
            if volume > 0:
                self.buy(best_ask, volume)
                
        # If bot sold, go SHORT
        elif bot_sold and len(order_depth.buy_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            volume = min(self.limit + position, order_depth.buy_orders[best_bid])
            if volume > 0:
                self.sell(best_bid, volume)
                
    def save(self) -> JSON:
        return {}
        
    def load(self, data: JSON) -> None:
        pass

class VolcanicRockStrategy(InitialStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.pablo_position = 0
        self.pablo_previous_position = 0
        self.pablo_position_history = []
        self.pablo_zero_action_taken = False
        self.history = []
        self.z_score = 0
        self.threshold = 1.9
        self.window_size = 50

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        position = state.position.get(self.symbol, 0)
        
        if not order_depth:
            return
            
        # Track Pablo's position
        self.pablo_previous_position = self.pablo_position
        
        # Update Pablo's position from market trades
        if state.market_trades and self.symbol in state.market_trades:
            trades = state.market_trades.get(self.symbol, [])
            for trade in trades:
                if trade.buyer == "Pablo":
                    self.pablo_position += trade.quantity
                elif trade.seller == "Pablo":
                    self.pablo_position -= trade.quantity
        
        # Record Pablo's position history
        self.pablo_position_history.append(self.pablo_position)
        
        # Check if Pablo's position has become zero
        pablo_zeroed_position = (
            self.pablo_position == 0 and 
            self.pablo_previous_position != 0 and
            not self.pablo_zero_action_taken
        )
        
        # Reset flag if Pablo's position becomes non-zero again
        if self.pablo_position != 0 and self.pablo_zero_action_taken:
            self.pablo_zero_action_taken = False
        
        # If Pablo zeroed his position, SELL our position
        if pablo_zeroed_position and order_depth and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            available_qty = order_depth.buy_orders[best_bid]
            
            # If we have a position, sell it all
            if position > 0:
                sell_qty = min(available_qty, position)
                if sell_qty > 0:
                    self.sell(best_bid, sell_qty)
                    self.pablo_zero_action_taken = True
            
        # Continue with the mean reversion strategy
        self.run_mean_reversion_strategy(state)
        
    def run_mean_reversion_strategy(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        position = state.position.get(self.symbol, 0)
        
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return
            
        # Calculate mid price
        rock_bid = max(order_depth.buy_orders.keys())
        rock_ask = min(order_depth.sell_orders.keys())
        rock_mid = (rock_bid + rock_ask) / 2
        
        # Update price history
        self.history.append(rock_mid)
        
        # Calculate z-score once we have enough data
        if len(self.history) > self.window_size:
            # Trim history to window size
            self.history = self.history[-self.window_size:]
            
            # Calculate mean and standard deviation
            rock_prices = np.array(self.history)
            mean = np.mean(rock_prices)
            std = np.std(rock_prices)
            
            # Calculate z-score
            if std > 0:
                self.z_score = (rock_mid - mean) / std
            
            # Trading logic based on z-score
            position_limit = self.limit
            
            if self.z_score < -self.threshold and position < position_limit:
                # Price is below mean, BUY
                best_ask = min(order_depth.sell_orders.keys())
                qty = -order_depth.sell_orders[best_ask]
                buy_qty = min(qty, position_limit - position)
                if buy_qty > 0:
                    self.buy(best_ask, buy_qty)
            
            elif self.z_score > self.threshold and position > -position_limit:
                # Price is above mean, SELL
                best_bid = max(order_depth.buy_orders.keys())
                qty = order_depth.buy_orders[best_bid]
                sell_qty = min(qty, position_limit + position)
                if sell_qty > 0:
                    self.sell(best_bid, sell_qty)
                    
    def save(self) -> JSON:
        return {
            "pablo_position": self.pablo_position,
            "pablo_zero_action_taken": self.pablo_zero_action_taken,
            "history": self.history,
            "z_score": self.z_score
        }
        
    def load(self, data: JSON) -> None:
        if data:
            self.pablo_position = data.get("pablo_position", 0)
            self.pablo_zero_action_taken = data.get("pablo_zero_action_taken", False)
            self.history = data.get("history", [])
            self.z_score = data.get("z_score", 0)

class CroissantsStrategy(SignalStrategy):
    def get_signal(self, state: TradingState) -> Signal | None:
        trades = state.market_trades.get(self.symbol, [])
        trades = [t for t in trades if t.timestamp == state.timestamp - 100]

        if any(t.buyer == "Olivia" for t in trades):
            return Signal.LONG

        if any(t.seller == "Olivia" for t in trades):
            return Signal.SHORT

class SquidInkStrategy(SignalStrategy):
    def get_signal(self, state: TradingState) -> Signal | None:
        trades = state.market_trades.get(self.symbol, [])
        trades = [t for t in trades if t.timestamp == state.timestamp - 100]

        if any(t.buyer == "Olivia" for t in trades):
            return Signal.LONG

        if any(t.seller == "Olivia" for t in trades):
            return Signal.SHORT

class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.position_limits = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK: 400,
            **{symbol: 200 for symbol in Product.VOUCHER_SYMBOLS}
        }
        limits = {
            "CROISSANTS": 250,
            "SQUID_INK": 50,
        }
        self.strikes = {symbol: k for symbol, k in zip(Product.VOUCHER_SYMBOLS, Product.STRIKES)}
        self.voucher_strategies = [
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_9500", 9500, absolute=True),
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_10000", 10000),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_9750", 200),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_10250", 200),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_10500", 200),
        ]
        self.history = {Product.VOLCANIC_ROCK: []}
        self.threshold = {Product.VOLCANIC_ROCK: 0}
        self.position_limit = {Product.VOLCANIC_ROCK: 0}
        self.z = {Product.VOLCANIC_ROCK: 0}
        self.jams_mid_prices = deque(maxlen=self.params[Product.JAMS]["sma_long"])
        self.jams_sma_short = None
        self.jams_sma_long = None
        self.jams_last_mid_price = None
        self.jams_trend_history = deque(maxlen=100)
        self.kelp_state = {}
        self.picnic1_strategy = PicnicBasketStrategy(
            Product.PICNIC_BASKET1,
            self.position_limits[Product.PICNIC_BASKET1] 
        )
        self.volcanic_rock_strategy = VolcanicRockStrategy(
        Product.VOLCANIC_ROCK, 
        self.position_limits[Product.VOLCANIC_ROCK]
        )
        self.rock_pablo_strategy = RockPabloStrategy(
        Product.VOLCANIC_ROCK, 
        self.position_limits[Product.VOLCANIC_ROCK]
        )
        self.squid_ink_strategy = SquidInkStrategy(
            Product.SQUID_INK, 
            self.position_limits[Product.SQUID_INK]
        )
        self.croissant_strategy = CroissantsStrategy(
            Product.CROISSANTS, 
            self.position_limits[Product.CROISSANTS]
        )
        
    def get_mid_price(self, symbol: str, state: TradingState) -> float | None:
        order_depth = state.order_depths.get(symbol)
        if not order_depth:
            return None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            return best_bid
        elif best_ask is not None:
            return best_ask
        return None

    def mid_price_2(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None

    def get_weighted_mid_price(self, symbol: str, state: TradingState) -> float | None:
        order_depth = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return self.get_mid_price(symbol, state)
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        if best_bid_vol + best_ask_vol == 0:
            return (best_bid + best_ask) / 2.0
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def calculate_volume_weighted_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        total_vol = 0
        total_val = 0
        for price, vol in order_depth.sell_orders.items():
            total_vol += abs(vol)
            total_val += price * abs(vol)
        for price, vol in order_depth.buy_orders.items():
            total_vol += abs(vol)
            total_val += price * abs(vol)
        if total_vol == 0:
            return None
        return total_val/total_vol

    def filtered_mid(self, product: str, order_depth: OrderDepth) -> float | None:
        """Return volume-filtered midpoint to mitigate tiny orders."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        threshold = self.params.get(product, {}).get("adverse_volume", 0)
        valid_asks = [p for p, v in order_depth.sell_orders.items() if abs(v) >= threshold]
        valid_bids = [p for p, v in order_depth.buy_orders.items() if abs(v) >= threshold]
        filt_ask = min(valid_asks) if valid_asks else None
        filt_bid = max(valid_bids) if valid_bids else None
        if filt_ask is not None and filt_bid is not None:
            return (filt_ask + filt_bid) / 2
        return (best_ask + best_bid) / 2

    def basket2_fair_value(
        self,
        basket_depth: OrderDepth,
        cro_depth: OrderDepth,
        jam_depth: OrderDepth,
        position: int,
        traderObject: dict,
    ) -> float | None:
        """Synthetic fair value for PICNIC_BASKET2 using croissants & jams."""
        mid = self.filtered_mid(Product.PICNIC_BASKET2, basket_depth)
        cro_mid = self.filtered_mid(Product.CROISSANTS, cro_depth)
        jam_mid = self.filtered_mid(Product.JAMS, jam_depth)
        if mid is None:
            return None
        if cro_mid is None or jam_mid is None:
            return mid
        synthetic_mid = (
            cro_mid * PICNIC2_WEIGHTS[Product.CROISSANTS]
            + jam_mid * PICNIC2_WEIGHTS[Product.JAMS]
        )
        weight = self.params[Product.PICNIC_BASKET2]["synthetic_weight"]
        if position:
            ratio = abs(position) / self.position_limits[Product.PICNIC_BASKET2]
            weight *= math.exp(-ratio)
        return (1 - weight) * mid + weight * synthetic_mid

    def take_best_orders(self, product: str, fair_value: float, take_width: float, orders: List[Order],
                        order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int,
                        adverse_volume: int, prevent_adverse: bool = False, traderObject: dict = None):
        position_limit = self.position_limits[product]
        if product == Product.SQUID_INK:
            if "currentSpike" not in traderObject:
                traderObject["currentSpike"] = False
            prev_price = traderObject.get("ink_last_price", fair_value)
            if traderObject["currentSpike"]:
                if abs(fair_value - prev_price) < self.params[Product.SQUID_INK]["spike_lb"]:
                    traderObject["currentSpike"] = False
                else:
                    if fair_value < traderObject["recoveryValue"]:
                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_amount = order_depth.sell_orders[best_ask]
                        quantity = min(abs(best_ask_amount), position_limit - position)
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            order_depth.sell_orders[best_ask] += quantity
                            if order_depth.sell_orders[best_ask] == 0:
                                del order_depth.sell_orders[best_ask]
                        return buy_order_volume, 0
                    else:
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_amount = order_depth.buy_orders[best_bid]
                        quantity = min(best_bid_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -quantity))
                            sell_order_volume += quantity
                            order_depth.buy_orders[best_bid] -= quantity
                            if order_depth.buy_orders[best_bid] == 0:
                                del order_depth.buy_orders[best_bid]
                        return 0, sell_order_volume
            if abs(fair_value - prev_price) > self.params[Product.SQUID_INK]["spike_ub"]:
                traderObject["currentSpike"] = True
                traderObject["recoveryValue"] = prev_price + self.params[Product.SQUID_INK]["offset"] if fair_value > prev_price else prev_price - self.params[Product.SQUID_INK]["offset"]
                if fair_value > prev_price:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
                    return 0, sell_order_volume
                else:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    quantity = min(abs(best_ask_amount), position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
                    return buy_order_volume, 0
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders1(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        position_limit: int,
    ) -> (int, int):
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amt = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amt, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amt = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amt, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
        return buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, position: int,
                    buy_order_volume: int, sell_order_volume: int):
        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, math.floor(bid), buy_quantity))
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, math.ceil(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order],
                            order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float,
                   position: int, adverse_volume: int, position_limit: int = 0, prevent_adverse: bool = False, traderObject: dict = None):
        orders = []
        if product == Product.RAINFOREST_RESIN or product == Product.SQUID_INK:
            buy_order_volume, sell_order_volume = self.take_best_orders1(
            product, fair_value, take_width, orders, 
            order_depth, position, 0, 0, position_limit
        )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product, fair_value, take_width, orders, order_depth, position, 0, 0,
                adverse_volume, prevent_adverse, traderObject
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: int,
                    position: int, buy_order_volume: int, sell_order_volume: int):
        orders = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(self, product, order_depth: OrderDepth, fair_value: float, position: int,
                    buy_order_volume: int, sell_order_volume: int, disregard_edge: float, join_edge: float,
                    default_edge: float, manage_position: bool = False, soft_position_limit: int = 0):
        adjustment = 0
        if product == Product.RAINFOREST_RESIN:
            total_buy_volume = sum(order_depth.buy_orders.values()) if order_depth.buy_orders else 0
            total_sell_volume = sum(abs(v) for v in order_depth.sell_orders.values()) if order_depth.sell_orders else 0
            total_volume = total_buy_volume + total_sell_volume if (total_buy_volume + total_sell_volume) > 0 else 1
            imbalance_ratio = (total_buy_volume - total_sell_volume) / total_volume
            adjustment = round(4.0 * imbalance_ratio)
        orders = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        ask = round(fair_value + default_edge)
        if best_ask_above_fair:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair + 1
            else:
                ask = best_ask_above_fair
        bid = round(fair_value - default_edge)
        if best_bid_below_fair:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1
        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume
    
    def rainforest_resin_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.RAINFOREST_RESIN) is not None:
            position = state.position.get(Product.RAINFOREST_RESIN, 0)
            order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            fair_value = self.params[Product.RAINFOREST_RESIN]["fair_value"]
            resin_take, bo, so = self.take_orders(
                Product.RAINFOREST_RESIN, order_depth, fair_value,
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                position, 0, self.position_limits[Product.RAINFOREST_RESIN]
            )
            resin_clear, bo, so = self.clear_orders(
                Product.RAINFOREST_RESIN, order_depth, fair_value,
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                position, bo, so,
            )
            resin_make, bo, so = self.make_orders(
                Product.RAINFOREST_RESIN, order_depth, fair_value, position, bo, so,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True, self.params[Product.RAINFOREST_RESIN]["soft_position_limit"]
            )
            orders = resin_take + resin_clear + resin_make
        return orders

    def kelp_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.KELP) is not None:
            position = state.position.get(Product.KELP, 0)
            order_depth = state.order_depths[Product.KELP]
            fair_value = self.kelp_fair_value(order_depth)
            if fair_value is None:
                logger.print("[KELP] No valid fair value, skipping trading.")
                return orders

            logger.print(f"[KELP] Fair Value: {fair_value:.2f}, Position: {position}")

            kelp_take, bo, so = self.take_orders(
                Product.KELP,
                order_depth,
                fair_value,
                self.params[Product.KELP]["take_width"],
                position, 0, 
                self.position_limits[Product.KELP],
            )
            kelp_clear, bo, so = self.clear_orders(
                Product.KELP,
                order_depth,
                fair_value,
                self.params[Product.KELP]["clear_width"],
                position,
                bo,
                so,
            )
            kelp_make, bo, so = self.make_orders(
                Product.KELP,
                order_depth,
                fair_value,
                position,
                bo,
                so,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            orders = kelp_take + kelp_clear + kelp_make
        return orders
    
    def kelp_fair_value(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return self.kelp_state.get("last_fair_value", None)

        mid_price = self.calculate_volume_weighted_mid_price(order_depth)
        if mid_price is None:
            return self.kelp_state.get("last_fair_value", None)

        last_price = self.kelp_state.get("last_price", None)
        if last_price is not None:
            last_returns = (mid_price - last_price) / last_price
            pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
            fair_value = mid_price + (mid_price * pred_returns)
        else:
            fair_value = mid_price

        self.kelp_state["last_price"] = mid_price
        self.kelp_state["last_fair_value"] = fair_value
        return fair_value

    def ink_fair_value(self, order_depth: OrderDepth, traderObject):
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                valid_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
                valid_buy = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
                mm_ask = min(valid_ask) if len(valid_ask) > 0 else None
                mm_bid = max(valid_buy) if len(valid_buy) > 0 else None
                if valid_ask and valid_buy:
                    mmmid_price = (mm_ask + mm_bid) / 2
                else:
                    mmmid_price = (best_ask + best_bid) / 2 if traderObject.get('ink_last_price', None) is None else traderObject['ink_last_price']
                if traderObject.get('ink_price_history', None) is None:
                    traderObject['ink_price_history'] = []
                traderObject['ink_price_history'].append(mmmid_price)
                if len(traderObject['ink_price_history']) > self.params[Product.SQUID_INK]["reversion_window"]:
                    traderObject['ink_price_history'] = traderObject['ink_price_history'][-self.params[Product.SQUID_INK]["reversion_window"]:]
                if len(traderObject['ink_price_history']) >= self.params[Product.SQUID_INK]["reversion_window"]:
                    prices = np.array(traderObject['ink_price_history'])
                    returns = (prices[1:] - prices[:-1]) / prices[:-1]
                    X = returns[:-1]
                    Y = returns[1:]
                    estimated_beta = -np.dot(X, Y) / np.dot(X, X) if np.dot(X, X) != 0 else self.params[Product.SQUID_INK]["reversion_beta"]
                    adaptive_beta = (self.params[Product.SQUID_INK]['reversion_weight'] * estimated_beta +
                                    (1 - self.params[Product.SQUID_INK]['reversion_weight']) * self.params[Product.SQUID_INK]["reversion_beta"])
                else:
                    adaptive_beta = self.params[Product.SQUID_INK]["reversion_beta"]
                fair = mmmid_price if traderObject.get('ink_last_price', None) is None else mmmid_price + (
                    mmmid_price * ((mmmid_price - traderObject["ink_last_price"]) / traderObject["ink_last_price"] * adaptive_beta))
                traderObject["ink_last_price"] = mmmid_price
                return fair
            return None

    def jams_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.JAMS) is not None:
            position = state.position.get(Product.JAMS, 0)
            order_depth = state.order_depths[Product.JAMS]
            mid_price = self.mid_price_2(order_depth)
            if mid_price is None:
                return []
            self.jams_mid_prices.append(mid_price)
            if len(self.jams_mid_prices) >= self.params[Product.JAMS]["sma_short"]:
                self.jams_sma_short = sum(list(self.jams_mid_prices)[-self.params[Product.JAMS]["sma_short"]:]) / self.params[Product.JAMS]["sma_short"]
            if len(self.jams_mid_prices) >= self.params[Product.JAMS]["sma_long"]:
                self.jams_sma_long = sum(self.jams_mid_prices) / len(self.jams_mid_prices)
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            if best_bid is None or best_ask is None:
                return []
            if self.jams_sma_short is None or self.jams_sma_long is None:
                return []
            trend = 0
            if self.jams_sma_short > self.jams_sma_long:
                trend = 1
            elif self.jams_sma_short < self.jams_sma_long:
                trend = -1
            self.jams_trend_history.append(trend)
            recent_trends = list(self.jams_trend_history)
            is_strongly_bearish = len(recent_trends) >= 50 and all(t == -1 for t in recent_trends[-50:])
            if position > 0 and trend == -1 and best_bid is not None:
                if self.jams_last_mid_price is not None and mid_price < self.jams_last_mid_price - self.params[Product.JAMS]["clear_threshold"]:
                    sell_volume = -position
                    orders.append(Order(Product.JAMS, best_bid, sell_volume))
            if position < 0 and trend == 1 and best_ask is not None:
                if self.jams_last_mid_price is not None and mid_price > self.jams_last_mid_price + self.params[Product.JAMS]["clear_threshold"]:
                    buy_volume = -position
                    orders.append(Order(Product.JAMS, best_ask, buy_volume))
            if is_strongly_bearish and position > self.params[Product.JAMS]["target_position_bearish"] and best_bid is not None:
                sell_volume = position - self.params[Product.JAMS]["target_position_bearish"]
                orders.append(Order(Product.JAMS, best_bid, -sell_volume))
            buy_price = round(mid_price - self.params[Product.JAMS]["min_edge"])
            sell_price = round(mid_price + self.params[Product.JAMS]["min_edge"])
            buy_volume = min(self.params[Product.JAMS]["order_volume"], self.params[Product.JAMS]["volume_limit"] - position)
            sell_volume = min(self.params[Product.JAMS]["order_volume"], self.params[Product.JAMS]["volume_limit"] + position)
            if trend == -1:
                buy_volume = int(buy_volume * self.params[Product.JAMS]["buy_volume_reduction_bearish"])
                sell_volume = int(sell_volume * self.params[Product.JAMS]["sell_volume_increase_bearish"])
            elif trend == 1:
                buy_volume = int(buy_volume * 1.5)
                sell_volume = int(sell_volume * 0.5)
            if trend == -1 and position > 0:
                buy_volume = 0
            if buy_volume > 0:
                orders.append(Order(Product.JAMS, buy_price, buy_volume))
            if sell_volume > 0:
                orders.append(Order(Product.JAMS, sell_price, -sell_volume))
            self.jams_last_mid_price = mid_price
        return orders

    def djembes_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.DJEMBES):
            position = state.position.get(Product.DJEMBES, 0)
            order_depth = state.order_depths[Product.DJEMBES]
            fair_value = self.params[Product.DJEMBES]["fair_value"]
            djembes_take, bo, so = self.take_orders(
                Product.DJEMBES, order_depth, fair_value,
                self.params[Product.DJEMBES]["take_width"],
                position, 0
            )
            djembes_clear, bo, so = self.clear_orders(
                Product.DJEMBES, order_depth, fair_value,
                self.params[Product.DJEMBES]["clear_width"],
                position, bo, so
            )
            djembes_make, bo, so = self.make_orders(
                Product.DJEMBES, order_depth, fair_value, position, bo, so,
                self.params[Product.DJEMBES]["disregard_edge"],
                self.params[Product.DJEMBES]["join_edge"],
                self.params[Product.DJEMBES]["default_edge"],
                True,
                self.params[Product.DJEMBES]["soft_position_limit"]
            )
            orders = djembes_take + djembes_clear + djembes_make
        return orders

    def artifical_order_depth(self, order_depths: Dict[str, OrderDepth], picnic1: bool = True):
        if picnic1:
            DJEMBES_PER_PICNIC, CROISSANT_PER_PICNIC, JAM_PER_PICNIC = PICNIC1_WEIGHTS[Product.DJEMBES], PICNIC1_WEIGHTS[Product.CROISSANTS], PICNIC1_WEIGHTS[Product.JAMS]
        else:
            CROISSANT_PER_PICNIC, JAM_PER_PICNIC = PICNIC2_WEIGHTS[Product.CROISSANTS], PICNIC2_WEIGHTS[Product.JAMS]
        artifical_order_price = OrderDepth()
        croissant_best_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys()) if order_depths[Product.CROISSANTS].buy_orders else 0
        croissant_best_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys()) if order_depths[Product.CROISSANTS].sell_orders else float("inf")
        jams_best_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else 0
        jams_best_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else float("inf")
        if picnic1:
            djembes_best_bid = max(order_depths[Product.DJEMBES].buy_orders.keys()) if order_depths[Product.DJEMBES].buy_orders else 0
            djembes_best_ask = min(order_depths[Product.DJEMBES].sell_orders.keys()) if order_depths[Product.DJEMBES].sell_orders else float("inf")
            art_bid = djembes_best_bid * DJEMBES_PER_PICNIC + croissant_best_bid * CROISSANT_PER_PICNIC + jams_best_bid * JAM_PER_PICNIC
            art_ask = djembes_best_ask * DJEMBES_PER_PICNIC + croissant_best_ask * CROISSANT_PER_PICNIC + jams_best_ask * JAM_PER_PICNIC
        else:
            art_bid = croissant_best_bid * CROISSANT_PER_PICNIC + jams_best_bid * JAM_PER_PICNIC
            art_ask = croissant_best_ask * CROISSANT_PER_PICNIC + jams_best_ask * JAM_PER_PICNIC
        if art_bid > 0:
            croissant_bid_volume = order_depths[Product.CROISSANTS].buy_orders.get(croissant_best_bid, 0) // CROISSANT_PER_PICNIC
            jams_bid_volume = order_depths[Product.JAMS].buy_orders.get(jams_best_bid, 0) // JAM_PER_PICNIC
            if picnic1:
                djembes_bid_volume = order_depths[Product.DJEMBES].buy_orders.get(djembes_best_bid, 0) // DJEMBES_PER_PICNIC
                artifical_bid_volume = min(djembes_bid_volume, croissant_bid_volume, jams_bid_volume)
            else:
                artifical_bid_volume = min(croissant_bid_volume, jams_bid_volume)
            artifical_order_price.buy_orders[art_bid] = artifical_bid_volume
        if art_ask < float("inf"):
            croissant_ask_volume = abs(order_depths[Product.CROISSANTS].sell_orders.get(croissant_best_ask, 0)) // CROISSANT_PER_PICNIC
            jams_ask_volume = abs(order_depths[Product.JAMS].sell_orders.get(jams_best_ask, 0)) // JAM_PER_PICNIC
            if picnic1:
                djembes_ask_volume = abs(order_depths[Product.DJEMBES].sell_orders.get(djembes_best_ask, 0)) // DJEMBES_PER_PICNIC
                artifical_ask_volume = min(djembes_ask_volume, croissant_ask_volume, jams_ask_volume)
            else:
                artifical_ask_volume = min(croissant_ask_volume, jams_ask_volume)
            artifical_order_price.sell_orders[art_ask] = -artifical_ask_volume
        return artifical_order_price

    def convert_orders(self, artifical_orders: List[Order], order_depths: Dict[str, OrderDepth], picnic1: bool = True):
        component_orders = {Product.DJEMBES: [], Product.CROISSANTS: [], Product.JAMS: []} if picnic1 else {Product.CROISSANTS: [], Product.JAMS: []}
        artifical_order_depth = self.artifical_order_depth(order_depths, picnic1)
        best_bid = max(artifical_order_depth.buy_orders.keys()) if artifical_order_depth.buy_orders else 0
        best_ask = min(artifical_order_depth.sell_orders.keys()) if artifical_order_depth.sell_orders else float("inf")
        for order in artifical_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                if not order_depths[Product.CROISSANTS].sell_orders or not order_depths[Product.JAMS].sell_orders or (picnic1 and not order_depths[Product.DJEMBES].sell_orders):
                    continue
                croissant_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
                if picnic1:
                    djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                if not order_depths[Product.CROISSANTS].buy_orders or not order_depths[Product.JAMS].buy_orders or (picnic1 and not order_depths[Product.DJEMBES].buy_orders):
                    continue
                croissant_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                if picnic1:
                    djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                continue
            croissaint_order = Order(Product.CROISSANTS, croissant_price, quantity * PICNIC1_WEIGHTS[Product.CROISSANTS] if picnic1 else quantity * PICNIC2_WEIGHTS[Product.CROISSANTS])
            jams_order = Order(Product.JAMS, jams_price, quantity * PICNIC1_WEIGHTS[Product.JAMS] if picnic1 else quantity * PICNIC2_WEIGHTS[Product.JAMS])
            if picnic1:
                component_orders[Product.DJEMBES].append(Order(Product.DJEMBES, djembes_price, quantity * PICNIC1_WEIGHTS[Product.DJEMBES]))
            component_orders[Product.CROISSANTS].append(croissaint_order)
            component_orders[Product.JAMS].append(jams_order)
        return component_orders

    def execute_spreads(self, target_position: int, picnic_position: int, order_depths: Dict[str, OrderDepth], picnic1: bool = True):
        if target_position == picnic_position:
            return None
        target_quantity = abs(target_position - picnic_position)
        picnic_order_depth = order_depths[Product.PICNIC_BASKET1] if picnic1 else order_depths[Product.PICNIC_BASKET2]
        artifical_order_depth = self.artifical_order_depth(order_depths, picnic1)
        if not picnic_order_depth.sell_orders or not artifical_order_depth.buy_orders or \
           not picnic_order_depth.buy_orders or not artifical_order_depth.sell_orders:
            return None
        if target_position > picnic_position:
            picnic_ask_price = min(picnic_order_depth.sell_orders.keys())
            picnic_ask_vol = abs(picnic_order_depth.sell_orders[picnic_ask_price])
            artifical_bid_price = max(artifical_order_depth.buy_orders.keys())
            artifical_bid_vol = abs(artifical_order_depth.buy_orders[artifical_bid_price])
            if picnic_ask_vol == 0 or artifical_bid_vol == 0:
                return None
            orderbook_volume = min(picnic_ask_vol, artifical_bid_vol)
            execute_volume = min(orderbook_volume, target_quantity)
            if execute_volume <= 0:
                return None
            picnic_orders = [Order(Product.PICNIC_BASKET1 if picnic1 else Product.PICNIC_BASKET2, picnic_ask_price, execute_volume)]
            artifical_orders = [Order(Product.ARTIFICAL1, artifical_bid_price, -execute_volume)]
            aggregate_orders = self.convert_orders(artifical_orders, order_depths, picnic1)
            if picnic1:
                aggregate_orders[Product.PICNIC_BASKET1] = picnic_orders
            else:
                aggregate_orders[Product.PICNIC_BASKET2] = picnic_orders
            return aggregate_orders
        else:
            picnic_bid_price = max(picnic_order_depth.buy_orders.keys())
            picnic_bid_vol = abs(picnic_order_depth.buy_orders[picnic_bid_price])
            artifical_ask_price = min(artifical_order_depth.sell_orders.keys())
            artifical_ask_vol = abs(artifical_order_depth.sell_orders[artifical_ask_price])
            if picnic_bid_vol == 0 or artifical_ask_vol == 0:
                return None
            orderbook_volume = min(picnic_bid_vol, artifical_ask_vol)
            execute_volume = min(orderbook_volume, target_quantity)
            if execute_volume <= 0:
                return None
            picnic_orders = [Order(Product.PICNIC_BASKET1 if picnic1 else Product.PICNIC_BASKET2, picnic_bid_price, -execute_volume)]
            artifical_orders = [Order(Product.ARTIFICAL1, artifical_ask_price, execute_volume)]
            aggregate_orders = self.convert_orders(artifical_orders, order_depths, picnic1)
            if picnic1:
                aggregate_orders[Product.PICNIC_BASKET1] = picnic_orders
            else:
                aggregate_orders[Product.PICNIC_BASKET2] = picnic_orders
            return aggregate_orders

    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: str, picnic_position: int,
                     spread_data: Dict[str, object], SPREAD, picnic1: bool = True):
        required_products = [Product.PICNIC_BASKET1, Product.DJEMBES, Product.CROISSANTS, Product.JAMS] if picnic1 else [Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS]
        if not all(p in order_depths for p in required_products):
            return None
        picnic_order_depth = order_depths[Product.PICNIC_BASKET1] if picnic1 else order_depths[Product.PICNIC_BASKET2]
        artifical_order_depth = self.artifical_order_depth(order_depths, picnic1)
        if not picnic_order_depth.buy_orders or not picnic_order_depth.sell_orders or \
           not artifical_order_depth.buy_orders or not artifical_order_depth.sell_orders:
            return None
        best_bid = max(picnic_order_depth.buy_orders.keys()) if picnic_order_depth.buy_orders else None
        best_bid_vol = picnic_order_depth.buy_orders.get(best_bid, 0) if best_bid else 0
        best_ask = min(picnic_order_depth.sell_orders.keys()) if picnic_order_depth.sell_orders else None
        best_ask_vol = abs(picnic_order_depth.sell_orders.get(best_ask, 0)) if best_ask else 0
        picnic_mprice = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol) if best_bid and best_ask and (best_bid_vol + best_ask_vol > 0) else 0
        best_bid = max(artifical_order_depth.buy_orders.keys()) if artifical_order_depth.buy_orders else None
        best_bid_vol = artifical_order_depth.buy_orders.get(best_bid, 0) if best_bid else 0
        best_ask = min(artifical_order_depth.sell_orders.keys()) if artifical_order_depth.sell_orders else None
        best_ask_vol = abs(artifical_order_depth.sell_orders.get(best_ask, 0)) if best_ask else 0
        artifical_mprice = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol) if best_bid and best_ask and (best_bid_vol + best_ask_vol > 0) else 0
        spread = picnic_mprice - artifical_mprice
        spread_data["spread_history"].append(spread)
        if len(spread_data["spread_history"]) < self.params[SPREAD]["spread_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[SPREAD]["spread_window"]:
            spread_data["spread_history"].pop(0)
        spread_std = np.std(spread_data["spread_history"])
        if spread_std == 0:
            return None
        spread_mean = self.params[SPREAD]["default_spread_mean"]
        zscore = (spread - spread_mean) / spread_std
        if zscore >= self.params[SPREAD]["zscore_threshold"]:
            if picnic_position != -self.params[SPREAD]["target_position"]:
                return self.execute_spreads(-self.params[SPREAD]["target_position"], picnic_position, order_depths, picnic1)
        if zscore <= -self.params[SPREAD]["zscore_threshold"]:
            if picnic_position != self.params[SPREAD]["target_position"]:
                return self.execute_spreads(self.params[SPREAD]["target_position"], picnic_position, order_depths, picnic1)
        spread_data["prev_zscore"] = zscore
        return None

    def compute_macaron_orders(self, state: TradingState):
        product = "MAGNIFICENT_MACARONS"
        orders: List[Order] = []

        current_pos = state.position.get(product, 0)

        conversions = 0
        bid_amount = 75
        ask_amount = 75
        observation = state.observations.conversionObservations[product]

        #- buying
        spread = 3.5
        offer_price = state.observations.conversionObservations[product].bidPrice - observation.exportTariff - spread
        orders.append(Order(product, int(offer_price), bid_amount))

        #- selling
        spread = 3.5 # find better param?
        offer_price = state.observations.conversionObservations[product].askPrice + observation.importTariff + spread
        orders.append(Order(product, math.ceil(offer_price), -ask_amount))

        if current_pos < 0:
            conversions = min(10, round(-current_pos))
        elif current_pos > 0:
            conversions = max(-10, round(current_pos))

        return orders, conversions

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        trader_state = {}
        if state.traderData not in [None, ""]:
            try:
                trader_state = jsonpickle.decode(state.traderData)
            except Exception:
                trader_state = {}
        
        # Load strategy states if they exist
        if "volcanic_rock_strategy" in trader_state:
            self.volcanic_rock_strategy.load(trader_state["volcanic_rock_strategy"])
        if "squid_ink_strategy" in trader_state:
            self.squid_ink_strategy.load(trader_state["squid_ink_strategy"])
        if "croissant_strategy" in trader_state:
            self.croissant_strategy.load(trader_state["croissant_strategy"])
        
        # Load other data from trader state
        self.kelp_state = trader_state.get("kelp_state", {})
        self.jams_mid_prices = trader_state.get("jams_mid_prices", deque(maxlen=self.params[Product.JAMS]["sma_long"]))
        self.jams_sma_short = trader_state.get("jams_sma_short", None)
        self.jams_sma_long = trader_state.get("jams_sma_long", None)
        self.jams_last_mid_price = trader_state.get("jams_last_mid_price", None)
        self.jams_trend_history = trader_state.get("jams_trend_history", deque(maxlen=100))
        trader_state.setdefault("base_iv_history", [])
        trader_state.setdefault(Product.SPREAD1, {"spread_history": [], "prev_zscore": 0, "clear_flag": False, "curr_avg": 0})
        trader_state.setdefault(Product.PICNIC_BASKET2, {"spread_history": [], "prev_zscore": 0, "clear_flag": False, "curr_avg": 0})
        trader_state.setdefault("last_ivs", {})

        result = {symbol: [] for symbol in state.listings.keys()}
        
        # Run strategies that are unchanged
        strats = [
            (Product.RAINFOREST_RESIN, self.rainforest_resin_strategy),
            (Product.KELP, self.kelp_strategy),
            (Product.JAMS, self.jams_strategy),
        ]

        for product, strategy_func in strats:
            if product in state.order_depths:
                orders = strategy_func(state)
                if orders:
                    result[product] = orders

        # Run the new SQUID_INK strategy
        if Product.SQUID_INK in state.order_depths:
            squid_orders, _ = self.squid_ink_strategy.run(state)
            result[Product.SQUID_INK] = squid_orders

        # Run the new CROISSANTS strategy  
        if Product.CROISSANTS in state.order_depths:
            croissant_orders, _ = self.croissant_strategy.run(state)
            result[Product.CROISSANTS] = croissant_orders

        # Run DJEMBES strategy (unchanged)
        if Product.DJEMBES in state.order_depths:
            djembes_orders = self.djembes_strategy(state)
            result[Product.DJEMBES].extend(djembes_orders)

        # PICNIC BASKET 1 strategy
        if all(p in state.order_depths
            for p in (Product.PICNIC_BASKET1,
                        Product.CROISSANTS,
                        Product.JAMS,
                        Product.DJEMBES)):
            pb1_result = self.picnic1_strategy.run(state)
            pb1_orders = pb1_result[0]
            result.setdefault(Product.PICNIC_BASKET1, [])
            result[Product.PICNIC_BASKET1].extend(pb1_orders)

        # PICNIC BASKET 2 strategy (unchanged)
        if Product.PICNIC_BASKET2 in self.params and all(p in state.order_depths for p in [Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS]):
            basket2_position = state.position.get(Product.PICNIC_BASKET2, 0)
            basket2_fair_value = self.basket2_fair_value(
                state.order_depths[Product.PICNIC_BASKET2],
                state.order_depths[Product.CROISSANTS],
                state.order_depths[Product.JAMS],
                basket2_position,
                trader_state
            )
            if basket2_fair_value is not None:
                b2_take, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.PICNIC_BASKET2,
                    state.order_depths[Product.PICNIC_BASKET2],
                    basket2_fair_value,
                    self.params[Product.PICNIC_BASKET2]["take_width"],
                    basket2_position,
                    self.params[Product.PICNIC_BASKET2]["adverse_volume"], 0, 
                    self.params[Product.PICNIC_BASKET2]["prevent_adverse"],
                    trader_state
                )
                b2_clear, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.PICNIC_BASKET2,
                    state.order_depths[Product.PICNIC_BASKET2],
                    basket2_fair_value,
                    self.params[Product.PICNIC_BASKET2]["clear_width"],
                    basket2_position,
                    buy_order_volume,
                    sell_order_volume
                )
                b2_make, _, _ = self.make_orders(
                    Product.PICNIC_BASKET2,
                    state.order_depths[Product.PICNIC_BASKET2],
                    basket2_fair_value,
                    basket2_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.PICNIC_BASKET2]["disregard_edge"],
                    self.params[Product.PICNIC_BASKET2]["join_edge"],
                    self.params[Product.PICNIC_BASKET2]["default_edge"]
                )
                result[Product.PICNIC_BASKET2].extend(b2_take + b2_clear + b2_make)

        # Instead of using volcanic_rock_strategy, use rock_pablo_strategy
        if Product.VOLCANIC_ROCK in state.order_depths:
            volcanic_orders, _ = self.rock_pablo_strategy.run(state)
            result[Product.VOLCANIC_ROCK] = volcanic_orders

        # VOUCHERS strategies (unchanged)
        if Product.VOLCANIC_ROCK in state.order_depths:
            for strategy in self.voucher_strategies:
                if isinstance(strategy, ParabolaFitIVStrategy):
                    orders, _, _ = strategy.run(state)
                    for symbol, order_list in orders.items():
                        result[symbol].extend(order_list)
                elif isinstance(strategy, VolcanicMarketMakingStrategy):
                    orders, _ = strategy.run(state)
                    result[strategy.symbol].extend(orders)

        # MAGNIFICENT_MACARONS strategy (unchanged)
        result["MAGNIFICENT_MACARONS"], conversions = self.compute_macaron_orders(state)

        # Save strategy states
        trader_state["volcanic_rock_strategy"] = self.volcanic_rock_strategy.save()
        trader_state["squid_ink_strategy"] = self.squid_ink_strategy.save()
        trader_state["croissant_strategy"] = self.croissant_strategy.save()
        
        # Save other trader state data
        trader_state["kelp_state"] = self.kelp_state
        trader_state["jams_mid_prices"] = self.jams_mid_prices
        trader_state["jams_sma_short"] = self.jams_sma_short
        trader_state["jams_sma_long"] = self.jams_sma_long
        trader_state["jams_last_mid_price"] = self.jams_last_mid_price
        trader_state["jams_trend_history"] = self.jams_trend_history

        final_result = {k: v for k, v in result.items() if v}
        traderData = jsonpickle.encode(trader_state)
        logger.flush(state, final_result, conversions, traderData)
        return final_result, conversions, traderData