
from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List, Dict, Any, TypeAlias, Tuple
from abc import abstractmethod
import json
import jsonpickle
from math import log, sqrt, exp, erf, pi, floor, ceil
import math
import numpy as np
from collections import deque

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

DAYS_LEFT = 3
VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
# Logger class
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

# Product class
class Product:
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
    VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
    STRIKES = [9500, 9750, 10000, 10250, 10500]
    VOUCHER_SYMBOLS = [f"{VOUCHER_PREFIX}{K}" for K in STRIKES]

# MarketData class
class MarketData:
    end_pos: Dict[str, int] = {}
    buy_sum: Dict[str, int] = {}
    sell_sum: Dict[str, int] = {}
    bid_prices: Dict[str, List[float]] = {}
    bid_volumes: Dict[str, List[int]] = {}
    ask_prices: Dict[str, List[float]] = {}
    ask_volumes: Dict[str, List[int]] = {}
    fair: Dict[str, float] = {}

# Parameters
PARAMS = {
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
        "fair_value": 15, "take_width": 0.5, "clear_width": 0.2, "disregard_edge": 0.5,
        "join_edge": 1, "default_edge": 1, "soft_position_limit": 175
    },
    Product.DJEMBES: {
        "fair_value": 20, "take_width": 0.5, "clear_width": 0.2, "disregard_edge": 0.5,
        "join_edge": 1, "default_edge": 1, "soft_position_limit": 30
    },
    Product.PICNIC_BASKET2: {
        "take_width": 2, "clear_width": 0, "prevent_adverse": True, "adverse_volume": 15,
        "disregard_edge": 2, "join_edge": 0, "default_edge": 2, "synthetic_weight": 0.03,
        "volatility_window_size": 10, "adverse_volatility": 0.1,
    },
    Product.VOLCANIC_ROCK: {
        "window_size": 50, "zscore_threshold": 2
    },
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000, "take_width": 1, "clear_width": 0, "disregard_edge": 1,
        "join_edge": 2, "default_edge": 1, "soft_position_limit": 50,
    },
    Product.KELP: {
        "take_width": 2, "clear_width": 0, "prevent_adverse": False, "adverse_volume": 15,
        "reversion_beta": -0.18, "disregard_edge": 2, "join_edge": 0, "default_edge": 1,
    },
    "VOLCANIC_ROCK_VOUCHER_9750": {
        "mean_volatility": 0.147417, "strike": 9750, "starting_time_to_expiry": 7 / 365,
        "std_window": 6, "z_score_threshold": 21,
    },
    "VOLCANIC_ROCK_VOUCHER_10000": {
        "mean_volatility": 0.140554, "strike": 10000, "starting_time_to_expiry": 7 / 365,
        "std_window": 6, "z_score_threshold": 21,
    },
}

PICNIC1_WEIGHTS = {Product.DJEMBES: 1, Product.CROISSANTS: 6, Product.JAMS: 3}
PICNIC2_WEIGHTS = {Product.CROISSANTS: 4, Product.JAMS: 2}

# Strategy base class
class Strategy:
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
        pass

# ParabolaFitIVStrategy for VOLCANIC_ROCK_VOUCHER_9500
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

# VolcanicMarketMakingStrategy for VOLCANIC_ROCK_VOUCHER_10250, 10500
class VolcanicMarketMakingStrategy(Strategy):
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

# SignalStrategy for PicnicBasketStrategy
class SignalStrategy(Strategy):
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

# PicnicBasketStrategy for CROISSANTS and PICNIC_BASKET1
class PicnicBasketStrategy(SignalStrategy):
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

# Trader class
class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.position_limits = {
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK: 400,
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.MAGNIFICENT_MACARONS: 75,
            **{symbol: 200 for symbol in Product.VOUCHER_SYMBOLS}
        }
        self.voucher_strategies = [
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_9500", 9500, absolute=True),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_10250", 200),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_10500", 200),
        ]
        self.history = {Product.VOLCANIC_ROCK: []}
        self.z = {Product.VOLCANIC_ROCK: 0}
        self.croissant_strategy = PicnicBasketStrategy("CROISSANTS", 250)
        self.picnic1_strategy = PicnicBasketStrategy("PICNIC_BASKET1", 60)
        self.recent_std = 0

    def _filtered_mid(self, product: str, order_depth: OrderDepth) -> float | None:
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

    def basket2_fair_value(self, basket_depth: OrderDepth, cro_depth: OrderDepth, jam_depth: OrderDepth, position: int, traderObject: dict) -> float | None:
        mid = self._filtered_mid(Product.PICNIC_BASKET2, basket_depth)
        cro_mid = self._filtered_mid(Product.CROISSANTS, cro_depth)
        jam_mid = self._filtered_mid(Product.JAMS, jam_depth)
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

    def take_best_orders(self, product: str, fair_value: float, take_width: float, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int, prevent_adverse: bool = False, adverse_volume: int = 0, traderObject: dict = None):
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

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, position: int, buy_order_volume: int, sell_order_volume: int):
        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, floor(bid), buy_quantity))
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ceil(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int):
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

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject):
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            valid_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            valid_buy = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            mm_ask = min(valid_ask) if len(valid_ask) > 0 else None
            mm_bid = max(valid_buy) if len(valid_buy) > 0 else None
            if valid_ask and valid_buy:
                mmmid_price = (mm_ask + mm_bid) / 2
            else:
                if traderObject.get('kelp_last_price', None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject['kelp_last_price']
            if traderObject.get('kelp_last_price', None) is None:
                fair = mmmid_price
            else:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (last_returns * self.params[Product.KELP]["reversion_beta"])
                fair = mmmid_price + (mmmid_price * pred_returns)
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float, position: int, prevent_adverse: bool = False, adverse_volume: int = 0, traderObject: dict = None):
        orders = []
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position, 0, 0,
            prevent_adverse, adverse_volume, traderObject
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: int, position: int, buy_order_volume: int, sell_order_volume: int):
        orders = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(self, product, order_depth: OrderDepth, fair_value: float, position: int, buy_order_volume: int, sell_order_volume: int, disregard_edge: float, join_edge: float, default_edge: float, manage_position: bool = False, soft_position_limit: int = 0):
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

    def croissants_strategy(self, state: TradingState) -> List[Order]:
        orders, _ = self.croissant_strategy.run(state)
        return orders

    def jams_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.JAMS):
            position = state.position.get(Product.JAMS, 0)
            order_depth = state.order_depths[Product.JAMS]
            fair_value = self.params[Product.JAMS]["fair_value"]
            jams_take, bo, so = self.take_orders(
                Product.JAMS, order_depth, fair_value,
                self.params[Product.JAMS]["take_width"],
                position
            )
            jams_clear, bo, so = self.clear_orders(
                Product.JAMS, order_depth, fair_value,
                self.params[Product.JAMS]["clear_width"],
                position, bo, so
            )
            jams_make, bo, so = self.make_orders(
                Product.JAMS, order_depth, fair_value, position, bo, so,
                self.params[Product.JAMS]["disregard_edge"],
                self.params[Product.JAMS]["join_edge"],
                self.params[Product.JAMS]["default_edge"],
                True,
                self.params[Product.JAMS]["soft_position_limit"]
            )
            orders = jams_take + jams_clear + jams_make
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
                position
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

    def update_volcanic_rock_history(self, state: TradingState) -> List[Order]:
        orders = []
        rock_depth = state.order_depths.get(Product.VOLCANIC_ROCK)
        if rock_depth and rock_depth.buy_orders and rock_depth.sell_orders:
            rock_bid = max(rock_depth.buy_orders.keys())
            rock_ask = min(rock_depth.sell_orders.keys())
            rock_mid = (rock_bid + rock_ask) / 2
            self.history[Product.VOLCANIC_ROCK].append(rock_mid)

        rock_prices = np.array(self.history[Product.VOLCANIC_ROCK])
        if len(rock_prices) > 50:
            recent = rock_prices[-50:]
            mean = np.mean(recent)
            std = np.std(recent)
            self.z[Product.VOLCANIC_ROCK] = (rock_prices[-1] - mean) / std if std > 0 else 0

        threshold = self.params[Product.VOLCANIC_ROCK]["zscore_threshold"]
        position = state.position.get(Product.VOLCANIC_ROCK, 0)
        position_limit = self.position_limits[Product.VOLCANIC_ROCK]

        if self.z[Product.VOLCANIC_ROCK] < -threshold and rock_depth.sell_orders:
            best_ask = min(rock_depth.sell_orders.keys())
            qty = -rock_depth.sell_orders[best_ask]
            buy_qty = min(qty, position_limit - position)
            if buy_qty > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, buy_qty))

        elif self.z[Product.VOLCANIC_ROCK] > threshold and rock_depth.buy_orders:
            best_bid = max(rock_depth.buy_orders.keys())
            qty = rock_depth.buy_orders[best_bid]
            sell_qty = min(qty, position_limit + position)
            if sell_qty > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -sell_qty))

        return orders

    def trade_resin(self, state, market_data):
        product = Product.RAINFOREST_RESIN
        end_pos = state.position.get(product, 0)
        buy_sum = 50 - end_pos
        sell_sum = 50 + end_pos
        orders = []
        order_depth: OrderDepth = state.order_depths[product]
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        bid_prices = list(bids.keys())
        bid_volumes = list(bids.values())
        ask_prices = list(asks.keys())
        ask_volumes = list(asks.values())
        if sell_sum > 0:
            for i in range(0, len(bid_prices)):
                if bid_prices[i] > 10000:
                    fill = min(bid_volumes[i], sell_sum)
                    orders.append(Order(product, bid_prices[i], -fill))
                    sell_sum -= fill
                    end_pos -= fill
                    bid_volumes[i] -= fill
        bid_prices, bid_volumes = zip(*[(ai, bi) for ai, bi in zip(bid_prices, bid_volumes) if bi != 0])
        bid_prices = list(bid_prices)
        bid_volumes = list(bid_volumes)
        if buy_sum > 0:
            for i in range(0, len(ask_prices)):
                if ask_prices[i] < 10000:
                    fill = min(-ask_volumes[i], buy_sum)
                    orders.append(Order(product, ask_prices[i], fill))
                    buy_sum -= fill
                    end_pos += fill
                    ask_volumes[i] += fill
        ask_prices, ask_volumes = zip(*[(ai, bi) for ai, bi in zip(ask_prices, ask_volumes) if bi != 0])
        ask_prices = list(ask_prices)
        ask_volumes = list(ask_volumes)
        if abs(ask_volumes[0]) > 1:
            orders.append(Order(product, max(ask_prices[0] - 1, 10000 + 1), -min(14, sell_sum)))
        else:
            orders.append(Order(product, max(10000 + 1, ask_prices[0]), -min(14, sell_sum)))
        sell_sum -= min(14, sell_sum)
        if bid_volumes[0] > 1:
            orders.append(Order(product, min(bid_prices[0] + 1, 10000 - 1), min(14, buy_sum)))
        else:
            orders.append(Order(product, min(10000 - 1, bid_prices[0]), min(14, buy_sum)))
        buy_sum -= min(14, buy_sum)
        if end_pos > 0:
            for i in range(0, len(bid_prices)):
                if bid_prices[i] == 10000:
                    fill = min(bid_volumes[i], sell_sum)
                    orders.append(Order(product, bid_prices[i], -fill))
                    sell_sum -= fill
                    end_pos -= fill
        if end_pos < 0:
            for i in range(0, len(ask_prices)):
                if ask_prices[i] == 10000:
                    fill = min(-ask_volumes[i], buy_sum)
                    orders.append(Order(product, ask_prices[i], fill))
                    buy_sum -= fill
                    end_pos += fill
        return orders

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes_call(self, S: float, K: float, T_days: float, r: float, sigma: float) -> float:
        T = T_days / 365.0
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)

    def implied_vol_call(self, market_price, S, K, T_days, r, tol=0.00000000000001, max_iter=250):
        sigma_low = 0.01
        sigma_high = 0.35
        if market_price <= 0 or S <= 0 or T_days <= 0:
            return 0.01
        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            if sigma_mid < 1e-5:
                sigma_mid = 1e-5
            price = self.black_scholes_call(S, K, T_days, r, sigma_mid)
            if abs(price - market_price) < tol:
                return sigma_mid
            if price > market_price:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid
        sigma_mid = (sigma_low + sigma_high) / 2
        return max(sigma_mid, 1e-5)

    def call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        r = 0
        T = T / 365
        if T == 0 or sigma == 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))

    def trade_9750(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_9750"  # Use string
        orders = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair[Product.VOLCANIC_ROCK]
        dte = DAYS_LEFT - state.timestamp / 1_000_000
        v_t = self.implied_vol_call(fair, underlying_fair, 9750, dte, 0)
        m_t = np.log(9750 / underlying_fair) / np.sqrt(dte / 365)
        base_coef = 0.264416
        linear_coef = 0.010031
        squared_coef = 0.147604
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        diff = v_t - fair_iv
        if "prices_9750" not in traderObject:
            traderObject["prices_9750"] = [diff]
        else:
            traderObject["prices_9750"].append(diff)
        threshold = 0.0055
        if len(traderObject["prices_9750"]) > 13:
            diff -= np.mean(traderObject["prices_9750"])
            traderObject["prices_9750"].pop(0)
        if diff > threshold:
            amount = market_data.sell_sum[product]
            amount = min(amount, sum(market_data.bid_volumes[product]))
            option_amount = amount
            for i in range(0, len(market_data.bid_prices[product])):
                fill = min(market_data.bid_volumes[product][i], option_amount)
                if fill != 0:
                    orders.append(Order(product, market_data.bid_prices[product][i], -fill))
                    market_data.sell_sum[product] -= fill
                    market_data.end_pos[product] -= fill
                    option_amount -= fill
        elif diff < -threshold:
            amount = market_data.buy_sum[product]
            amount = min(amount, -sum(market_data.ask_volumes[product]))
            option_amount = amount
            for i in range(0, len(market_data.ask_prices[product])):
                fill = min(-market_data.ask_volumes[product][i], option_amount)
                if fill != 0:
                    orders.append(Order(product, market_data.ask_prices[product][i], fill))
                    market_data.buy_sum[product] -= fill
                    market_data.end_pos[product] += fill
                    option_amount -= fill
        return orders

    def trade_10000(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_10000"  # Use string
        orders = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair[Product.VOLCANIC_ROCK]
        dte = DAYS_LEFT - state.timestamp / 1_000_000
        v_t = self.implied_vol_call(fair, underlying_fair, 10000, dte, 0)
        m_t = np.log(10000 / underlying_fair) / np.sqrt(dte / 365)
        base_coef = 0.14786181
        linear_coef = 0.00099561
        squared_coef = 0.23544086
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        diff = v_t - fair_iv
        if "prices_10000" not in traderObject:
            traderObject["prices_10000"] = [diff]
        else:
            traderObject["prices_10000"].append(diff)
        threshold = 0.0035
        if len(traderObject["prices_10000"]) > 20:
            diff -= np.mean(traderObject["prices_10000"])
            traderObject["prices_10000"].pop(0)
            if diff > threshold:
                amount = market_data.sell_sum[product]
                amount = min(amount, sum(market_data.bid_volumes[product]))
                option_amount = amount
                for i in range(0, len(market_data.bid_prices[product])):
                    fill = min(market_data.bid_volumes[product][i], option_amount)
                    if fill != 0:
                        orders.append(Order(product, market_data.bid_prices[product][i], -fill))
                        market_data.sell_sum[product] -= fill
                        market_data.end_pos[product] -= fill
                        option_amount -= fill
            elif diff < -threshold:
                amount = market_data.buy_sum[product]
                amount = min(amount, -sum(market_data.ask_volumes[product]))
                option_amount = amount
                for i in range(0, len(market_data.ask_prices[product])):
                    fill = min(-market_data.ask_volumes[product][i], option_amount)
                    if fill != 0:
                        orders.append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill
                        option_amount -= fill
        return orders

    def calculate_sunlight_rate_of_change(self, traderObject):
        if len(traderObject["sunlight_history"]) < 5:
            return 0
        changes = []
        for i in range(1, len(traderObject["sunlight_history"])):
            changes.append(traderObject["sunlight_history"][i] - traderObject["sunlight_history"][i - 1])
        return sum(changes) / len(changes)

    def take_macaron(self, state, market_data, traderObject):
        product = Product.MAGNIFICENT_MACARONS
        orders = []
        fair = market_data.fair[product]
        conversions = 0
        overseas_ask = state.observations.conversionObservations[product].askPrice + state.observations.conversionObservations[product].transportFees + state.observations.conversionObservations[product].importTariff
        overseas_bid = state.observations.conversionObservations[product].bidPrice - state.observations.conversionObservations[product].transportFees - state.observations.conversionObservations[product].exportTariff
        if 'last_sunlight' in traderObject:
            if state.observations.conversionObservations[product].sunlightIndex < traderObject["last_sunlight"]:
                direction = -1
            elif state.observations.conversionObservations[product].sunlightIndex == traderObject["last_sunlight"]:
                direction = 0
            else:
                direction = 1
        else:
            direction = 0
        if "sunlight_history" in traderObject:
            traderObject["sunlight_history"].append(state.observations.conversionObservations[product].sunlightIndex)
        else:
            traderObject["sunlight_history"] = [state.observations.conversionObservations[product].sunlightIndex]
        if len(traderObject["sunlight_history"]) > 5:
            traderObject["sunlight_history"].pop(0)
        traderObject['last_sunlight'] = state.observations.conversionObservations[product].sunlightIndex
        total_bids = sum(market_data.bid_volumes[product])
        total_asks = -sum(market_data.ask_volumes[product])
        current_sunlight = state.observations.conversionObservations[product].sunlightIndex
        mean_price = 640
        std_dev = 55
        current_price = fair
        z_score = (current_price - mean_price) / std_dev
        if current_sunlight < 50:
            if direction == -1 and market_data.buy_sum[product] > 0:
                amount = min(market_data.buy_sum[product], -sum(market_data.ask_volumes[product]))
                for i in range(0, len(market_data.ask_prices[product])):
                    fill = min(-market_data.ask_volumes[product][i], amount)
                    if fill != 0:
                        orders.append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill
                        amount -= fill
            elif direction == 1 and market_data.sell_sum[product] > 0 and self.calculate_sunlight_rate_of_change(traderObject) > 0.008:
                amount = min(market_data.sell_sum[product], sum(market_data.bid_volumes[product]))
                for i in range(0, len(market_data.bid_prices[product])):
                    fill = min(market_data.bid_volumes[product][i], amount)
                    if fill != 0:
                        orders.append(Order(product, market_data.bid_prices[product][i], -fill))
                        market_data.sell_sum[product] -= fill
                        market_data.end_pos[product] -= fill
                        amount -= fill
            elif abs(current_sunlight - 49) < 1 and market_data.end_pos[product] < 0:
                amount = min(market_data.buy_sum[product], -market_data.end_pos[product])
                for i in range(0, len(market_data.ask_prices[product])):
                    fill = min(-market_data.ask_volumes[product][i], amount)
                    if fill != 0:
                        orders.append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill
                        amount -= fill
        elif current_sunlight > 50:
            if z_score < -1.2 and market_data.buy_sum[product] > 0:
                amount = min(market_data.buy_sum[product], -sum(market_data.ask_volumes[product]))
                for i in range(0, len(market_data.ask_prices[product])):
                    fill = min(-market_data.ask_volumes[product][i], amount)
                    if fill != 0:
                        orders.append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill
                        amount -= fill
            elif z_score > 1.2 and market_data.sell_sum[product] > 0:
                amount = min(market_data.sell_sum[product], sum(market_data.bid_volumes[product]))
                for i in range(0, len(market_data.bid_prices[product])):
                    fill = min(market_data.bid_volumes[product][i], amount)
                    if fill != 0:
                        orders.append(Order(product, market_data.bid_prices[product][i], -fill))
                        market_data.sell_sum[product] -= fill
                        market_data.end_pos[product] -= fill
                        amount -= fill
        return orders, conversions

    def make_macaron(self, state, market_data):
        product = Product.MAGNIFICENT_MACARONS
        orders: List[Order] = []
        order_depth = state.order_depths[product]
        fair_mid = market_data.fair[product]
        pos = market_data.end_pos[product]
        bid_px = math.floor(fair_mid - 4)
        ask_px = math.ceil(fair_mid + 4)
        size = 14
        buy_cap = self.position_limits[product] - pos
        sell_cap = self.position_limits[product] + pos
        if buy_cap > 0:
            qty = min(size, buy_cap)
            orders.append(Order(product, bid_px, qty))
        if sell_cap > 0:
            qty = min(size, sell_cap)
            orders.append(Order(product, ask_px, -qty))
        return orders

    def clear_macaron(self, state, market_data):
        product = Product.MAGNIFICENT_MACARONS
        orders: List[Order] = []
        fair = market_data.fair[product]
        pos = market_data.end_pos[product]
        width = 3
        if pos > 0:
            orders.append(Order(product, round(fair + width), -pos))
        elif pos < 0:
            orders.append(Order(product, round(fair - width), -pos))
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}
        traderObject.setdefault(Product.PICNIC_BASKET2, {"spread_history": [], "prev_zscore": 0, "clear_flag": False, "curr_avg": 0})

        result = {symbol: [] for symbol in state.listings.keys()}
        conversions = 0
        market_data = MarketData()
    def __init__(self, params=None):
        self.params = params or PARAMS
        
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK: 50, 
                      Product.PICNIC_BASKET1: 60, Product.CROISSANTS: 250, Product.JAMS: 350,
                      Product.PICNIC_BASKET2: 100,  Product.VOLCANIC_ROCK: 400,
                      Product.VOLCANIC_ROCK_VOUCHER_10000:200, Product.VOLCANIC_ROCK_VOUCHER_10250:200,
                        Product.VOLCANIC_ROCK_VOUCHER_10500:200, Product.VOLCANIC_ROCK_VOUCHER_9750:200,
                        Product.VOLCANIC_ROCK_VOUCHER_9500:200, Product.MACARONS: 65}
                      
        self.signal = {
            Product.RAINFOREST_RESIN: 0,
            Product.KELP: 0,
            Product.SQUID_INK: 0,
            Product.PICNIC_BASKET1: 0,
            Product.CROISSANTS: 0,
            Product.JAMS: 0,
            Product.DJEMBES: 0,
            Product.PICNIC_BASKET2: 0,
            Product.VOLCANIC_ROCK: 0,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 0,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 0,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 0,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 0,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 0,
        }


    def VOLCANIC_ROCK_price(self, state):
        depth = state.order_depths["VOLCANIC_ROCK"]
        if not depth.sell_orders or not depth.buy_orders:
            return 0
        buy = max(list(depth.buy_orders.keys()))
        sell = min(list(depth.sell_orders.keys()))
        if (buy == 0 or sell == 0):
            return 0
        return (buy + sell) / 2
    
    
    def update_signal(self, state: TradingState, traderObject, product) -> None:
        
        if not state.order_depths[product].sell_orders or not state.order_depths[product].buy_orders:
            return None

        
        order_depth = state.order_depths[product]
        sell_vol = sum(abs(qty) for qty in order_depth.sell_orders.values())
        buy_vol = sum(abs(qty) for qty in order_depth.buy_orders.values())
        sell_money = sum(price * abs(qty) for price, qty in order_depth.sell_orders.items())
        buy_money = sum(price * abs(qty) for price, qty in order_depth.buy_orders.items())
        if sell_vol == 0 or buy_vol == 0:
            return None
        fair_value = (sell_money + buy_money) / (sell_vol + buy_vol)

        vwap = fair_value
        last_prices = traderObject.get(f"{product}_last_prices", [])
        last_prices.append(vwap)
        
        if len(last_prices) > self.params[product]["ma_length"]:
            last_prices.pop(0)
        
        traderObject[f"{product}_last_prices"] = last_prices

        if len(last_prices) < self.params[product]["ma_length"]:
            return None
        
        long_ma = np.mean(last_prices)
        sd = np.std(last_prices)
        zscore = (vwap - long_ma) / sd
        sma_short = last_prices[-self.params[product]["short_ma_length"] :]
        sma_diffed = np.diff(sma_short, n=1)

        buy_signal = zscore < self.params[product]["open_threshold"] and sma_diffed[-1] > 0 and sma_diffed[-2] > 0 and sma_short[-1] > sma_short[-2] and sma_diffed[-1] > sma_diffed[-2]
        sell_signal = zscore > self.params[product]["close_threshold"] and sma_diffed[-1] < 0 and sma_diffed[-2] > 0 and sma_short[-1] < sma_short[-2] and sma_diffed[-1] < sma_diffed[-2]

        extreme_buy_signal = zscore < -4 or fair_value < 20
        buy_signal |= extreme_buy_signal
        extreme_sell_signal = zscore > 4
        sell_signal |= extreme_sell_signal

        neutral_signal = abs(zscore) < 0

        if buy_signal:
            self.signal[product] = 1
        elif sell_signal:
            self.signal[product] = -1


        if extreme_sell_signal:
            self.signal[product] = -2
        if extreme_buy_signal:
            self.signal[product] = 2

    
        
    def spam_orders(self, state : TradingState, product, signal_product):

        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders

        if not buy_orders or not sell_orders:
            return []
        
        
        orders = []
        pos = state.position.get(product, 0)

        if self.signal[signal_product] == 2:
            # take all sell orders
            orderdepth = state.order_depths[product]
            for price, qty in orderdepth.sell_orders.items():
                if pos + abs(qty) > self.LIMIT[product]:
                    break
                orders.append(Order(product, price, abs(qty)))
                pos += abs(qty)
            rem_buy = self.LIMIT[product] - pos
            best_buy = max(orderdepth.buy_orders.keys())
            orders.append(Order(product, best_buy + 1, rem_buy))
            return orders
        
        elif self.signal[signal_product] == -2:
            # take all buy orders
            orderdepth = state.order_depths[product]
            for price, qty in orderdepth.buy_orders.items():
                if pos - abs(qty) < -self.LIMIT[product]:
                    break
                orders.append(Order(product, price, -abs(qty)))
                pos -= abs(qty)
            rem_sell = self.LIMIT[product] + pos
            best_sell = min(orderdepth.sell_orders.keys())
            orders.append(Order(product, best_sell - 1, -rem_sell))
            return orders


        if self.signal[signal_product] > 0:
            rem_buy = self.LIMIT[product] - pos
            orderdepth = state.order_depths[product]
            # add our own buy order at best_buy + 1
            best_buy = max(orderdepth.buy_orders.keys())
            orders.append(Order(product, best_buy + 1, rem_buy))
        
        elif self.signal[signal_product] < 0:
            rem_sell = self.LIMIT[product] + pos
            orderdepth = state.order_depths[product]
            # add our own sell order at best_sell - 1
            best_sell = min(orderdepth.sell_orders.keys())
            orders.append(Order(product, best_sell - 1, -rem_sell))
        
        elif self.signal[signal_product] == 0:
            best_buy = max(state.order_depths[product].buy_orders.keys())
            best_sell = min(state.order_depths[product].sell_orders.keys())

            if pos > 0:
                # close buy position
                orders.append(Order(product, best_buy + 1, -pos))
            elif pos < 0:
                # close sell position
                orders.append(Order(product, best_sell - 1, -pos))
        
        return orders


        # Initialize MarketData for all products
        products = [
            Product.RAINFOREST_RESIN,
            Product.KELP,
            Product.VOLCANIC_ROCK,
            "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000",
            Product.MAGNIFICENT_MACARONS,
            Product.SQUID_INK,
            Product.CROISSANTS,
            Product.JAMS,
            Product.DJEMBES,
            Product.PICNIC_BASKET1,
            Product.PICNIC_BASKET2,
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500",
        ]
        for product in products:
            if product not in state.order_depths:
                continue
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            bids, asks = order_depth.buy_orders, order_depth.sell_orders
            mm_bid = max(bids.keys()) if bids else None
            mm_ask = min(asks.keys()) if asks else None
            if bids and asks:
                fair_price = (mm_ask + mm_bid) / 2
            elif asks:
                fair_price = mm_ask
            elif bids:
                fair_price = mm_bid
            else:
                fair_price = traderObject.get(f"prev_fair_{product}", 0)
            traderObject[f"prev_fair_{product}"] = fair_price
            market_data.end_pos[product] = position
            market_data.buy_sum[product] = self.position_limits[product] - position
            market_data.sell_sum[product] = self.position_limits[product] + position
            market_data.bid_prices[product] = list(bids.keys())
            market_data.bid_volumes[product] = list(bids.values())
            market_data.ask_prices[product] = list(asks.keys())
            market_data.ask_volumes[product] = list(asks.values())
            market_data.fair[product] = fair_price

        # SQUID_INK
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            ink_position = state.position.get(Product.SQUID_INK, 0)
            ink_od_copy = OrderDepth()
            ink_od_copy.buy_orders = state.order_depths[Product.SQUID_INK].buy_orders.copy()
            ink_od_copy.sell_orders = state.order_depths[Product.SQUID_INK].sell_orders.copy()
            ink_fair_value = self.ink_fair_value(ink_od_copy, traderObject)
            if ink_fair_value is not None:
                ink_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.SQUID_INK, ink_od_copy, ink_fair_value,
                    self.params[Product.SQUID_INK]['take_width'], ink_position,
                    self.params[Product.SQUID_INK]['prevent_adverse'], self.params[Product.SQUID_INK]['adverse_volume'], traderObject
                )
                ink_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.SQUID_INK, ink_od_copy, ink_fair_value,
                    self.params[Product.SQUID_INK]['clear_width'], ink_position, buy_order_volume, sell_order_volume
                )
                ink_make_orders, _, _ = self.make_orders(
                    Product.SQUID_INK, state.order_depths[Product.SQUID_INK], ink_fair_value, ink_position,
                    buy_order_volume, sell_order_volume, self.params[Product.SQUID_INK]['disregard_edge'],
                    self.params[Product.SQUID_INK]['join_edge'], self.params[Product.SQUID_INK]['default_edge']
                )
                result[Product.SQUID_INK].extend(ink_take_orders + ink_clear_orders + ink_make_orders)

        # CROISSANTS
        if Product.CROISSANTS in state.order_depths:
            croissant_orders = self.croissants_strategy(state)
            result[Product.CROISSANTS].extend(croissant_orders)

        # JAMS
        if Product.JAMS in state.order_depths:
            jams_orders = self.jams_strategy(state)
            result[Product.JAMS].extend(jams_orders)

        # DJEMBES
        if Product.DJEMBES in state.order_depths:
            djembes_orders = self.djembes_strategy(state)
            result[Product.DJEMBES].extend(djembes_orders)

        # PICNIC_BASKET1
        if all(p in state.order_depths for p in (Product.PICNIC_BASKET1, Product.CROISSANTS, Product.JAMS, Product.DJEMBES)):
            pb1_result = self.picnic1_strategy.run(state)
            pb1_orders = pb1_result[0]
            result.setdefault(Product.PICNIC_BASKET1, [])
            result[Product.PICNIC_BASKET1].extend(pb1_orders)

        # PICNIC_BASKET2
        if Product.PICNIC_BASKET2 in self.params and all(p in state.order_depths for p in [Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS]):
            basket2_position = state.position.get(Product.PICNIC_BASKET2, 0)
            basket2_fair_value = self.basket2_fair_value(
                state.order_depths[Product.PICNIC_BASKET2],
                state.order_depths[Product.CROISSANTS],
                state.order_depths[Product.JAMS],
                basket2_position,
                traderObject
            )
            if basket2_fair_value is not None:
                b2_take, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.PICNIC_BASKET2,
                    state.order_depths[Product.PICNIC_BASKET2],
                    basket2_fair_value,
                    self.params[Product.PICNIC_BASKET2]["take_width"],
                    basket2_position,
                    self.params[Product.PICNIC_BASKET2]["prevent_adverse"],
                    self.params[Product.PICNIC_BASKET2]["adverse_volume"],
                    traderObject
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

        # VOLCANIC_ROCK
        result[Product.VOLCANIC_ROCK].extend(self.update_volcanic_rock_history(state))

        # VOUCHERS (9500, 10250, 10500)
        if Product.VOLCANIC_ROCK in state.order_depths:
            for strategy in self.voucher_strategies:
                if isinstance(strategy, ParabolaFitIVStrategy):
                    orders, _, _ = strategy.run(state)
                    for symbol, order_list in orders.items():
                        result[symbol].extend(order_list)
                elif isinstance(strategy, VolcanicMarketMakingStrategy):
                    orders, _ = strategy.run(state)
                    result[strategy.symbol].extend(orders)

        # RAINFOREST_RESIN
        if Product.RAINFOREST_RESIN in state.order_depths:
            result[Product.RAINFOREST_RESIN] = self.trade_resin(state, market_data)

        # KELP
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject)
            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.KELP, state.order_depths[Product.KELP], kelp_fair_value,
                self.params[Product.KELP]['take_width'], kelp_position,
                self.params[Product.KELP]['prevent_adverse'], self.params[Product.KELP]['adverse_volume'], traderObject
            )
        kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
            Product.KELP, state.order_depths[Product.KELP], kelp_fair_value,
            self.params[Product.KELP]['clear_width'], kelp_position, buy_order_volume, sell_order_volume
        )
        kelp_make_orders, _, _ = self.make_orders(
            Product.KELP, state.order_depths[Product.KELP], kelp_fair_value, kelp_position,
            buy_order_volume, sell_order_volume, self.params[Product.KELP]['disregard_edge'],
            self.params[Product.KELP]['join_edge'], self.params[Product.KELP]['default_edge']
        )
        result[Product.KELP].extend(kelp_take_orders + kelp_clear_orders + kelp_make_orders)

        # VOLCANIC_ROCK_VOUCHER_9750
        voucher_9750_symbol = "VOLCANIC_ROCK_VOUCHER_9750" # Use the string literal
        if voucher_9750_symbol in state.order_depths:
            # Ensure the key exists in the result dictionary before extending
            result.setdefault(voucher_9750_symbol, [])
            result[voucher_9750_symbol].extend(self.trade_9750(state, market_data, traderObject))

       # VOLCANIC_ROCK_VOUCHER_10000
        voucher_10000_symbol = "VOLCANIC_ROCK_VOUCHER_10000" # Use the string literal
        if voucher_10000_symbol in state.order_depths:
            # Ensure the key exists in the result dictionary before extending
            result.setdefault(voucher_10000_symbol, [])
            result[voucher_10000_symbol].extend(self.trade_10000(state, market_data, traderObject))


        # MAGNIFICENT_MACARONS
        if Product.MAGNIFICENT_MACARONS in state.order_depths:
            macaron_take_orders, macaron_conversions = self.take_macaron(state, market_data, traderObject)
            macaron_make_orders = self.make_macaron(state, market_data)
            macaron_clear_orders = self.clear_macaron(state, market_data)
            result[Product.MAGNIFICENT_MACARONS].extend(macaron_take_orders + macaron_make_orders + macaron_clear_orders)
            conversions += macaron_conversions

        # Serialize traderObject for next iteration
        trader_data = jsonpickle.encode(traderObject)

        # Log and flush
        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data