import json
from abc import abstractmethod
from typing import Any, TypeAlias, List, Dict, Tuple
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import jsonpickle
import math
import numpy as np
from collections import deque
from math import erf, sqrt, log

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# Logger class (unchanged)
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

# Product class (only relevant commodities)
class Product:
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
    STRIKES = [9500, 10250, 10500]
Product.VOUCHER_SYMBOLS = [f"{Product.VOUCHER_PREFIX}{K}" for K in Product.STRIKES]

# PARAMS (only for relevant commodities)
PARAMS = {
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
    Product.SQUID_INK: {
        "take_width": 2, "clear_width": 1, "prevent_adverse": False, "adverse_volume": 15,
        "reversion_beta": -0.228, "disregard_edge": 2, "join_edge": 0, "default_edge": 1,
        "spike_lb": 3, "spike_ub": 5.6, "offset": 2, "reversion_window": 55,
        "reversion_weight": 0.12
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
        "zscore_threshold": 2
    }
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

    def basket2_fair_value(
        self,
        basket_depth: OrderDepth,
        cro_depth: OrderDepth,
        jam_depth: OrderDepth,
        position: int,
        traderObject: dict,
    ) -> float | None:
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

    def take_best_orders(self, product: str, fair_value: float, take_width: float, orders: List[Order],
                        order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int,
                        prevent_adverse: bool = False, adverse_volume: int = 0, traderObject: dict = None):
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

    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float,
                   position: int, prevent_adverse: bool = False, adverse_volume: int = 0, traderObject: dict = None):
        orders = []
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position, 0, 0,
            prevent_adverse, adverse_volume, traderObject
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

        # VOUCHERS
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
        traderData = jsonpickle.encode(traderObject, unpicklable=False)
        logger.flush(state, final_result, conversions, traderData)
        return final_result, conversions, traderData