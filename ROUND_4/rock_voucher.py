from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder
from typing import List, Dict, Tuple
import json, jsonpickle
import math
import numpy as np
from collections import deque
from math import log, sqrt, erf
import statistics

# Logger class (unchanged)
class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def flush(self, state: TradingState, orders: dict[str, list[Order]], conversions: int, trader_data: str):
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
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in trades.values() for t in arr]

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

# Product class (fixed to ensure VOUCHER_PREFIX is defined before VOUCHER_SYMBOLS)
class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
    VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
    STRIKES = [9500, 9750, 10000, 10250, 10500]
    VOUCHER_SYMBOLS = [f"{VOUCHER_PREFIX}{K}" for K in STRIKES]

# PARAMS for VOLCANIC_ROCK (unchanged)
PARAMS = {
    Product.VOLCANIC_ROCK: {
        "window_size": 50,
        "zscore_threshold": 2
    }
}

# ParabolaFitIVStrategy class (unchanged)
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

            if mispricing > 2 and position < self.position_limit:
                qty = min(20, self.position_limit - position)
                result.append(Order(self.voucher, best_ask, qty))
            elif mispricing < -2 and position > -self.position_limit:
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
            qty = min(30, self.position_limit - position)
            result.append(Order(self.voucher, best_ask, qty))
        elif v_t > current_fit + 0.013 and position > -self.position_limit:
            qty = min(30, self.position_limit + position)
            result.append(Order(self.voucher, best_bid, -qty))

        orders[self.voucher] = result
        return orders, 0, ""

class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.position_limits = {
            Product.VOLCANIC_ROCK: 400,
            Product.MAGNIFICENT_MACARONS: 75,
            **{symbol: 200 for symbol in Product.VOUCHER_SYMBOLS}
        }
        self.conversion_limit = 10
        self.strikes = {symbol: k for symbol, k in zip(Product.VOUCHER_SYMBOLS, Product.STRIKES)}
        self.voucher_strategies = [
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_10000", 10000),
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_10250", 10250),
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_10500", 10500),
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_9500", 9500, absolute=True),
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_9750", 9750, absolute=True)
        ]
        self.history = {Product.VOLCANIC_ROCK: []}
        self.z = {Product.VOLCANIC_ROCK: 0}
        self.macaron_data = {
            "buy_price_history": [],
            "position_history": {Product.MAGNIFICENT_MACARONS: []},
            "acquisition_price": 0,
            "fee_history": [],
            "last_sell_price": float('inf')
        }

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

    def update_magnificent_macarons(self, state: TradingState) -> Tuple[List[Order], int]:
        orders = []
        conversions = 0
        product = Product.MAGNIFICENT_MACARONS
        order_depth: OrderDepth = state.order_depths.get(product, OrderDepth())
        position = state.position.get(product, 0)
        position_limit = self.position_limits[product]

        conv_obs = state.observations.conversionObservations.get(product)
        if not conv_obs:
            return orders, conversions

        bid_price = conv_obs.bidPrice
        ask_price = conv_obs.askPrice
        transport_fees = conv_obs.transportFees
        import_tariff = conv_obs.importTariff
        export_tariff = conv_obs.exportTariff

        buy_price = ask_price + transport_fees + import_tariff
        sell_price_conv = bid_price - transport_fees - export_tariff
        self.macaron_data["buy_price_history"].append(buy_price)

        self.macaron_data["fee_history"].append({
            "timestamp": state.timestamp,
            "transport_fees": transport_fees,
            "import_tariff": import_tariff,
            "export_tariff": export_tariff
        })

        best_bid = max(order_depth.buy_orders.keys(), default=0)
        bid_quantity = order_depth.buy_orders.get(best_bid, 0)

        recent_prices = self.macaron_data["buy_price_history"][-20:] if len(self.macaron_data["buy_price_history"]) >= 20 else self.macaron_data["buy_price_history"]
        price_stable = buy_price <= (statistics.mean(recent_prices) + statistics.stdev(recent_prices)) if len(recent_prices) > 1 else True
        if position < 0 and self.macaron_data.get("last_sell_price", float('inf')) > buy_price + 0.5 and price_stable and best_bid > buy_price + 1.0:
            conversions = min(-position, self.conversion_limit, bid_quantity)
            self.macaron_data["acquisition_price"] = buy_price
            print(f"CONVERSION REQUEST: Import {conversions} units at {buy_price}")

        max_short = min(bid_quantity, position_limit + position, 10)
        if max_short > 0 and bid_quantity >= 5:
            base_sell_price = max(int(bid_price - 0.5), int(buy_price + 1.5))
            sell_price = best_bid if best_bid >= base_sell_price else base_sell_price
            if sell_price > best_bid + 0.5:
                sell_price = int(best_bid + 0.5) if best_bid > 0 else base_sell_price
            quantity = -max_short
            orders.append(Order(product, sell_price, quantity))
            self.macaron_data["last_sell_price"] = sell_price
            print(f"SELL {product} {-quantity}x at {sell_price}")

        if position > 0:
            acquisition_price = self.macaron_data.get("acquisition_price", buy_price)
            storage_cost = position * 0.1
            if best_bid >= acquisition_price + storage_cost + 0.5:
                quantity = -position
                orders.append(Order(product, best_bid, quantity))
                print(f"SELL {product} {-quantity}x at {best_bid} to avoid storage")
            elif best_bid == 0 and sell_price_conv > acquisition_price + storage_cost + 0.5:
                conversions = -min(position, self.conversion_limit)
                print(f"CONVERSION REQUEST: Export {conversions} units at {sell_price_conv}")

        print(f"Timestamp: {state.timestamp}, Position: {position}, Best Bid: {best_bid}, Buy Price: {buy_price}, Conversions: {conversions}, Orders: {orders}")

        self.macaron_data["position_history"][product].append(position)
        return orders, conversions

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}
        traderObject.setdefault("base_iv_history", {})
        traderObject.setdefault("last_ivs", {})
        traderObject.setdefault("macaron_data", self.macaron_data)

        result = {symbol: [] for symbol in state.listings.keys()}
        total_conversions = 0

        result[Product.VOLCANIC_ROCK].extend(self.update_volcanic_rock_history(state))

        if Product.VOLCANIC_ROCK in state.order_depths:
            for strategy in self.voucher_strategies:
                orders, _, _ = strategy.run(state)
                for symbol, order_list in orders.items():
                    result[symbol].extend(order_list)

        if Product.MAGNIFICENT_MACARONS in state.order_depths:
            macaron_orders, macaron_conversions = self.update_magnificent_macarons(state)
            result[Product.MAGNIFICENT_MACARONS].extend(macaron_orders)
            total_conversions += macaron_conversions

        traderObject["macaron_data"] = self.macaron_data

        final_result = {k: v for k, v in result.items() if v}
        traderData = jsonpickle.encode(traderObject, unpicklable=False)
        logger.flush(state, final_result, total_conversions, traderData)
        return final_result, total_conversions, traderData