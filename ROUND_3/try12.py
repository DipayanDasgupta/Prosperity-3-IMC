# -*- coding: utf-8 -*-
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder
from typing import List, Dict, Tuple
import json, jsonpickle, math, numpy as np, statistics

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
    # VOUCHER_SYMBOLS definition moved outside the class block below

# Define VOUCHER_SYMBOLS after the class is defined to avoid NameError
Product.VOUCHER_SYMBOLS = [f"{Product.VOUCHER_PREFIX}{K}" for K in Product.STRIKES]


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000, "take_width": 1, "clear_width": 0, "disregard_edge": 1,
        "join_edge": 2, "default_edge": 1, "soft_position_limit": 50
    },
    Product.KELP: {
        "take_width": 2, "clear_width": 0, "prevent_adverse": False, "adverse_volume": 15,
        "reversion_beta": -0.18, "disregard_edge": 2, "join_edge": 0, "default_edge": 1,
        "ink_adjustment_factor": 0.05
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
    Product.SPREAD1: {
        "default_spread_mean": 48.777856, "default_spread_std": 85.119723,
        "spread_window": 55, "zscore_threshold": 4, "target_position": 100
    },
    Product.SPREAD2: {
        "default_spread_mean": 30.2336, "default_spread_std": 59.8536,
        "spread_window": 59, "zscore_threshold": 6, "target_position": 100
    },
    Product.PICNIC_BASKET1: {
        "b2_adjustment_factor": 0.05
    }
}

PICNIC1_WEIGHTS = {Product.DJEMBES: 1, Product.CROISSANTS: 6, Product.JAMS: 3}
PICNIC2_WEIGHTS = {Product.CROISSANTS: 4, Product.JAMS: 2}

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
        self.strikes = {symbol: k for symbol, k in zip(Product.VOUCHER_SYMBOLS, Product.STRIKES)}
        self.smile_coeffs = [-1.12938253, -0.0421697, 0.00938588]
        self.risk_free_rate = 0
        self.order_quantity = 10
        self.iv_history_length = 100
        self.mispricing_threshold_std_dev = 1.0
        self.base_iv = 0.009385877054057754

    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def norm_pdf(self, x):
        return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

    def black_scholes_call(self, S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)

    def vega(self, S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * self.norm_pdf(d1) * math.sqrt(T)

    def implied_volatility(self, target_price, S, K, T, r, initial_guess=0.1, max_iterations=100, tolerance=1e-6):
        sigma = initial_guess
        intrinsic_value = max(0.0, S - K * math.exp(-r * T))
        if target_price < intrinsic_value - tolerance or target_price < 0:
            return 0.0
        for _ in range(max_iterations):
            price = self.black_scholes_call(S, K, T, r, sigma)
            v = self.vega(S, K, T, r, sigma)
            diff = price - target_price
            if abs(diff) < tolerance:
                return sigma
            if v < 1e-10:
                if sigma < 0.01:
                    sigma = 0.01
                price_perturbed = self.black_scholes_call(S, K, T, r, sigma * 1.01)
                v_perturbed = self.vega(S, K, T, r, sigma * 1.01)
                if v_perturbed < 1e-10:
                    break
                sigma *= 1.01
                continue
            sigma -= diff / v
            if sigma <= 0:
                sigma = tolerance
            elif sigma > 5.0:
                sigma = 5.0
        final_price = self.black_scholes_call(S, K, T, r, sigma)
        if abs(final_price - target_price) < tolerance * 10:
            return sigma
        return sigma

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

    def calculate_tte(self, timestamp: int) -> float:
        if timestamp < 0:
            timestamp = 0
        day_number = math.floor(timestamp / 10000)
        time_within_day = timestamp % 10000
        steps_per_day = 10000
        start_days_remaining = 8 - day_number
        current_days_remaining = start_days_remaining - (time_within_day + 1) / steps_per_day
        return max(0.0, current_days_remaining / 365.0)

    def calculate_m_t(self, S: float, K: int, TTE: float) -> float | None:
        if S <= 0 or K <= 0 or TTE <= 0:
            return None
        try:
            return math.log(K / S) / math.sqrt(TTE)
        except ValueError:
            return None

    def get_smile_volatility(self, m_t: float) -> float:
        a, b, c = self.smile_coeffs
        smile_iv = a * (m_t ** 2) + b * m_t + c
        return max(0.0001, smile_iv)

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

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject, ink_order_depth: OrderDepth):
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
                mmmid_price = (best_ask + best_bid) / 2 if traderObject.get('kelp_last_price', None) is None else traderObject['kelp_last_price']
            fair = mmmid_price if traderObject.get('kelp_last_price', None) is None else mmmid_price + (
                mmmid_price * ((mmmid_price - traderObject["kelp_last_price"]) / traderObject["kelp_last_price"] * self.params[Product.KELP]["reversion_beta"]))
            if traderObject.get("ink_last_price", None) is not None:
                old_ink_price = traderObject["ink_last_price"]
                valid_ask_ink = [price for price in ink_order_depth.sell_orders.keys() if abs(ink_order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
                valid_buy_ink = [price for price in ink_order_depth.buy_orders.keys() if abs(ink_order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
                new_ink_mid = (min(valid_ask_ink) + max(valid_buy_ink)) / 2 if valid_ask_ink and valid_buy_ink else (
                    min(ink_order_depth.sell_orders.keys()) + max(ink_order_depth.buy_orders.keys())) / 2
                ink_return = (new_ink_mid - old_ink_price) / old_ink_price
                fair -= (self.params[Product.KELP].get("ink_adjustment_factor", 0.5) * ink_return * mmmid_price)
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

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

    def croissants_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.CROISSANTS) is not None:
            position = state.position.get(Product.CROISSANTS, 0)
            order_depth = state.order_depths[Product.CROISSANTS]
            fair_value = self.params[Product.CROISSANTS]["fair_value"]
            croissants_take, bo, so = self.take_orders(
                Product.CROISSANTS, order_depth, fair_value,
                self.params[Product.CROISSANTS]["take_width"],
                position # Removed position_limits here, take_orders doesn't need it directly
            )
            croissants_clear, bo, so = self.clear_orders(
                Product.CROISSANTS, order_depth, fair_value,
                self.params[Product.CROISSANTS]["clear_width"],
                position, bo, so # Removed position_limits here, clear_orders doesn't need it directly
            )
            croissants_make, bo, so = self.make_orders(
                Product.CROISSANTS, order_depth, fair_value, position, bo, so,
                self.params[Product.CROISSANTS]["disregard_edge"],
                self.params[Product.CROISSANTS]["join_edge"],
                self.params[Product.CROISSANTS]["default_edge"],
                # Removed position_limits here, make_orders takes individual limit
                 True, # manage_position
                self.params[Product.CROISSANTS]["soft_position_limit"]
            )
            orders = croissants_take + croissants_clear + croissants_make
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
            # FIX: Use .get(key, 0) to handle cases where best_bid is 0 (no bids)
            croissant_bid_volume = order_depths[Product.CROISSANTS].buy_orders.get(croissant_best_bid, 0) // CROISSANT_PER_PICNIC
            jams_bid_volume = order_depths[Product.JAMS].buy_orders.get(jams_best_bid, 0) // JAM_PER_PICNIC
            if picnic1:
                djembes_bid_volume = order_depths[Product.DJEMBES].buy_orders.get(djembes_best_bid, 0) // DJEMBES_PER_PICNIC
                artifical_bid_volume = min(djembes_bid_volume, croissant_bid_volume, jams_bid_volume)
            else:
                artifical_bid_volume = min(croissant_bid_volume, jams_bid_volume)
            artifical_order_price.buy_orders[art_bid] = artifical_bid_volume
        if art_ask < float("inf"):
             # FIX: Use .get(key, 0) to handle cases where best_ask is inf (no asks) - though less likely to cause KeyError
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
                 # Ensure there are sell orders before trying to find the min
                if not order_depths[Product.CROISSANTS].sell_orders or not order_depths[Product.JAMS].sell_orders or (picnic1 and not order_depths[Product.DJEMBES].sell_orders): continue
                croissant_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
                if picnic1:
                    djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Ensure there are buy orders before trying to find the max
                if not order_depths[Product.CROISSANTS].buy_orders or not order_depths[Product.JAMS].buy_orders or (picnic1 and not order_depths[Product.DJEMBES].buy_orders): continue
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

        # Ensure books are not empty before proceeding
        if not picnic_order_depth.sell_orders or not artifical_order_depth.buy_orders or \
           not picnic_order_depth.buy_orders or not artifical_order_depth.sell_orders:
             return None # Cannot execute if any side is missing

        if target_position > picnic_position: # Buy picnic, Sell components (buy artifical negative)
            picnic_ask_price = min(picnic_order_depth.sell_orders.keys())
            picnic_ask_vol = abs(picnic_order_depth.sell_orders[picnic_ask_price])
            artifical_bid_price = max(artifical_order_depth.buy_orders.keys())
            artifical_bid_vol = abs(artifical_order_depth.buy_orders[artifical_bid_price])

            if picnic_ask_vol == 0 or artifical_bid_vol == 0: return None # Check volume exists

            orderbook_volume = min(picnic_ask_vol, artifical_bid_vol)
            execute_volume = min(orderbook_volume, target_quantity)
            if execute_volume <= 0: return None # Ensure positive execution volume

            picnic_orders = [Order(Product.PICNIC_BASKET1 if picnic1 else Product.PICNIC_BASKET2, picnic_ask_price, execute_volume)]
            artifical_orders = [Order(Product.ARTIFICAL1, artifical_bid_price, -execute_volume)]
            aggregate_orders = self.convert_orders(artifical_orders, order_depths, picnic1)

            # Add picnic orders to the correct key
            if picnic1:
                aggregate_orders[Product.PICNIC_BASKET1] = picnic_orders
            else:
                aggregate_orders[Product.PICNIC_BASKET2] = picnic_orders
            return aggregate_orders
        else: # Sell picnic, Buy components (sell artifical negative)
            picnic_bid_price = max(picnic_order_depth.buy_orders.keys())
            picnic_bid_vol = abs(picnic_order_depth.buy_orders[picnic_bid_price])
            artifical_ask_price = min(artifical_order_depth.sell_orders.keys())
            artifical_ask_vol = abs(artifical_order_depth.sell_orders[artifical_ask_price])

            if picnic_bid_vol == 0 or artifical_ask_vol == 0: return None # Check volume exists

            orderbook_volume = min(picnic_bid_vol, artifical_ask_vol)
            execute_volume = min(orderbook_volume, target_quantity)
            if execute_volume <= 0: return None # Ensure positive execution volume

            picnic_orders = [Order(Product.PICNIC_BASKET1 if picnic1 else Product.PICNIC_BASKET2, picnic_bid_price, -execute_volume)]
            artifical_orders = [Order(Product.ARTIFICAL1, artifical_ask_price, execute_volume)] # Positive quantity for artificial order
            aggregate_orders = self.convert_orders(artifical_orders, order_depths, picnic1)

            # Add picnic orders to the correct key
            if picnic1:
                aggregate_orders[Product.PICNIC_BASKET1] = picnic_orders
            else:
                aggregate_orders[Product.PICNIC_BASKET2] = picnic_orders
            return aggregate_orders


    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: str, picnic_position: int,
                     spread_data: Dict[str, object], SPREAD, picnic1: bool = True):
        # Simplified check: ensure all necessary components are present
        required_products = [Product.PICNIC_BASKET1, Product.DJEMBES, Product.CROISSANTS, Product.JAMS] if picnic1 else [Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS]
        if not all(p in order_depths for p in required_products):
            return None

        picnic_order_depth = order_depths[Product.PICNIC_BASKET1] if picnic1 else order_depths[Product.PICNIC_BASKET2]
        artifical_order_depth = self.artifical_order_depth(order_depths, picnic1)

        # Check if order books have prices before calculating midprice
        if not picnic_order_depth.buy_orders or not picnic_order_depth.sell_orders or \
           not artifical_order_depth.buy_orders or not artifical_order_depth.sell_orders:
             return None # Cannot calculate spread if any side is missing

        best_bid_picnic = max(picnic_order_depth.buy_orders.keys())
        best_bid_vol_picnic = abs(picnic_order_depth.buy_orders[best_bid_picnic])
        best_ask_picnic = min(picnic_order_depth.sell_orders.keys())
        best_ask_vol_picnic = abs(picnic_order_depth.sell_orders[best_ask_picnic])
        picnic_mprice = (best_bid_picnic * best_ask_vol_picnic + best_ask_picnic * best_bid_vol_picnic) / (best_bid_vol_picnic + best_ask_vol_picnic) if best_bid_vol_picnic + best_ask_vol_picnic > 0 else (best_bid_picnic + best_ask_picnic) / 2

        best_bid_art = max(artifical_order_depth.buy_orders.keys())
        best_bid_vol_art = abs(artifical_order_depth.buy_orders[best_bid_art])
        best_ask_art = min(artifical_order_depth.sell_orders.keys())
        best_ask_vol_art = abs(artifical_order_depth.sell_orders[best_ask_art])
        artifical_mprice = (best_bid_art * best_ask_vol_art + best_ask_art * best_bid_vol_art) / (best_bid_vol_art + best_ask_vol_art) if best_bid_vol_art + best_ask_vol_art > 0 else (best_bid_art + best_ask_art) / 2


        spread = picnic_mprice - artifical_mprice
        spread_data["spread_history"].append(spread)

        if len(spread_data["spread_history"]) < self.params[SPREAD]["spread_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[SPREAD]["spread_window"]:
            spread_data["spread_history"].pop(0)

        spread_mean = np.mean(spread_data["spread_history"]) # Use rolling mean
        spread_std = np.std(spread_data["spread_history"])

        if spread_std == 0: return None # Avoid division by zero

        # Use rolling mean instead of fixed default mean
        zscore = (spread - spread_mean) / spread_std


        target_pos = self.params[SPREAD]["target_position"]
        if zscore >= self.params[SPREAD]["zscore_threshold"]:
            if picnic_position > -target_pos: # Check if we can still sell more
                 # Calculate quantity to reach target, respecting current position
                qty_to_trade = -target_pos - picnic_position
                return self.execute_spreads(picnic_position + qty_to_trade, picnic_position, order_depths, picnic1)
        elif zscore <= -self.params[SPREAD]["zscore_threshold"]:
             if picnic_position < target_pos: # Check if we can still buy more
                 # Calculate quantity to reach target, respecting current position
                 qty_to_trade = target_pos - picnic_position
                 return self.execute_spreads(picnic_position + qty_to_trade, picnic_position, order_depths, picnic1)
        # Logic to potentially close position if z-score reverts towards mean (optional)
        # elif abs(zscore) < some_closing_threshold and picnic_position != 0:
        #     return self.execute_spreads(0, picnic_position, order_depths, picnic1)

        spread_data["prev_zscore"] = zscore
        return None


    def trade_resin(self, state):
        product = Product.RAINFOREST_RESIN
        position_limit = self.position_limits[product]
        fair_value = self.params[product]["fair_value"]
        current_pos = state.position.get(product, 0)
        buy_sum = position_limit - current_pos
        sell_sum = position_limit + current_pos # Max sell is limit + current position
        orders = []
        order_depth = state.order_depths[product]
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders

        # Take aggressive orders first
        if asks:
             best_ask = min(asks.keys())
             best_ask_vol = abs(asks[best_ask])
             if best_ask < fair_value:
                 fill = min(best_ask_vol, buy_sum)
                 if fill > 0:
                     orders.append(Order(product, best_ask, fill))
                     buy_sum -= fill
                     current_pos += fill

        if bids:
            best_bid = max(bids.keys())
            best_bid_vol = abs(bids[best_bid])
            if best_bid > fair_value:
                 fill = min(best_bid_vol, sell_sum)
                 if fill > 0:
                     orders.append(Order(product, best_bid, -fill))
                     sell_sum -= fill
                     current_pos -= fill

        # Market making around fair value
        if asks:
             best_ask = min(asks.keys())
             ask_price = max(fair_value + 1, best_ask -1) # Undercut best ask or quote above fair
        else:
             ask_price = fair_value + 1

        if bids:
            best_bid = max(bids.keys())
            bid_price = min(fair_value - 1, best_bid + 1) # Overcut best bid or quote below fair
        else:
             bid_price = fair_value - 1

        # Adjust based on position only if not hitting limits aggressively
        # Soft limit adjustment - maybe integrated into price setting or quantity logic later
        # if current_pos > self.params[product].get("soft_position_limit", position_limit / 2):
        #     ask_price -= 1 # Be more aggressive selling
        # elif current_pos < -self.params[product].get("soft_position_limit", position_limit / 2):
        #     bid_price += 1 # Be more aggressive buying

        # Place maker orders respecting remaining limits
        if buy_sum > 0:
             orders.append(Order(product, bid_price, buy_sum))

        if sell_sum > 0:
            orders.append(Order(product, ask_price, -sell_sum))

        return orders


    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}
        traderObject.setdefault("base_iv_history", [])
        traderObject.setdefault(Product.SPREAD1, {"spread_history": [], "prev_zscore": 0, "clear_flag": False, "curr_avg": 0})
        traderObject.setdefault(Product.SPREAD2, {"spread_history": [], "prev_zscore": 0, "clear_flag": False, "curr_avg": 0})
        traderObject.setdefault("last_ivs", {}) # Initialize last_ivs

        result = {symbol: [] for symbol in state.listings.keys()} # Use listings for all possible products
        conversions = 0

        # RAINFOREST_RESIN
        if Product.RAINFOREST_RESIN in state.order_depths:
            result[Product.RAINFOREST_RESIN].extend(self.trade_resin(state)) # Use extend

        # KELP
        if Product.KELP in self.params and Product.KELP in state.order_depths and Product.SQUID_INK in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            # Make a copy of order depth to avoid modifying the original state during calculations
            kelp_od_copy = OrderDepth()
            kelp_od_copy.buy_orders = state.order_depths[Product.KELP].buy_orders.copy()
            kelp_od_copy.sell_orders = state.order_depths[Product.KELP].sell_orders.copy()
            ink_od_copy = OrderDepth()
            ink_od_copy.buy_orders = state.order_depths[Product.SQUID_INK].buy_orders.copy()
            ink_od_copy.sell_orders = state.order_depths[Product.SQUID_INK].sell_orders.copy()

            kelp_fair_value = self.kelp_fair_value(kelp_od_copy, traderObject, ink_od_copy)

            if kelp_fair_value is not None:
                # Operate on copies for take/clear calculations
                kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.KELP, kelp_od_copy, kelp_fair_value,
                    self.params[Product.KELP]['take_width'], kelp_position,
                    self.params[Product.KELP]['prevent_adverse'], self.params[Product.KELP]['adverse_volume'], traderObject
                )
                kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.KELP, kelp_od_copy, kelp_fair_value,
                    self.params[Product.KELP]['clear_width'], kelp_position, buy_order_volume, sell_order_volume
                )
                # Use original order depth for making orders
                kelp_make_orders, _, _ = self.make_orders(
                    Product.KELP, state.order_depths[Product.KELP], kelp_fair_value, kelp_position,
                    buy_order_volume, sell_order_volume, self.params[Product.KELP]['disregard_edge'],
                    self.params[Product.KELP]['join_edge'], self.params[Product.KELP]['default_edge']
                )
                result[Product.KELP].extend(kelp_take_orders + kelp_clear_orders + kelp_make_orders) # Use extend

        # SQUID_INK
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            ink_position = state.position.get(Product.SQUID_INK, 0)
            # Make a copy
            ink_od_copy = OrderDepth()
            ink_od_copy.buy_orders = state.order_depths[Product.SQUID_INK].buy_orders.copy()
            ink_od_copy.sell_orders = state.order_depths[Product.SQUID_INK].sell_orders.copy()

            ink_fair_value = self.ink_fair_value(ink_od_copy, traderObject)

            if ink_fair_value is not None:
                 # Pass copy to take_orders to avoid modifying state during spike logic
                ink_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.SQUID_INK, ink_od_copy, ink_fair_value,
                    self.params[Product.SQUID_INK]['take_width'], ink_position,
                    self.params[Product.SQUID_INK]['prevent_adverse'], self.params[Product.SQUID_INK]['adverse_volume'], traderObject
                )
                # Pass potentially modified copy (if take_orders modified it) to clear_orders
                ink_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.SQUID_INK, ink_od_copy, ink_fair_value,
                    self.params[Product.SQUID_INK]['clear_width'], ink_position, buy_order_volume, sell_order_volume
                )
                 # Use original order depth for making orders
                ink_make_orders, _, _ = self.make_orders(
                    Product.SQUID_INK, state.order_depths[Product.SQUID_INK], ink_fair_value, ink_position,
                    buy_order_volume, sell_order_volume, self.params[Product.SQUID_INK]['disregard_edge'],
                    self.params[Product.SQUID_INK]['join_edge'], self.params[Product.SQUID_INK]['default_edge']
                )
                result[Product.SQUID_INK].extend(ink_take_orders + ink_clear_orders + ink_make_orders) # Use extend

        # CROISSANTS - Calculated first, potentially modified by spreads later
        if Product.CROISSANTS in state.order_depths:
            if Product.JAMS in state.order_depths and Product.DJEMBES in state.order_depths: # Check dependencies
                croissant_orders = self.croissants_strategy(state)
                result[Product.CROISSANTS].extend(croissant_orders) # Use extend


        # SPREAD1 (PICNIC_BASKET1) - Check dependencies
        if all(p in state.order_depths for p in [Product.PICNIC_BASKET1, Product.DJEMBES, Product.CROISSANTS, Product.JAMS]):
            picnic1_position = state.position.get(Product.PICNIC_BASKET1, 0)
            spread1_orders_dict = self.spread_orders(
                state.order_depths, Product.PICNIC_BASKET1, picnic1_position,
                traderObject[Product.SPREAD1], SPREAD=Product.SPREAD1, picnic1=True
            )
            if spread1_orders_dict:
                for product, orders in spread1_orders_dict.items():
                    result[product].extend(orders) # Use extend


        # SPREAD2 (PICNIC_BASKET2) - Check dependencies
        if all(p in state.order_depths for p in [Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS]):
            picnic2_position = state.position.get(Product.PICNIC_BASKET2, 0)
            spread2_orders_dict = self.spread_orders(
                state.order_depths, Product.PICNIC_BASKET2, picnic2_position,
                traderObject[Product.SPREAD2], SPREAD=Product.SPREAD2, picnic1=False
            )
            if spread2_orders_dict:
                for product, orders in spread2_orders_dict.items():
                     result[product].extend(orders) # Use extend


        # VOLCANIC_ROCK and VOUCHERS
        if Product.VOLCANIC_ROCK in state.order_depths:
            TTE = self.calculate_tte(state.timestamp)
            S = self.get_weighted_mid_price(Product.VOLCANIC_ROCK, state)
            calculated_ivs = {}
            if S is not None and TTE > 0: # Ensure S and TTE are valid
                for symbol in Product.VOUCHER_SYMBOLS:
                    if symbol not in state.order_depths:
                        continue # Skip if no order depth for this voucher
                    K = self.strikes[symbol]
                    limit = self.position_limits[symbol]
                    current_pos = state.position.get(symbol, 0)
                    voucher_price = self.get_weighted_mid_price(symbol, state)
                    if voucher_price is None:
                        continue # Skip if no valid price
                    m_t = self.calculate_m_t(S, K, TTE)
                    if m_t is None:
                        continue # Skip if cannot calculate moneyness

                    last_iv = traderObject.get("last_ivs", {}).get(symbol, self.base_iv)
                    iv_market = self.implied_volatility(voucher_price, S, K, TTE, self.risk_free_rate, initial_guess=last_iv)

                    if iv_market <= 0: # Check if IV calculation was successful
                         # If IV calc fails, maybe try BS price with smile IV? Or just skip.
                         # For now, skip if market IV is invalid.
                         continue

                    calculated_ivs[symbol] = iv_market
                    iv_smile = self.get_smile_volatility(m_t)
                    mispricing = iv_market - iv_smile
                    threshold = 0.001 # Consider making this dynamic or part of params
                    order_depth = state.order_depths.get(symbol)
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

                    # Ensure result[symbol] exists
                    if symbol not in result: result[symbol] = []

                    if mispricing > threshold: # Sell overpriced voucher
                        volume_to_sell = self.order_quantity
                        available_sell_limit = limit + current_pos # Correct limit for selling (negative position)
                        actual_sell_volume = min(volume_to_sell, available_sell_limit)
                        if actual_sell_volume > 0:
                            sell_price = best_bid if best_bid is not None else int(math.floor(voucher_price))
                            if sell_price > 0:
                                result[symbol].append(Order(symbol, sell_price, -actual_sell_volume))
                    elif mispricing < -threshold: # Buy underpriced voucher
                        volume_to_buy = self.order_quantity
                        available_buy_limit = limit - current_pos # Correct limit for buying (positive position)
                        actual_buy_volume = min(volume_to_buy, available_buy_limit)
                        if actual_buy_volume > 0:
                            buy_price = best_ask if best_ask is not None else int(math.ceil(voucher_price))
                            if buy_price > 0:
                                result[symbol].append(Order(symbol, buy_price, actual_buy_volume))
            traderObject["last_ivs"] = calculated_ivs
            atm_symbol = f"{Product.VOUCHER_PREFIX}10000"
            if atm_symbol in calculated_ivs:
                current_base_iv = calculated_ivs[atm_symbol]
                traderObject["base_iv_history"].append(current_base_iv)
                if len(traderObject["base_iv_history"]) > self.iv_history_length:
                    traderObject["base_iv_history"] = traderObject["base_iv_history"][-self.iv_history_length:]

        # Clean up result dictionary - remove keys with empty lists
        final_result = {k: v for k, v in result.items() if v}

        traderData = jsonpickle.encode(traderObject, unpicklable=False)
        #logger.flush(state, final_result, conversions, traderData) # Log final_result
        return final_result, conversions, traderData