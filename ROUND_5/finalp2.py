from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List, Dict, Any
import string
import json
import jsonpickle
from math import log, sqrt, exp, erf, pi
import math
import numpy as np

DAYS_LEFT = 3

class MarketData:
    end_pos: Dict[str, int] = {}
    buy_sum: Dict[str, int] = {}
    sell_sum: Dict[str, int] = {}
    bid_prices: Dict[str, List[float]] = {}
    bid_volumes: Dict[str, List[int]] = {}
    ask_prices: Dict[str, List[float]] = {}
    ask_volumes: Dict[str, List[int]] = {}
    fair: Dict[str, float] = {}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 1,
        "soft_position_limit": 50,
    },
    Product.KELP: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.18,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.147417,
        "strike": 10000,
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.140554,
        "strike": 10000,
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 21,
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.PRODUCT_LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.MAGNIFICENT_MACARONS: 75,
        }

    def take_best_orders(self, product: str, fair_value: float, take_width: float, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int, prevent_adverse: bool = False, adverse_volume: int = 0, traderObject: dict = None):
        position_limit = self.PRODUCT_LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
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
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, position: int, buy_order_volume: int, sell_order_volume: int):
        buy_quantity = self.PRODUCT_LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, math.floor(bid), buy_quantity))
        sell_quantity = self.PRODUCT_LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, math.ceil(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.PRODUCT_LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.PRODUCT_LIMIT[product] + (position - sell_order_volume)
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
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = 0, 0
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, prevent_adverse, adverse_volume, traderObject)
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: int, position: int, buy_order_volume: int, sell_order_volume: int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume)
        return orders, buy_order_volume, sell_order_volume

    def make_orders(self, product, order_depth: OrderDepth, fair_value: float, position: int, buy_order_volume: int, sell_order_volume: int, disregard_edge: float, join_edge: float, default_edge: float, manage_position: bool = False, soft_position_limit: int = 0):
        orders: List[Order] = []
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
            elif position < -1 * soft_position_limit:
                bid += 1
        buy_order_volume, sell_order_volume = self.market_make(product, orders, bid, ask, position, buy_order_volume, sell_order_volume)
        return orders, buy_order_volume, sell_order_volume

    def trade_resin(self, state, market_data):
        product = "RAINFOREST_RESIN"
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
        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            price = self.black_scholes_call(S, K, T_days, r, sigma_mid)
            if abs(price - market_price) < tol:
                return sigma_mid
            if price > market_price:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid
        return (sigma_low + sigma_high) / 2

    def call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        r = 0
        T = T / 365
        if T == 0 or sigma == 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))

    def trade_10000(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_10000"
        orders = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
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
                amount = market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10000"]
                amount = min(amount, sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10000"]))
                option_amount = amount
                for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_10000"])):
                    fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10000"][i], option_amount)
                    if fill != 0:
                        orders.append(Order("VOLCANIC_ROCK_VOUCHER_10000", market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_10000"][i], -fill))
                        market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10000"] -= fill
                        market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10000"] -= fill
                        option_amount -= fill
            elif diff < -threshold:
                amount = market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10000"]
                amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10000"]))
                option_amount = amount
                for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_10000"])):
                    fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10000"][i], option_amount)
                    if fill != 0:
                        orders.append(Order("VOLCANIC_ROCK_VOUCHER_10000", market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_10000"][i], fill))
                        market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10000"] -= fill
                        market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10000"] += fill
                        option_amount -= fill
        return orders

    def trade_9750(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_9750"
        orders = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
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
            amount = market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_9750"]
            amount = min(amount, sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_9750"]))
            option_amount = amount
            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_9750"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_9750"][i], option_amount)
                if fill != 0:
                    orders.append(Order("VOLCANIC_ROCK_VOUCHER_9750", market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_9750"][i], -fill))
                    market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_9750"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_9750"] -= fill
                    option_amount -= fill
        elif diff < -threshold:
            amount = market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_9750"]
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_9750"]))
            option_amount = amount
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_9750"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_9750"][i], option_amount)
                if fill != 0:
                    orders.append(Order("VOLCANIC_ROCK_VOUCHER_9750", market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_9750"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_9750"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_9750"] += fill
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
        product = "MAGNIFICENT_MACARONS"
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
        product = "MAGNIFICENT_MACARONS"
        orders: List[Order] = []
        order_depth = state.order_depths[product]
        fair_mid = market_data.fair[product]
        pos = market_data.end_pos[product]
        bid_px = math.floor(fair_mid - 4)
        ask_px = math.ceil(fair_mid + 4)
        size = 14
        buy_cap = self.PRODUCT_LIMIT[product] - pos
        sell_cap = self.PRODUCT_LIMIT[product] + pos
        if buy_cap > 0:
            qty = min(size, buy_cap)
            orders.append(Order(product, bid_px, qty))
        if sell_cap > 0:
            qty = min(size, sell_cap)
            orders.append(Order(product, ask_px, -qty))
        return orders

    def clear_macaron(self, state, market_data):
        product = "MAGNIFICENT_MACARONS"
        orders: List[Order] = []
        fair = market_data.fair[product]
        pos = market_data.end_pos[product]
        width = 3
        if pos > 0:
            orders.append(Order(product, round(fair + width), -pos))
        elif pos < 0:
            orders.append(Order(product, round(fair - width), -pos))
        return orders

    def run(self, state: TradingState):
        traderObject = {}
        result = {}
        market_data = MarketData()
        products = ["RAINFOREST_RESIN", "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK", "MAGNIFICENT_MACARONS", "KELP"]
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        for product in products:
            if product not in state.order_depths:
                continue
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            bids, asks = order_depth.buy_orders, order_depth.sell_orders
            if order_depth.buy_orders:
                mm_bid = max(bids.items(), key=lambda tup: tup[0])[0]
            if order_depth.sell_orders:
                mm_ask = min(asks.items(), key=lambda tup: tup[0])[0]
            if order_depth.sell_orders and order_depth.buy_orders:
                fair_price = (mm_ask + mm_bid) / 2
            elif order_depth.sell_orders:
                fair_price = mm_ask
            elif order_depth.buy_orders:
                fair_price = mm_bid
            else:
                fair_price = traderObject.get(f"prev_fair_{product}", 0)
            traderObject[f"prev_fair_{product}"] = fair_price
            market_data.end_pos[product] = position
            market_data.buy_sum[product] = self.PRODUCT_LIMIT[product] - position
            market_data.sell_sum[product] = self.PRODUCT_LIMIT[product] + position
            market_data.bid_prices[product] = list(bids.keys())
            market_data.bid_volumes[product] = list(bids.values())
            market_data.ask_prices[product] = list(asks.keys())
            market_data.ask_volumes[product] = list(asks.values())
            market_data.fair[product] = fair_price
        result["RAINFOREST_RESIN"] = self.trade_resin(state, market_data)
        result[Product.VOLCANIC_ROCK_VOUCHER_9750] = self.trade_9750(state, market_data, traderObject)
        result[Product.VOLCANIC_ROCK_VOUCHER_10000] = self.trade_10000(state, market_data, traderObject)
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject)
            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(Product.KELP, state.order_depths[Product.KELP], kelp_fair_value, self.params[Product.KELP]['take_width'], kelp_position, self.params[Product.KELP]['prevent_adverse'], self.params[Product.KELP]['adverse_volume'], traderObject)
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(Product.KELP, state.order_depths[Product.KELP], kelp_fair_value, self.params[Product.KELP]['clear_width'], kelp_position, buy_order_volume, sell_order_volume)
            kelp_make_orders, _, _ = self.make_orders(Product.KELP, state.order_depths[Product.KELP], kelp_fair_value, kelp_position, buy_order_volume, sell_order_volume, self.params[Product.KELP]['disregard_edge'], self.params[Product.KELP]['join_edge'], self.params[Product.KELP]['default_edge'])
            result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders
        if "prev_mac_prices" not in traderObject:
            traderObject["prev_mac_prices"] = [market_data.fair["MAGNIFICENT_MACARONS"]]
        else:
            traderObject["prev_mac_prices"].append(market_data.fair["MAGNIFICENT_MACARONS"])
        self.recent_std = np.std(traderObject["prev_mac_prices"])
        if len(traderObject["prev_mac_prices"]) > 13:
            traderObject["prev_mac_prices"].pop(0)
        mac_take = mac_make = mac_clear = []
        conversions = 0
        mac_take, conversions = self.take_macaron(state, market_data, traderObject)
        if self.recent_std < 8:
            mac_make = self.make_macaron(state, market_data)
            mac_clear = self.clear_macaron(state, market_data)
        result["MAGNIFICENT_MACARONS"] = mac_take + mac_make + mac_clear
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData