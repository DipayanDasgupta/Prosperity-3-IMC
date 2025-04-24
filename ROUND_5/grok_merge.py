from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List, Dict, Any
import string
import json
import jsonpickle
from math import log, sqrt, exp, erf, pi
import math
import numpy as np

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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

PARAMS = {
    Product.MAGNIFICENT_MACARONS: {}
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS

        self.params = params
        self.PRODUCT_LIMIT = {
            Product.MAGNIFICENT_MACARONS: 75
        }

    def calculate_sunlight_rate_of_change(self, traderObject):
        """Calculate the average rate of change of sunlight over the last 5 ticks
        :param traderObject:
        """
        if len(traderObject["sunlight_history"]) < 5:
            return 0
        changes = []
        for i in range(1, len(traderObject["sunlight_history"])):
            changes.append(traderObject["sunlight_history"][i] - traderObject["sunlight_history"][i - 1])
        return sum(changes) / len(changes)

    def take_macaron(self, state, market_data, traderObject):
        product = "MAGNIFICENT_MACARONS"
        orders = {}
        for p in ["MAGNIFICENT_MACARONS"]:
            orders[p] = []
        fair = market_data.fair[product]
        conversions = 0
        # print(state.observations.conversionObservations[product])
        x = state.observations.conversionObservations
        overseas_ask = state.observations.conversionObservations[product].askPrice + \
                       state.observations.conversionObservations[product].transportFees + \
                       state.observations.conversionObservations[product].importTariff
        overseas_bid = state.observations.conversionObservations[product].bidPrice - \
                       state.observations.conversionObservations[product].transportFees - \
                       state.observations.conversionObservations[product].exportTariff
        if 'last_sunlight' in traderObject:
            if state.observations.conversionObservations[product].sunlightIndex < traderObject["last_sunlight"]:
                direction = -1
            elif state.observations.conversionObservations[product].sunlightIndex == traderObject["last_sunlight"]:
                direction = 0
            else:
                direction = 1
        else:
            direction = 0

        # Update sunlight history
        if "sunlight_history" in traderObject:
            traderObject["sunlight_history"].append(state.observations.conversionObservations[product].sunlightIndex)
        else:
            traderObject["sunlight_history"] = [state.observations.conversionObservations[product].sunlightIndex]
        if len(traderObject["sunlight_history"]) > 5:
            traderObject["sunlight_history"].pop(0)

        traderObject['last_sunlight'] = state.observations.conversionObservations[product].sunlightIndex

        # New trading strategy based on bid/ask volumes and sunlight
        total_bids = sum(market_data.bid_volumes[product])
        total_asks = -sum(market_data.ask_volumes[product])

        current_sunlight = state.observations.conversionObservations[product].sunlightIndex

        # Calculate z-score for position management
        mean_price = 640
        std_dev = 55  # Based on range 550-750
        current_price = fair  # Using the fair price as current price
        z_score = (current_price - mean_price) / std_dev

        # Strategy for sunlight below 50
        if current_sunlight < 50:
            # Buy if sunlight dropped below 50 and is less than previous day
            if direction == -1 and market_data.buy_sum[product] > 0:
                amount = min(market_data.buy_sum[product], -sum(market_data.ask_volumes[product]))
                for i in range(0, len(market_data.ask_prices[product])):
                    fill = min(-market_data.ask_volumes[product][i], amount)
                    if fill != 0:
                        orders[product].append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill
                        amount -= fill
            # Go short if sunlight is increasing rapidly from below 50
            elif direction == 1 and market_data.sell_sum[
                product] > 0 and self.calculate_sunlight_rate_of_change(traderObject) > 0.008:
                amount = min(market_data.sell_sum[product], sum(market_data.bid_volumes[product]))
                for i in range(0, len(market_data.bid_prices[product])):
                    fill = min(market_data.bid_volumes[product][i], amount)
                    if fill != 0:
                        orders[product].append(Order(product, market_data.bid_prices[product][i], -fill))
                        market_data.sell_sum[product] -= fill
                        market_data.end_pos[product] -= fill
                        amount -= fill
            # Close short position if sunlight reaches 49
            elif abs(current_sunlight - 49) < 1 and market_data.end_pos[product] < 0:
                amount = min(market_data.buy_sum[product], -market_data.end_pos[product])
                for i in range(0, len(market_data.ask_prices[product])):
                    fill = min(-market_data.ask_volumes[product][i], amount)
                    if fill != 0:
                        orders[product].append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill
                        amount -= fill

        elif current_sunlight > 50:
            # Mean reversion strategy with z-score
            if z_score < -1.2 and market_data.buy_sum[product] > 0:  # Price is significantly below mean
                # Buy when price is significantly below mean
                amount = min(market_data.buy_sum[product], -sum(market_data.ask_volumes[product]))
                for i in range(0, len(market_data.ask_prices[product])):
                    fill = min(-market_data.ask_volumes[product][i], amount)
                    if fill != 0:
                        orders[product].append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill
                        amount -= fill
            elif z_score > 1.2 and market_data.sell_sum[product] > 0:  # Price is significantly above mean
                # Sell when price is significantly above mean
                amount = min(market_data.sell_sum[product], sum(market_data.bid_volumes[product]))
                for i in range(0, len(market_data.bid_prices[product])):
                    fill = min(market_data.bid_volumes[product][i], amount)
                    if fill != 0:
                        orders[product].append(Order(product, market_data.bid_prices[product][i], -fill))
                        market_data.sell_sum[product] -= fill
                        market_data.end_pos[product] -= fill
                        amount -= fill

        return orders["MAGNIFICENT_MACARONS"], conversions

    def make_macaron(self, state, market_data):
        product = "MAGNIFICENT_MACARONS"
        orders: List[Order] = []

        order_depth = state.order_depths[product]
        fair_mid = market_data.fair[product]
        pos = market_data.end_pos[product]
        # Ik this market looks really wide, but it made the most money
        bid_px = math.floor(fair_mid - 4)
        ask_px = math.ceil(fair_mid + 4)
        size = 14  # hyperparam - slice of the market book

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
        width = 3 if self.recent_std > 7 else 4   # one‐tick clearance

        if pos > 0:
            orders.append(Order(product, round(fair + width), -pos))
        elif pos < 0:
            orders.append(Order(product, round(fair - width), -pos))
        return orders

    def run(self, state: TradingState):
        traderObject = {}
        result = {}
        market_data = MarketData()
        products = ["MAGNIFICENT_MACARONS"]
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        for product in products:
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            bids, asks = order_depth.buy_orders, order_depth.sell_orders
            if order_depth.buy_orders:
                mm_bid = max(bids.items(), key=lambda tup: tup[1])[0]
            if order_depth.sell_orders:
                mm_ask = min(asks.items(), key=lambda tup: tup[1])[0]
            if order_depth.sell_orders and order_depth.buy_orders:
                fair_price = (mm_ask + mm_bid) / 2
            elif order_depth.sell_orders:
                fair_price = mm_ask
            elif order_depth.buy_orders:
                fair_price = mm_bid
            else:
                fair_price = traderObject[f"prev_fair_{product}"]
            traderObject[f"prev_fair_{product}"] = fair_price

            market_data.end_pos[product] = position
            market_data.buy_sum[product] = self.PRODUCT_LIMIT[product] - position
            market_data.sell_sum[product] = self.PRODUCT_LIMIT[product] + position
            market_data.bid_prices[product] = list(bids.keys())
            market_data.bid_volumes[product] = list(bids.values())
            market_data.ask_prices[product] = list(asks.keys())
            market_data.ask_volumes[product] = list(asks.values())
            market_data.fair[product] = fair_price

        # Round 4
        if "prev_mac_prices" not in traderObject:
            traderObject["prev_mac_prices"] = [market_data.fair["MAGNIFICENT_MACARONS"]]
        else:
            traderObject["prev_mac_prices"].append(market_data.fair["MAGNIFICENT_MACARONS"])
        self.recent_std = np.std(traderObject["prev_mac_prices"])
        if len(traderObject["prev_mac_prices"]) > 13:
            traderObject["prev_mac_prices"].pop(-1)
        high_thresh = 53
        mac_take = mac_make = mac_clear = []
        conversions = 0
        mac_take, conversions = self.take_macaron(state, market_data, traderObject)
        if self.recent_std < high_thresh:
            mac_make = self.make_macaron(state, market_data)
            mac_clear = self.clear_macaron(state, market_data)
        result["MAGNIFICENT_MACARONS"] = mac_take + mac_make + mac_clear

        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData