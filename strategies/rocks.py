from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import List, Any, Dict, Tuple
import json
import jsonpickle
import numpy as np
import math
import collections
import pandas as pd
import copy
from collections import deque
from math import sqrt, log, erf

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    MACARONS = "MAGNIFICENT_MACARONS"
    options = [
        VOLCANIC_ROCK_VOUCHER_10000,
        VOLCANIC_ROCK_VOUCHER_9500,
        VOLCANIC_ROCK_VOUCHER_9750,
        VOLCANIC_ROCK_VOUCHER_10250,
        VOLCANIC_ROCK_VOUCHER_10500,
    ]

PARAMS = {
    Product.VOLCANIC_ROCK: {
        "ma_length": 100,
        "open_threshold": -1.6,
        "close_threshold": 1.6,
        "short_ma_length": 10,
    },

}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
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
                    self.truncate("", max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            "",
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_observations(self, observations: Observation) -> list[Any]:
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

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()


class Trader:
    
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

    def run(self, state: TradingState):
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}
        result = {}

        conversions = 0



        product = Product.VOLCANIC_ROCK 
        self.update_signal(state, traderObject, product)
        result[product] = self.spam_orders(state, product, product)

        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData