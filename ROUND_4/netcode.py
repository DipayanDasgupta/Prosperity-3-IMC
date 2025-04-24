from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

import numpy as np
from collections import deque
from statistics import linear_regression

class Trader:
    def __init__(self):
        self.history = dict()
        self.sp_match = dict()
        self.best_ask = dict()
        self.best_bid = dict()
        self.avg_buy_price = dict()
        self.last_position = dict()
        self.last_buy_price = dict()
        
        self.std_len = 100
        self.regression_window = 75

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = dict()

        if len(self.sp_match) == 0:
            for symbol in state.listings.keys():
                self.sp_match[state.listings[symbol].product] = symbol
                self.history[symbol] = []
                self.last_position[symbol] = 0
                self.avg_buy_price[symbol] = 0
                self.last_buy_price[symbol] = 0 

        for symbol in state.order_depths.keys():
            order_depth: OrderDepth = state.order_depths[symbol]
            orders: list[Order] = []

            self.best_ask[symbol] = min(order_depth.sell_orders.keys())
            self.best_bid[symbol] = max(order_depth.buy_orders.keys())
            mid_price = (self.best_ask[symbol] + self.best_bid[symbol]) / 2
            
            hist_data = self.history[symbol]
            hist_data.append(mid_price)
            hist_data = hist_data[-(self.std_len + 1):]
            self.history[symbol] = hist_data

        for symbol in state.order_depths.keys():
            if symbol in state.position.keys():
                position = state.position[self.sp_match[symbol]]
            else:
                position = 0

            if symbol == 'RAINFOREST_RESIN':
                orders = self.rainforest(symbol, position)
            elif symbol == 'MAGNIFICENT_MACARONS':
                orders = self.macarons(symbol, position, state)
            elif symbol == 'VOLCANIC_ROCK': # 'VOLCANIC_ROCK_VOUCHER_XXXX'
                orders = self.vrock(symbol, position)

            result[symbol] = orders
                
        traderData = "SAMPLE"
        conversions = 1
        return result, conversions, traderData
    
    def rainforest(self, symbol, position):
        # return [Order(symbol, 9999, max(0, 50 - position)),
        #         Order(symbol, 10001, min(0, -50 - position))]
    
        return []

    def macarons(self, symbol, position, state):
        orders = []
        alpha = 1.05
        obs = state.observations.conversionObservations[self.sp_match[symbol]]

        if self.last_position[symbol] < position:
            self.avg_buy_price[symbol] = (self.avg_buy_price[symbol] * self.last_position[symbol] +\
                                   self.last_buy_price[symbol] * (position - self.last_position[symbol])) / position
            self.last_position[symbol] = position
            self.last_buy_price[symbol] = self.best_ask[symbol] + obs.transportFees - obs.importTariff
        else:
            self.last_position[symbol] = position
            self.last_buy_price[symbol] = self.best_ask[symbol] + obs.transportFees - obs.importTariff

        if self.last_position[symbol] == 0:
            self.avg_buy_price[symbol] = 1e9

        if obs.sunlightIndex < 50:
            beta = np.array([0, -94.56880778, -87.24115437, 6.51608416, 0])
        else:
            beta = np.array([47.58327099, -42.1295798, -38.42240284, 4.90408068, -2.21331757])
        
        x = np.array([obs.transportFees, obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex])
        predicted_price = x @ beta - 2 * obs.transportFees + obs.importTariff - obs.exportTariff

        if predicted_price > self.best_ask[symbol] * alpha\
              and self.last_buy_price[symbol] < self.avg_buy_price[symbol]\
              and self.last_position[symbol] < 70:
            orders.append(Order(symbol, self.best_ask[symbol], max(0, min(70 - self.last_position[symbol], 10))))

        if self.avg_buy_price[symbol] * alpha + obs.transportFees + obs.exportTariff < self.best_bid[symbol]:
            orders.append(Order(symbol, self.best_bid[symbol], -position))

        return orders

        # return []
    
    def vrock(self, symbol, position):
        return []