from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import jsonpickle

class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"

class Trader:
    def __init__(self):
        self.price_history = []
        self.position = 0
        self.account_balance = 100000  # Initial account balance

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None

    def run(self, state: TradingState) -> (Dict[str, List[Order]], int, str):
        traderObject = {}
        if state.traderData and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        result = {}
        conversions = 0

        if Product.VOLCANIC_ROCK in state.order_depths:
            order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            mid_price = self.get_mid_price(order_depth)
            if mid_price is not None:
                self.price_history.append(mid_price)
                if len(self.price_history) > 1:
                    # Compare current price with previous price
                    prev_price = self.price_history[-2]
                    orders = []

                    if mid_price < prev_price:
                        # Price is falling, buy aggressively against the trend
                        trade_size = 50  # Fixed large size, no risk management
                        orders.append(Order(Product.VOLCANIC_ROCK, int(mid_price), trade_size))
                        self.position += trade_size
                    elif mid_price > prev_price:
                        # Price is rising, sell everything against the trend
                        if self.position > 0:
                            orders.append(Order(Product.VOLCANIC_ROCK, int(mid_price), -self.position))
                            self.position = 0

                    result[Product.VOLCANIC_ROCK] = orders

        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData
