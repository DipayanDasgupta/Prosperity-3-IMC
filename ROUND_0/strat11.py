from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import math

# Define product constants
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"

# Define parameters inspired by the runner-up strategy
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "base_fair_value": 10000,
        "trade_adjustment_weight": 0.4,
        "take_width": 0.1,
        "disregard_edge": 1,  # For pennying/joining
        "join_edge": 2,
        "default_edge": 0.8,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 0.1,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,  # From runner-up strategy
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1.0,
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.kelp_prices = []  # For KELP price history
        self.kelp_vwap = []    # For KELP VWAP history
        self.kelp_price_history = []  # For mean-reversion calculation
        self.resin_trade_prices = []  # To track recent trade prices for RAINFOREST_RESIN
        self.resin_price_history = []  # For longer-term fair value average
        self.kelp_fair_value_history = []  # For longer-term fair value average
        self.position_limits = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50}

    def resin_fair_value(self, state: TradingState) -> int:
        # Base fair value
        base_fair_value = self.params[Product.RAINFOREST_RESIN]["base_fair_value"]
        
        # Incorporate order book mid-price
        order_depth = state.order_depths.get(Product.RAINFOREST_RESIN, OrderDepth())
        mid_price = base_fair_value
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

        # Adjust based on recent trades
        trade_adjustment = 0
        if Product.RAINFOREST_RESIN in state.market_trades and state.market_trades[Product.RAINFOREST_RESIN]:
            recent_trade = state.market_trades[Product.RAINFOREST_RESIN][-1]
            self.resin_trade_prices.append(recent_trade.price)
            if len(self.resin_trade_prices) > 15:
                self.resin_trade_prices.pop(0)
            avg_trade_price = sum(self.resin_trade_prices) / len(self.resin_trade_prices)
            trade_adjustment = (avg_trade_price - mid_price) * self.params[Product.RAINFOREST_RESIN]["trade_adjustment_weight"]

        # Combine mid-price, base, and trade adjustment
        fair_value = base_fair_value * 0.3 + mid_price * 0.4 + trade_adjustment
        self.resin_price_history.append(fair_value)
        if len(self.resin_price_history) > 50:
            self.resin_price_history.pop(0)

        return round(fair_value)

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject: dict) -> int:
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return 5000

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [
            price for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
        ]
        filtered_bid = [
            price for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
        ]
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid
        if mm_ask is None or mm_bid is None:
            if traderObject.get("kelp_last_price", None) is None:
                mid_price = (best_ask + best_bid) / 2
            else:
                mid_price = traderObject["kelp_last_price"]
        else:
            mid_price = (mm_ask + mm_bid) / 2

        # Mean-reversion model
        if traderObject.get("kelp_last_price", None) is not None:
            last_price = traderObject["kelp_last_price"]
            last_returns = (mid_price - last_price) / last_price
            pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
            fair_value = mid_price + (mid_price * pred_returns)
        else:
            fair_value = mid_price

        traderObject["kelp_last_price"] = mid_price
        self.kelp_fair_value_history.append(fair_value)
        if len(self.kelp_fair_value_history) > 50:
            self.kelp_fair_value_history.pop(0)

        return round(fair_value)

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):  # Updated return type to include orders
        position_limit = self.position_limits[product]

        # Adjust take width based on trend
        trend_factor = 1.0
        if product == Product.RAINFOREST_RESIN and len(self.resin_price_history) >= 50:
            long_term_avg = sum(self.resin_price_history[-50:]) / 50
            if fair_value > long_term_avg:
                trend_factor = 0.9  # More aggressive buying
            elif fair_value < long_term_avg:
                trend_factor = 0.9  # More aggressive selling
        elif product == Product.KELP and len(self.kelp_fair_value_history) >= 50:
            long_term_avg = sum(self.kelp_fair_value_history[-50:]) / 50
            if fair_value > long_term_avg:
                trend_factor = 0.9
            elif fair_value < long_term_avg:
                trend_factor = 0.9

        adjusted_take_width = take_width * trend_factor

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - adjusted_take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    quantity = min(quantity * (1.2 if trend_factor < 1 else 1.0), position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, int(quantity)))
                        buy_order_volume += int(quantity)
                        order_depth.sell_orders[best_ask] += int(quantity)
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + adjusted_take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    quantity = min(quantity * (1.2 if trend_factor < 1 else 1.0), position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -int(quantity)))
                        sell_order_volume += int(quantity)
                        order_depth.buy_orders[best_bid] -= int(quantity)
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return orders, buy_order_volume, sell_order_volume  # Now returning orders as well

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value)
        fair_for_ask = round(fair_value)

        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)

        threshold = 30
        stop_loss_threshold = 30
        profit_take_threshold = 20

        # Clear positions if over threshold
        if position_after_take > threshold:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < -threshold:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        # Stop-loss
        if position_after_take > 0 and fair_for_ask < fair_value - stop_loss_threshold:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid > fair_value + stop_loss_threshold:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        # Profit-taking
        if position_after_take > 0 and fair_for_ask > fair_value + profit_take_threshold:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid < fair_value - profit_take_threshold:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ) -> (List[Order], int, int):  # Updated return type
        orders: List[Order] = []
        asks_above_fair = [
            price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # Join
            else:
                ask = best_ask_above_fair - 1  # Penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        # Adjust prices based on position
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        # Place market-making orders
        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, int(buy_quantity)))

        sell_quantity = self.position_limits[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -int(sell_quantity)))

        return orders, buy_order_volume, sell_order_volume

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Take orders
        take_orders, buy_order_volume, sell_order_volume = self.take_best_orders(
            Product.RAINFOREST_RESIN,
            fair_value,
            self.params[Product.RAINFOREST_RESIN]["take_width"],
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        # Clear positions and take profits
        buy_order_volume, sell_order_volume = self.clear_position_order(
            Product.RAINFOREST_RESIN,
            fair_value,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        # Market making with pennying/joining
        make_orders, buy_order_volume, sell_order_volume = self.make_orders(
            Product.RAINFOREST_RESIN,
            order_depth,
            fair_value,
            position,
            buy_order_volume,
            sell_order_volume,
            self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
            self.params[Product.RAINFOREST_RESIN]["join_edge"],
            self.params[Product.RAINFOREST_RESIN]["default_edge"],
            manage_position=True,
            soft_position_limit=self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
        )

        return take_orders + orders + make_orders

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, position: int, position_limit: int, state: TradingState, traderObject: dict) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        fair_value = self.kelp_fair_value(order_depth, traderObject)

        # Take orders
        take_orders, buy_order_volume, sell_order_volume = self.take_best_orders(
            Product.KELP,
            fair_value,
            self.params[Product.KELP]["take_width"],
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse=self.params[Product.KELP]["prevent_adverse"],
            adverse_volume=self.params[Product.KELP]["adverse_volume"],
        )

        # Clear positions and take profits
        buy_order_volume, sell_order_volume = self.clear_position_order(
            Product.KELP,
            fair_value,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        # Market making with pennying/joining
        make_orders, buy_order_volume, sell_order_volume = self.make_orders(
            Product.KELP,
            order_depth,
            fair_value,
            position,
            buy_order_volume,
            sell_order_volume,
            self.params[Product.KELP]["disregard_edge"],
            self.params[Product.KELP]["join_edge"],
            self.params[Product.KELP]["default_edge"],
            manage_position=False,  # KELP doesn't use position-based price adjustment
            soft_position_limit=0,
        )

        return take_orders + orders + make_orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_fair_value = self.resin_fair_value(state)
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_orders = self.resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                resin_fair_value,
                resin_position,
                self.position_limits[Product.RAINFOREST_RESIN]
            )
            result[Product.RAINFOREST_RESIN] = resin_orders

        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_orders = self.kelp_orders(
                state.order_depths[Product.KELP],
                10,  # timespan
                kelp_position,
                self.position_limits[Product.KELP],
                state,
                traderObject
            )
            result[Product.KELP] = kelp_orders

        traderData = jsonpickle.encode({
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "kelp_price_history": self.kelp_price_history,
            "resin_trade_prices": self.resin_trade_prices,
            "resin_price_history": self.resin_price_history,
            "kelp_fair_value_history": self.kelp_fair_value_history,
            "kelp_last_price": traderObject.get("kelp_last_price", None)
        })
        conversions = 0  # No conversions in tutorial round

        return result, conversions, traderData