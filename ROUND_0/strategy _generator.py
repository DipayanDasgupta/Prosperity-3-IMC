# Import necessary libraries
import random

# Define the base strategy as a string template
base_strategy = """
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import math

class Trader:
    def __init__(self):
        self.kelp_prices = []  # For KELP price history
        self.kelp_vwap = []    # For KELP VWAP history
        self.kelp_price_history = []  # For moving average
        self.resin_trade_prices = []  # To track recent trade prices for RAINFOREST_RESIN
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}

    def resin_fair_value(self, state: TradingState) -> int:
        # Base fair value
        base_fair_value = {base_fair_value}
        
        # Adjust based on recent trades
        if "RAINFOREST_RESIN" in state.market_trades and state.market_trades["RAINFOREST_RESIN"]:
            recent_trade = state.market_trades["RAINFOREST_RESIN"][-1]  # Most recent trade
            self.resin_trade_prices.append(recent_trade.price)
            if len(self.resin_trade_prices) > 10:
                self.resin_trade_prices.pop(0)
            avg_trade_price = sum(self.resin_trade_prices) / len(self.resin_trade_prices)
            # Adjust fair value slightly towards the average trade price and round to integer
            return round(base_fair_value * 0.9 + avg_trade_price * 0.1)
        return base_fair_value

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value], default=fair_value + 1)
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value], default=fair_value - 1)

        # Take opportunities more aggressively
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value:  # No take_width threshold
                quantity = min(best_ask_amount, position_limit - position, 30)  # Increased cap
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value:
                quantity = min(best_bid_amount, position_limit + position, 30)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -quantity))
                    sell_order_volume += quantity

        # Clear positions
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, 1
        )

        # Market making with dynamic width
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            buy_price = round(fair_value - width)  # Ensure integer price
            orders.append(Order("RAINFOREST_RESIN", buy_price, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            sell_price = round(fair_value + width)  # Ensure integer price
            orders.append(Order("RAINFOREST_RESIN", sell_price, -sell_quantity))

        return orders

    def kelp_fair_value(self, order_depth: OrderDepth, state: TradingState) -> int:
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return 5000  # Fallback value (already an integer)

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 10]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 10]
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid
        mid_price = (mm_ask + mm_bid) / 2

        # Incorporate recent market trades
        trade_adjustment = 0
        if "KELP" in state.market_trades and state.market_trades["KELP"]:
            recent_trade = state.market_trades["KELP"][-1]
            trade_adjustment = (recent_trade.price - mid_price) * 0.2  # Small adjustment based on trade direction

        # Use a weighted moving average
        self.kelp_price_history.append(mid_price)
        if len(self.kelp_price_history) > {timespan}:  # Shorter window
            self.kelp_price_history.pop(0)
        
        weights = {weights}[:len(self.kelp_price_history)]
        weighted_sum = sum(p * w for p, w in zip(self.kelp_price_history, weights))
        total_weight = sum(weights)
        fair_value = (weighted_sum / total_weight if total_weight > 0 else mid_price) + trade_adjustment

        return round(fair_value)  # Ensure integer fair value

    def get_dynamic_widths(self, order_depth: OrderDepth) -> tuple:
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return {base_make_width}, {base_take_width}  # Default values
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid
        
        # Dynamic adjustment based on spread (proxy for volatility)
        base_make_width = {base_make_width}
        base_take_width = {base_take_width}
        if spread > 5:  # High volatility
            make_width = base_make_width * 0.8  # Tighter spread
            take_width = base_take_width * 0.8
        elif spread < 2:  # Low volatility
            make_width = base_make_width * 1.2  # Wider spread
            take_width = base_take_width * 1.2
        else:
            make_width = base_make_width
            take_width = base_take_width

        return make_width, take_width

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, position: int, position_limit: int, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            fair_value = self.kelp_fair_value(order_depth, state)
            make_width, take_width = self.get_dynamic_widths(order_depth)

            # Take opportunities more aggressively
            if best_ask <= fair_value - take_width:
                ask_amount = -order_depth.sell_orders[best_ask]
                quantity = min(ask_amount, position_limit - position, 30)  # Increased cap
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity

            if best_bid >= fair_value + take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                quantity = min(bid_amount, position_limit + position, 30)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -quantity))
                    sell_order_volume += quantity

            # Clear positions
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 1
            )

            # Market making with dynamic width
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                buy_price = round(fair_value - make_width)  # Ensure integer price
                orders.append(Order("KELP", buy_price, buy_quantity))

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                sell_price = round(fair_value + make_width)  # Ensure integer price
                orders.append(Order("KELP", sell_price, -sell_quantity))

        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int) -> tuple:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # Allow larger positions before clearing
        if position_after_take > {position_clear_threshold}:  # Increased threshold
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

        if position_after_take < -{position_clear_threshold}:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def run(self, state: TradingState) -> tuple:
        result = {}
        resin_fair_value = self.resin_fair_value(state)
        resin_width = {resin_width}
        timespan = {timespan}  # Shorter window for KELP

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position.get("RAINFOREST_RESIN", 0)
            resin_orders = self.resin_orders(
                state.order_depths["RAINFOREST_RESIN"], resin_fair_value, resin_width, resin_position, self.position_limits["RAINFOREST_RESIN"]
            )
            result["RAINFOREST_RESIN"] = resin_orders

        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"], timespan, kelp_position, self.position_limits["KELP"], state
            )
            result["KELP"] = kelp_orders

        traderData = jsonpickle.encode({
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "kelp_price_history": self.kelp_price_history,
            "resin_trade_prices": self.resin_trade_prices
        })
        conversions = 0  # No conversions in tutorial round

        return result, conversions, traderData
"""

# Define parameter ranges for tweaking
parameter_ranges = {
    "base_fair_value": [9900, 9950, 10000, 10050, 10100],  # Fair value for RAINFOREST_RESIN
    "resin_width": [0.5, 1, 1.5, 2],  # Width for RAINFOREST_RESIN market making
    "timespan": [2, 3, 4, 5, 6],  # Window for KELP moving average
    "base_make_width": [1.0, 1.5, 2.0, 2.5],  # Base make width for KELP
    "base_take_width": [0.2, 0.3, 0.4, 0.5],  # Base take width for KELP
    "position_clear_threshold": [15, 20, 25, 30],  # Position clearing threshold
    "weights": [
        "[1, 2, 3]",  # Original weights
        "[1, 3, 5]",  # More emphasis on recent prices
        "[1, 1, 1]",  # Equal weights
        "[2, 2, 2]",  # Equal but higher weights
        "[1, 2, 4]"   # Slightly more emphasis on recent prices
    ]
}

# Function to generate a random strategy by tweaking parameters
def generate_strategy(strategy_num):
    params = {
        "base_fair_value": random.choice(parameter_ranges["base_fair_value"]),
        "resin_width": random.choice(parameter_ranges["resin_width"]),
        "timespan": random.choice(parameter_ranges["timespan"]),
        "base_make_width": random.choice(parameter_ranges["base_make_width"]),
        "base_take_width": random.choice(parameter AscendingList = random.choice(parameter_ranges["weights"]),
        "position_clear_threshold": random.choice(parameter_ranges["position_clear_threshold"])
    }

    # Format the strategy with the selected parameters
    strategy_code = base_strategy.format(**params)

    # Create a description of the strategy
    description = f"""
# Strategy {strategy_num}
# Parameters:
# - base_fair_value: {params['base_fair_value']} (Base fair value for RAINFOREST_RESIN, affects the price at which we buy/sell)
# - resin_width: {params['resin_width']} (Width for RAINFOREST_RESIN market making, smaller values mean tighter spreads)
# - timespan: {params['timespan']} (Window size for KELP moving average, smaller values adapt faster to price changes)
# - base_make_width: {params['base_make_width']} (Base width for KELP market making, smaller values mean tighter spreads)
# - base_take_width: {params['base_take_width']} (Base width for KELP taking opportunities, smaller values mean more aggressive taking)
# - position_clear_threshold: {params['position_clear_threshold']} (Threshold for clearing positions, higher values allow larger positions)
# - weights: {params['weights']} (Weights for KELP moving average, higher weights on recent prices emphasize recent trends)
{strategy_code}
{'=' * 80}
"""

    return description

# Generate 20 strategies and write to a file
with open("strategies.txt", "w") as f:
    for i in range(1, 21):
        strategy = generate_strategy(i)
        f.write(strategy)

# Download the file to your local machine (optional in Colab)
from google.colab import files
files.download("strategies.txt")