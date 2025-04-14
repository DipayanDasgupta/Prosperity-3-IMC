import math
import jsonpickle
from typing import List, Dict, Tuple

# Import necessary classes from the provided datamodel
from datamodel import OrderDepth, TradingState, Order, Trade, ProsperityEncoder

class Trader:

    def __init__(self):
        # Position limits for each product, fetched dynamically if possible, default here
        self.position_limits = {"PEARLS": 20, "BANANAS": 20} # Add other products as they appear
        # Store calculated fair values between runs (optional, for smoothing)
        self.last_fair_values = {}
        self.ema_alpha = 0.1 # Exponential Moving Average alpha for smoothing fair value (optional)

    def calculate_fair_value(self, order_depth: OrderDepth, product: str) -> float:
        """
        Calculates a fair value estimate based on the order book.
        Uses weighted midpoint, falling back to simple midpoint or last known value.
        """
        if not order_depth.sell_orders and not order_depth.buy_orders:
            # No orders, return last known fair value or a default (e.g., 10000 for PEARLS)
            return self.last_fair_values.get(product, 10000.0) # Default to 10000 if no history

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        if best_ask is not None and best_bid is not None:
            best_ask_vol = abs(order_depth.sell_orders[best_ask])
            best_bid_vol = order_depth.buy_orders[best_bid]
            # Weighted midpoint calculation
            if best_ask_vol + best_bid_vol > 0:
                 # Weighted average: Price * Opposite Volume / Total Volume
                fair_value = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
            else:
                # Fallback to simple midpoint if volumes are zero (shouldn't happen often)
                fair_value = (best_ask + best_bid) / 2.0
        elif best_ask is not None:
            fair_value = best_ask # Use best ask if no bids
        elif best_bid is not None:
            fair_value = best_bid # Use best bid if no asks
        else:
            # Should not happen if first check passed, but as safety:
            return self.last_fair_values.get(product, 10000.0)

        # Optional: Smooth fair value using EMA
        prev_fair = self.last_fair_values.get(product, fair_value) # Initialize with current if no history
        fair_value = self.ema_alpha * fair_value + (1 - self.ema_alpha) * prev_fair
        self.last_fair_values[product] = fair_value # Update stored value

        return fair_value
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main trading logic method.
        """
        # Initialize result dictionary for orders
        result: Dict[str, List[Order]] = {}
        conversions = 0 # No conversions needed for PEARLS usually

        # Optional: Load state from traderData
        try:
            # Use jsonpickle to decode traderData, default to empty dict if it's empty or invalid
            loaded_data = jsonpickle.decode(state.traderData) if state.traderData else {}
            self.last_fair_values = loaded_data.get('last_fair_values', self.last_fair_values)
        except Exception as e:
            print(f"Error decoding traderData: {e}")
            # Keep existing self.last_fair_values or reset if needed
            self.last_fair_values = {}

        # Process each product
        for product, order_depth in state.order_depths.items():

            orders: List[Order] = []
            position_limit = self.position_limits.get(product, 20) # Default limit if not specified
            current_position = state.position.get(product, 0)

            # Calculate dynamic fair value
            fair_value = self.calculate_fair_value(order_depth, product)
            # Store for next iteration (if using state)
            self.last_fair_values[product] = fair_value

            print(f"Product: {product}, Fair Value: {fair_value:.2f}, Position: {current_position}")

            # Define target buy/sell prices around fair value
            # Ensure integer prices as required by the Order class
            target_buy_price = math.floor(fair_value - 1)
            target_sell_price = math.ceil(fair_value + 1)

            # --- Aggressive Trading Logic ---
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

            # Buy aggressively if best ask is below fair value
            if best_ask is not None and best_ask < fair_value:
                price_to_buy = best_ask
                available_volume = abs(order_depth.sell_orders[best_ask])
                # Calculate max volume we can buy based on position limit
                max_buy_volume = position_limit - current_position
                # Order size is minimum of available volume, max allowed by limit, and a base size (optional)
                order_volume = min(available_volume, max_buy_volume) # Consider adding a max order size param here too

                if order_volume > 0:
                    print(f"AGGRESSIVE BUY {product}: {order_volume}x @ {price_to_buy} (Fair: {fair_value:.2f})")
                    orders.append(Order(product, price_to_buy, order_volume))
                    # Update position assumption for subsequent logic in this iteration
                    current_position += order_volume

            # Sell aggressively if best bid is above fair value
            if best_bid is not None and best_bid > fair_value:
                price_to_sell = best_bid
                available_volume = order_depth.buy_orders[best_bid]
                # Calculate max volume we can sell based on position limit
                max_sell_volume = position_limit + current_position # How many we can sell to reach -limit
                # Order size is minimum of available volume, max allowed by limit, and a base size (optional)
                order_volume = min(available_volume, max_sell_volume) # Consider adding a max order size param

                if order_volume > 0:
                    print(f"AGGRESSIVE SELL {product}: {order_volume}x @ {price_to_sell} (Fair: {fair_value:.2f})")
                    orders.append(Order(product, price_to_sell, -order_volume)) # Negative quantity for sell
                    # Update position assumption
                    current_position -= order_volume


            # --- Passive Market Making Logic ---
            # Place passive orders only if we didn't just fill aggressively on that side

            # Place passive BUY order
            # Check if we still have room to buy
            if position_limit - current_position > 0:
                # Check if we didn't just buy aggressively at or above our target passive price
                did_aggressive_buy = any(o.price >= target_buy_price and o.quantity > 0 for o in orders)
                if not did_aggressive_buy:
                    buy_volume = position_limit - current_position # Try to fill remaining capacity towards limit
                    # Optionally cap passive order size (e.g., max 5 or 10 at a time)
                    buy_volume = min(buy_volume, 5) # Example cap
                    if buy_volume > 0:
                       print(f"PASSIVE BUY {product}: {buy_volume}x @ {target_buy_price} (Fair: {fair_value:.2f})")
                       orders.append(Order(product, target_buy_price, buy_volume))

            # Place passive SELL order
            # Check if we still have room to sell
            if position_limit + current_position > 0:
                 # Check if we didn't just sell aggressively at or below our target passive price
                did_aggressive_sell = any(o.price <= target_sell_price and o.quantity < 0 for o in orders)
                if not did_aggressive_sell:
                    sell_volume = position_limit + current_position # Try to fill remaining capacity towards limit
                    # Optionally cap passive order size
                    sell_volume = min(sell_volume, 5) # Example cap
                    if sell_volume > 0:
                        print(f"PASSIVE SELL {product}: {sell_volume}x @ {target_sell_price} (Fair: {fair_value:.2f})")
                        orders.append(Order(product, target_sell_price, -sell_volume)) # Negative quantity

            result[product] = orders

        # --- State Persistence ---
        # Serialize necessary data (like last fair values) into traderData string
        trader_data_to_save = {'last_fair_values': self.last_fair_values}
        traderData = jsonpickle.encode(trader_data_to_save, unpicklable=False) # unpicklable=False for simpler JSON

        # Return orders, conversions, and state data
        return result, conversions, traderData