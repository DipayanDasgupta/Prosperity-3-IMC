from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle

class Trader:
    def __init__(self):
        # Optional: Initialize any persistent state if needed (though not guaranteed between calls)
        self.position_limits = {"PEARLS": 20}  # Adjust based on round-specific limits
        self.fair_values = {"PEARLS": 10000}   # Starting fair value, refine with data later

    def run(self, state: TradingState):
        # Initialize result dictionary for orders
        result = {}
        
        # Process each product in the order depths
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Get current position (default to 0 if not present)
            current_position = state.position.get(product, 0)
            position_limit = self.position_limits.get(product, 20)  # Default to 20 if unknown
            
            # Calculate fair value (for now, use hardcoded; later, refine with market data)
            fair_value = self.fair_values.get(product, 10000)
            
            # Extract best bid and ask from order book
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
            
            # Define spread and order parameters
            spread = 2  # Tight spread for market-making (adjustable)
            order_size = 5  # Conservative size to stay within limits (adjustable)
            
            # Buy logic: Place buy order below fair value if within position limit
            if best_ask < fair_value and (current_position + order_size) <= position_limit:
                buy_price = min(fair_value - 1, best_ask)  # Buy slightly below fair value or at best ask
                buy_quantity = min(order_size, -order_depth.sell_orders.get(best_ask, 0))
                if buy_quantity > 0:
                    orders.append(Order(product, buy_price, buy_quantity))
                    print(f"BUY {product} {buy_quantity}x {buy_price}")
            
            # Sell logic: Place sell order above fair value if within position limit
            if best_bid > 0 and best_bid > fair_value and (current_position - order_size) >= -position_limit:
                sell_price = max(fair_value + 1, best_bid)  # Sell slightly above fair value or at best bid
                sell_quantity = min(order_size, order_depth.buy_orders.get(best_bid, 0))
                if sell_quantity > 0:
                    orders.append(Order(product, sell_price, -sell_quantity))
                    print(f"SELL {product} {sell_quantity}x {sell_price}")
            
            # Add orders to result
            result[product] = orders
        
        # State persistence (optional: serialize data for next iteration)
        trader_data = jsonpickle.encode({"last_fair_values": self.fair_values})
        
        # No conversions for now (can be added later if needed)
        conversions = 0
        
        return result, conversions, trader_data