#!/usr/bin/env python3
import json
import math
import statistics
from collections import deque
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod # Import ABC and abstractmethod

# -------------------------------------------
# Data Types (Minimal for KELP Trading)
# -------------------------------------------
Symbol = str
Price = int
Volume = int

class Order:
    # Use __init__ for constructor
    def __init__(self, symbol: Symbol, price: Price, quantity: Volume) -> None:
        self.symbol = symbol
        # Ensure price is an integer after rounding
        self.price = int(round(price))
        # Positive=BUY, Negative=SELL
        self.quantity = quantity

    def __repr__(self):
        trade_type = "BUY" if self.quantity > 0 else "SELL"
        return f"Order({trade_type} {abs(self.quantity)} {self.symbol} @ {self.price})"

class OrderDepth:
    # Use __init__ for constructor
    def __init__(self) -> None:
        self.buy_orders: Dict[Price, Volume] = {}
        self.sell_orders: Dict[Price, Volume] = {}

class Trade:
    # Use __init__ for constructor
    def __init__(self, symbol: Symbol, price: Price, quantity: Volume, buyer: Optional[str], seller: Optional[str], timestamp: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

class TradingState:
    # Use __init__ for constructor
    def __init__(self,
                 timestamp: int,
                 traderData: str,
                 order_depths: Dict[Symbol, OrderDepth],
                 position: Dict[Symbol, Volume]) -> None:
        self.timestamp = timestamp
        self.traderData = traderData
        self.order_depths = order_depths
        self.position = position

class ProsperityEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, deque):
            return list(o)
        # Add handling for Order objects if needed in traderData, though typically not
        # if isinstance(o, Order):
        #     return {'symbol': o.symbol, 'price': o.price, 'quantity': o.quantity}
        return super().default(o)

# -------------------------------------------
# Base Strategy Class (Definition Added)
# -------------------------------------------
class Strategy(ABC): # Use ABC for abstract methods
    # Use __init__ for constructor
    def __init__(self, symbol: Symbol, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: List[Order] = [] # Initialize orders list here

    @abstractmethod
    def act(self, state: TradingState) -> None:
        """Core logic to determine trades."""
        raise NotImplementedError

    def run(self, state: TradingState) -> List[Order]:
        """Runs the strategy for a given state."""
        self.orders = [] # Reset orders for the current tick
        self.act(state)
        return self.orders

    # Keep buy/sell naming consistent with R2 strategy usage
    def buy(self, price: Price, quantity: Volume) -> None:
        """Helper to place buy orders."""
        # Ensure quantity is positive for a buy helper
        if quantity > 0:
            self.orders.append(Order(self.symbol, int(round(price)), int(quantity)))
            # print(f"PLACING BUY : {int(quantity)} {self.symbol} @ {int(round(price))}")

    def sell(self, price: Price, quantity: Volume) -> None:
        """Helper to place sell orders (stores quantity as negative)."""
        # Ensure quantity is positive for a sell helper (it will be stored negative)
        if quantity > 0:
            self.orders.append(Order(self.symbol, int(round(price)), -int(quantity)))
            # print(f"PLACING SELL: {int(quantity)} {self.symbol} @ {int(round(price))}")

    # Methods for state persistence (can be overridden)
    def save(self) -> Any:
        """Returns data to be saved in traderData."""
        return None # Default: no state saved

    def load(self, data: Any) -> None:
        """Loads data from traderData."""
        pass # Default: no state loaded

# -------------------------------------------
# KELP Strategy (Based on Profitable R2 InkMeanReversionTrader)
# -------------------------------------------
class Round2KelpStrategy(Strategy): # Now inherits correctly
    # Use __init__ for constructor
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        # Ensure maxlen is set correctly for deque
        self.window = 300
        self.prices = deque(maxlen=self.window)
        self.std_dev_cutoff = 0.01
        self.aggr_threshold = 2.0
        self.pass_upper_threshold = 1.7
        self.pass_lower_threshold = 0.8
        self.exit_threshold_factor = 0.25

    def get_best_bbo(self, order_depth: OrderDepth) -> Optional[tuple[Price, Price]]:
        """Gets Best Bid and Offer"""
        # Check if order_depth exists and has orders
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        try:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return best_bid, best_ask
        except ValueError: # Handles case where buy_orders or sell_orders might be empty after check (race condition unlikely but possible)
            return None

    def get_mid_price(self, best_bid: Price, best_ask: Price) -> Optional[float]:
        """Calculates Mid Price"""
        # BBO check already happened in get_best_bbo
        return (best_bid + best_ask) / 2.0

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        # Ensure order depth exists for the symbol
        if not order_depth:
            print(f"Warning: No order depth found for {self.symbol}")
            return

        bbo = self.get_best_bbo(order_depth)
        if bbo is None:
            # print(f"Warning: Could not get BBO for {self.symbol}")
            return
        best_bid, best_ask = bbo

        # Mid price calculation is safe here as bbo is not None
        mid_price = self.get_mid_price(best_bid, best_ask)
        if mid_price is None: # Should not happen if BBO is valid, but defensive check
             return

        self.prices.append(mid_price)

        # Wait until the window is filled
        if len(self.prices) < self.window:
            # print(f"Filling price window for {self.symbol}: {len(self.prices)}/{self.window}")
            return

        mean = 0.0
        std_dev = 0.0
        try:
            mean = statistics.mean(self.prices)
            # Need at least 2 points for stdev
            if len(self.prices) > 1:
                std_dev = statistics.stdev(self.prices)
            else:
                # If only one price, std dev is 0, handle gracefully
                std_dev = 0

            # Avoid trading if volatility is too low
            if std_dev < self.std_dev_cutoff:
                # print(f"Std Dev too low ({std_dev:.4f} < {self.std_dev_cutoff}), skipping trade for {self.symbol}")
                return

        except statistics.StatisticsError:
            print(f"Error calculating statistics for {self.symbol}, skipping.")
            return # Avoid trading if stats fail

        # Get current position, default to 0 if symbol not in position dict
        position = state.position.get(self.symbol, 0)
        buy_capacity = self.limit - position
        sell_capacity = self.limit + position # Note: position is negative if short

        # === EXIT ZONE ===
        # Check if the mid price is close to the mean, indicating potential mean reversion completion
        exit_deviation = self.exit_threshold_factor * std_dev
        if abs(mid_price - mean) <= exit_deviation:
            if position > 0: # Holding a long position, exit by selling
                # print(f"EXIT LONG: Selling {position} {self.symbol} near mean")
                self.sell(best_bid, position) # Sell at best bid to exit quickly
                return # Exit logic takes precedence
            elif position < 0: # Holding a short position, exit by buying
                # print(f"EXIT SHORT: Buying {-position} {self.symbol} near mean")
                self.buy(best_ask, -position) # Buy at best ask to exit quickly
                return # Exit logic takes precedence

        # === ENTRY LOGIC ===
        entry_order_placed = False

        # --- BUY ---
        # Only consider buying if we have capacity
        if buy_capacity > 0:
            # Aggressive Buy: Price is significantly below the mean
            if mid_price < mean - self.aggr_threshold * std_dev:
                # print(f"AGGRESSIVE BUY: Price {mid_price:.2f} < Mean {mean:.2f} - {self.aggr_threshold}*StdDev {std_dev:.2f}")
                self.buy(best_ask, buy_capacity) # Buy aggressively at best ask
                entry_order_placed = True
            # Passive Buy Zone: Price is somewhat below the mean
            elif not entry_order_placed and \
                 mean - self.pass_upper_threshold * std_dev < mid_price < mean - self.pass_lower_threshold * std_dev:
                 # Calculate how deep into the passive zone the price is
                 distance = mean - mid_price # How far below mean
                 denominator = (self.pass_upper_threshold - self.pass_lower_threshold) * std_dev
                 if denominator > 0: # Avoid division by zero if thresholds are same or std_dev is zero
                     factor = (distance - self.pass_lower_threshold * std_dev) / denominator
                     factor = min(max(factor, 0), 1) # Clamp factor between 0 and 1
                     # Scale volume based on factor, use math.ceil to ensure at least 1 if factor > 0
                     volume = math.ceil(factor * buy_capacity)
                     if volume > 0:
                         # print(f"PASSIVE BUY: Price {mid_price:.2f} in passive zone, Factor {factor:.2f}, Volume {volume}")
                         self.buy(best_bid, volume) # Passive buy at best bid
                         entry_order_placed = True

        # --- SELL ---
        # Only consider selling if we have capacity and no buy order was placed
        if not entry_order_placed and sell_capacity > 0:
            # Aggressive Sell: Price is significantly above the mean
            if mid_price > mean + self.aggr_threshold * std_dev:
                # print(f"AGGRESSIVE SELL: Price {mid_price:.2f} > Mean {mean:.2f} + {self.aggr_threshold}*StdDev {std_dev:.2f}")
                self.sell(best_bid, sell_capacity) # Sell aggressively at best bid
                entry_order_placed = True
            # Passive Sell Zone: Price is somewhat above the mean
            elif not entry_order_placed and \
                 mean + self.pass_lower_threshold * std_dev < mid_price < mean + self.pass_upper_threshold * std_dev:
                 # Calculate how deep into the passive zone the price is
                 distance = mid_price - mean # How far above mean
                 denominator = (self.pass_upper_threshold - self.pass_lower_threshold) * std_dev
                 if denominator > 0:
                     factor = (distance - self.pass_lower_threshold * std_dev) / denominator
                     factor = min(max(factor, 0), 1) # Clamp factor
                     volume = math.ceil(factor * sell_capacity)
                     if volume > 0:
                         # print(f"PASSIVE SELL: Price {mid_price:.2f} in passive zone, Factor {factor:.2f}, Volume {volume}")
                         self.sell(best_ask, volume) # Passive sell at best ask
                         entry_order_placed = True

    # Save the price history deque
    def save(self) -> Any:
        return list(self.prices)

    # Load the price history deque
    def load(self, data: Any) -> None:
        if isinstance(data, list):
            # Recreate deque with loaded data and correct maxlen
            self.prices = deque(data, maxlen=self.window)
            # print(f"Loaded {len(self.prices)} prices for {self.symbol}")
        else:
            # If data is invalid or not present, initialize fresh
            self.prices = deque(maxlen=self.window)
            # print(f"No valid price data found for {self.symbol}, starting fresh.")


# -------------------------------------------
# Trader Class (Simplified KELP-only entry point)
# -------------------------------------------
class Trader:
    # Use __init__ for constructor
    def __init__(self) -> None:
        self.kelp_symbol = "KELP"
        self.kelp_limit = 50 # Match limit from your R2 strategy example
        self.kelp_strategy = Round2KelpStrategy(
            symbol=self.kelp_symbol,
            limit=self.kelp_limit
        )
        # Cache for loaded trader data to avoid repeated JSON parsing if multiple strategies needed it
        self.trader_data_cache: Dict[str, Any] = {}
        print(f"Initialized Trader with Round 2 Logic for {self.kelp_symbol} (Limit: {self.kelp_limit})")

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        all_orders: Dict[Symbol, List[Order]] = {}
        conversions = 0 # KELP doesn't involve conversions
        new_trader_data_payload: Dict[str, Any] = {} # Data to be saved for the next tick

        # 1. Load State from traderData
        self.trader_data_cache = {} # Reset cache for this tick
        if state.traderData:
            try:
                # Attempt to load the entire traderData JSON string
                loaded_data = json.loads(state.traderData)
                # Check if it's a dictionary (expected format)
                if isinstance(loaded_data, dict):
                    self.trader_data_cache = loaded_data
                    # Specifically load data for the KELP strategy if present
                    kelp_state_data = self.trader_data_cache.get(self.kelp_symbol)
                    if kelp_state_data is not None:
                        self.kelp_strategy.load(kelp_state_data)
                    # else:
                        # print(f"No previous state found for {self.kelp_symbol} in traderData.")
                # else:
                    # print(f"Warning: traderData was not a dictionary: {state.traderData}")

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to decode traderData JSON: {e}. Data: '{state.traderData}'")
            except Exception as e:
                 print(f"Warning: An unexpected error occurred during traderData loading: {e}")

        # 2. Run Strategy Logic for KELP
        try:
            # Execute the KELP strategy's run method
            kelp_orders = self.kelp_strategy.run(state)
            # If the strategy generated any orders, add them to the results
            if kelp_orders:
                all_orders[self.kelp_symbol] = kelp_orders
        except Exception as e:
             print(f"ERROR running KELP strategy for {self.kelp_symbol}: {e}")
             # Optional: Print traceback for detailed debugging
             import traceback
             print(traceback.format_exc())

        # 3. Save State from KELP Strategy
        try:
            # Get the state data to be saved from the KELP strategy
            kelp_saved_state = self.kelp_strategy.save()
            # Only add to payload if the strategy actually returned something to save
            if kelp_saved_state is not None:
                new_trader_data_payload[self.kelp_symbol] = kelp_saved_state
        except Exception as e:
            print(f"ERROR saving KELP strategy state for {self.kelp_symbol}: {e}")

        # 4. Encode Trader Data for next tick
        trader_data_string = ""
        # Only encode if there's data to save
        if new_trader_data_payload:
            try:
                # Use the custom encoder (ProsperityEncoder) to handle deques
                # Use separators=(",", ":") for compact JSON recommended by Prosperity
                trader_data_string = json.dumps(new_trader_data_payload, cls=ProsperityEncoder, separators=(",", ":"))
            except Exception as e:
                 print(f"ERROR encoding traderData: {e}")
                 # Fallback to empty JSON object if encoding fails
                 trader_data_string = "{}"

        # Return orders, conversions (0 for KELP), and the serialized trader data string
        return all_orders, conversions, trader_data_string