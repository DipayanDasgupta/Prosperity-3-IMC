# IMC Prosperity Round 2 Strategy - v1
# Incorporates EDA insights & Basket Arbitrage

import json
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# Assuming datamodel classes are available (Order, TradingState, etc.)
from datamodel import Order, TradingState, OrderDepth, Trade, Symbol, ProsperityEncoder, Listing, Observation

import numpy as np
import pandas as pd # Keep for potential state analysis if needed offline
import jsonpickle # Use for robust state serialization

# Type Alias for JSON data
JSON = Dict[str, Any] | List[Any] | str | int | float | bool | None

class Product: # Define Product names
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"

# --- Basket Compositions ---
BASKET1_COMPOSITION = {Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1}
BASKET2_COMPOSITION = {Product.CROISSANTS: 4, Product.JAMS: 2}


# Logger Class (Using robust version)
class Logger:
    # ... (Include the robust Logger class from your previous successful code) ...
    # Make sure the flush method uses the compression helpers correctly
    def __init__(self) -> None: self.logs = ""; self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
         if len(self.logs) < self.max_log_length * 2: self.logs += sep.join(map(str, objects)) + end
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # --- Full Compression Logic ---
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
            compressed_conversion_observations = {}
            plain_obs = state.observations.plainValueObservations if hasattr(state.observations, 'plainValueObservations') and state.observations is not None else {}
            compressed_observations = [plain_obs, compressed_conversion_observations]
            pos = state.position if state.position is not None else {}
            return [state.timestamp, trader_data, compress_listings(state.listings),
                    compress_order_depths(state.order_depths), compress_trades(state.own_trades),
                    compress_trades(state.market_trades), pos, compressed_observations,]
        def compress_listings(listings: Optional[dict[Symbol, Listing]]) -> list[list[Any]]:
            compressed = [];
            if listings:
                for listing in listings.values():
                     if listing: compressed.append([listing.symbol, listing.product, listing.denomination])
            return compressed
        def compress_order_depths(order_depths: Optional[dict[Symbol, OrderDepth]]) -> dict[Symbol, list[Any]]:
            compressed = {};
            if order_depths:
                for symbol, order_depth in order_depths.items():
                    buy_orders_dict = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
                    sell_orders_dict = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
                    compressed[symbol]=[buy_orders_dict, sell_orders_dict]
            return compressed
        def compress_trades(trades: Optional[dict[Symbol, list[Trade]]]) -> list[list[Any]]:
            compressed = [];
            if trades:
                for arr in trades.values():
                     if arr:
                        for trade in arr:
                             if trade: compressed.append([trade.symbol,trade.price,trade.quantity,trade.buyer or "",trade.seller or "",trade.timestamp])
            return compressed
        def compress_orders(orders: Optional[dict[Symbol, list[Order]]]) -> list[list[Any]]:
            compressed = [];
            if orders:
                for arr in orders.values():
                    if arr:
                        for order in arr:
                            if order: compressed.append([order.symbol, order.price, order.quantity])
            return compressed
        def to_json(value: Any) -> str:
             cls_to_use = ProsperityEncoder if "ProsperityEncoder" in globals() else None
             try: return json.dumps(value, cls=cls_to_use, separators=(",", ":"))
             except Exception: return json.dumps(value, separators=(",", ":")) # Fallback
        def truncate(value: Any, max_length: int) -> str:
            s_value = "";
            try: s_value = str(value)
            except Exception: s_value = ""
            if len(s_value) <= max_length: return s_value
            return s_value[:max_length - 3] + "..."
        # --- End Compression Helpers ---
        try:
            min_obs = Observation({}, {})
            min_state = TradingState(state.timestamp, "", {}, {}, {}, {}, {}, min_obs)
            base_value = [compress_state(min_state, ""), compress_orders({}), conversions, "", ""]
            base_json = to_json(base_value); base_length=len(base_json)
            available_length = self.max_log_length - base_length - 200
            if available_length < 0: available_length = 0
            max_item_length = max(100, available_length // 3)
            trader_data_in = state.traderData if isinstance(state.traderData, str) else ""
            trader_data_out = trader_data if isinstance(trader_data, str) else ""
            logs_out = self.logs if isinstance(self.logs, str) else ""
            truncated_trader_data_state = truncate(trader_data_in, max_item_length)
            truncated_trader_data_out = truncate(trader_data_out, max_item_length)
            truncated_logs = truncate(logs_out, max_item_length)
            compressed_orders_val = compress_orders(orders)
            compressed_state_val = compress_state(state, truncated_trader_data_state)
            log_entry = [compressed_state_val, compressed_orders_val, conversions, truncated_trader_data_out, truncated_logs]
            final_json_output = to_json(log_entry)
            if len(final_json_output) > self.max_log_length:
                final_json_output = final_json_output[:self.max_log_length - 5] + "...]}"
            print(final_json_output)
        except Exception as e: print(json.dumps({"error": f"Logging failed: {e}", "timestamp": state.timestamp}))
        self.logs = ""
logger = Logger()

# --- Helper Functions ---
def calculate_mid_price(order_depth: Optional[OrderDepth]) -> Optional[float]:
    if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders: return None
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0

def calculate_vwap(order_depth: Optional[OrderDepth], side: str, depth: int = 1) -> Optional[float]:
    """Calculates Volume Weighted Average Price up to a certain depth."""
    if not order_depth: return None
    orders = order_depth.sell_orders if side == 'ask' else order_depth.buy_orders
    if not orders: return None

    sorted_levels = sorted(orders.items(), key=lambda x: x[0], reverse=(side == 'bid'))
    total_vol = 0
    total_val = 0
    levels_considered = 0
    for price, volume in sorted_levels:
        vol = abs(volume)
        total_vol += vol
        total_val += price * vol
        levels_considered += 1
        if levels_considered >= depth:
            break

    return total_val / total_vol if total_vol > 0 else None

def calculate_nav(composition: Dict[Symbol, int], state: TradingState) -> Optional[float]:
    """Calculates the Net Asset Value of a basket."""
    nav = 0.0
    for product, quantity in composition.items():
        order_depth = state.order_depths.get(product)
        mid_price = calculate_mid_price(order_depth)
        if mid_price is None:
            # logger.print(f"Warning: Missing mid_price for {product} to calculate NAV")
            return None # Cannot calculate NAV if a component price is missing
        nav += mid_price * quantity
    return nav


# --- Base Strategy Class ---
class BaseStrategy:
    def __init__(self, symbol: Symbol, params: Dict, limit: int) -> None:
        self.symbol = symbol
        self.params = params
        self.limit = limit
        self.orders: List[Order] = []

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def run(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> List[Order]:
        self.orders = [] # Reset orders for this tick
        try:
            self.act(state, current_orders)
        except Exception as e:
            logger.print(f"ERROR in {self.symbol} strategy: {e} at ts {state.timestamp}")
            # Add basic flatten logic on error
            current_pos = state.position.get(self.symbol, 0)
            if current_pos != 0:
                 ods = state.order_depths.get(self.symbol)
                 if ods:
                     if current_pos > 0 and ods.buy_orders: self.sell(max(ods.buy_orders.keys()), abs(current_pos))
                     elif current_pos < 0 and ods.sell_orders: self.buy(min(ods.sell_orders.keys()), abs(current_pos))
        return self.orders

    def act(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> None:
        # Base implementation - strategies should override this
        raise NotImplementedError

    def buy(self, price: float, quantity: float) -> None:
        q = int(round(quantity))
        if q > 0: self.orders.append(Order(self.symbol, int(round(price)), q))

    def sell(self, price: float, quantity: float) -> None:
        q = int(round(quantity))
        if q > 0: self.orders.append(Order(self.symbol, int(round(price)), -q))

    def save_state(self) -> Optional[JSON]: return None
    def load_state(self, data: Optional[JSON]) -> None: pass


# --- Standard Market Making Strategy (EMA based) ---
class EMAMarketMakingStrategy(BaseStrategy):
    def __init__(self, symbol: Symbol, params: Dict, limit: int) -> None:
        super().__init__(symbol, params, limit)
        self.ema_value: Optional[float] = None

    def load_state(self, data: Optional[JSON]) -> None:
        if isinstance(data, dict): self.ema_value = data.get("ema_value")
        else: self.ema_value = None

    def save_state(self) -> Optional[JSON]:
        return {"ema_value": self.ema_value} if self.ema_value is not None else None

    def calculate_ema(self, current_price: float, alpha: float) -> float:
         if self.ema_value is None: self.ema_value = current_price # Initialize
         else: self.ema_value = alpha * current_price + (1 - alpha) * self.ema_value
         return self.ema_value

    def act(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> None:
        order_depth = state.order_depths.get(self.symbol)
        mid_price = calculate_mid_price(order_depth)
        if mid_price is None: return # Cannot trade without mid-price

        # Update EMA
        ema_alpha = self.get_param('ema_alpha', 0.4) # Default alpha
        fair_value = self.calculate_ema(mid_price, ema_alpha)
        rounded_fair_value = int(round(fair_value))

        # Parameters
        spread = self.get_param('spread', 1) # Half-spread
        trade_size = self.get_param('trade_size', 5)
        skew_factor = self.get_param('skew_factor', 0.1) # Optional inventory skew
        pos_limit_ratio = self.get_param('pos_limit_ratio', 0.9) # Don't quote near limit

        # Basic Inventory Skew (Optional)
        position = state.position.get(self.symbol, 0)
        skew = round(position * skew_factor)
        buy_quote = rounded_fair_value - spread - skew
        sell_quote = rounded_fair_value + spread - skew

        # Adjust quotes based on BBO
        if order_depth.sell_orders: buy_quote = min(buy_quote, min(order_depth.sell_orders.keys()) - 1)
        if order_depth.buy_orders: sell_quote = max(sell_quote, max(order_depth.buy_orders.keys()) + 1)

        # Place Orders
        buy_cap = self.limit - position
        sell_cap = self.limit + position

        if buy_cap > 0 and abs(position) < self.limit * pos_limit_ratio:
            qty = min(trade_size, buy_cap)
            self.buy(buy_quote, qty)

        if sell_cap > 0 and abs(position) < self.limit * pos_limit_ratio:
            qty = min(trade_size, sell_cap)
            self.sell(sell_quote, qty)

# --- Fixed Value Market Making Strategy ---
class FixedValueMarketMakingStrategy(BaseStrategy):
    def act(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> None:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders: return

        fair_value = self.get_param('fair_value', 10000)
        spread = self.get_param('spread', 7) # Half-spread
        trade_size = self.get_param('trade_size', 10)
        pos_limit_ratio = self.get_param('pos_limit_ratio', 0.9)

        position = state.position.get(self.symbol, 0)
        buy_cap = self.limit - position
        sell_cap = self.limit + position

        buy_quote = fair_value - spread
        sell_quote = fair_value + spread

        if buy_cap > 0 and abs(position) < self.limit * pos_limit_ratio:
             qty = min(trade_size, buy_cap)
             self.buy(buy_quote, qty)

        if sell_cap > 0 and abs(position) < self.limit * pos_limit_ratio:
             qty = min(trade_size, sell_cap)
             self.sell(sell_quote, qty)

# --- Inventory Skew MM Strategy (Used for Squid Ink) ---
class InventorySkewMMStrategy(EMAMarketMakingStrategy): # Inherits EMA logic
     def act(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> None:
        order_depth = state.order_depths.get(self.symbol)
        mid_price = calculate_mid_price(order_depth)
        if mid_price is None: return

        # Update EMA
        ema_alpha = self.get_param('ema_alpha', 0.35)
        fair_value = self.calculate_ema(mid_price, ema_alpha)
        rounded_fair_value = int(round(fair_value))

        # Parameters
        spread = self.get_param('spread', 3.0) # Half-spread
        trade_size = self.get_param('trade_size', 8)
        skew_factor = self.get_param('skew_factor', 0.4) # Stronger skew
        pos_limit_ratio = self.get_param('pos_limit_ratio', 0.85)
        clear_pos_ratio = self.get_param('clear_pos_ratio', 0.7) # Ratio to start clearing
        clear_spread_mult = self.get_param('clear_spread_mult', 1.5) # How much wider to quote for clearing

        position = state.position.get(self.symbol, 0)
        buy_cap = self.limit - position
        sell_cap = self.limit + position

        # Calculate Skewed Quotes
        skew = round(position * skew_factor)
        base_buy_quote = rounded_fair_value - spread - skew
        base_sell_quote = rounded_fair_value + spread - skew

        final_buy_quote = base_buy_quote
        final_sell_quote = base_sell_quote
        if order_depth.sell_orders: final_buy_quote = min(base_buy_quote, min(order_depth.sell_orders.keys()) - 1)
        if order_depth.buy_orders: final_sell_quote = max(base_sell_quote, max(order_depth.buy_orders.keys()) + 1)

        # Passive Quoting
        if abs(position) < self.limit * pos_limit_ratio:
            if buy_cap > 0:
                qty = min(trade_size, buy_cap)
                self.buy(final_buy_quote, qty)
            if sell_cap > 0:
                qty = min(trade_size, sell_cap)
                self.sell(final_sell_quote, qty)

        # Clearing Logic
        if position > self.limit * clear_pos_ratio and sell_cap > 0:
            clear_qty = min(sell_cap, trade_size, position - int(self.limit * (clear_pos_ratio - 0.1))) # Clear towards target
            clear_price = rounded_fair_value - (spread * clear_spread_mult) - skew # Aggressive sell price
            if order_depth.buy_orders: clear_price = max(clear_price, max(order_depth.buy_orders.keys()) + 1) # Try to post, not hit
            self.sell(clear_price, clear_qty)
            logger.print(f"CLEARING LONG {self.symbol}: {clear_qty}@{clear_price}")
        elif position < -self.limit * clear_pos_ratio and buy_cap > 0:
            clear_qty = min(buy_cap, trade_size, abs(position) - int(self.limit * (clear_pos_ratio - 0.1)))
            clear_price = rounded_fair_value + (spread * clear_spread_mult) - skew # Aggressive buy price
            if order_depth.sell_orders: clear_price = min(clear_price, min(order_depth.sell_orders.keys()) - 1) # Try to post, not hit
            self.buy(clear_price, clear_qty)
            logger.print(f"CLEARING SHORT {self.symbol}: {clear_qty}@{clear_price}")


# --- Basket Arbitrage Strategy ---
class BasketArbitrageStrategy(BaseStrategy):
    def __init__(self, symbol: Symbol, params: Dict, limit: int, composition: Dict[Symbol, int]) -> None:
        super().__init__(symbol, params, limit)
        self.composition = composition
        self.component_symbols = list(composition.keys())
        # State tracking for open arb position (optional, simple version doesn't use state)
        # self.arb_position = 0 # e.g., +1 for long basket/short components, -1 for short basket/long components

    # Simple version doesn't need save/load state for now

    def act(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> None:
        basket_depth = state.order_depths.get(self.symbol)
        basket_mid_price = calculate_mid_price(basket_depth)
        if basket_mid_price is None: return # Need basket price

        # Calculate NAV
        nav = calculate_nav(self.composition, state)
        if nav is None: return # Need all component prices

        # Parameters
        entry_threshold = self.get_param('entry_threshold', 5.0) # Spread widens by this amount to enter
        exit_threshold = self.get_param('exit_threshold', 1.0) # Spread narrows to this to exit
        trade_size = self.get_param('trade_size', 10) # How many Baskets to trade per signal
        max_basket_pos = self.get_param('max_basket_pos', 50) # Max arb position in baskets

        # Current Positions
        basket_pos = state.position.get(self.symbol, 0)
        # Note: Need to consider component positions for accurate limit checks, simplified here

        # Calculate Spread
        spread = basket_mid_price - nav
        logger.print(f"{self.symbol} | Mid: {basket_mid_price:.1f} | NAV: {nav:.1f} | Spread: {spread:.2f} | Pos: {basket_pos}")

        # --- Trading Logic ---

        # Exit Logic: If position exists and spread crosses back
        if basket_pos > 0 and spread < exit_threshold: # Exit long basket / buy components
            qty_to_trade = min(basket_pos, trade_size) # Exit in chunks
            logger.print(f"EXIT LONG ARB {self.symbol}: {qty_to_trade}")
            self.sell(basket_mid_price - 1, qty_to_trade) # Sell basket slightly passively
            for product, ratio in self.composition.items():
                 comp_qty = qty_to_trade * ratio
                 comp_depth = state.order_depths.get(product)
                 comp_price = calculate_mid_price(comp_depth)
                 if comp_price: self.buy(comp_price + 1, comp_qty) # Buy components slightly passively

        elif basket_pos < 0 and spread > -exit_threshold: # Exit short basket / sell components
             qty_to_trade = min(abs(basket_pos), trade_size)
             logger.print(f"EXIT SHORT ARB {self.symbol}: {qty_to_trade}")
             self.buy(basket_mid_price + 1, qty_to_trade) # Buy basket slightly passively
             for product, ratio in self.composition.items():
                 comp_qty = qty_to_trade * ratio
                 comp_depth = state.order_depths.get(product)
                 comp_price = calculate_mid_price(comp_depth)
                 if comp_price: self.sell(comp_price - 1, comp_qty) # Sell components slightly passively

        # Entry Logic: If near flat and spread crosses entry threshold
        elif abs(basket_pos) < max_basket_pos: # Only enter if not already at max arb position
            if spread < -entry_threshold: # Enter long basket / short components
                 qty_to_trade = trade_size
                 logger.print(f"ENTER LONG ARB {self.symbol}: {qty_to_trade}")
                 self.buy(basket_mid_price + 1, qty_to_trade) # Buy basket slightly passively
                 for product, ratio in self.composition.items():
                     comp_qty = qty_to_trade * ratio
                     comp_depth = state.order_depths.get(product)
                     comp_price = calculate_mid_price(comp_depth)
                     if comp_price: self.sell(comp_price - 1, comp_qty) # Sell components slightly passively

            elif spread > entry_threshold: # Enter short basket / long components
                 qty_to_trade = trade_size
                 logger.print(f"ENTER SHORT ARB {self.symbol}: {qty_to_trade}")
                 self.sell(basket_mid_price - 1, qty_to_trade) # Sell basket slightly passively
                 for product, ratio in self.composition.items():
                     comp_qty = qty_to_trade * ratio
                     comp_depth = state.order_depths.get(product)
                     comp_price = calculate_mid_price(comp_depth)
                     if comp_price: self.buy(comp_price + 1, comp_qty) # Buy components slightly passively


# --- Trader Class ---
class Trader:
    def __init__(self) -> None:
        # Define limits (Check official rules for Round 2 limits - using placeholders)
        self.limits = {
            Product.CROISSANTS: 60, Product.JAMS: 60, Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 600, Product.PICNIC_BASKET2: 600, # Baskets usually have higher limits
            Product.KELP: 50, Product.RAINFOREST_RESIN: 50, Product.SQUID_INK: 50
        }

        # Define Strategy Parameters
        self.params = {
            Product.CROISSANTS: {'ema_alpha': 0.5, 'spread': 1, 'trade_size': 15},
            Product.JAMS: {'ema_alpha': 0.5, 'spread': 1, 'trade_size': 15},
            Product.DJEMBES: {'ema_alpha': 0.5, 'spread': 1, 'trade_size': 15},
            Product.PICNIC_BASKET1: {'entry_threshold': 4.0, 'exit_threshold': 1.0, 'trade_size': 10, 'max_basket_pos': 50},
            Product.PICNIC_BASKET2: {'entry_threshold': 3.0, 'exit_threshold': 0.5, 'trade_size': 10, 'max_basket_pos': 50},
            Product.KELP: {'ema_alpha': 0.3, 'spread': 2, 'trade_size': 15},
            Product.RAINFOREST_RESIN: {'fair_value': 10000, 'spread': 7, 'trade_size': 10},
            Product.SQUID_INK: {'ema_alpha': 0.35, 'spread': 3.0, 'trade_size': 8, 'skew_factor': 0.4, 'pos_limit_ratio': 0.85, 'clear_pos_ratio': 0.7, 'clear_spread_mult': 1.5}
        }

        # Instantiate Strategies
        self.strategies: Dict[Symbol, BaseStrategy] = {
            Product.CROISSANTS: EMAMarketMakingStrategy(Product.CROISSANTS, self.params[Product.CROISSANTS], self.limits[Product.CROISSANTS]),
            Product.JAMS: EMAMarketMakingStrategy(Product.JAMS, self.params[Product.JAMS], self.limits[Product.JAMS]),
            Product.DJEMBES: EMAMarketMakingStrategy(Product.DJEMBES, self.params[Product.DJEMBES], self.limits[Product.DJEMBES]),
            Product.PICNIC_BASKET1: BasketArbitrageStrategy(Product.PICNIC_BASKET1, self.params[Product.PICNIC_BASKET1], self.limits[Product.PICNIC_BASKET1], BASKET1_COMPOSITION),
            Product.PICNIC_BASKET2: BasketArbitrageStrategy(Product.PICNIC_BASKET2, self.params[Product.PICNIC_BASKET2], self.limits[Product.PICNIC_BASKET2], BASKET2_COMPOSITION),
            Product.KELP: EMAMarketMakingStrategy(Product.KELP, self.params[Product.KELP], self.limits[Product.KELP]),
            Product.RAINFOREST_RESIN: FixedValueMarketMakingStrategy(Product.RAINFOREST_RESIN, self.params[Product.RAINFOREST_RESIN], self.limits[Product.RAINFOREST_RESIN]),
            Product.SQUID_INK: InventorySkewMMStrategy(Product.SQUID_INK, self.params[Product.SQUID_INK], self.limits[Product.SQUID_INK])
        }

        self.trader_data_cache = {} # For storing strategy states

    def load_trader_data(self, traderData: str):
        if not traderData:
            self.trader_data_cache = {}
            for strategy in self.strategies.values():
                if hasattr(strategy, 'load_state'): strategy.load_state(None)
        else:
            try:
                decoded_data = jsonpickle.decode(traderData)
                if isinstance(decoded_data, dict):
                    self.trader_data_cache = decoded_data
                    for symbol, strategy in self.strategies.items():
                        if hasattr(strategy, 'load_state'):
                            strategy.load_state(self.trader_data_cache.get(symbol)) # Pass None if key missing
                else:
                    logger.print("Decoded traderData is not dict, resetting state.")
                    self.trader_data_cache = {}
                    for strategy in self.strategies.values():
                        if hasattr(strategy, 'load_state'): strategy.load_state(None)
            except Exception as e:
                logger.print(f"ERROR decoding traderData: {e}")
                self.trader_data_cache = {}
                for strategy in self.strategies.values():
                    if hasattr(strategy, 'load_state'): strategy.load_state(None)

    def save_trader_data(self) -> str:
        self.trader_data_cache = {}
        for symbol, strategy in self.strategies.items():
            if hasattr(strategy, 'save_state'):
                saved = strategy.save_state()
                if saved is not None: self.trader_data_cache[symbol] = saved
        try:
            return jsonpickle.encode(self.trader_data_cache, unpicklable=False)
        except Exception as e:
            logger.print(f"ERROR encoding traderData: {e}")
            return ""

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        self.load_trader_data(state.traderData)
        all_orders: Dict[Symbol, List[Order]] = {symbol: [] for symbol in self.strategies} # Initialize empty lists
        conversions = 0 # Placeholder for round 2 conversions (loot mechanic?)

        # Run strategies sequentially
        # Important: Basket strategy runs first to place its orders,
        # then component strategies run (they might modify component orders slightly,
        # but the main arb direction is set by the basket strategy).
        # This order might need refinement.
        run_order = [
            Product.PICNIC_BASKET1, Product.PICNIC_BASKET2, # Run baskets first
            Product.CROISSANTS, Product.JAMS, Product.DJEMBES, # Then components
            Product.KELP, Product.RAINFOREST_RESIN, Product.SQUID_INK # Then others
        ]

        for symbol in run_order:
            if symbol in self.strategies and symbol in state.order_depths:
                strategy = self.strategies[symbol]
                # Pass existing orders placed by previous strategies in this tick (esp. for arb)
                product_orders = strategy.run(state, all_orders)
                all_orders[symbol].extend(product_orders) # Append orders from this strategy

        # TODO: Implement loot container opening logic here if needed
        # This likely involves checking state.observations for container info
        # and setting the 'conversions' variable based on chosen container(s).
        # Example placeholder:
        # if state.timestamp == 0: # Example: Choose container at start
        #     container_choice = 1 # Choose container 1 (free)
        #     conversions = container_choice
        #     logger.print("Choosing loot container:", container_choice)


        traderData_output = self.save_trader_data()
        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output