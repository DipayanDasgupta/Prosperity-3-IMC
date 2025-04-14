# IMC Prosperity Round 2 Strategy - v2 (Improved)
# Features: Statistical Arbitrage for Baskets, Refined MM, Limit Checks

import json
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# Assuming datamodel classes are available
from datamodel import Order, TradingState, OrderDepth, Trade, Symbol, ProsperityEncoder, Listing, Observation

import numpy as np
import statistics # For mean/stdev calculations in arb
import jsonpickle

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
    def __init__(self) -> None: self.logs = ""; self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
         log_line = sep.join(map(str, objects)) + end
         # Prevent excessive log growth in memory
         if len(self.logs) + len(log_line) < self.max_log_length * 5:
              self.logs += log_line
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # --- Full Compression Logic ---
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
            compressed_conversion_observations = {}
            plain_obs = state.observations.plainValueObservations if hasattr(state.observations, 'plainValueObservations') and state.observations is not None else {}
            compressed_observations = [plain_obs, compressed_conversion_observations]
            pos = state.position if state.position is not None else {}
            own_trades_comp = compress_trades(state.own_trades)
            market_trades_comp = compress_trades(state.market_trades)
            return [state.timestamp, trader_data, compress_listings(state.listings),
                    compress_order_depths(state.order_depths), own_trades_comp,
                    market_trades_comp, pos, compressed_observations,]
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
    try:
        best_bid = max(map(int, order_depth.buy_orders.keys()))
        best_ask = min(map(int, order_depth.sell_orders.keys()))
        return (best_bid + best_ask) / 2.0
    except (ValueError, TypeError): return None # Handle potential empty keys or non-numeric conversion

def calculate_nav(composition: Dict[Symbol, int], state: TradingState) -> Optional[float]:
    nav = 0.0
    for product, quantity in composition.items():
        mid_price = calculate_mid_price(state.order_depths.get(product))
        if mid_price is None: return None
        nav += mid_price * quantity
    return nav

# --- Base Strategy Class ---
class BaseStrategy:
    # ... (Keep BaseStrategy as defined in v1) ...
    def __init__(self, symbol: Symbol, params: Dict, limit: int) -> None:
        self.symbol = symbol
        self.params = params
        self.limit = limit
        self.orders: List[Order] = []

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def run(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> List[Order]:
        self.orders = []
        try:
            self.act(state, current_orders)
        except Exception as e:
            logger.print(f"ERROR in {self.symbol} strategy: {e} at ts {state.timestamp}")
            current_pos = state.position.get(self.symbol, 0)
            if current_pos != 0:
                 ods = state.order_depths.get(self.symbol)
                 if ods:
                     if current_pos > 0 and ods.buy_orders: self.sell(max(ods.buy_orders.keys()), abs(current_pos))
                     elif current_pos < 0 and ods.sell_orders: self.buy(min(ods.sell_orders.keys()), abs(current_pos))
        return self.orders

    def act(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> None:
        raise NotImplementedError

    def buy(self, price: float, quantity: float) -> None:
        q = int(round(quantity))
        if q > 0: self.orders.append(Order(self.symbol, int(round(price)), q))

    def sell(self, price: float, quantity: float) -> None:
        q = int(round(quantity))
        if q > 0: self.orders.append(Order(self.symbol, int(round(price)), -q))

    def save_state(self) -> Optional[JSON]: return None
    def load_state(self, data: Optional[JSON]) -> None: pass

# --- EMA Market Making Strategy ---
class EMAMarketMakingStrategy(BaseStrategy):
    # ... (Keep EMAMarketMakingStrategy as defined in v1) ...
    def __init__(self, symbol: Symbol, params: Dict, limit: int) -> None:
        super().__init__(symbol, params, limit)
        self.ema_value: Optional[float] = None
    def load_state(self, data: Optional[JSON]) -> None:
        if isinstance(data, dict): self.ema_value = data.get("ema_value")
        else: self.ema_value = None
    def save_state(self) -> Optional[JSON]:
        return {"ema_value": self.ema_value} if self.ema_value is not None else None
    def calculate_ema(self, current_price: float, alpha: float) -> float:
         if self.ema_value is None: self.ema_value = current_price
         else: self.ema_value = alpha * current_price + (1 - alpha) * self.ema_value
         return self.ema_value
    def act(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> None:
        order_depth = state.order_depths.get(self.symbol)
        mid_price = calculate_mid_price(order_depth)
        if mid_price is None: return
        ema_alpha = self.get_param('ema_alpha', 0.4); fair_value = self.calculate_ema(mid_price, ema_alpha)
        rounded_fair_value = int(round(fair_value))
        spread = self.get_param('spread', 1.5); trade_size = self.get_param('trade_size', 15)
        skew_factor = self.get_param('skew_factor', 0.0); pos_limit_ratio = self.get_param('pos_limit_ratio', 0.9) # Reduced default skew
        position = state.position.get(self.symbol, 0)
        skew = round(position * skew_factor)
        buy_quote = rounded_fair_value - math.ceil(spread) - skew # Use ceil/floor for integer spread
        sell_quote = rounded_fair_value + math.ceil(spread) - skew
        if order_depth.sell_orders: buy_quote = min(buy_quote, min(order_depth.sell_orders.keys()) - 1)
        if order_depth.buy_orders: sell_quote = max(sell_quote, max(order_depth.buy_orders.keys()) + 1)
        buy_cap = self.limit - position; sell_cap = self.limit + position
        if buy_cap > 0 and abs(position) < self.limit * pos_limit_ratio:
            qty = min(trade_size, buy_cap); self.buy(buy_quote, qty)
        if sell_cap > 0 and abs(position) < self.limit * pos_limit_ratio:
            qty = min(trade_size, sell_cap); self.sell(sell_quote, qty)

# --- Fixed Value Market Making Strategy ---
class FixedValueMarketMakingStrategy(BaseStrategy):
    # ... (Keep FixedValueMarketMakingStrategy as defined in v1, maybe widen spread) ...
     def act(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> None:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders: return
        fair_value = self.get_param('fair_value', 10000); spread = self.get_param('spread', 8) # Slightly wider
        trade_size = self.get_param('trade_size', 15); pos_limit_ratio = self.get_param('pos_limit_ratio', 0.9)
        position = state.position.get(self.symbol, 0)
        buy_cap = self.limit - position; sell_cap = self.limit + position
        buy_quote = fair_value - spread; sell_quote = fair_value + spread
        if buy_cap > 0 and abs(position) < self.limit * pos_limit_ratio:
             qty = min(trade_size, buy_cap); self.buy(buy_quote, qty)
        if sell_cap > 0 and abs(position) < self.limit * pos_limit_ratio:
             qty = min(trade_size, sell_cap); self.sell(sell_quote, qty)

# --- Inventory Skew MM Strategy (For Squid Ink) ---
class InventorySkewMMStrategy(EMAMarketMakingStrategy): # Inherits EMA logic
    # ... (Keep InventorySkewMMStrategy as defined in v1, maybe increase trade_size) ...
    def act(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> None:
        order_depth = state.order_depths.get(self.symbol); mid_price = calculate_mid_price(order_depth)
        if mid_price is None: return
        ema_alpha = self.get_param('ema_alpha', 0.35); fair_value = self.calculate_ema(mid_price, ema_alpha)
        rounded_fair_value = int(round(fair_value))
        spread = self.get_param('spread', 3.0); trade_size = self.get_param('trade_size', 10) # Increased size
        skew_factor = self.get_param('skew_factor', 0.4); pos_limit_ratio = self.get_param('pos_limit_ratio', 0.85)
        clear_pos_ratio = self.get_param('clear_pos_ratio', 0.7); clear_spread_mult = self.get_param('clear_spread_mult', 1.5)
        position = state.position.get(self.symbol, 0)
        buy_cap = self.limit - position; sell_cap = self.limit + position
        skew = round(position * skew_factor)
        base_buy_quote = rounded_fair_value - spread - skew; base_sell_quote = rounded_fair_value + spread - skew
        final_buy_quote = base_buy_quote; final_sell_quote = base_sell_quote
        if order_depth.sell_orders: final_buy_quote = min(base_buy_quote, min(order_depth.sell_orders.keys()) - 1)
        if order_depth.buy_orders: final_sell_quote = max(base_sell_quote, max(order_depth.buy_orders.keys()) + 1)
        # Passive Quoting
        if abs(position) < self.limit * pos_limit_ratio:
            if buy_cap > 0: qty = min(trade_size, buy_cap); self.buy(final_buy_quote, qty)
            if sell_cap > 0: qty = min(trade_size, sell_cap); self.sell(final_sell_quote, qty)
        # Clearing Logic
        if position > self.limit * clear_pos_ratio and sell_cap > 0:
            clear_qty = min(sell_cap, trade_size, position - int(self.limit * (clear_pos_ratio - 0.1)))
            clear_price = rounded_fair_value - (spread * clear_spread_mult) - skew
            if order_depth.buy_orders: clear_price = max(clear_price, max(order_depth.buy_orders.keys()) + 1)
            self.sell(clear_price, clear_qty); logger.print(f"CLEARING LONG {self.symbol}: {clear_qty}@{clear_price}")
        elif position < -self.limit * clear_pos_ratio and buy_cap > 0:
            clear_qty = min(buy_cap, trade_size, abs(position) - int(self.limit * (clear_pos_ratio - 0.1)))
            clear_price = rounded_fair_value + (spread * clear_spread_mult) - skew
            if order_depth.sell_orders: clear_price = min(clear_price, min(order_depth.sell_orders.keys()) - 1)
            self.buy(clear_price, clear_qty); logger.print(f"CLEARING SHORT {self.symbol}: {clear_qty}@{clear_price}")

# --- Statistical Arbitrage Strategy (IMPROVED) ---
class StatisticalArbitrageStrategy(BaseStrategy):
    def __init__(self, symbol: Symbol, params: Dict, limit: int, composition: Dict[Symbol, int]) -> None:
        super().__init__(symbol, params, limit)
        self.composition = composition
        self.component_symbols = list(composition.keys())
        self.spread_history: deque[float] = deque(maxlen=self.get_param('spread_window', 100))
        self.min_spread_samples = 20 # Need at least this many samples for stdev

    def load_state(self, data: Optional[JSON]) -> None:
        if isinstance(data, dict) and 'spread_history' in data:
             # Load deque, ensuring maxlen is reapplied
            self.spread_history = deque(data['spread_history'], maxlen=self.get_param('spread_window', 100))
        else:
             self.spread_history = deque(maxlen=self.get_param('spread_window', 100)) # Initialize fresh

    def save_state(self) -> Optional[JSON]:
        return {"spread_history": list(self.spread_history)}

    def check_limits(self, state: TradingState, basket_qty_change: int) -> bool:
        """Checks if proposed trade violates limits for basket or components."""
        # Check basket limit
        current_basket_pos = state.position.get(self.symbol, 0)
        if abs(current_basket_pos + basket_qty_change) > self.limit:
            logger.print(f"LIMIT CHECK FAIL ({self.symbol}): {current_basket_pos} + {basket_qty_change} > {self.limit}")
            return False

        # Check component limits
        component_limits = Trader.LIMITS # Access global limits defined in Trader
        for product, ratio in self.composition.items():
            component_limit = component_limits.get(product, 0) # Get limit for component
            current_comp_pos = state.position.get(product, 0)
            # If buying basket (+basket_qty), selling components (-ratio*basket_qty)
            # If selling basket (-basket_qty), buying components (+ratio*basket_qty)
            proposed_comp_change = -ratio * basket_qty_change
            if abs(current_comp_pos + proposed_comp_change) > component_limit:
                 logger.print(f"LIMIT CHECK FAIL ({product}): {current_comp_pos} + {proposed_comp_change} > {component_limit}")
                 return False
        return True # All limits OK

    def act(self, state: TradingState, current_orders: Dict[Symbol, List[Order]]) -> None:
        basket_depth = state.order_depths.get(self.symbol)
        basket_mid = calculate_mid_price(basket_depth)
        if basket_mid is None: return

        nav = calculate_nav(self.composition, state)
        if nav is None: return

        spread = basket_mid - nav
        self.spread_history.append(spread)

        if len(self.spread_history) < self.min_spread_samples: return # Wait for enough data

        # Calculate rolling stats
        spread_mean = statistics.mean(self.spread_history)
        spread_stdev = statistics.stdev(self.spread_history) if len(self.spread_history) > 1 else 0

        if spread_stdev < 0.1: return # Avoid trading if spread is flat

        # Parameters
        entry_z = self.get_param('entry_z', 1.5) # Enter when spread is N stdevs away
        exit_z = self.get_param('exit_z', 0.25) # Exit when spread is back within M stdevs
        trade_size = self.get_param('trade_size', 50) # Increased size
        max_total_exposure = self.get_param('max_total_exposure', 500) # Limit total basket abs position

        # Positions
        basket_pos = state.position.get(self.symbol, 0)

        logger.print(f"{self.symbol} | Mid: {basket_mid:.1f} | NAV: {nav:.1f} | Spread: {spread:.2f} | Mean: {spread_mean:.2f} | Std: {spread_stdev:.2f} | Pos: {basket_pos}")

        # --- Trading Logic ---
        upper_band = spread_mean + entry_z * spread_stdev
        lower_band = spread_mean - entry_z * spread_stdev
        exit_upper = spread_mean + exit_z * spread_stdev
        exit_lower = spread_mean - exit_z * spread_stdev

        # Exit Logic
        if basket_pos > 0 and spread < exit_upper: # Exit long basket / buy components
            qty_to_trade = min(basket_pos, trade_size)
            logger.print(f"EXIT LONG STAT ARB {self.symbol}: {qty_to_trade}")
            # Sell Basket (Hit Bid or Mid)
            if basket_depth.buy_orders: self.sell(max(basket_depth.buy_orders.keys()), qty_to_trade)
            # Buy Components (Hit Ask or Mid)
            for product, ratio in self.composition.items():
                 comp_qty = qty_to_trade * ratio; comp_depth = state.order_depths.get(product)
                 if comp_depth and comp_depth.sell_orders: self.buy(min(comp_depth.sell_orders.keys()), comp_qty)

        elif basket_pos < 0 and spread > exit_lower: # Exit short basket / sell components
             qty_to_trade = min(abs(basket_pos), trade_size)
             logger.print(f"EXIT SHORT STAT ARB {self.symbol}: {qty_to_trade}")
             # Buy Basket (Hit Ask or Mid)
             if basket_depth.sell_orders: self.buy(min(basket_depth.sell_orders.keys()), qty_to_trade)
             # Sell Components (Hit Bid or Mid)
             for product, ratio in self.composition.items():
                 comp_qty = qty_to_trade * ratio; comp_depth = state.order_depths.get(product)
                 if comp_depth and comp_depth.buy_orders: self.sell(max(comp_depth.buy_orders.keys()), comp_qty)

        # Entry Logic
        elif abs(basket_pos) < max_total_exposure:
            if spread < lower_band: # Enter long basket / short components
                 qty_to_trade = trade_size
                 # Check limits BEFORE placing orders
                 if self.check_limits(state, qty_to_trade):
                     logger.print(f"ENTER LONG STAT ARB {self.symbol}: {qty_to_trade}")
                     # Buy Basket (Hit Ask or Mid)
                     if basket_depth.sell_orders: self.buy(min(basket_depth.sell_orders.keys()), qty_to_trade)
                     # Sell Components (Hit Bid or Mid)
                     for product, ratio in self.composition.items():
                         comp_qty = qty_to_trade * ratio; comp_depth = state.order_depths.get(product)
                         if comp_depth and comp_depth.buy_orders: self.sell(max(comp_depth.buy_orders.keys()), comp_qty)
                 else: logger.print(f"SKIPPING LONG STAT ARB {self.symbol} due to limits")


            elif spread > upper_band: # Enter short basket / long components
                 qty_to_trade = trade_size
                 # Check limits BEFORE placing orders (note: qty is negative for selling basket)
                 if self.check_limits(state, -qty_to_trade):
                     logger.print(f"ENTER SHORT STAT ARB {self.symbol}: {qty_to_trade}")
                     # Sell Basket (Hit Bid or Mid)
                     if basket_depth.buy_orders: self.sell(max(basket_depth.buy_orders.keys()), qty_to_trade)
                     # Buy Components (Hit Ask or Mid)
                     for product, ratio in self.composition.items():
                         comp_qty = qty_to_trade * ratio; comp_depth = state.order_depths.get(product)
                         if comp_depth and comp_depth.sell_orders: self.buy(min(comp_depth.sell_orders.keys()), comp_qty)
                 else: logger.print(f"SKIPPING SHORT STAT ARB {self.symbol} due to limits")


# --- Trader Class ---
class Trader:
    # Class variable for limits, accessible by strategies
    LIMITS = {
        Product.CROISSANTS: 60, Product.JAMS: 60, Product.DJEMBES: 60,
        Product.PICNIC_BASKET1: 600, Product.PICNIC_BASKET2: 600,
        Product.KELP: 50, Product.RAINFOREST_RESIN: 50, Product.SQUID_INK: 50
    }

    def __init__(self) -> None:
        self.limits = Trader.LIMITS # Use class variable

        # Define Strategy Parameters (Refined)
        self.params = {
            Product.CROISSANTS: {'ema_alpha': 0.4, 'spread': 1, 'trade_size': 20}, # Tighter spread, higher size
            Product.JAMS: {'ema_alpha': 0.4, 'spread': 1.5, 'trade_size': 15}, # Slightly wider spread
            Product.DJEMBES: {'ema_alpha': 0.4, 'spread': 1, 'trade_size': 20}, # Tighter spread, higher size
            Product.PICNIC_BASKET1: {'entry_z': 1.5, 'exit_z': 0.2, 'trade_size': 60, 'max_total_exposure': 500, 'spread_window': 100}, # Stat arb params
            Product.PICNIC_BASKET2: {'entry_z': 1.5, 'exit_z': 0.2, 'trade_size': 60, 'max_total_exposure': 500, 'spread_window': 100}, # Stat arb params
            Product.KELP: {'ema_alpha': 0.3, 'spread': 2, 'trade_size': 20}, # Slightly larger size
            Product.RAINFOREST_RESIN: {'fair_value': 10000, 'spread': 8, 'trade_size': 15}, # Wider spread, larger size
            Product.SQUID_INK: {'ema_alpha': 0.35, 'spread': 3.0, 'trade_size': 10, 'skew_factor': 0.4, 'pos_limit_ratio': 0.85, 'clear_pos_ratio': 0.7, 'clear_spread_mult': 1.5} # Increased size
        }

        # Instantiate Strategies
        self.strategies: Dict[Symbol, BaseStrategy] = {
            Product.CROISSANTS: EMAMarketMakingStrategy(Product.CROISSANTS, self.params[Product.CROISSANTS], self.limits[Product.CROISSANTS]),
            Product.JAMS: EMAMarketMakingStrategy(Product.JAMS, self.params[Product.JAMS], self.limits[Product.JAMS]),
            Product.DJEMBES: EMAMarketMakingStrategy(Product.DJEMBES, self.params[Product.DJEMBES], self.limits[Product.DJEMBES]),
            Product.PICNIC_BASKET1: StatisticalArbitrageStrategy(Product.PICNIC_BASKET1, self.params[Product.PICNIC_BASKET1], self.limits[Product.PICNIC_BASKET1], BASKET1_COMPOSITION),
            Product.PICNIC_BASKET2: StatisticalArbitrageStrategy(Product.PICNIC_BASKET2, self.params[Product.PICNIC_BASKET2], self.limits[Product.PICNIC_BASKET2], BASKET2_COMPOSITION),
            Product.KELP: EMAMarketMakingStrategy(Product.KELP, self.params[Product.KELP], self.limits[Product.KELP]),
            Product.RAINFOREST_RESIN: FixedValueMarketMakingStrategy(Product.RAINFOREST_RESIN, self.params[Product.RAINFOREST_RESIN], self.limits[Product.RAINFOREST_RESIN]),
            Product.SQUID_INK: InventorySkewMMStrategy(Product.SQUID_INK, self.params[Product.SQUID_INK], self.limits[Product.SQUID_INK])
        }
        self.trader_data_cache = {}

    def load_trader_data(self, traderData: str):
        # ... (Keep load_trader_data using jsonpickle as defined in v1) ...
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
        # ... (Keep save_trader_data using jsonpickle as defined in v1) ...
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
        all_orders: Dict[Symbol, List[Order]] = {symbol: [] for symbol in self.strategies}
        conversions = 0 # Placeholder for loot mechanic

        # Define run order - potentially run arb last to consider final component prices?
        # Or run first to set the primary trade? Let's keep Baskets first for now.
        run_order = [
            Product.PICNIC_BASKET1, Product.PICNIC_BASKET2,
            Product.CROISSANTS, Product.JAMS, Product.DJEMBES,
            Product.KELP, Product.RAINFOREST_RESIN, Product.SQUID_INK
        ]

        for symbol in run_order:
            if symbol in self.strategies and symbol in state.order_depths:
                strategy = self.strategies[symbol]
                # Pass the current state of orders (might be modified by arb strategy)
                # This is tricky - simpler to let strategies generate independently for now
                # and rely on exchange logic / limits.
                product_orders = strategy.run(state, {}) # Pass empty dict for now
                # Accumulate orders, overwriting might be complex if arb places component orders
                # Let's just extend the list for now, assuming limit checks prevent issues
                all_orders[symbol].extend(product_orders)

        # TODO: Implement loot container opening logic here

        traderData_output = self.save_trader_data()
        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output