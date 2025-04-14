# Final Strategy for Prosperity Round 1 - Optimized Squid Ink v1
# Combines best performing logic for each product:
# - RAINFOREST_RESIN: Fixed Value Market Making
# - KELP: EMA Market Making with Simple Inventory Thresholds
# - SQUID_INK: EMA Market Making with Aggressive Inventory Skew (Optimized Params + Clearing)

import json
import math
from collections import deque, defaultdict
from typing import Any, TypeAlias, List, Dict, Optional, Tuple
from abc import abstractmethod
import jsonpickle
import numpy as np # Keep for potential use

# Assuming datamodel classes are available
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# Type Alias for JSON data
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

# Parameters dictionary - Tuned for Round 1 Products - OPTIMIZED SQUID INK
PARAMS = {
    Product.RAINFOREST_RESIN: { # For FixedValueStrategy (Unchanged)
        "fair_value": 10000,
        "take_width": 1,
        "make_spread": 2,
        "aggr_size": 2,
        "make_size": 5,
    },
    Product.KELP: { # For OriginalMarketMakingStrategy (Unchanged)
        "ema_alpha": 0.3,
        "take_width": 1,
        "aggr_size": 3,
        "make_size": 4,
    },
    Product.SQUID_INK: { # For InventorySkewMMStrategy (Optimized)
        "ema_alpha": 0.35,     # Faster EMA responsiveness
        "take_width": 2,       # Keep threshold for aggressive takes
        "clear_width": 2,      # ENABLED: Width to cross fair value for clearing
        "make_spread": 3.0,    # Widen base half-spread slightly
        "aggr_size": 3,        # Increased aggressive size
        "make_size": 4,        # Increased passive size
        "skew_factor": 0.4,    # INCREASED inventory skew factor
        "max_pos_ratio_passive": 0.85, # Keep passive quoting threshold
        "clear_pos_ratio": 0.7  # ENABLED: Position ratio to trigger clearing (e.g., > 35 / < -35)
    }
}

# Logger class (Unchanged)
class Logger:
    def __init__(self) -> None: self.logs = ""; self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None: self.logs += sep.join(map(str, objects)) + end
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
            compressed_conversion_observations = {}
            plain_obs = state.observations.plainValueObservations if hasattr(state.observations, 'plainValueObservations') and state.observations is not None else {}
            compressed_observations = [plain_obs, compressed_conversion_observations]
            return [
                state.timestamp, trader_data, compress_listings(state.listings),
                compress_order_depths(state.order_depths), compress_trades(state.own_trades or {}),
                compress_trades(state.market_trades or {}), state.position or {}, compressed_observations,
            ]
        def compress_listings(listings: dict[Symbol, Listing]) -> list[list[Any]]:
            compressed = [];
            if listings:
                for listing in listings.values():
                     if listing: compressed.append([listing.symbol, listing.product, listing.denomination])
            return compressed
        def compress_order_depths(order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
            compressed = {};
            if order_depths:
                for symbol, order_depth in order_depths.items():
                    if order_depth: buy_orders_dict=order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}; sell_orders_dict=order_depth.sell_orders if isinstance(order_depth.sell_orders,dict) else {}; compressed[symbol]=[buy_orders_dict, sell_orders_dict]
                    else: compressed[symbol] = [{}, {}]
            return compressed
        def compress_trades(trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
            compressed = [];
            if trades:
                for arr in trades.values():
                     if arr:
                        for trade in arr:
                             if trade: compressed.append([trade.symbol,trade.price,trade.quantity,trade.buyer or "",trade.seller or "",trade.timestamp])
            return compressed
        def compress_orders(orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
            compressed = [];
            if orders:
                for arr in orders.values():
                    if arr:
                        for order in arr:
                            if order: compressed.append([order.symbol, order.price, order.quantity])
            return compressed
        def to_json(value: Any) -> str:
             try: return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
             except Exception: return json.dumps(value, separators=(",", ":"))
        def truncate(value: str, max_length: int) -> str:
            if not isinstance(value, str): value = str(value)
            if len(value) <= max_length: return value
            return value[:max_length - 3] + "..."
        try:
            min_obs=Observation({}, {}); min_state=TradingState(state.timestamp,"",{},{},{},{},{},min_obs)
            base_value=[compress_state(min_state,""),compress_orders({}),conversions,"",""]; base_json=to_json(base_value); base_length=len(base_json)
            available_length=self.max_log_length-base_length-200;
            if available_length < 0: available_length=0
            max_item_length=available_length//3
            truncated_trader_data_state=truncate(state.traderData if state.traderData else "", max_item_length)
            truncated_trader_data_out=truncate(trader_data, max_item_length); truncated_logs=truncate(self.logs, max_item_length)
            log_entry=[compress_state(state, truncated_trader_data_state), compress_orders(orders), conversions, truncated_trader_data_out, truncated_logs]
            final_json_output = to_json(log_entry)
            if len(final_json_output)>self.max_log_length: final_json_output=final_json_output[:self.max_log_length-5]+"...]}"
            print(final_json_output)
        except Exception as e: print(json.dumps({"error": f"Logging failed: {e}", "timestamp": state.timestamp}))
        self.logs = ""

logger = Logger()


# --- Base Strategy Class --- (Unchanged)
class BaseStrategy:
    def __init__(self, symbol: Symbol, limit: int, params: Dict) -> None:
        self.symbol = symbol
        self.limit = limit
        self.params = params
        self.orders: List[Order] = []

    def run(self, state: TradingState, trader_data_for_product: Dict) -> List[Order]:
        self.orders = []
        try:
            self.load_state(trader_data_for_product)
            self.act(state)
        except Exception as e:
            logger.print(f"ERROR running {self.symbol} strategy at ts {state.timestamp}: {e}")
            # Fallback: cancel orders if strategy errors
            self.orders = [Order(self.symbol, p, 0) for p in range(int(self.params.get("fair_value",1))-100,int(self.params.get("fair_value",1))+100) ] if self.symbol==Product.RAINFOREST_RESIN else []
            if self.symbol != Product.RAINFOREST_RESIN: # Attempt generic cancel if not Resin
                 ods = state.order_depths.get(self.symbol)
                 if ods:
                     if ods.buy_orders: self.orders.append(Order(self.symbol, max(ods.buy_orders.keys()), 0))
                     if ods.sell_orders: self.orders.append(Order(self.symbol, min(ods.sell_orders.keys()), 0))
        return self.orders

    @abstractmethod
    def act(self, state: TradingState) -> None: raise NotImplementedError()

    def buy(self, price: float, quantity: float) -> None:
        if quantity > 0: self.orders.append(Order(self.symbol, int(round(price)), int(round(quantity))))
    def sell(self, price: float, quantity: float) -> None:
        if quantity > 0: self.orders.append(Order(self.symbol, int(round(price)), -int(round(quantity))))
    def save_state(self) -> JSON: return None
    def load_state(self, data: JSON) -> None: pass

# --- EMA Calculation Helper --- (Unchanged)
def calculate_ema(current_price: float, prev_ema: Optional[float], alpha: float) -> float:
    if prev_ema is None: return current_price
    return alpha * current_price + (1 - alpha) * prev_ema

def get_current_midprice(order_depth: Optional[OrderDepth]) -> Optional[float]:
    if not order_depth: return None
    bids=order_depth.buy_orders; asks=order_depth.sell_orders;
    if not bids or not asks: return min(asks.keys()) if asks else (max(bids.keys()) if bids else None)
    best_ask=min(asks.keys()); best_bid=max(bids.keys());
    if best_ask not in asks or best_bid not in bids: return (best_ask + best_bid) / 2.0
    best_ask_vol=abs(asks[best_ask]); best_bid_vol=bids[best_bid]
    if best_ask_vol+best_bid_vol==0: return (best_ask+best_bid)/2.0
    return (best_bid*best_ask_vol+best_ask*best_bid_vol)/(best_bid_vol+best_ask_vol)

# --- Concrete Strategies ---

# Fixed Value Strategy (for Rainforest Resin) - (Unchanged)
class FixedValueStrategy(BaseStrategy):
    def act(self, state: TradingState) -> None:
        ods = state.order_depths.get(self.symbol)
        if not ods: return
        pos = state.position.get(self.symbol, 0)
        fair_value = self.params["fair_value"]
        make_spread_half = self.params["make_spread"] # Param is already half-spread
        aggr_size = self.params["aggr_size"]
        make_size = self.params["make_size"]
        take_width = self.params["take_width"]
        buy_cap = self.limit - pos; sell_cap = self.limit + pos
        best_ask = min(ods.sell_orders.keys()) if ods.sell_orders else float('inf')
        best_bid = max(ods.buy_orders.keys()) if ods.buy_orders else float('-inf')
        # Aggressive orders
        if best_ask <= fair_value - take_width and buy_cap > 0: vol=min(buy_cap, abs(ods.sell_orders[best_ask]), aggr_size); self.buy(best_ask, vol); buy_cap-=vol
        if best_bid >= fair_value + take_width and sell_cap > 0: vol=min(sell_cap, ods.buy_orders[best_bid], aggr_size); self.sell(best_bid, vol); sell_cap-=vol
        # Passive orders
        if buy_cap > 0: self.buy(fair_value - make_spread_half, min(buy_cap, make_size))
        if sell_cap > 0: self.sell(fair_value + make_spread_half, min(sell_cap, make_size))
    def save_state(self) -> JSON: return None
    def load_state(self, data: JSON) -> None: pass

# Original Market Making Logic (for Kelp) - (Unchanged)
class OriginalMarketMakingStrategy(BaseStrategy):
    def __init__(self, symbol: Symbol, limit: int, params: Dict) -> None:
        super().__init__(symbol, limit, params)
        self.ema_value: Optional[float] = None

    def act(self, state: TradingState) -> None:
        ods = state.order_depths.get(self.symbol)
        if not ods: return

        current_mid = get_current_midprice(ods)
        if current_mid is not None:
            self.ema_value = calculate_ema(current_mid, self.ema_value, self.params["ema_alpha"])
        elif self.ema_value is None: return

        fair_value = int(round(self.ema_value))
        buy_orders=sorted(ods.buy_orders.items(),reverse=True); sell_orders=sorted(ods.sell_orders.items())
        pos=state.position.get(self.symbol,0); buy_cap=self.limit-pos; sell_cap=self.limit+pos
        best_ask = min(sell_orders, key=lambda x: x[0])[0] if sell_orders else float('inf')
        best_bid = max(buy_orders, key=lambda x: x[0])[0] if buy_orders else float('-inf')

        take_width = self.params["take_width"]
        aggr_size = self.params["aggr_size"]
        make_size = self.params["make_size"]

        taken_buy=0
        if best_ask <= fair_value - take_width and buy_cap > 0: vol=min(abs(ods.sell_orders[best_ask]), buy_cap, aggr_size); self.buy(best_ask, vol); buy_cap-=vol; taken_buy+=vol
        taken_sell=0
        if best_bid >= fair_value + take_width and sell_cap > 0: vol=min(ods.buy_orders[best_bid], sell_cap, aggr_size); self.sell(best_bid, vol); sell_cap-=vol; taken_sell+=vol

        passive_buy_price = fair_value - 2 if pos > self.limit * 0.5 else fair_value - 1
        passive_sell_price = fair_value + 2 if pos < -self.limit * 0.5 else fair_value + 1

        if buy_cap > 0 and taken_buy == 0:
             final_buy_price = min(passive_buy_price, best_ask - 1) if best_ask != float('inf') else passive_buy_price
             self.buy(final_buy_price, min(buy_cap, make_size))
        if sell_cap > 0 and taken_sell == 0:
             final_sell_price = max(passive_sell_price, best_bid + 1) if best_bid != float('-inf') else passive_sell_price
             self.sell(final_sell_price, min(sell_cap, make_size))

    def save_state(self) -> JSON: return {"ema_value": self.ema_value}
    def load_state(self, data: JSON) -> None:
        if isinstance(data,dict): self.ema_value=data.get("ema_value", None)
        else: self.ema_value=None

# Inventory Skew Strategy (for Squid Ink) - OPTIMIZED + ENABLED CLEARING
class InventorySkewMMStrategy(BaseStrategy):
    def __init__(self, symbol: Symbol, limit: int, params: Dict) -> None:
        super().__init__(symbol, limit, params)
        self.ema_value: Optional[float] = None

    def act(self, state: TradingState) -> None:
        ods = state.order_depths.get(self.symbol)
        if not ods: return

        current_mid = get_current_midprice(ods)
        if current_mid is not None:
            self.ema_value = calculate_ema(current_mid, self.ema_value, self.params["ema_alpha"])
        elif self.ema_value is None: # Need an EMA value to proceed
            # Try to initialize EMA from trades if available and EMA is None
            market_trades = state.market_trades.get(self.symbol, [])
            own_trades = state.own_trades.get(self.symbol, [])
            all_trades = sorted(market_trades + own_trades, key=lambda t: t.timestamp)
            if all_trades:
                self.ema_value = all_trades[-1].price
            else:
                 return # Still cannot proceed

        # Ensure ema_value is not None here
        if self.ema_value is None:
             logger.print(f"ERROR: {self.symbol} EMA is still None at {state.timestamp}")
             return

        fair_value = self.ema_value # Use float EMA directly for calculations before rounding
        buy_orders=sorted(ods.buy_orders.items(),reverse=True); sell_orders=sorted(ods.sell_orders.items())
        pos=state.position.get(self.symbol,0); buy_cap=self.limit-pos; sell_cap=self.limit+pos
        best_ask = min(sell_orders, key=lambda x: x[0])[0] if sell_orders else float('inf')
        best_bid = max(buy_orders, key=lambda x: x[0])[0] if buy_orders else float('-inf')

        # Read parameters
        take_width = self.params["take_width"]
        clear_width = self.params["clear_width"]
        make_spread_half = self.params["make_spread"] # Param is already half-spread
        aggr_size = self.params["aggr_size"]
        make_size = self.params["make_size"]
        skew_factor = self.params["skew_factor"]
        max_pos_ratio_passive = self.params.get("max_pos_ratio_passive", 1.0) # Default to 1 if not set
        clear_pos_ratio = self.params.get("clear_pos_ratio", 0.6) # Default to 0.6 if not set

        # Calculate skew based on position and factor
        # Skew affects the midpoint around which we quote passively
        price_skew = pos * skew_factor # Simplified skew factor application relative to position size
        skewed_fair_value = fair_value - price_skew # If long (pos>0), lower quote midpoint; if short (pos<0), raise it

        # --- Aggressive Orders (Taking Liquidity) ---
        taken_buy = 0
        if best_ask <= fair_value - take_width and buy_cap > 0:
            vol = min(abs(ods.sell_orders[best_ask]), buy_cap, aggr_size)
            self.buy(best_ask, vol)
            buy_cap -= vol
            taken_buy += vol
            # logger.print(f"TAKE BUY {self.symbol}: {vol}@{best_ask} (FV: {fair_value:.1f})")

        taken_sell = 0
        if best_bid >= fair_value + take_width and sell_cap > 0:
            vol = min(ods.buy_orders[best_bid], sell_cap, aggr_size)
            self.sell(best_bid, vol)
            sell_cap -= vol
            taken_sell += vol
            # logger.print(f"TAKE SELL {self.symbol}: {vol}@{best_bid} (FV: {fair_value:.1f})")


        # --- Passive Orders (Making Liquidity) ---
        # Only place passive orders if position is within the passive limit ratio
        if abs(pos) < self.limit * max_pos_ratio_passive:
            pass_buy_p = skewed_fair_value - make_spread_half
            pass_sell_p = skewed_fair_value + make_spread_half

            # Improve order placement - ensure price level is valid (integer) and avoid crossing BBO
            pass_buy_p_int = int(round(pass_buy_p))
            pass_sell_p_int = int(round(pass_sell_p))

            # Adjust to be inside BBO if possible, but don't cross our own midpoint logic
            if best_ask != float('inf'): pass_buy_p_int = min(pass_buy_p_int, best_ask - 1)
            if best_bid != float('-inf'): pass_sell_p_int = max(pass_sell_p_int, best_bid + 1)

            # Ensure buy < sell after adjustments
            if pass_buy_p_int < pass_sell_p_int:
                 # Place orders only if we haven't taken aggressively on that side
                 if buy_cap > 0 and taken_buy == 0:
                     self.buy(pass_buy_p_int, min(buy_cap, make_size))
                     # logger.print(f"MAKE BUY {self.symbol}: {min(buy_cap, make_size)}@{pass_buy_p_int} (SkewFV: {skewed_fair_value:.1f})")
                 if sell_cap > 0 and taken_sell == 0:
                     self.sell(pass_sell_p_int, min(sell_cap, make_size))
                     # logger.print(f"MAKE SELL {self.symbol}: {min(sell_cap, make_size)}@{pass_sell_p_int} (SkewFV: {skewed_fair_value:.1f})")

        # --- Inventory Clearing Logic (Aggressive reduction of large positions) ---
        # Activated only if clear_width > 0
        if clear_width > 0:
             # Clear LONG position if too large
             if pos > self.limit * clear_pos_ratio and sell_cap > 0:
                 clear_price = int(round(fair_value - clear_width)) # Aim below fair value
                 # Try to hit best bid if it's better than our target clear price
                 if best_bid > float('-inf'): clear_price = max(clear_price, best_bid)

                 # Calculate volume to clear - aim to reduce position, use make_size for volume
                 vol_to_clear = min(pos - int(self.limit * clear_pos_ratio * 0.5), sell_cap, make_size) # Target slightly below threshold
                 if vol_to_clear > 0:
                     self.sell(clear_price, vol_to_clear)
                     # logger.print(f"CLEAR LONG {self.symbol}: {vol_to_clear}@{clear_price} (Pos: {pos})")

             # Clear SHORT position if too large
             elif pos < -self.limit * clear_pos_ratio and buy_cap > 0:
                  clear_price = int(round(fair_value + clear_width)) # Aim above fair value
                  # Try to hit best ask if it's better than our target clear price
                  if best_ask < float('inf'): clear_price = min(clear_price, best_ask)

                  # Calculate volume to clear - aim to reduce position, use make_size for volume
                  vol_to_clear = min(abs(pos) - int(self.limit * clear_pos_ratio * 0.5), buy_cap, make_size) # Target slightly below threshold
                  if vol_to_clear > 0:
                     self.buy(clear_price, vol_to_clear)
                     # logger.print(f"CLEAR SHORT {self.symbol}: {vol_to_clear}@{clear_price} (Pos: {pos})")


    def save_state(self) -> JSON: return {"ema_value": self.ema_value}
    def load_state(self, data: JSON) -> None:
        if isinstance(data,dict): self.ema_value=data.get("ema_value", None)
        else: self.ema_value=None


# --- Main Trader Class --- (Unchanged Structure, uses updated params/strategies)
class Trader:
    def __init__(self, params=None):
        if params is None: params = PARAMS # Use updated PARAMS
        self.params = params
        self.LIMIT = { # Define limits for Round 1 products
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50
        }
        # Instantiate strategies using the correct classes based on performance
        self.strategies: Dict[Symbol, BaseStrategy] = {}
        if Product.RAINFOREST_RESIN in self.params:
            self.strategies[Product.RAINFOREST_RESIN] = FixedValueStrategy(Product.RAINFOREST_RESIN, self.LIMIT[Product.RAINFOREST_RESIN], self.params[Product.RAINFOREST_RESIN])
        if Product.KELP in self.params:
            self.strategies[Product.KELP] = OriginalMarketMakingStrategy(Product.KELP, self.LIMIT[Product.KELP], self.params[Product.KELP])
        if Product.SQUID_INK in self.params:
             # Ensure InventorySkewMMStrategy is used with the updated SQUID_INK params
             self.strategies[Product.SQUID_INK] = InventorySkewMMStrategy(Product.SQUID_INK, self.LIMIT[Product.SQUID_INK], self.params[Product.SQUID_INK])

        self.trader_data_cache = {} # Cache for strategy states

    def load_trader_data(self, traderData: str):
        if not traderData: self.trader_data_cache = {}
        else:
            try: self.trader_data_cache = jsonpickle.decode(traderData)
            except Exception as e: logger.print(f"ERROR decoding traderData: {e}"); self.trader_data_cache = {}
        # Ensure sub-dictionaries exist for each strategy needing state
        for symbol, strategy in self.strategies.items():
            # Check if the strategy actually saves state before creating an empty dict
             if hasattr(strategy, 'save_state') and callable(strategy.save_state):
                 # Check if save_state requires arguments (it shouldn't based on design)
                 # A simple check if it needs state is if its save_state returns non-None
                 # We'll rely on the save_trader_data logic to populate the cache correctly
                 # Here, we just ensure the top-level key exists if needed later
                 if symbol not in self.trader_data_cache and isinstance(self.trader_data_cache, dict):
                    # Add only if state saving might occur (EMA strategies)
                    if isinstance(strategy, (OriginalMarketMakingStrategy, InventorySkewMMStrategy)):
                       self.trader_data_cache[symbol] = {}


    def save_trader_data(self) -> str:
        # Ensure trader_data_cache is a dict
        if not isinstance(self.trader_data_cache, dict): self.trader_data_cache = {}

        for symbol, strategy in self.strategies.items():
             if hasattr(strategy, 'save_state'):
                saved = strategy.save_state()
                if saved is not None:
                     self.trader_data_cache[symbol] = saved # Update or add state
                 # If save_state returns None (like FixedValueStrategy), remove old state if present
                elif symbol in self.trader_data_cache:
                     del self.trader_data_cache[symbol]

        try:
            # Use ensure_ascii=False for potentially better compression if non-ASCII appears (unlikely here)
            return jsonpickle.encode(self.trader_data_cache, unpicklable=False)
        except Exception as e:
            logger.print(f"ERROR encoding traderData: {e}")
            return "" # Return empty string on error


    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        self.load_trader_data(state.traderData)
        all_orders: Dict[Symbol, List[Order]] = {}
        conversions = 0 # No conversions in Round 1

        for symbol, strategy in self.strategies.items():
            product_orders: List[Order] = []
            # Ensure order depths exist for the symbol before running strategy
            if symbol in state.order_depths:
                # Pass the specific part of the cache relevant to this product's strategy
                product_state_data = self.trader_data_cache.get(symbol, {}) if isinstance(self.trader_data_cache, dict) else {}
                product_orders = strategy.run(state, product_state_data)
                # Strategy internally updates its state; save_state will retrieve it later
            all_orders[symbol] = product_orders

        # Update the state cache *after* all strategies have run for this timestamp
        traderData_output = self.save_trader_data()

        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output