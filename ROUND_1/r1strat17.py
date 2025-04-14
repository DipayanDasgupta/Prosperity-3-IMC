import json
import math
from collections import deque, defaultdict
from typing import Any, TypeAlias, List, Dict, Optional, Tuple
from abc import abstractmethod
import jsonpickle # Make sure this is imported
import numpy as np # Keep for potential future use

# Assuming datamodel classes are available as in the competition environment
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# Type Alias for JSON data
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Product:
    # Define product names for Round 1
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

# Parameters dictionary - Tuned based on previous backtests
PARAMS = {
    Product.RAINFOREST_RESIN: { # Using FixedValueStrategy
        "fair_value": 10000,
        "take_width": 1,
        "make_spread": 2, # Half-spread (total 4)
        "aggr_size": 2,
        "make_size": 5,
    },
    Product.KELP: { # Using OriginalMarketMakingStrategy
        "ema_alpha": 0.3,
        "take_width": 1,
        "aggr_size": 3,
        "make_size": 4,
    },
    Product.SQUID_INK: { # Using InventorySkewMMStrategyV21
        "ema_alpha": 0.25,
        "take_width": 2,
        "clear_width": 2,  # Re-enabled clearing, trigger earlier
        "make_spread": 2.5, # Base half-spread
        "aggr_size": 2,
        "make_size": 3,
        "skew_factor": 0.30, # Increased skew factor
        "max_pos_ratio_passive": 0.80, # Stop passive quoting slightly earlier
        "clear_pos_ratio": 0.5 # Start clearing if |pos| > 50% limit
    }
}


# Logger class (Defined before use)
class Logger:
    # (Keep the full Logger class implementation from previous correct responses)
    def __init__(self) -> None: self.logs = ""; self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None: self.logs += sep.join(map(str, objects)) + end
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # --- Full Compression Logic ---
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
            compressed_conversion_observations = {} # Empty for R1
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
        # --- End Compression Helpers ---
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

# --- Base Strategy Class ---
class BaseStrategy:
    # (Keep implementation as before)
    def __init__(self, symbol: Symbol, limit: int, params: Dict) -> None: self.symbol = symbol; self.limit = limit; self.params = params; self.orders: List[Order] = []
    def run(self, state: TradingState, trader_data_for_product: Dict) -> List[Order]:
        self.orders = []
        try: self.load_state(trader_data_for_product); self.act(state)
        except Exception as e: logger.print(f"ERROR {self.symbol}: {e}"); self.orders = []
        return self.orders
    @abstractmethod
    def act(self, state: TradingState) -> None: raise NotImplementedError()
    def buy(self, price: float, quantity: float) -> None:
        if quantity > 0: self.orders.append(Order(self.symbol, int(round(price)), int(round(quantity))))
    def sell(self, price: float, quantity: float) -> None:
        if quantity > 0: self.orders.append(Order(self.symbol, int(round(price)), -int(round(quantity))))
    def save_state(self) -> JSON: return None
    def load_state(self, data: JSON) -> None: pass

# --- EMA Calculation Helper ---
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

class FixedValueStrategy(BaseStrategy):
    """Basic MM for fixed fair value products like Rainforest Resin."""
    def act(self, state: TradingState) -> None:
        ods = state.order_depths.get(self.symbol)
        if not ods: return
        pos = state.position.get(self.symbol, 0)
        fair_value = self.params["fair_value"]
        make_spread_half = self.params["make_spread"] / 2
        aggr_size = self.params["aggr_size"]
        make_size = self.params["make_size"]
        take_width = self.params["take_width"]
        buy_cap = self.limit - pos; sell_cap = self.limit + pos
        best_ask = min(ods.sell_orders.keys()) if ods.sell_orders else float('inf')
        best_bid = max(ods.buy_orders.keys()) if ods.buy_orders else float('-inf')
        if best_ask <= fair_value - take_width and buy_cap > 0: vol=min(buy_cap, abs(ods.sell_orders[best_ask]), aggr_size); self.buy(best_ask, vol); buy_cap-=vol
        if best_bid >= fair_value + take_width and sell_cap > 0: vol=min(sell_cap, ods.buy_orders[best_bid], aggr_size); self.sell(best_bid, vol); sell_cap-=vol
        if buy_cap > 0: self.buy(fair_value - make_spread_half, min(buy_cap, make_size))
        if sell_cap > 0: self.sell(fair_value + make_spread_half, min(sell_cap, make_size))
    def save_state(self) -> JSON: return None
    def load_state(self, data: JSON) -> None: pass


class OriginalMarketMakingStrategy(BaseStrategy):
    """Market making using EMA and simple inventory thresholds (for Kelp)."""
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

        # Passive Orders - Pricing based on inventory thresholds
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


class InventorySkewMMStrategy(BaseStrategy):
    """Market making using EMA and aggressive inventory skewing (for Squid Ink)."""
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

        take_width=self.params["take_width"]; clear_width=self.params["clear_width"]; make_spread_half=self.params["make_spread"]
        aggr_size=self.params["aggr_size"]; make_size=self.params["make_size"]; skew_factor=self.params["skew_factor"]
        max_pos_ratio=self.params.get("max_pos_ratio_passive", 1.0)
        clear_pos_ratio = self.params.get("clear_pos_ratio", 0.5) # Use param for clearing threshold

        price_skew = int(round(pos * skew_factor * make_spread_half * 2))
        skewed_fair_value = fair_value - price_skew

        taken_buy=0
        if best_ask <= fair_value - take_width and buy_cap > 0: vol=min(abs(ods.sell_orders[best_ask]),buy_cap,aggr_size); self.buy(best_ask,vol); buy_cap-=vol; taken_buy+=vol
        taken_sell=0
        if best_bid >= fair_value + take_width and sell_cap > 0: vol=min(ods.buy_orders[best_bid],sell_cap,aggr_size); self.sell(best_bid,vol); sell_cap-=vol; taken_sell+=vol

        pass_buy_p = skewed_fair_value - make_spread_half
        pass_sell_p = skewed_fair_value + make_spread_half
        pass_buy_p = int(round(min(pass_buy_p, best_ask - 1))) if best_ask != float('inf') else int(round(pass_buy_p))
        pass_sell_p = int(round(max(pass_sell_p, best_bid + 1))) if best_bid != float('-inf') else int(round(pass_sell_p))

        if abs(pos) < self.limit * max_pos_ratio:
            if buy_cap > 0 and taken_buy==0: self.buy(pass_buy_p, min(buy_cap, make_size))
            if sell_cap > 0 and taken_sell==0: self.sell(pass_sell_p, min(sell_cap, make_size))

        # Clearing Logic (Trigger earlier based on clear_pos_ratio)
        if clear_width > 0:
             if pos > self.limit * clear_pos_ratio and sell_cap > 0: # Heavy long
                 clear_price = int(round(fair_value - clear_width)); clear_price = max(clear_price, best_bid + 1) if best_bid > float('-inf') else clear_price
                 vol=min(pos - int(self.limit * (clear_pos_ratio*0.5)), sell_cap, make_size*2) # Try to clear more aggressively back towards neutral/half limit
                 if vol > 0: self.sell(clear_price, vol)
             elif pos < -self.limit * clear_pos_ratio and buy_cap > 0: # Heavy short
                  clear_price = int(round(fair_value + clear_width)); clear_price = min(clear_price, best_ask - 1) if best_ask < float('inf') else clear_price
                  vol=min(abs(pos) - int(self.limit * (clear_pos_ratio*0.5)), buy_cap, make_size*2) # Try to clear more aggressively
                  if vol > 0: self.buy(clear_price, vol)

    def save_state(self) -> JSON: return {"ema_value": self.ema_value}
    def load_state(self, data: JSON) -> None:
        if isinstance(data,dict): self.ema_value=data.get("ema_value", None)
        else: self.ema_value=None


# --- Main Trader Class ---
class Trader:
    def __init__(self, params=None):
        if params is None: params = PARAMS
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
             self.strategies[Product.SQUID_INK] = InventorySkewMMStrategy(Product.SQUID_INK, self.LIMIT[Product.SQUID_INK], self.params[Product.SQUID_INK])

        self.trader_data_cache = {} # Cache for strategy states

    def load_trader_data(self, traderData: str):
        if not traderData: self.trader_data_cache = {}
        else:
            try: self.trader_data_cache = jsonpickle.decode(traderData)
            except Exception as e: logger.print(f"ERROR decoding traderData: {e}"); self.trader_data_cache = {}
        for symbol in self.strategies:
            if hasattr(self.strategies[symbol], 'save_state') and self.strategies[symbol].save_state() is not None:
                 if symbol not in self.trader_data_cache: self.trader_data_cache[symbol] = {}

    def save_trader_data(self) -> str:
        for symbol, strategy in self.strategies.items():
             if hasattr(strategy, 'save_state'):
                saved = strategy.save_state()
                if saved is not None: self.trader_data_cache[symbol] = saved
        try: return jsonpickle.encode(self.trader_data_cache, unpicklable=False)
        except Exception as e: logger.print(f"ERROR encoding traderData: {e}"); return ""

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        self.load_trader_data(state.traderData)
        all_orders: Dict[Symbol, List[Order]] = {}
        conversions = 0

        for symbol, strategy in self.strategies.items():
            product_orders: List[Order] = []
            if symbol in state.order_depths:
                product_state_data = self.trader_data_cache.get(symbol, {})
                product_orders = strategy.run(state, product_state_data)
                updated_state = strategy.save_state()
                if updated_state is not None: self.trader_data_cache[symbol] = updated_state
            all_orders[symbol] = product_orders

        traderData_output = self.save_trader_data()
        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output