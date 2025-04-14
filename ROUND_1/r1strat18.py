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

# Parameters dictionary - Using parameters aligned with the chosen
    def save_state(self) -> JSON: return None
    def load_state(self, data: JSON) -> None: pass


# Original Market Making Logic (for Kelp) - Reverted to simpler logic
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
        buy_orders=sorted(ods.buy best strategies
PARAMS = {
    Product.RAINFOREST_RESIN: { # For FixedValueStrategy
        "fair_value": 10000,
        "take_width": 1,
        "make_spread": 2, # Half-spread
        "aggr_size": 2,
        "make_size": 5,
    },
    Product.KELP: { # For OriginalMarketMakingStrategy
        "ema_alpha": 0.3,
        "take_width": 1,
        "aggr_size": 3,
        "make_size": 4,
        # Note: Spread/skew uses built-in position thresholds
    },
    Product.SQUID_INK: { # For InventorySkewMMStrategy (V19 logic)
        "ema_alpha": 0.25,
        "take_width": 2,
        "clear_width": 0, # Keep clearing disabled for now
        "make_spread": 2.5, # Base half-spread
        "aggr_size": 2,
        "make_size": 3,
        _orders.items(),reverse=True); sell_orders=sorted(ods.sell_orders.items())
        pos=state.position.get(self.symbol,0); buy_cap=self.limit-pos; sell_cap=self.limit+pos
        best_ask = min(sell_orders, key=lambda x: x[0])[0] if sell_orders else float('inf')
        best_bid = max(buy_orders, key=lambda x: x[0])[0] if buy"skew_factor": 0.25, # Inventory skew factor
        "max_pos_ratio_passive": 0.85 # Stop passive quoting threshold
    }
}


# Logger class (Defined before use)
class Logger:
_orders else float('-inf')

        take_width = self.params    # (Keep the full Logger class implementation from previous correct responses)
    def __init__(self) -> None: self.logs = ""; self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None: self.logs += sep.join(map(str, objects["take_width"]
        aggr_size = self.params[")) + end
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # --- Full Compression Logic ---
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
            compressed_conversion_observations = {}aggr_size"]
        make_size = self.params["make_size"]

        taken_buy=0
        if best_ask <= fair_value - take
            if hasattr(state, 'observations') and state.observations is not None and hasattr(state.observations, 'conversionObservations') and state.observations.conversionObservations is not None:
                 for product, observation in state.observations_width and buy_cap > 0: vol=min(abs(ods.sell_orders[best_ask]), buy_cap, aggr_size); self.buy(best_ask, vol); buy_cap-=vol; taken_buy+=vol
        taken_sell=0
        if best_bid >= fair_value + take.conversionObservations.items():
                     if observation is not None: compressed_conversion_observations[product]=[getattr(observation,'bidPrice',None),getattr_width and sell_cap > 0: vol=min(ods.buy_orders[best_bid(observation,'askPrice',None),getattr(observation,'transportFees',0], sell_cap, aggr_size); self.sell(best_bid, vol); sell_cap-=vol; taken_sell+=vol

        # Original Passive Logic: Price based on simple inventory threshold
        passive_buy_price =.0),getattr(observation,'exportTariff',0.0),getattr(observation,'importTariff',0.0),getattr(observation,'sun fair_value - 2 if pos > self.limit * 0.5 else fair_value - 1
        passive_sell_price =light',None),getattr(observation,'humidity',None)]
                     else: compressed_conversion_observations[product]=[None]*7
            plain_obs = state.observations.plainValueObservations if hasattr(state.observations, ' fair_value + 2 if pos < -self.limit * 0plainValueObservations') and state.observations is not None else {}
            compressed.5 else fair_value + 1

        # Ensure passive orders don't cross best market price
        final_buy_price = min(passive_buy_price, best_ask - 1) if best_ask != float('inf') else passive__observations = [plain_obs, compressed_conversion_observations]
            return [
                state.timestamp, trader_data, compress_listings(state.listings),
                compress_order_depths(state.order_depths), compressbuy_price
        final_sell_price = max(passive_sell_price, best_bid + 1) if best_bid != float('-inf') else_trades(state.own_trades or {}),
                compress_trades(state.market_trades or {}), state.position or {}, compressed_observations,
            ]
        def compress_ passive_sell_price

        if buy_cap > 0 and takenlistings(listings: dict[Symbol, Listing]) -> list[list[Any]]:
            compressed = [];
            if listings:
                for listing in listings.values():
                     if listing: compressed.append([listing.symbol, listing.product, listing.denomination])
            return_buy == 0: self.buy(final_buy_price, min(buy_cap, make_size))
        if sell_cap compressed
        def compress_order_depths(order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
            compressed = {};
            if order_depths:
                for symbol, order_depth > 0 and taken_sell == 0: self.sell(final_sell_price, min(sell_cap, make_size))

    def in order_depths.items():
                    if order_depth: buy_orders_dict=order_depth.buy_orders if isinstance(order save_state(self) -> JSON: return {"ema_value": self.ema_value}
_depth.buy_orders, dict) else {}; sell_orders_dict=order_depth.sell_orders if isinstance(order_depth.sell_orders,dict) else {}; compressed[symbol]=[buy_orders_dict, sell_orders_dict]    def load_state(self, data: JSON) -> None:
        if isinstance(data,dict): self.ema_value=data.
                    else: compressed[symbol] = [{}, {}]
            return compressed
        def compress_get("ema_value", None)
        else: self.ema_value=None


# Inventory Skew Strategy (for Squid Ink) - Keep this logic
class InventorySkewMMStrategy(BaseStrategy):
    def __init__(self, symbol: Symbol, limit: int, params: Dict) -> None:
        trades(trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
            compressed = [];
            if trades:
                for arr in trades.values():super().__init__(symbol, limit, params)
        self.ema_
                     if arr:
                        for trade in arr:
                             if trade: compressed.append([trade.symbol,trade.price,trade.quantity,trade.buyer or "",trade.seller or "",tradevalue: Optional[float] = None

    def act(self, state: TradingState) -> None:
        ods = state.order_depth.timestamp])
            return compressed
        def compress_orders(orders:s.get(self.symbol)
        if not ods: return

 dict[Symbol, list[Order]]) -> list[list[Any]]:
            compressed = [];
            if orders:
                for arr in orders.        current_mid = get_current_midprice(ods)
        if current_mid is not None:
            self.ema_value = calculate_ema(currentvalues():
                    if arr:
                        for order in arr:
                            if order: compressed.append([order.symbol, order.price, order_mid, self.ema_value, self.params["ema_alpha"])
        elif self.ema_value is None: return

        fair.quantity])
            return compressed
        def to_json(value: Any) -> str:
             try: return json.dumps(value,_value = int(round(self.ema_value))
        buy_orders=sorted(ods.buy_orders.items(),reverse=True); sell_orders=sorted(ods.sell_orders.items())
        pos=state.position. cls=ProsperityEncoder, separators=(",", ":"))
             except Exceptionget(self.symbol,0); buy_cap=self.limit-pos; sell_cap: return json.dumps(value, separators=(",", ":"))
        =self.limit+pos
        best_ask = min(sell_orders, key=lambdadef truncate(value: str, max_length: int) -> str: x: x[0])[0] if sell_orders else float('inf
            if not isinstance(value, str): value = str(value)')
        best_bid = max(buy_orders, key=lambda x: x[0])[0] if buy_orders else float('-inf
            if len(value) <= max_length: return value
            ')

        take_width=self.params["take_width"]; clear_width=self.params["clear_width"]; make_spread_halfreturn value[:max_length - 3] + "..."
        # --- End Compression Helpers ---
        try:
            min_obs=Observation({}, {}); min_state=TradingState(state.timestamp,"",{},{},{},{},{},min_obs)
=self.params["make_spread"]
        aggr_size=self.params["aggr_size"]; make_size=self.params["make_size"]; skew_factor=self.params["skew_factor"]
        max_pos_ratio=self.params.get("max_pos_ratio_passive", 1.0); clear_pos_ratio = self.params.get            base_value=[compress_state(min_state,""),compress_orders({}),conversions,"",""]; base_json=to_json(base_value); base_("clear_pos_ratio", 0.5) # Use param

        price_skew = int(length=len(base_json)
            available_length=self.max_log_length-base_length-200;
            if available_length < 0round(pos * skew_factor * make_spread_half * 2))
        skewed_fair_value = fair_value - price_skew

        taken_buy=: available_length=0
            max_item_length=available_length//3
            truncated_trader_data_state=truncate(state.traderData if state.traderData else0
        if best_ask <= fair_value - take_width and "", max_item_length)
            truncated_trader_data_out=truncate(trader_data, max_item_length); truncated_logs=truncate(self. buy_cap > 0: vol=min(abs(ods.selllogs, max_item_length)
            log_entry=[compress_state(state, truncated_trader_data_state), compress_orders(orders), conversions, truncated_trader_data_out, truncated_logs]
_orders[best_ask]),buy_cap,aggr_size);            final_json_output = to_json(log_entry)
 self.buy(best_ask,vol); buy_cap-=vol; taken_buy+=vol
        taken_sell=0
        if best_bid >= fair_value + take_width and sell_cap >             if len(final_json_output)>self.max_log_0: vol=min(ods.buy_orders[best_bid],sell_cap,aggr_length: final_json_output=final_json_output[:self.size); self.sell(best_bid,vol); sell_cap-=max_log_length-5]+"...]}"
            print(final_json_output)
        except Exception as e: print(json.dumpsvol; taken_sell+=vol

        pass_buy_p = skewed_fair_value - make_spread_half
        pass_sell_({"error": f"Logging failed: {e}", "timestamp": state.p = skewed_fair_value + make_spread_half
        pass_buy_p = int(round(min(pass_buy_p, best_ask - 1))) if best_ask != float('inf') else int(round(pass_buy_p))
        pass_sell_ptimestamp}))
        self.logs = ""

logger = Logger()


# --- Base Strategy Class ---
class BaseStrategy:
    # (Keep implementation as before)
 = int(round(max(pass_sell_p, best_bid + 1))) if best_bid != float('-inf') else int(round(pass_sell_p))

        if abs(pos) < self.limit * max_pos_ratio:
                def __init__(self, symbol: Symbol, limit: int, params: Dict) -> None: self.symbol = symbol; self.limit =if buy_cap > 0 and taken_buy==0: self.buy(pass_buy_p, min(buy_cap, make_size))
            if sell_cap > 0 and taken_sell==0: self.sell(pass_sell limit; self.params = params; self.orders: List[Order] = []
    def run(self, state: TradingState, trader_data_for_product: Dict) -> List[Order]:
        self.orders = []
        try: self.load_state_p, min(sell_cap, make_size))

        if clear_width > 0:
             if pos > self.limit * clear_pos_ratio and sell_cap > 0:
                 clear_price = int(round(trader_data_for_product); self.act(state)
        except Exception as e:(fair_value - clear_width)); clear_price = max(clear_price, best_bid + 1) if best_bid > float('-inf') else clear_price
                 vol=min(pos - int(self.limit * ( logger.print(f"ERROR {self.symbol}: {e}"); self.orders = []
        return self.orders
    @abstractmethod
    def act(self, state: TradingState) -> None: raise NotImplementedError()
    def buy(self, price:clear_pos_ratio*0.5)), sell_cap, make_size* float, quantity: float) -> None:
        if quantity > 0: self.orders.append(Order(self.symbol, int(round(price)), int2) # Clear more aggressively
                 if vol > 0: self.sell(clear_price, vol)
             elif pos < -self.limit * clear_pos_(round(quantity))))
    def sell(self, price: float,ratio and buy_cap > 0:
                  clear_price = int(round( quantity: float) -> None:
        if quantity > 0: self.orders.append(Order(self.symbol, int(round(pricefair_value + clear_width)); clear_price = min(clear_price, best_ask - 1) if best_ask < float(')), -int(round(quantity))))
    def save_state(self) -> JSON: returninf') else clear_price
                  vol=min(abs(pos) - int(self None
    def load_state(self, data: JSON) -> None: pass


# --- EMA Calculation Helper ---
def calculate_ema(current_price: float, prev.limit * (clear_pos_ratio*0.5)), buy_cap, make_size*_ema: Optional[float], alpha: float) -> float:
    2); # Clear more aggressively
                  if vol > 0: self.buy(clear_price,if prev_ema is None: return current_price
    return alpha * current_price + (1 - alpha) * prev_ema

def get_current_midprice(order_depth: Optional[OrderDepthvol)

    def save_state(self) -> JSON: return {"ema_value": self.ema_value}
    def load_state(self,]) -> Optional[float]:
    if not order_depth: return None
    bids= data: JSON) -> None:
        if isinstance(data,dict): self.ema_value=data.get("ema_value", None)order_depth.buy_orders; asks=order_depth.sell_
        else: self.ema_value=None


# --- Main Traderorders;
    if not bids or not asks: return min(asks.keys()) if asks else ( Class ---
class Trader:
    def __init__(self, params=None):
        if params is None: params = PARAMS
        self.params = params
        max(bids.keys()) if bids else None)
    best_ask=min(asks.keys()); best_bid=max(bidsself.LIMIT = { # Define limits for Round 1 products
            Product.RAINFOREST_RES.keys());
    if best_ask not in asks or best_bid notIN: 50,
            Product.KELP: 50,
            Product.SQUID in bids: return (best_ask + best_bid) / 2._INK: 50
        }
        # Instantiate strategies using the correct classes based on performance
        0
    best_ask_vol=abs(asks[best_askself.strategies: Dict[Symbol, BaseStrategy] = {}
        if Product.RAINFOREST_RESIN in self.params:
            self]); best_bid_vol=bids[best_bid]
    if best_ask.strategies[Product.RAINFOREST_RESIN] = FixedValueStrategy_vol+best_bid_vol==0: return (best_ask+best_bid(Product.RAINFOREST_RESIN, self.LIMIT[Product.RAINFOREST_RESIN], self.params[Product.RAINFOREST)/2.0
    return (best_bid*best_ask_vol+best_ask*best_bid_vol)/(best_bid_vol+best_ask_vol)

# --- Concrete Strategies ---

# Fixed Value Strategy (for Rainforest Resin)
_RESIN])
        if Product.KELP in self.params:class FixedValueStrategy(BaseStrategy):
    def act(self, state
             # Using OriginalMarketMakingStrategy for Kelp
            self.strategies[Product.KELP] = OriginalMarketMakingStrategy(Product.KELP,: TradingState) -> None:
        ods = state.order_depth self.LIMIT[Product.KELP], self.params[Product.KELs.get(self.symbol)
        if not ods: return
        pos = state.position.get(self.symbol, 0)
        fair_value = self.params["fair_value"]
        make_spread_half =P])
        if Product.SQUID_INK in self.params: self.params["make_spread"] / 2
        aggr_size = self.
             # Using InventorySkewMMStrategy for Squid Ink
             self.strategiesparams["aggr_size"]
        make_size = self.params["make_size"]
        take_width = self.params["take_width"]
        buy_cap = self.limit - pos; sell[Product.SQUID_INK] = InventorySkewMMStrategy(Product.SQUID_INK, self.LIMIT[Product.SQUID_INK], self.params[Product.SQUID_INK])

        self.tr_cap = self.limit + pos
        best_ask = min(odsader_data_cache = {} # Cache for strategy states

    # (Keep load_trader_data, save_trader_data, and run methods exactly as in the previous version)
    def load_trader_data(self,.sell_orders.keys()) if ods.sell_orders else float(' traderData: str):
        if not traderData: self.traderinf')
        best_bid = max(ods.buy_orders._data_cache = {}
        else:
            try: self.keys()) if ods.buy_orders else float('-inf')
        if best_ask <=trader_data_cache = jsonpickle.decode(traderData)
            except Exception as e: logger.print(f"ERROR decoding traderData: {e}"); self.trader_data_cache = fair_value - take_width and buy_cap > 0: vol {}
        for symbol in self.strategies:
            if hasattr(self=min(buy_cap, abs(ods.sell_orders[best_ask]), ag.strategies[symbol], 'save_state') and self.strategies[symbol].save_state() is not None:
                 if symbol not in selfgr_size); self.buy(best_ask, vol); buy_cap-=vol
        if best_bid >= fair_value + take_width and sell_cap > 0: vol.trader_data_cache: self.trader_data_cache=min(sell_cap, ods.buy_orders[best_bid], aggr_size); self.sell(best_bid, vol); sell_cap-=vol
[symbol] = {}

    def save_trader_data(self) -> str:
        for symbol, strategy in self.strategies.items():
             if hasattr(strategy, 'save_state'):
                saved = strategy.save_state()
                if saved is not None: self.trader        if buy_cap > 0: self.buy(fair_value - make_spread_half_data_cache[symbol] = saved
        try: return jsonpickle, min(buy_cap, make_size))
        if sell_.encode(self.trader_data_cache, unpicklable=cap > 0: self.sell(fair_value + make_spread_halfFalse)
        except Exception as e: logger.print(f"ERROR, min(sell_cap, make_size))
    def save_state(self) -> JSON: return None
    def load_state(self, data: JSON) -> None: pass


# Original Market Making Logic (for Kelp)
 encoding traderData: {e}"); return ""

    def run(self,class OriginalMarketMakingStrategy(BaseStrategy):
    def __init__(self state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        self.load_trader_data(state, symbol: Symbol, limit: int, params: Dict) -> None:.traderData)
        all_orders: Dict[Symbol, List[Order]] = {}
        conversions = 0
        for symbol, strategy in self.strategies.items():
            product_orders: List[Order] = []
            if symbol in state.
        super().__init__(symbol, limit, params)
        self.order_depths:
                product_state_data = self.trader_data_cache.get(symbol, {})
                product_orders = strategy.run(stateema_value: Optional[float] = None

    def act(self, state: TradingState) -> None:
        ods = state.order, product_state_data)
                updated_state = strategy.save_state()
                if_depths.get(self.symbol)
        if not ods: updated_state is not None: self.trader_data_cache[ return

        current_mid = get_current_midprice(ods)
        if current_mid issymbol] = updated_state
            all_orders[symbol] = product_orders

        tr not None:
            self.ema_value = calculate_ema(current_mid, self.ema_value, self.params["ema_alphaaderData_output = self.save_trader_data()
        "])
        elif self.ema_value is None: return

        fairlogger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output