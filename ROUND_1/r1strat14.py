import json
import math
from collections import deque, defaultdict
from typing import Any, TypeAlias, List, Dict, Optional, Tuple
from abc import abstractmethod
import jsonpickle # Make sure this is imported
import numpy as np # Needed for standard deviation (though not used in V20)

# Assuming datamodel classes are available as in the competition environment
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation

# Type Alias for JSON data
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Product:
    # Define product names used in the strategy
    AMETHYSTS = "AMETHYSTS"
    STARFRUIT = "STARFRUIT"
    # Add Round 1 products if they weren't defined before
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    # Add others if needed later from the template
    ORCHIDS = "ORCHIDS"
    GIFT_BASKET = "GIFT_BASKET"
    CHOCOLATE = "CHOCOLATE"
    STRAWBERRIES = "STRAWBERRIES"
    ROSES = "ROSES"


# Parameters dictionary - Adapt values for R1 products
PARAMS = {
    Product.AMETHYSTS: {
        "fair_value": 10000,
        "take_width": 1,  # How far price must cross fair value to take aggressively
        "clear_width": 1, # How far inside fair value to place clearing orders
        "make_spread": 2, # Half-spread for passive market making
        "aggr_size": 2,   # Size for aggressive orders
        "make_size": 5,   # Size for passive orders
        "skew_factor": 0.0, # No skew for Amethysts
    },
    Product.KELP: { # Using similar structure, will need EMA fair value
        "ema_alpha": 0.3, # Faster EMA for potentially ranging Kelp
        "take_width": 1,
        "clear_width": 1,
        "make_spread": 1.5, # Slightly tighter spread than Amethyst maybe
        "aggr_size": 3,
        "make_size": 4,
        "skew_factor": 0.1, # Moderate inventory skew
    },
    Product.SQUID_INK: { # New parameters for Squid Ink based on V19 idea
        "ema_alpha": 0.25, # Moderate EMA responsiveness
        "take_width": 2,   # Require larger edge to take aggressively due to vol
        "clear_width": 2,  # Place clearing orders further inside FV
        "make_spread": 2.5, # Wider base spread
        "aggr_size": 2,    # Small aggressive size
        "make_size": 3,    # Small passive size
        "skew_factor": 0.25, # Stronger inventory skew
        "max_pos_ratio_passive": 0.8 # Allow quoting until 80% of limit
    }
    # Add other product params if needed
}


# Logger class (Ensure this definition is complete and correct *before* instantiation)
class Logger:
    # (Keep the full Logger class implementation from previous correct responses)
    def __init__(self) -> None: self.logs = ""; self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None: self.logs += sep.join(map(str, objects)) + end
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # --- Full Compression Logic ---
        def compress_state(state: TradingState, trader_data: str) -> list[Any]:
            compressed_conversion_observations = {}
            if hasattr(state, 'observations') and state.observations is not None and hasattr(state.observations, 'conversionObservations') and state.observations.conversionObservations is not None:
                 for product, observation in state.observations.conversionObservations.items():
                     if observation is not None: compressed_conversion_observations[product]=[getattr(observation,'bidPrice',None),getattr(observation,'askPrice',None),getattr(observation,'transportFees',0.0),getattr(observation,'exportTariff',0.0),getattr(observation,'importTariff',0.0),getattr(observation,'sunlight',None),getattr(observation,'humidity',None)]
                     else: compressed_conversion_observations[product]=[None]*7
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


# --- Base Strategy Class (Simplified) ---
class BaseStrategy:
    def __init__(self, symbol: Symbol, limit: int, params: Dict) -> None:
        self.symbol = symbol
        self.limit = limit
        self.params = params
        self.orders: List[Order] = []

    @abstractmethod
    def run(self, state: TradingState, trader_data_for_product: Dict) -> List[Order]:
        raise NotImplementedError

    def buy(self, price: float, quantity: float) -> None:
        if quantity > 0: self.orders.append(Order(self.symbol, int(round(price)), int(round(quantity))))
    def sell(self, price: float, quantity: float) -> None:
        if quantity > 0: self.orders.append(Order(self.symbol, int(round(price)), -int(round(quantity))))
    def save_state(self) -> JSON: return None # Strategies needing state override this
    def load_state(self, data: JSON) -> None: pass # Strategies needing state override this


# --- EMA Calculation Helper ---
def calculate_ema(current_price: float, prev_ema: Optional[float], alpha: float) -> float:
    if prev_ema is None: return current_price
    return alpha * current_price + (1 - alpha) * prev_ema

def get_current_midprice(order_depth: OrderDepth) -> Optional[float]:
    if not order_depth: return None
    bids=order_depth.buy_orders; asks=order_depth.sell_orders;
    if not bids or not asks: return min(asks.keys()) if asks else (max(bids.keys()) if bids else None)
    best_ask=min(asks.keys()); best_bid=max(bids.keys()); best_ask_vol=abs(asks[best_ask]); best_bid_vol=bids[best_bid]
    if best_ask_vol+best_bid_vol==0: return (best_ask+best_bid)/2.0
    return (best_bid*best_ask_vol+best_ask*best_bid_vol)/(best_bid_vol+best_ask_vol)

# --- Concrete Strategies ---

class AmethystStrategy(BaseStrategy):
    def run(self, state: TradingState, trader_data_for_product: Dict) -> List[Order]:
        self.orders = []
        ods = state.order_depths[self.symbol]
        pos = state.position.get(self.symbol, 0)
        fair_value = self.params["fair_value"]
        make_spread_half = self.params["make_spread"] / 2
        aggr_size = self.params["aggr_size"]
        make_size = self.params["make_size"]
        buy_cap = self.limit - pos; sell_cap = self.limit + pos

        best_ask = min(ods.sell_orders.keys()) if ods.sell_orders else float('inf')
        best_bid = max(ods.buy_orders.keys()) if ods.buy_orders else float('-inf')

        # Aggressive orders
        if best_ask < fair_value and buy_cap > 0:
            vol=min(buy_cap, abs(ods.sell_orders[best_ask]), aggr_size); self.buy(best_ask, vol); buy_cap-=vol
        if best_bid > fair_value and sell_cap > 0:
            vol=min(sell_cap, ods.buy_orders[best_bid], aggr_size); self.sell(best_bid, vol); sell_cap-=vol

        # Passive orders
        if buy_cap > 0: self.buy(fair_value - make_spread_half, min(buy_cap, make_size))
        if sell_cap > 0: self.sell(fair_value + make_spread_half, min(sell_cap, make_size))

        return self.orders

class GenericMMStrategy(BaseStrategy):
    """Market making using EMA and inventory skew, base for Kelp/Squid."""
    def __init__(self, symbol: Symbol, limit: int, params: Dict) -> None:
        super().__init__(symbol, limit, params)
        self.ema_value: Optional[float] = None

    def run(self, state: TradingState, trader_data_for_product: Dict) -> List[Order]:
        self.orders = []
        self.load_state(trader_data_for_product.get(self.symbol, None)) # Load EMA state

        current_mid = get_current_midprice(state.order_depths[self.symbol])
        if current_mid is not None:
            self.ema_value = calculate_ema(current_mid, self.ema_value, self.params["ema_alpha"])
        else: # If no midprice, try to use last EMA or skip
            if self.ema_value is None: return self.orders # Cannot proceed without fair value

        fair_value = int(round(self.ema_value))
        ods=state.order_depths[self.symbol]; buy_orders=sorted(ods.buy_orders.items(),reverse=True); sell_orders=sorted(ods.sell_orders.items())
        pos=state.position.get(self.symbol,0); buy_cap=self.limit-pos; sell_cap=self.limit+pos
        best_ask = min(sell_orders, key=lambda x: x[0])[0] if sell_orders else float('inf')
        best_bid = max(buy_orders, key=lambda x: x[0])[0] if buy_orders else float('-inf')

        # Parameters from self.params
        take_width = self.params["take_width"]
        clear_width = self.params["clear_width"]
        make_spread_half = self.params["make_spread"] / 2
        aggr_size = self.params["aggr_size"]
        make_size = self.params["make_size"]
        skew_factor = self.params["skew_factor"]
        max_pos_ratio = self.params.get("max_pos_ratio_passive", 1.0) # Default 1.0 if not set

        # --- Calculate Inventory Skew ---
        price_skew = int(round(pos * skew_factor * make_spread_half * 2)) # Skew proportional to spread & position
        skewed_fair_value = fair_value - price_skew
        # logger.print(f"{self.symbol}: Pos={pos}, FV={fair_value}, Skew={price_skew}, SkewedFV={skewed_fair_value}")


        # --- Aggressive Orders (taking vs FAIR value, maybe skewed later) ---
        taken_buy=0
        if best_ask <= fair_value - take_width and buy_cap > 0:
            vol=min(abs(ods.sell_orders[best_ask]), buy_cap, aggr_size); self.buy(best_ask, vol); buy_cap-=vol; taken_buy+=vol
        taken_sell=0
        if best_bid >= fair_value + take_width and sell_cap > 0:
            vol=min(ods.buy_orders[best_bid], sell_cap, aggr_size); self.sell(best_bid, vol); sell_cap-=vol; taken_sell+=vol

        # --- Passive Orders (around SKEWED fair value) ---
        passive_buy_price = skewed_fair_value - make_spread_half
        passive_sell_price = skewed_fair_value + make_spread_half

        # Ensure passive orders don't cross the best market
        passive_buy_price = int(round(min(passive_buy_price, best_ask - 1))) if best_ask != float('inf') else int(round(passive_buy_price))
        passive_sell_price = int(round(max(passive_sell_price, best_bid + 1))) if best_bid != float('-inf') else int(round(passive_sell_price))

        # Place orders only if position is not too large
        if abs(pos) < self.limit * max_pos_ratio:
            if buy_cap > 0: # Always quote buy side if capacity allows (unless just aggressively bought)
                if taken_buy == 0: self.buy(passive_buy_price, min(buy_cap, make_size))
            if sell_cap > 0: # Always quote sell side if capacity allows (unless just aggressively sold)
                 if taken_sell == 0: self.sell(passive_sell_price, min(sell_cap, make_size))
        # else: logger.print(f"{self.symbol}: Position {pos} too high, reducing passive quoting.")


        # --- Clearing Logic (Optional - try aggressive clear first) ---
        # If position is still large after passive quoting, try to clear aggressively closer to fair value
        if clear_width > 0:
             if pos > self.limit * 0.6 and sell_cap > 0: # Heavy long
                 clear_price = int(round(fair_value - clear_width))
                 clear_price = max(clear_price, best_bid + 1) if best_bid > float('-inf') else clear_price # Don't cross bid
                 vol_to_clear = min(pos // 2, sell_cap) # Clear half the excess inventory aggressively
                 if vol_to_clear > 0: self.sell(clear_price, vol_to_clear)
             elif pos < -self.limit * 0.6 and buy_cap > 0: # Heavy short
                  clear_price = int(round(fair_value + clear_width))
                  clear_price = min(clear_price, best_ask - 1) if best_ask < float('inf') else clear_price # Don't cross ask
                  vol_to_clear = min(abs(pos) // 2, buy_cap) # Clear half the excess inventory aggressively
                  if vol_to_clear > 0: self.buy(clear_price, vol_to_clear)

        return self.orders

    def save_state(self) -> JSON:
        # Save EMA value
        return {"ema_value": self.ema_value}

    def load_state(self, data: JSON) -> None:
        # Load EMA value
        if isinstance(data, dict):
            self.ema_value = data.get("ema_value", None)
        else:
            self.ema_value = None


# --- Main Trader Class ---
class Trader:
    def __init__(self, params=None):
        if params is None: params = PARAMS # Use default PARAMS if none provided
        self.params = params
        self.LIMIT = { # Define limits for all potential products
            Product.AMETHYSTS: 20, Product.STARFRUIT: 20, Product.ORCHIDS: 100,
            Product.GIFT_BASKET: 60, Product.CHOCOLATE: 250,
            Product.STRAWBERRIES: 350, Product.ROSES: 60,
            # Add Round 1 products
            Product.KELP: 50, Product.SQUID_INK: 50
        }
        # Instantiate strategies based on available params
        self.strategies: Dict[Symbol, BaseStrategy] = {}
        if Product.AMETHYSTS in self.params:
            self.strategies[Product.AMETHYSTS] = AmethystStrategy(Product.AMETHYSTS, self.LIMIT[Product.AMETHYSTS], self.params[Product.AMETHYSTS])
        if Product.KELP in self.params:
            self.strategies[Product.KELP] = GenericMMStrategy(Product.KELP, self.LIMIT[Product.KELP], self.params[Product.KELP])
        if Product.SQUID_INK in self.params:
             self.strategies[Product.SQUID_INK] = GenericMMStrategy(Product.SQUID_INK, self.LIMIT[Product.SQUID_INK], self.params[Product.SQUID_INK])
        # Add other strategies here if needed (Starfruit, Orchids, Basket using their specific logic if you want to merge later)

        self.trader_data_cache = {} # Cache for strategy states

    def load_trader_data(self, traderData: str):
        if not traderData: self.trader_data_cache = {}
        else:
            try: self.trader_data_cache = jsonpickle.decode(traderData)
            except Exception as e: logger.print(f"ERROR decoding traderData: {e}"); self.trader_data_cache = {}
        # Ensure sub-dictionaries exist for each strategy
        for symbol in self.strategies:
            if symbol not in self.trader_data_cache: self.trader_data_cache[symbol] = {}


    def save_trader_data(self) -> str:
        # Update cache with latest state from strategies
        for symbol, strategy in self.strategies.items():
            saved = strategy.save_state()
            if saved is not None: # Only update if strategy saves something
                 self.trader_data_cache[symbol] = saved # Store state under the symbol key

        try: return jsonpickle.encode(self.trader_data_cache, unpicklable=False)
        except Exception as e: logger.print(f"ERROR encoding traderData: {e}"); return ""

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        self.load_trader_data(state.traderData) # Load state into self.trader_data_cache

        all_orders: Dict[Symbol, List[Order]] = {}
        conversions = 0

        for symbol, strategy in self.strategies.items():
            if symbol in state.order_depths:
                # Pass the relevant part of the trader data to the strategy
                product_state = self.trader_data_cache.get(symbol, {})
                all_orders[symbol] = strategy.run(state, product_state)
                # Update cache with any state changes made by the strategy run (if save_state modified it)
                saved = strategy.save_state()
                if saved is not None: self.trader_data_cache[symbol] = saved
            else:
                 all_orders[symbol] = []


        traderData_output = self.save_trader_data() # Save the updated cache
        logger.flush(state, all_orders, conversions, traderData_output)
        return all_orders, conversions, traderData_output