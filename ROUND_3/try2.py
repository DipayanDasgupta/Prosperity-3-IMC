import json
import math
from abc import abstractmethod, ABC
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
# Scipy not needed for this version

# --- Constants ---
# No BS constants needed

# --- Data Models (Keep as is) ---
class Symbol(str): pass
class Listing:
    def __init__(self, s: Symbol, p: str, d: str) -> None: self.symbol, self.product, self.denomination = s, p, d
class Order:
    def __init__(self, s: Symbol, p: int, q: int) -> None: self.symbol, self.price, self.quantity = s, p, q
    def __repr__(self): return f"Order({self.symbol}, {self.price}, {self.quantity})"
class OrderDepth:
    buy_orders: Dict[int, int]; sell_orders: Dict[int, int]
    @staticmethod
    def fix_buy_sell_keys(obj):
        if isinstance(obj, dict):
            if "buy_orders" in obj and isinstance(obj["buy_orders"], list): obj["buy_orders"] = {int(p): v for p,v in obj["buy_orders"]}
            if "sell_orders" in obj and isinstance(obj["sell_orders"], list): obj["sell_orders"] = {int(p): v for p,v in obj["sell_orders"]}
            for v in obj.values(): OrderDepth.fix_buy_sell_keys(v)
        elif isinstance(obj, list):
            for i in obj: OrderDepth.fix_buy_sell_keys(i)
    def __init__(self, buy: Union[Dict[int, int], List]={}, sell: Union[Dict[int, int], List]={}) -> None:
        if isinstance(buy, list): self.buy_orders = {int(p): v for p,v in buy}
        else: self.buy_orders = buy
        if isinstance(sell, list): self.sell_orders = {int(p): v for p,v in sell}
        else: self.sell_orders = sell
class Trade:
    def __init__(self, s: Symbol, p: int, q: int, b: str="", sl: str="", t: int=0) -> None:
        self.symbol, self.price, self.quantity, self.buyer, self.seller, self.timestamp = s, p, q, b, sl, t
class Observation:
    def __init__(self, pvo: Dict[str, Any]={}, cvo: Dict[str, Any]={}) -> None:
        self.plainValueObservations, self.conversionObservations = pvo, cvo
class TradingState:
    def __init__(self, ts: int, td: str, lst: Dict[Symbol, Listing], od: Dict[Symbol, OrderDepth], ot: Dict[Symbol, List[Trade]], mt: Dict[Symbol, List[Trade]], pos: Dict[Symbol, int], obs: Observation) -> None:
        self.timestamp, self.traderData, self.listings, self.order_depths, self.own_trades, self.market_trades, self.position, self.observations = ts, td, lst, od, ot, mt, pos, obs
        OrderDepth.fix_buy_sell_keys(self.order_depths)

# --- JSON Encoder ---
class ProsperityEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, deque): return list(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o) if not np.isnan(o) else None
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, float) and not math.isfinite(o): return None
        if hasattr(o, "__dict__"): return {k: v for k, v in o.__dict__.items() if not k.startswith('_') and not callable(v)}
        return super().default(o)

# --- Logger ---
class Logger:
    def __init__(self) -> None: self.logs, self.max_log_length = "", 3750
    def print(self, *o: Any, s: str=" ", e: str="\n") -> None: self.logs += s.join(map(str,o)) + e
    def flush(self, st: TradingState, ords: Dict[Symbol, List[Order]], conv: int, td: str) -> None:
        truncated_td_in = self.truncate(st.traderData, 500); truncated_td_out = self.truncate(td, 500)
        valid_ods = {s: o for s, o in st.order_depths.items() if isinstance(o, OrderDepth) and o.buy_orders is not None and o.sell_orders is not None}
        log_output = {
            "state": {"timestamp": st.timestamp, "traderData": truncated_td_in, "listings": list(st.listings.keys()), "order_depths": {s: {"bids": len(o.buy_orders), "asks": len(o.sell_orders)} for s, o in valid_ods.items()}, "own_trades": {s: len(t) for s, t in st.own_trades.items()}, "market_trades": {s: len(t) for s, t in st.market_trades.items()}, "position": st.position, "observations": st.observations},
            "orders": [[o.symbol, o.price, o.quantity] for ol in ords.values() for o in ol],
            "conversions": conv, "trader_data_next": truncated_td_out, "log_messages": self.truncate(self.logs, self.max_log_length - 1500)}
        try: print(json.dumps(log_output, cls=ProsperityEncoder, separators=(",", ":"), allow_nan=False))
        except (TypeError, ValueError) as e: print(f"JSON Encoding Error: {e}\nFallback Log: TS={st.timestamp}, Pos={st.position}, Orders={len(log_output['orders'])}"); print(f"Problematic Data Snippet: {str(log_output)[:500]}")
        self.logs = ""
    def truncate(self, v: str, ml: int) -> str: return v if len(v) <= ml else v[:ml-3]+"..."
logger = Logger()

# --- Shared Helper Functions ---
def calculate_weighted_mid_price(order_depth: OrderDepth) -> Optional[float]:
    if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders: return None
    try:
        buy_keys = [int(k) for k in order_depth.buy_orders.keys()]; sell_keys = [int(k) for k in order_depth.sell_orders.keys()]
        if not buy_keys or not sell_keys: return None
        best_bid = max(buy_keys); best_ask = min(sell_keys)
        if best_bid not in order_depth.buy_orders or best_ask not in order_depth.sell_orders: return None
        bid_vol = order_depth.buy_orders[best_bid]; ask_vol = abs(order_depth.sell_orders[best_ask])
    except (ValueError, TypeError, KeyError): return None
    if bid_vol + ask_vol == 0: return (best_bid + best_ask) / 2.0
    return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

def calculate_volatility(price_history: deque, window_size: int) -> float:
    # Use simple standard deviation of prices as volatility proxy
    if len(price_history) < max(5, window_size // 4): return 1.0 # Return base vol if not enough data
    prices = np.array(list(price_history), dtype=float)
    if len(prices) < 5: return 1.0
    std_dev = np.std(prices)
    if np.isnan(std_dev) or not math.isfinite(std_dev): return 1.0
    return max(0.5, std_dev) # Ensure minimum volatility value


# --- Base Strategy Class ---
class Strategy(ABC):
    def __init__(self, symbol: Symbol, limit: int, window_size: int, smooth_window: int) -> None:
        self.symbol = symbol; self.limit = limit; self.window_size = window_size
        self.smooth_window = smooth_window # Window for smoothing FV
        self.wmp_history = deque(maxlen=smooth_window); self.orders: List[Order] = []
        self.vol_price_history = deque(maxlen=window_size) # Separate history for volatility calc
        self.fair_value: Optional[float] = None
        self.volatility: float = 1.0

    @abstractmethod
    def run(self, state: TradingState, *args) -> List[Order]: raise NotImplementedError()

    def place_order(self, side: str, price: int, quantity: int, current_pos: int, max_abs_pos: int):
        if quantity <= 0 or not math.isfinite(price) or price <= 0: return
        if side == "BUY":
            max_buy = max_abs_pos - current_pos; order_qty = min(quantity, max_buy)
            if order_qty > 0: self.orders.append(Order(self.symbol, price, order_qty))
        elif side == "SELL":
            max_sell = max_abs_pos + current_pos; order_qty = min(quantity, max_sell)
            if order_qty > 0: self.orders.append(Order(self.symbol, price, -order_qty))

    def get_smoothed_fair_value(self) -> Optional[float]:
         if not self.wmp_history: return None
         valid_wmp = [v for v in self.wmp_history if v is not None and math.isfinite(v)]
         if not valid_wmp: return None
         return sum(valid_wmp) / len(valid_wmp)

    def update_state(self, state: TradingState):
        """Updates internal state like fair value and volatility."""
        order_depth = state.order_depths.get(self.symbol)
        current_wmid = calculate_weighted_mid_price(order_depth) if order_depth else None

        if current_wmid is not None and math.isfinite(current_wmid):
            self.wmp_history.append(current_wmid)
            self.vol_price_history.append(current_wmid) # Add to vol history

        self.fair_value = self.get_smoothed_fair_value()
        # Only calculate volatility if fair value is valid
        if self.fair_value is not None and math.isfinite(self.fair_value):
            self.volatility = calculate_volatility(self.vol_price_history, self.window_size)
            if not math.isfinite(self.volatility): self.volatility = 1.0 # Fallback
        else:
             self.volatility = 1.0 # Default if FV is bad

    def save(self) -> Any:
        return {
            'wmp_history': list(self.wmp_history),
            'vol_price_history': list(self.vol_price_history),
            'fair_value': self.fair_value,
            'volatility': self.volatility
            }
    def load(self, data: Any) -> None:
        # Load WMP history for fair value smoothing
        wmp_hist = data.get('wmp_history', []) if data else []
        valid_wmp_hist = [p for p in wmp_hist if isinstance(p, (int, float)) and math.isfinite(p)]
        sw = getattr(self, 'smooth_window', 5)
        self.wmp_history = deque(valid_wmp_hist, maxlen=sw)

        # Load price history for volatility calculation
        vol_hist = data.get('vol_price_history', []) if data else []
        valid_vol_hist = [p for p in vol_hist if isinstance(p, (int, float)) and math.isfinite(p)]
        ws = getattr(self, 'window_size', 20)
        self.vol_price_history = deque(valid_vol_hist, maxlen=ws)

        # Load last calculated values
        self.fair_value = data.get('fair_value', None)
        self.volatility = data.get('volatility', 1.0)
        # Ensure loaded values are valid numbers
        if self.fair_value is not None and not math.isfinite(self.fair_value): self.fair_value = None
        if not math.isfinite(self.volatility): self.volatility = 1.0


# --- Statistical Market Maker Strategy ---
class StatisticalMarketMakerStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, window_size: int = 30, alpha: float = 0.20, k: float = 1.0, base_spread: int = 4, order_size: int = 15, smooth_window: int = 5) -> None:
        super().__init__(symbol, limit, window_size, smooth_window)
        self.alpha = alpha # Inventory skew parameter
        self.k = k         # Volatility multiplier for spread
        self.base_spread = base_spread # Minimum spread in ticks
        self.order_size = order_size   # Base size for orders

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.update_state(state) # Update FV and Volatility first

        if self.fair_value is None: # Cannot quote without fair value
             # logger.print(f"Skipping {self.symbol}: No valid fair value.")
             return self.orders

        position = state.position.get(self.symbol, 0)
        adj_fv = self.fair_value - self.alpha * position # Apply inventory skew
        if not math.isfinite(adj_fv): return self.orders

        # Spread calculation based on product's own volatility
        # Scale k by fair value to make spread wider for higher priced items
        price_scale_factor = max(1.0, self.fair_value / 1000.0) if self.fair_value > 0 else 1.0 # Example scaling
        dynamic_spread = self.k * self.volatility * price_scale_factor
        total_spread = max(self.base_spread, dynamic_spread if not math.isnan(dynamic_spread) else self.base_spread)
        half_spread = math.ceil(total_spread / 2.0)

        target_bid = int(math.floor(adj_fv - half_spread))
        target_ask = int(math.ceil(adj_fv + half_spread))
        target_bid = max(1, target_bid)
        if target_ask <= target_bid: target_ask = target_bid + 1

        # Simple sanity check - don't quote zero spread
        if target_ask - target_bid < 1: return self.orders

        # Place orders with adjusted size based on position? (Optional)
        # current_trade_size = max(1, int(self.order_size * (1 - 0.5 * abs(position)/self.limit))) # Example size reduction
        current_trade_size = self.order_size

        # logger.print(f"{self.symbol} | FV:{self.fair_value:.1f} AdjFV:{adj_fv:.1f} Vol:{self.volatility:.2f} | Sprd:{total_spread:.1f} Bid:{target_bid} Ask:{target_ask} | Pos:{position}")

        self.place_order("BUY", target_bid, current_trade_size, position, self.limit)
        self.place_order("SELL", target_ask, current_trade_size, position, self.limit)
        return self.orders

# --- Trader Class ---
class Trader:
    def __init__(self) -> None:
        self.LIMIT = {"VOLCANIC_ROCK": 400, "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200, "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200, "VOLCANIC_ROCK_VOUCHER_10500": 200}
        self.volcanic_symbols = list(self.LIMIT.keys())
        self.strategies: Dict[Symbol, Strategy] = {}

        # === Defensive Statistical MM Parameters ===
        # Rock: Moderate alpha, decent size, reasonable spread
        rock_params = {"alpha": 0.05, "k": 0.7, "base_spread": 3, "order_size": 30, "smooth_window": 5, "window_size": 30}
        # Vouchers: HIGH alpha, WIDE base spread, LOW size, low K (less dynamic spread)
        # Start very defensively
        voucher_params = {'alpha': 0.30, 'k': 0.5, 'base_spread': 6, 'order_size': 10, 'smooth_window': 5, 'window_size': 30}
        # =========================================

        self.strategies["VOLCANIC_ROCK"] = StatisticalMarketMakerStrategy(symbol="VOLCANIC_ROCK", limit=self.LIMIT["VOLCANIC_ROCK"], **rock_params)
        for sym in self.volcanic_symbols:
             if "VOUCHER" in sym:
                # Apply voucher params - potentially differentiate ITM/ATM/OTM later if needed
                self.strategies[sym] = StatisticalMarketMakerStrategy(symbol=sym, limit=self.LIMIT[sym], **voucher_params)

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        conversions = 0; orders: Dict[Symbol, List[Order]] = {}; strategy_states_to_save = {}

        # Load state for all strategies first
        if state.traderData:
            try:
                loaded_data = json.loads(state.traderData)
                # Assumes traderData stores states keyed by symbol
                for symbol, strategy_data in loaded_data.items():
                    if symbol in self.strategies and hasattr(self.strategies[symbol], 'load'):
                        self.strategies[symbol].load(strategy_data if isinstance(strategy_data, dict) else None)
            except (json.JSONDecodeError, TypeError) as e: logger.print(f"Error decoding traderData: {e}")

        # Run each strategy independently
        for symbol, strategy in self.strategies.items():
            if symbol not in state.order_depths: continue # Skip if no market data
            try:
                orders[symbol] = strategy.run(state)
                strategy_states_to_save[symbol] = strategy.save()
            except Exception as e:
                logger.print(f"ERROR running strategy for {symbol}: {e}"); import traceback; logger.print(traceback.format_exc())
                orders[symbol] = [] # Place no orders if strategy errors

        # Save all strategy states
        trader_data_str = "{}" # Default empty
        try:
            cleaned_data = json.loads(json.dumps(strategy_states_to_save, cls=ProsperityEncoder, allow_nan=True).replace('NaN', 'null').replace('Infinity','null').replace('-Infinity','null'))
            trader_data_str = json.dumps(cleaned_data, separators=(",",":"))
        except Exception as json_e: logger.print(f"Error cleaning/dumping trader data: {json_e}")

        logger.flush(state, orders, conversions, trader_data_str)
        return orders, conversions, trader_data_str