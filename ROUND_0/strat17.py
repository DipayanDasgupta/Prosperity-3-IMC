import json
from abc import abstractmethod
from collections import deque
# Ensure datamodel.py is in the same directory
try:
    from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
except ImportError:
    # Basic placeholders if datamodel isn't available
    Symbol = str
    class ProsperityEncoder(json.JSONEncoder):
        def default(self, o): return super().default(o)
    class Order:
        def __init__(self, symbol, price, quantity):
            self.symbol, self.price, self.quantity = symbol, price, quantity
    # Define other types minimally if needed
    class Listing: # Define as a class for local testing
        def __init__(self, symbol, product, denomination):
            self.symbol = symbol
            self.product = product
            self.denomination = denomination
    Observation = object(); OrderDepth = object(); Trade = object(); TradingState = object()

from typing import Any, TypeAlias
import math # For rounding

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# --- Logger Class (Corrected compress_listings) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750 # Adjust if needed

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        log_string = sep.join(map(str, objects)) + end
        if len(self.logs) + len(log_string) < self.max_log_length:
             self.logs += log_string

    def flush(self, state: TradingState | None, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        if state is None:
             print(json.dumps(["No state", self.compress_orders(orders), conversions, trader_data, self.logs]))
             self.logs = ""
             return
        try:
            compressed_state = self.compress_state(state, "")
            compressed_orders = self.compress_orders(orders)
            base_info = [compressed_state, compressed_orders, conversions, "", ""]
            base_length = len(json.dumps(base_info, cls=ProsperityEncoder, default=str))
        except Exception:
            base_length = 200

        max_item_length = max(0, (self.max_log_length - base_length) // 3)

        log_output = [
            self.compress_state(state, self.truncate(getattr(state, 'traderData', ""), max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]
        try:
            print(self.to_json(log_output))
        except Exception as e:
            print(f"Error encoding log data: {e}")
            print(f"Conversion: {conversions}, TraderData: {self.truncate(trader_data, 100)}, Logs: {self.truncate(self.logs, 100)}")
        self.logs = ""


    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            getattr(state, 'timestamp', 0), trader_data,
            self.compress_listings(getattr(state, 'listings', {})),
            self.compress_order_depths(getattr(state, 'order_depths', {})),
            self.compress_trades(getattr(state, 'own_trades', {})),
            self.compress_trades(getattr(state, 'market_trades', {})),
            getattr(state, 'position', {}),
            self.compress_observations(getattr(state, 'observations', None)),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        """Correctly compresses listings using attribute access."""
        compressed = []
        if listings:
            for listing_obj in listings.values():
                 # *** FIX: Use attribute access (.symbol) not .get() ***
                 # getattr is safe if the attribute *might* not exist, but
                 # for standard datamodel types, direct access is fine.
                 try:
                     sym = listing_obj.symbol
                     prod = listing_obj.product
                     den = listing_obj.denomination
                     compressed.append([sym, prod, den])
                 except AttributeError as e:
                     # Log if a listing object doesn't have expected attributes
                     # self.print(f"WARN: Listing object missing attribute: {e}")
                     compressed.append(['ERROR', 'ERROR', 'ERROR']) # Placeholder
        return compressed

    # --- Keep other compress methods, to_json, truncate as before ---
    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        if order_depths:
             for symbol, order_depth in order_depths.items():
                  compressed[symbol] = [getattr(order_depth, 'buy_orders', {}), getattr(order_depth, 'sell_orders', {})]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if trades:
            for arr in trades.values():
                 if arr:
                    for trade in arr:
                        compressed.append([
                            getattr(trade, 'symbol', ''), getattr(trade, 'price', 0),
                            getattr(trade, 'quantity', 0), getattr(trade, 'buyer', ''),
                            getattr(trade, 'seller', ''), getattr(trade, 'timestamp', 0),
                        ])
        return compressed

    def compress_observations(self, observations: Observation | None) -> list[Any]:
        if not observations: return [{}, {}]
        conversion_observations = {}
        if hasattr(observations, 'conversionObservations') and observations.conversionObservations:
            for product, observation in observations.conversionObservations.items():
                conversion_observations[product] = [
                    getattr(observation, 'bidPrice', 0), getattr(observation, 'askPrice', 0),
                    getattr(observation, 'transportFees', 0), getattr(observation, 'exportTariff', 0),
                    getattr(observation, 'importTariff', 0), getattr(observation, 'sunlight', 0),
                    getattr(observation, 'humidity', 0),
                ]
        plain_observations = getattr(observations, 'plainValueObservations', {})
        return [plain_observations, conversion_observations]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if orders:
            for arr in orders.values():
                 if arr:
                    for order in arr:
                        compressed.append([getattr(order, 'symbol', ''), getattr(order, 'price', 0), getattr(order, 'quantity', 0)])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"), default=str)

    def truncate(self, value: str, max_length: int) -> str:
        if not isinstance(value, str): value = str(value)
        if len(value) <= max_length: return value
        if max_length < 3: return value[:max_length]
        return value[:max_length - 3] + "..."


logger = Logger()

# --- Base Strategy Class (Unchanged) ---
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: list[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        try:
             self.act(state)
        except Exception as e:
             logger.print(f"ERROR in strategy {self.symbol}: {e}")
             import traceback
             logger.print(traceback.format_exc()) # Print full traceback for debugging
             pass
        return self.orders

    def buy(self, price: float | int, quantity: int) -> None:
        if quantity > 0:
             self.orders.append(Order(self.symbol, int(round(price)), quantity))

    def sell(self, price: float | int, quantity: int) -> None:
         if quantity > 0:
            self.orders.append(Order(self.symbol, int(round(price)), -quantity))

    def save(self) -> JSON: return None
    def load(self, data: JSON) -> None: pass

# --- Improved Market Making Strategy (Unchanged from previous version) ---
class ImprovedMarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, base_spread: float = 1.5, skew_factor: float = 0.05, aggression: float = 0.1) -> None:
        super().__init__(symbol, limit)
        self.base_spread = base_spread
        self.skew_factor = skew_factor
        self.aggression = aggression
        self.last_wap: float | None = None

    def calculate_wap(self, order_depth: OrderDepth) -> float | None:
        buy_orders = getattr(order_depth, 'buy_orders', {})
        sell_orders = getattr(order_depth, 'sell_orders', {})

        if not sell_orders or not buy_orders:
            return self.last_wap

        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())

        if best_bid >= best_ask:
            mid = (best_bid + best_ask) / 2
            # logger.print(f"WARN: Book crossed for {self.symbol}. Bid={best_bid}, Ask={best_ask}. Using mid: {mid}")
            return mid

        bid_vol = buy_orders[best_bid]
        ask_vol = abs(sell_orders[best_ask])

        if bid_vol == 0 and ask_vol == 0:
             mid_price = (best_bid + best_ask) / 2
             if self.last_wap and abs(self.last_wap - mid_price) < (self.base_spread * 2):
                  # logger.print(f"WARN: Zero volume at top level {self.symbol}. Using last WAP: {self.last_wap}")
                  return self.last_wap
             else:
                 # logger.print(f"WARN: Zero volume at top level {self.symbol}. Using mid: {mid_price}")
                 return mid_price
        if bid_vol == 0:
             # logger.print(f"WARN: Zero bid volume at top level {self.symbol}. Using best ask: {best_ask}")
             return best_ask
        if ask_vol == 0:
             # logger.print(f"WARN: Zero ask volume at top level {self.symbol}. Using best bid: {best_bid}")
             return best_bid

        wap = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
        self.last_wap = wap
        return wap

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return

        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)

        wap = self.calculate_wap(order_depth)
        if wap is None:
             # logger.print(f"WARN: Could not calculate WAP for {self.symbol}, skipping tick.")
             return

        skew = round(self.skew_factor * position)
        target_bid_price = math.floor(wap - self.base_spread - skew)
        target_ask_price = math.ceil(wap + self.base_spread - skew)

        ideal_buy_qty = self.limit - position
        ideal_sell_qty = self.limit + position

        # Limit total quantity submitted per side
        buy_capacity = self.limit // 2
        sell_capacity = self.limit // 2

        final_buy_qty = max(0, min(ideal_buy_qty, buy_capacity))
        final_sell_qty = max(0, min(ideal_sell_qty, sell_capacity))

        # --- Aggressive Taking Logic ---
        if final_buy_qty > 0 and order_depth.sell_orders:
            best_ask_price = min(order_depth.sell_orders.keys())
            if best_ask_price <= target_bid_price + self.aggression:
                best_ask_volume = abs(order_depth.sell_orders[best_ask_price])
                qty_to_take = min(final_buy_qty, best_ask_volume)
                if qty_to_take > 0:
                    # logger.print(f"{self.symbol} Aggressive BUY: Price={best_ask_price}, Qty={qty_to_take}")
                    self.buy(best_ask_price, qty_to_take)
                    final_buy_qty -= qty_to_take

        if final_sell_qty > 0 and order_depth.buy_orders:
             best_bid_price = max(order_depth.buy_orders.keys())
             if best_bid_price >= target_ask_price - self.aggression:
                 best_bid_volume = abs(order_depth.buy_orders[best_bid_price])
                 qty_to_take = min(final_sell_qty, best_bid_volume)
                 if qty_to_take > 0:
                    # logger.print(f"{self.symbol} Aggressive SELL: Price={best_bid_price}, Qty={qty_to_take}")
                    self.sell(best_bid_price, qty_to_take)
                    final_sell_qty -= qty_to_take

        # --- Limit Order Placement ---
        if final_buy_qty > 0:
            # logger.print(f"{self.symbol} Placing BUY: Price={target_bid_price}, Qty={final_buy_qty}")
            self.buy(target_bid_price, final_buy_qty)

        if final_sell_qty > 0:
             # logger.print(f"{self.symbol} Placing SELL: Price={target_ask_price}, Qty={final_sell_qty}")
             self.sell(target_ask_price, final_sell_qty)


# --- Concrete Strategy Implementations (Unchanged) ---
class RainforestResinStrategy(ImprovedMarketMakingStrategy):
     def __init__(self, symbol: Symbol, limit: int):
          super().__init__(symbol, limit, base_spread=1.0, skew_factor=0.02, aggression=0)

class KelpStrategy(ImprovedMarketMakingStrategy):
     def __init__(self, symbol: Symbol, limit: int):
          super().__init__(symbol, limit, base_spread=1.5, skew_factor=0.08, aggression=0.1)


# --- Trader Class (Unchanged) ---
class Trader:
    def __init__(self) -> None:
        self.limit = 50 # Use the actual limit

        limits = {
            "RAINFOREST_RESIN": self.limit,
            "KELP": self.limit,
        }
        self.strategies: dict[Symbol, Strategy] = {}
        available_strategies = {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
        }
        for symbol, limit_val in limits.items():
             if symbol in available_strategies:
                  self.strategies[symbol] = available_strategies[symbol](symbol, limit_val)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        orders: dict[Symbol, list[Order]] = {}
        trader_data = "{}" # Sending empty JSON

        for symbol, strategy in self.strategies.items():
            if strategy.limit != self.limit:
                 logger.print(f"WARN: Strategy limit ({strategy.limit}) != Trader limit ({self.limit}) for {symbol}")
                 strategy.limit = self.limit

            if symbol in state.order_depths:
                try:
                     strategy_orders = strategy.run(state)
                     if isinstance(strategy_orders, list):
                          orders[symbol] = strategy_orders
                     else:
                          logger.print(f"WARN: Strategy {symbol} did not return list")
                          orders[symbol] = []
                except Exception as e:
                     logger.print(f"ERROR running strategy for {symbol}: {e}")
                     import traceback
                     logger.print(traceback.format_exc())
                     orders[symbol] = []
            else:
                 orders[symbol] = []

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data