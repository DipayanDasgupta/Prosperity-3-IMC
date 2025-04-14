import json
from abc import abstractmethod
from collections import deque
# Make sure all necessary imports from datamodel are present
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias, List, Dict, Optional, Tuple # Added List, Dict, Optional, Tuple
import statistics # Ensure statistics is imported

# Type Alias for JSON data
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Product: # Define Product names
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        # Simple check to prevent excessively long logs accumulating in memory
        if len(self.logs) < self.max_log_length * 2:
             self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Using the more robust compression logic from previous examples
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
                    # Handle potential None or non-dict buy/sell orders gracefully
                    buy_orders_dict = order_depth.buy_orders if isinstance(order_depth.buy_orders, dict) else {}
                    sell_orders_dict = order_depth.sell_orders if isinstance(order_depth.sell_orders, dict) else {}
                    compressed[symbol] = [buy_orders_dict, sell_orders_dict]
            return compressed

        def compress_trades(trades: Optional[dict[Symbol, list[Trade]]]) -> list[list[Any]]: # Added Optional
            compressed = [];
            if trades: # Check if trades is not None
                for arr in trades.values():
                     if arr: # Check if list is not empty
                        for trade in arr:
                             if trade: # Check if trade object is not None
                                compressed.append([trade.symbol,trade.price,trade.quantity,trade.buyer or "",trade.seller or "",trade.timestamp])
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
             # Use ProsperityEncoder if available, otherwise standard json
             cls_to_use = ProsperityEncoder if "ProsperityEncoder" in globals() else None
             try: return json.dumps(value, cls=cls_to_use, separators=(",", ":"))
             except Exception: return json.dumps(value, separators=(",", ":")) # Fallback

        def truncate(value: Any, max_length: int) -> str: # Accept Any type
            # Convert value to string safely
            if not isinstance(value, str):
                try: value = str(value)
                except Exception: value = "" # Fallback to empty string if str() fails

            if len(value) <= max_length: return value
            return value[:max_length - 3] + "..."
        # --- End Compression Helpers ---
        try:
            # Estimate base length more safely
            min_state_data = [state.timestamp, "", [], {}, [], [], {}, [{}, {}]]
            base_value=[min_state_data,[],conversions,"",""]; base_json=to_json(base_value); base_length=len(base_json)

            available_length=self.max_log_length-base_length-200; # Keep buffer
            if available_length < 0: available_length=0
            max_item_length=max(available_length//3, 100) # Ensure minimum length

            truncated_trader_data_state=truncate(state.traderData, max_item_length)
            truncated_trader_data_out=truncate(trader_data, max_item_length); truncated_logs=truncate(self.logs, max_item_length)
            log_entry=[compress_state(state, truncated_trader_data_state), compress_orders(orders), conversions, truncated_trader_data_out, truncated_logs]
            final_json_output = to_json(log_entry)
            if len(final_json_output)>self.max_log_length: final_json_output=final_json_output[:self.max_log_length-5]+"...]}"
            print(final_json_output)
        except Exception as e: print(json.dumps({"error": f"Logging failed: {e}", "timestamp": state.timestamp}))
        self.logs = ""


logger = Logger()

class Strategy:
    # Keep original simple init
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: List[Order] = [] # Initialize orders here

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = [] # Reset orders for this run
        try:
            self.act(state)
        except Exception as e:
            # Log the error
            logger.print(f"ERROR in {self.symbol} strategy run: {e}")
            # Basic flatten attempt on error
            self.orders = [] # Clear potentially bad orders
            current_pos = state.position.get(self.symbol, 0)
            if current_pos != 0:
                ods = state.order_depths.get(self.symbol)
                if ods:
                    if current_pos > 0 and ods.buy_orders:
                        self.sell(max(ods.buy_orders.keys()), abs(current_pos))
                    elif current_pos < 0 and ods.sell_orders:
                        self.buy(min(ods.sell_orders.keys()), abs(current_pos))
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
         # Ensure quantity is positive integer
         q = int(round(quantity))
         if q > 0:
            self.orders.append(Order(self.symbol, int(round(price)), q))

    def sell(self, price: int, quantity: int) -> None:
         # Ensure quantity is positive integer for selling logic
         q = int(round(quantity))
         if q > 0:
            self.orders.append(Order(self.symbol, int(round(price)), -q))

    # Keep original simple save/load structure
    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window: deque[bool] = deque(maxlen=10) # Correct type hint and maxlen
        self.window_size = 10 # Match maxlen

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int: # Keep original signature
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        # Check if symbol exists in order depths first
        if self.symbol not in state.order_depths:
             # logger.print(f"No order depth for {self.symbol}")
             return

        true_value = self.get_true_value(state)
        order_depth = state.order_depths[self.symbol]

        # Check if buy/sell orders exist and are not empty
        if not order_depth.buy_orders or not order_depth.sell_orders:
             # logger.print(f"Empty book for {self.symbol}")
             return

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Use maxlen for window operations
        self.window.append(abs(position) >= self.limit * 0.95) # Append before checking length
        # Removed redundant check len(self.window) > self.window_size as deque handles it
        soft_liquidate = len(self.window) == self.window.maxlen and sum(self.window) >= self.window.maxlen / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window.maxlen and sum(self.window) >= self.window.maxlen * 0.8 # Use 80% threshold

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < -self.limit * 0.5 else true_value

        # Aggressive Buys
        for price, volume in sell_orders:
            if to_buy <= 0: break # Stop if nothing left to buy
            if price <= max_buy_price:
                quantity = min(to_buy, abs(volume)) # Use abs(volume)
                if quantity > 0:
                    self.buy(price, quantity)
                    to_buy -= quantity
            else: # Prices are sorted, no need to check further
                 break

        # Liquidation Buys (if short)
        liq_buy_size = max(1, self.limit // 4) # Use a fraction of limit for liquidation size
        if position < 0: # Only liquidate buys if position is short
            if hard_liquidate and to_buy > 0:
                quantity = min(abs(position), to_buy, liq_buy_size) # Limit liquidation size
                if quantity > 0: self.buy(true_value, quantity); to_buy -= quantity
            elif soft_liquidate and to_buy > 0:
                quantity = min(abs(position), to_buy, liq_buy_size) # Limit liquidation size
                if quantity > 0: self.buy(true_value - 1, quantity); to_buy -= quantity # Slightly more passive price

        # Passive Buys
        if to_buy > 0:
            # Check if buy_orders has items before calling max
            if buy_orders:
                popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                price = min(max_buy_price, popular_buy_price + 1)
                self.buy(price, to_buy) # Place remaining capacity passively
            else: # Fallback if no bids exist yet
                self.buy(max_buy_price, to_buy)


        # Aggressive Sells
        for price, volume in buy_orders:
             if to_sell <= 0: break # Stop if nothing left to sell
             if price >= min_sell_price:
                quantity = min(to_sell, abs(volume)) # Use abs(volume)
                if quantity > 0:
                    self.sell(price, quantity)
                    to_sell -= quantity
             else: # Prices are sorted reverse, no need to check further
                  break

        # Liquidation Sells (if long)
        liq_sell_size = max(1, self.limit // 4) # Use a fraction of limit for liquidation size
        if position > 0: # Only liquidate sells if position is long
            if hard_liquidate and to_sell > 0:
                quantity = min(position, to_sell, liq_sell_size) # Limit liquidation size
                if quantity > 0: self.sell(true_value, quantity); to_sell -= quantity
            elif soft_liquidate and to_sell > 0:
                quantity = min(position, to_sell, liq_sell_size) # Limit liquidation size
                if quantity > 0: self.sell(true_value + 1, quantity); to_sell -= quantity # Slightly more passive price

        # Passive Sells
        if to_sell > 0:
            # Check if sell_orders has items before calling min
            if sell_orders:
                popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
                price = max(min_sell_price, popular_sell_price - 1)
                self.sell(price, to_sell) # Place remaining capacity passively
            else: # Fallback if no asks exist yet
                 self.sell(min_sell_price, to_sell)


    def save(self) -> JSON:
        # Ensure window state is saved correctly
        return {"window": list(self.window)} # Save as dict

    def load(self, data: JSON) -> None:
        # Load window state safely
        if isinstance(data, dict) and "window" in data and isinstance(data["window"], list):
            # Ensure maxlen is set correctly
            self.window = deque(data["window"], maxlen=self.window_size)
        else:
             # Initialize correctly if data is missing or wrong format
            self.window = deque(maxlen=self.window_size)


# --- Ink Mean Reversion Strategy (SQUID_INK) ---
class InkMeanReversionTrader(Strategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.prices = deque(maxlen=100) # *** Using shorter window 100 ***
        self.window_size = 100 # Match deque maxlen

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return # Need book data

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        mid_price = (best_bid + best_ask) / 2.0
        self.prices.append(mid_price)

        # Require reasonable number of data points
        if len(self.prices) < self.window_size * 0.5: # Wait for 50% fill
            return

        try:
            mean = statistics.mean(self.prices)
            # Use population stdev if full, sample stdev otherwise
            std_dev = statistics.pstdev(self.prices) if len(self.prices) == self.window_size else statistics.stdev(self.prices)
        except statistics.StatisticsError:
            # Not enough data points or all points are the same
            return

        # Avoid trading if stdev is zero or extremely small
        if std_dev < 0.1:
            return

        position = state.position.get(self.symbol, 0)
        buy_cap = self.limit - position
        sell_cap = self.limit + position

        # Define Bands
        upper_band_2 = mean + 2.0 * std_dev
        upper_band_1 = mean + 1.0 * std_dev
        lower_band_1 = mean - 1.0 * std_dev
        lower_band_2 = mean - 2.0 * std_dev
        exit_threshold = 0.25 * std_dev # Exit when price is within 0.25 std dev of mean

        # *** Using reduced trade size ***
        trade_size = 8 # Smaller trade size for Ink

        # === Aggressive BUY (Market Take) ===
        if mid_price < lower_band_2 and buy_cap > 0:
            qty = min(buy_cap, trade_size) # Limit by trade_size
            if qty > 0: self.buy(best_ask, qty)

        # === Passive BUY (Limit Order) ===
        elif mid_price < lower_band_1 and buy_cap > 0:
            qty = min(buy_cap, trade_size) # Limit by trade_size
            if qty > 0: self.buy(best_bid, qty) # Place at best bid

        # === Aggressive SELL (Market Take) ===
        elif mid_price > upper_band_2 and sell_cap > 0:
            qty = min(sell_cap, trade_size) # Limit by trade_size
            if qty > 0: self.sell(best_bid, qty)

        # === Passive SELL (Limit Order) ===
        elif mid_price > upper_band_1 and sell_cap > 0:
            qty = min(sell_cap, trade_size) # Limit by trade_size
            if qty > 0: self.sell(best_ask, qty) # Place at best ask

        # === EXIT ZONE ===
        elif abs(mid_price - mean) <= exit_threshold:
            if position > 0 and sell_cap > 0: # Exit Long
                qty = min(position, sell_cap, trade_size) # Exit in chunks
                if qty > 0: self.sell(best_bid, qty) # Hit bid
            elif position < 0 and buy_cap > 0: # Exit Short
                qty = min(abs(position), buy_cap, trade_size) # Exit in chunks
                if qty > 0: self.buy(best_ask, qty) # Hit ask

    # Save/Load needs to handle the deque correctly
    def save(self) -> JSON:
        return {"prices": list(self.prices)} # Save as dict

    def load(self, data: JSON) -> None:
         # Load prices deque safely
        if isinstance(data, dict) and "prices" in data and isinstance(data["prices"], list):
            # Ensure maxlen is set correctly
            self.prices = deque(data["prices"], maxlen=self.window_size)
        else:
             # Initialize correctly if data is missing or wrong format
            self.prices = deque(maxlen=self.window_size)


# --- Concrete Strategy Implementations (Unused but kept for structure) ---
class RainforestResinStrategy(MarketMakingStrategy):
     def __init__(self, symbol: Symbol, limit: int): # Add init
         super().__init__(symbol, limit)
     def get_true_value(self, state: TradingState) -> int:
        return 10000

class KelpStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int): # Add init
         super().__init__(symbol, limit)
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths.get(self.symbol)
        # Add checks for empty order books before proceeding
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            # Fallback: maybe return previous estimate or a default if possible
            # For now, just return a placeholder or handle the error upstream
            logger.print(f"Warning: Empty book for {self.symbol} in get_true_value")
            # Returning a fixed value or raising an error might be better options
            # Depending on how MarketMakingStrategy handles exceptions or None values
            # Let's return a dummy value to avoid crashing, but this needs refinement
            return 0 # Needs better handling - maybe use EMA or last price

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # Check if orders exist before accessing
        if not buy_orders or not sell_orders:
             logger.print(f"Warning: Empty buy or sell orders for {self.symbol} in get_true_value")
             return 0 # Needs better handling

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)

# --- Main Trader Class (Modified for SQUID_INK only) ---
class Trader:
    def __init__(self) -> None:
        limits = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
        }

        self.strategies: Dict[Symbol, Strategy] = {} # Initialize as empty dict

        # --- ONLY INSTANTIATE SQUID_INK STRATEGY ---
        self.strategies[Product.SQUID_INK] = InkMeanReversionTrader(
            Product.SQUID_INK, limits[Product.SQUID_INK]
        )
        # --- Instantiation for RESIN and KELP commented out ---
        # self.strategies[Product.RAINFOREST_RESIN] = RainforestResinStrategy(
        #     Product.RAINFOREST_RESIN, limits[Product.RAINFOREST_RESIN]
        # )
        # self.strategies[Product.KELP] = KelpStrategy(
        #     Product.KELP, limits[Product.KELP]
        # )
        # ----------------------------------------------------

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        orders: Dict[Symbol, list[Order]] = { # Initialize orders dict for all potential symbols
             Product.RAINFOREST_RESIN: [],
             Product.KELP: [],
             Product.SQUID_INK: []
        }

        # Load trader data (only Ink state will be present/relevant)
        # Using simple json load/dump as per the original r1strat20 structure
        old_trader_data = {}
        if state.traderData:
             try:
                 old_trader_data = json.loads(state.traderData)
             except json.JSONDecodeError:
                 logger.print("Error decoding traderData")
                 old_trader_data = {} # Reset if decode fails

        new_trader_data = {}

        # --- ONLY RUN SQUID_INK STRATEGY ---
        symbol_to_run = Product.SQUID_INK
        if symbol_to_run in self.strategies:
            strategy = self.strategies[symbol_to_run]

            # Load state for Ink strategy
            if symbol_to_run in old_trader_data:
                try:
                    strategy.load(old_trader_data[symbol_to_run])
                except Exception as e:
                     logger.print(f"Error loading state for {symbol_to_run}: {e}")


            # Run strategy only if order depth exists
            if symbol_to_run in state.order_depths:
                orders[symbol_to_run] = strategy.run(state)

            # Save state for Ink strategy
            saved_state = strategy.save()
            if saved_state is not None: # Only save if state is actually returned
                 new_trader_data[symbol_to_run] = saved_state
        # ------------------------------------

        # Convert new_trader_data to string
        trader_data = ""
        if new_trader_data: # Only dump if not empty
             try:
                 trader_data = json.dumps(new_trader_data, separators=(",", ":"))
             except Exception as e:
                 logger.print(f"Error encoding traderData: {e}")
                 trader_data = "" # Reset on error

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data