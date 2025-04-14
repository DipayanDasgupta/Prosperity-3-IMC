import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import Any, List
import jsonpickle
import math

# --------------------------------------------------------------------------------------------------
# Logger Class (Unchanged from the first provided block)
# --------------------------------------------------------------------------------------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Sanitize trader_data to ensure it's a string
        if not isinstance(trader_data, str):
            # Attempt to encode if it's not a string (e.g., if jsonpickle failed or returned bytes)
            try:
                trader_data = str(trader_data)
            except Exception:
                trader_data = "Error encoding trader data" # Fallback

        # Sanitize state.traderData
        state_trader_data = state.traderData
        if not isinstance(state_trader_data, str):
             try:
                state_trader_data = str(state_trader_data)
             except Exception:
                state_trader_data = "Error encoding state trader data" # Fallback


        try:
            base_length = len(self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions,
                "",
                "",
            ]))

            # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
            max_item_length = (self.max_log_length - base_length) // 3
            if max_item_length < 0: max_item_length = 0 # Ensure non-negative length

            # Use the sanitized and potentially stringified trader data
            compressed_state = self.compress_state(state, self.truncate(state_trader_data, max_item_length))
            compressed_orders = self.compress_orders(orders)
            truncated_trader_data = self.truncate(trader_data, max_item_length)
            truncated_logs = self.truncate(self.logs, max_item_length)

            log_output = self.to_json([
                compressed_state,
                compressed_orders,
                conversions,
                truncated_trader_data,
                truncated_logs,
            ])

        except Exception as e:
            # Fallback in case of JSON encoding errors during compression/truncation
            print(f"Error during log flushing: {e}")
            log_output = json.dumps({ # Simple JSON fallback
                 "timestamp": state.timestamp,
                 "error": "Log flushing failed",
                 "traderData_preview": trader_data[:50], # Log only a snippet
                 "log_preview": self.logs[:50]
            })
            # Ensure the fallback log output doesn't exceed limits either
            if len(log_output) > self.max_log_length:
                 log_output = json.dumps({"error": "Log flushing failed, output too long"})


        print(log_output)
        self.logs = "" # Reset logs after flushing

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Ensure trader_data is string before using
        trader_data_str = trader_data if isinstance(trader_data, str) else str(trader_data)
        return [
            state.timestamp,
            trader_data_str, # Use the string version
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        # Ensure listings is a dict
        if isinstance(listings, dict):
            for listing in listings.values():
                 # Check if listing is a dictionary before accessing keys
                 if isinstance(listing, dict) and "symbol" in listing and "product" in listing and "denomination" in listing:
                     compressed.append([listing["symbol"], listing["product"], listing["denomination"]])
                 elif isinstance(listing, Listing): # Handle if it's the actual datamodel object
                      compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        if isinstance(order_depths, dict):
            for symbol, order_depth in order_depths.items():
                 # Ensure order_depth is the correct type or has the expected attributes
                 if isinstance(order_depth, OrderDepth):
                     compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
                 elif isinstance(order_depth, dict) and "buy_orders" in order_depth and "sell_orders" in order_depth: # Handle dict representation
                     compressed[symbol] = [order_depth["buy_orders"], order_depth["sell_orders"]]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if isinstance(trades, dict):
            for arr in trades.values():
                 if isinstance(arr, list):
                     for trade in arr:
                          # Check if trade is the correct type or has the expected attributes
                          if isinstance(trade, Trade):
                              compressed.append([
                                  trade.symbol,
                                  trade.price,
                                  trade.quantity,
                                  trade.buyer,
                                  trade.seller,
                                  trade.timestamp,
                              ])
                          elif isinstance(trade, dict): # Handle dict representation
                               compressed.append([
                                    trade.get("symbol", ""),
                                    trade.get("price", 0),
                                    trade.get("quantity", 0),
                                    trade.get("buyer", ""),
                                    trade.get("seller", ""),
                                    trade.get("timestamp", 0),
                                ])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
         # Check if observations object is valid
         if not isinstance(observations, Observation) or not hasattr(observations, 'conversionObservations'):
             return [[], {}] # Return default empty structure

         conversion_observations = {}
         if isinstance(observations.conversionObservations, dict):
             for product, observation in observations.conversionObservations.items():
                  # Check if observation is the correct type or has the expected attributes
                  if hasattr(observation, 'bidPrice'): # Duck typing for ConversionObservation attributes
                     conversion_observations[product] = [
                         observation.bidPrice,
                         observation.askPrice,
                         observation.transportFees,
                         observation.exportTariff,
                         observation.importTariff,
                         observation.sunlight,
                         observation.humidity,
                     ]
                  elif isinstance(observation, dict): # Handle dict representation
                      conversion_observations[product] = [
                          observation.get("bidPrice", 0),
                          observation.get("askPrice", 0),
                          observation.get("transportFees", 0),
                          observation.get("exportTariff", 0),
                          observation.get("importTariff", 0),
                          observation.get("sunlight", 0),
                          observation.get("humidity", 0),
                      ]


         plain_value_observations = getattr(observations, 'plainValueObservations', {})
         if not isinstance(plain_value_observations, dict):
             plain_value_observations = {} # Ensure it's a dict


         return [plain_value_observations, conversion_observations]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if isinstance(orders, dict):
            for arr in orders.values():
                if isinstance(arr, list):
                    for order in arr:
                        # Ensure order is the correct type or has the expected attributes
                        if isinstance(order, Order):
                            compressed.append([order.symbol, order.price, order.quantity])
                        elif isinstance(order, dict) and "symbol" in order and "price" in order and "quantity" in order: # Handle dict representation
                             compressed.append([order["symbol"], order["price"], order["quantity"]])
        return compressed

    def to_json(self, value: Any) -> str:
        # Use ProsperityEncoder for custom types, handle potential errors
        try:
            return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
        except TypeError as e:
             print(f"JSON encoding error: {e}. Value causing error: {value}")
             # Provide a safe fallback JSON representation
             return json.dumps({"error": "Encoding failed", "value_type": str(type(value))})


    def truncate(self, value: str, max_length: int) -> str:
        # Ensure value is a string before truncating
        if not isinstance(value, str):
             try:
                 value = str(value) # Attempt conversion
             except Exception:
                 return "..." # Fallback if conversion fails

        if len(value) <= max_length:
            return value
        # Ensure max_length is reasonable before slicing
        if max_length < 3:
             return "..."[:max_length] # Handle very small max_length

        return value[:max_length - 3] + "..."

logger = Logger()

# --------------------------------------------------------------------------------------------------
# Trader Class (Combined and Refined)
# --------------------------------------------------------------------------------------------------
class Trader:
    def __init__(self):
        # Initialize state variables. These will be loaded from traderData if available.
        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_price_history = []
        self.resin_trade_prices = []
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        # Logger.print("Trader initialized.") # Example log

    def load_trader_data(self, trader_data_str: str):
        """Loads trader state from the traderData string."""
        if trader_data_str:
            try:
                data = jsonpickle.decode(trader_data_str)
                if isinstance(data, dict): # Check if decoding was successful and is a dict
                    self.kelp_prices = data.get('kelp_prices', [])
                    self.kelp_vwap = data.get('kelp_vwap', [])
                    self.kelp_price_history = data.get('kelp_price_history', [])
                    self.resin_trade_prices = data.get('resin_trade_prices', [])
                    # Logger.print("Trader data loaded successfully.")
                else:
                    # Logger.print("Failed to decode trader data or unexpected format.")
                    self._initialize_state() # Re-initialize if decode fails
            except Exception as e:
                # Logger.print(f"Error loading trader data: {e}. Re-initializing state.")
                self._initialize_state() # Re-initialize on error
        else:
            # Logger.print("No trader data found. Initializing state.")
            self._initialize_state() # Initialize if no data string

    def _initialize_state(self):
        """Helper to initialize or reset state variables."""
        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_price_history = []
        self.resin_trade_prices = []
        # self.position_limits remain as defined in __init__

    def resin_fair_value(self, state: TradingState) -> int:
        """Calculates the fair value for RAINFOREST_RESIN."""
        base_fair_value = 10000 # Set a default or initial base fair value

        # Check for valid order depths to calculate mid-price if no recent trades
        order_depth = state.order_depths.get("RAINFOREST_RESIN")
        if order_depth and order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            base_fair_value = mid_price # Use mid-price as a better starting point

        # Adjust based on recent trades
        market_trades = state.market_trades.get("RAINFOREST_RESIN", [])
        if market_trades:
            # Ensure self.resin_trade_prices is initialized correctly
            if not hasattr(self, 'resin_trade_prices') or not isinstance(self.resin_trade_prices, list):
                self.resin_trade_prices = []

            # Add the latest trade price
            self.resin_trade_prices.append(market_trades[-1].price)
            # Keep the history length limited
            max_history = 15
            if len(self.resin_trade_prices) > max_history:
                self.resin_trade_prices = self.resin_trade_prices[-max_history:] # Keep only the last N trades

            if self.resin_trade_prices: # Ensure list is not empty after potential slicing
                avg_trade_price = sum(self.resin_trade_prices) / len(self.resin_trade_prices)
                # Increased weight on recent trades: 70% base/mid, 30% recent avg trade price
                fair_value = round(base_fair_value * 0.7 + avg_trade_price * 0.3)
                # logger.print(f"RESIN Fair Value (Trade Adjusted): {fair_value}, Base: {base_fair_value}, Avg Trade: {avg_trade_price}")
                return fair_value
            else:
                 # logger.print(f"RESIN Fair Value (Base/Mid): {round(base_fair_value)}")
                 return round(base_fair_value)
        else:
            # logger.print(f"RESIN Fair Value (Base/Mid - No Trades): {round(base_fair_value)}")
            return round(base_fair_value) # Return base/mid if no trades


    def get_resin_dynamic_widths(self, order_depth: OrderDepth) -> tuple:
        """Introduce simple dynamic widths for RAINFOREST_RESIN based on spread."""
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return 1.0, 0.2  # Default values if order depth is missing or empty

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid

        base_make_width = 1.0  # Reduced base width for tighter spreads
        base_take_width = 0.2  # Reduced for more aggressive taking

        if spread > 5:  # High volatility/spread -> slightly tighter MM, same take
            make_width = base_make_width * 0.9
            take_width = base_take_width # * 0.9 - maybe keep take width stable or even wider?
        elif spread < 2:  # Low volatility/spread -> slightly wider MM, same take
            make_width = base_make_width * 1.1
            take_width = base_take_width # * 1.1
        else: # Normal spread
            make_width = base_make_width
            take_width = base_take_width

        # Ensure widths are not negative
        make_width = max(0.1, make_width) # Minimum make width
        take_width = max(0.1, take_width) # Minimum take width

        # logger.print(f"RESIN Dynamic Widths: Make={make_width:.2f}, Take={take_width:.2f}, Spread={spread}")
        return make_width, take_width


    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        """Generates orders for RAINFOREST_RESIN."""
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        product = "RAINFOREST_RESIN"

        if not order_depth: # Handle missing order depth
            # logger.print("Warning: No order depth found for RAINFOREST_RESIN.")
            return orders

        make_width, take_width = self.get_resin_dynamic_widths(order_depth)

        # --- Taking Logic ---
        # Buy cheaper asks (take sell orders)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask <= fair_value - take_width:
                best_ask_amount = abs(order_depth.sell_orders[best_ask]) # Amount is positive
                available_buy_limit = position_limit - position # How much more we can buy
                quantity_to_take = min(best_ask_amount, available_buy_limit, 35) # Max take size cap
                if quantity_to_take > 0:
                    orders.append(Order(product, best_ask, quantity_to_take))
                    buy_order_volume += quantity_to_take
                    # logger.print(f"RESIN TAKE BUY: Price={best_ask}, Qty={quantity_to_take}, Fair={fair_value}")


        # Sell to higher bids (take buy orders)
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid >= fair_value + take_width:
                best_bid_amount = abs(order_depth.buy_orders[best_bid]) # Amount is positive
                available_sell_limit = position_limit + position # How much more we can sell (abs value)
                quantity_to_take = min(best_bid_amount, available_sell_limit, 35) # Max take size cap
                if quantity_to_take > 0:
                    orders.append(Order(product, best_bid, -quantity_to_take))
                    sell_order_volume -= quantity_to_take # Volume is negative for sells
                    # logger.print(f"RESIN TAKE SELL: Price={best_bid}, Qty={-quantity_to_take}, Fair={fair_value}")


        # --- Position Management Logic (before Market Making) ---
        # This section tries to reduce position if it exceeds certain thresholds
        # Note: The clear_position_order logic is complex and might need refinement.
        # Let's simplify it for now: if position is too large/small, place orders closer to fair value.
        position_after_take = position + buy_order_volume + sell_order_volume # Note sell_order_volume is negative
        threshold = 25 # If position exceeds this (abs), try to reduce

        # If too long, place sell orders closer to fair value
        if position_after_take > threshold:
             reduce_qty = position_after_take - threshold # Amount to reduce by
             sell_price_clear = math.floor(fair_value) # Sell slightly below fair? Or at fair bid?
             orders.append(Order(product, sell_price_clear, -min(reduce_qty, position_limit + (position - abs(sell_order_volume))))) # Ensure doesn't exceed sell limit
             # logger.print(f"RESIN CLEAR LONG: Price={sell_price_clear}, Qty={-min(reduce_qty, position_limit + (position - abs(sell_order_volume)))}")


        # If too short, place buy orders closer to fair value
        if position_after_take < -threshold:
             reduce_qty = abs(position_after_take) - threshold # Amount to reduce by (positive)
             buy_price_clear = math.ceil(fair_value) # Buy slightly above fair? Or at fair ask?
             orders.append(Order(product, buy_price_clear, min(reduce_qty, position_limit - (position + buy_order_volume)))) # Ensure doesn't exceed buy limit
             # logger.print(f"RESIN CLEAR SHORT: Price={buy_price_clear}, Qty={min(reduce_qty, position_limit - (position + buy_order_volume))}")


        # --- Market Making Logic ---
        # Calculate remaining capacity after takes and potential clearing orders
        current_buy_capacity = position_limit - (position + buy_order_volume + sell_order_volume) # How much more we can buy
        current_sell_capacity = position_limit + (position + buy_order_volume + sell_order_volume) # How much more we can sell (absolute)


        # Place buy orders (make)
        if current_buy_capacity > 0:
            buy_price = math.floor(fair_value - make_width) # Use floor for buy price
            orders.append(Order(product, buy_price, current_buy_capacity))
            # logger.print(f"RESIN MAKE BUY: Price={buy_price}, Qty={current_buy_capacity}")


        # Place sell orders (make)
        if current_sell_capacity > 0:
            sell_price = math.ceil(fair_value + make_width) # Use ceil for sell price
            orders.append(Order(product, sell_price, -current_sell_capacity))
            # logger.print(f"RESIN MAKE SELL: Price={sell_price}, Qty={-current_sell_capacity}")


        return orders


    def kelp_fair_value(self, order_depth: OrderDepth, state: TradingState) -> int:
        """Calculates the fair value for KELP."""
        # Ensure self.kelp_price_history is initialized
        if not hasattr(self, 'kelp_price_history') or not isinstance(self.kelp_price_history, list):
             self.kelp_price_history = []

        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
             # Fallback: Use last known fair value or a default if history is empty
             if self.kelp_price_history:
                 return round(self.kelp_price_history[-1])
             else:
                 return 5000 # Default value

        # Calculate mid-price based on best bid/ask or filtered MM levels
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # Optional: Filter for larger volume levels to estimate market maker mid-price
        # min_mm_volume = 10
        # filtered_ask = [p for p, vol in order_depth.sell_orders.items() if abs(vol) >= min_mm_volume]
        # filtered_bid = [p for p, vol in order_depth.buy_orders.items() if abs(vol) >= min_mm_volume]
        # mm_ask = min(filtered_ask) if filtered_ask else best_ask
        # mm_bid = max(filtered_bid) if filtered_bid else best_bid
        # mid_price = (mm_ask + mm_bid) / 2
        mid_price = (best_ask + best_bid) / 2 # Using simple mid-price for now


        # Incorporate recent market trades adjustment (optional, reduce weight if causing issues)
        trade_adjustment = 0
        market_trades = state.market_trades.get("KELP", [])
        if market_trades:
            recent_trade_price = market_trades[-1].price
            trade_adjustment = (recent_trade_price - mid_price) * 0.10 # Reduced weight

        # Use a weighted moving average (WMA) of historical mid-prices
        self.kelp_price_history.append(mid_price)
        max_history = 5 # Window size for WMA
        if len(self.kelp_price_history) > max_history:
            self.kelp_price_history = self.kelp_price_history[-max_history:] # Keep last N prices

        # Calculate WMA
        if self.kelp_price_history:
            weights = list(range(1, len(self.kelp_price_history) + 1)) # Simple linear weights [1, 2, 3...]
            weighted_sum = sum(p * w for p, w in zip(self.kelp_price_history, weights))
            total_weight = sum(weights)
            wma_mid_price = weighted_sum / total_weight if total_weight > 0 else mid_price
        else:
            wma_mid_price = mid_price # Use current mid if no history

        # Combine WMA and trade adjustment
        fair_value = wma_mid_price + trade_adjustment
        # logger.print(f"KELP Fair Value: {round(fair_value)}, WMA: {wma_mid_price:.2f}, TradeAdj: {trade_adjustment:.2f}, Mid: {mid_price:.2f}")

        return round(fair_value)


    def get_kelp_dynamic_widths(self, order_depth: OrderDepth) -> tuple:
        """Calculate dynamic widths for KELP based on spread."""
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return 1.2, 0.2  # Default values

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid

        # Base widths - potentially adjust based on KELP's typical behavior
        base_make_width = 1.2
        base_take_width = 0.2

        # Adjust based on spread (similar logic to RESIN, could be tuned differently)
        if spread > 5: # Wider spread
            make_width = base_make_width * 0.9 # Tighter MM spread
            take_width = base_take_width # * 0.9
        elif spread < 2: # Tighter spread
            make_width = base_make_width * 1.1 # Wider MM spread
            take_width = base_take_width # * 1.1
        else: # Normal spread
            make_width = base_make_width
            take_width = base_take_width

        # Ensure widths are not negative
        make_width = max(0.1, make_width)
        take_width = max(0.1, take_width)

        # logger.print(f"KELP Dynamic Widths: Make={make_width:.2f}, Take={take_width:.2f}, Spread={spread}")
        return make_width, take_width


    def kelp_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        """Generates orders for KELP. Structure similar to resin_orders."""
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        product = "KELP"

        if not order_depth:
            # logger.print("Warning: No order depth found for KELP.")
            return orders

        make_width, take_width = self.get_kelp_dynamic_widths(order_depth)

        # --- Taking Logic ---
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask <= fair_value - take_width:
                best_ask_amount = abs(order_depth.sell_orders[best_ask])
                available_buy_limit = position_limit - position
                quantity_to_take = min(best_ask_amount, available_buy_limit, 35) # Max take size cap
                if quantity_to_take > 0:
                    orders.append(Order(product, best_ask, quantity_to_take))
                    buy_order_volume += quantity_to_take
                    # logger.print(f"KELP TAKE BUY: Price={best_ask}, Qty={quantity_to_take}, Fair={fair_value}")


        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid >= fair_value + take_width:
                best_bid_amount = abs(order_depth.buy_orders[best_bid])
                available_sell_limit = position_limit + position
                quantity_to_take = min(best_bid_amount, available_sell_limit, 35) # Max take size cap
                if quantity_to_take > 0:
                    orders.append(Order(product, best_bid, -quantity_to_take))
                    sell_order_volume -= quantity_to_take
                    # logger.print(f"KELP TAKE SELL: Price={best_bid}, Qty={-quantity_to_take}, Fair={fair_value}")


        # --- Position Management Logic ---
        # Simplified clearing logic (similar to RESIN)
        position_after_take = position + buy_order_volume + sell_order_volume
        threshold = 25 # Adjust threshold for KELP if needed

        if position_after_take > threshold:
             reduce_qty = position_after_take - threshold
             sell_price_clear = math.floor(fair_value)
             sell_qty = min(reduce_qty, position_limit + (position - abs(sell_order_volume)))
             if sell_qty > 0:
                 orders.append(Order(product, sell_price_clear, -sell_qty))
                 # logger.print(f"KELP CLEAR LONG: Price={sell_price_clear}, Qty={-sell_qty}")


        if position_after_take < -threshold:
             reduce_qty = abs(position_after_take) - threshold
             buy_price_clear = math.ceil(fair_value)
             buy_qty = min(reduce_qty, position_limit - (position + buy_order_volume))
             if buy_qty > 0:
                 orders.append(Order(product, buy_price_clear, buy_qty))
                 # logger.print(f"KELP CLEAR SHORT: Price={buy_price_clear}, Qty={buy_qty}")


        # --- Market Making Logic ---
        current_buy_capacity = position_limit - (position + buy_order_volume + sell_order_volume)
        current_sell_capacity = position_limit + (position + buy_order_volume + sell_order_volume)

        if current_buy_capacity > 0:
            buy_price = math.floor(fair_value - make_width)
            orders.append(Order(product, buy_price, current_buy_capacity))
            # logger.print(f"KELP MAKE BUY: Price={buy_price}, Qty={current_buy_capacity}")


        if current_sell_capacity > 0:
            sell_price = math.ceil(fair_value + make_width)
            orders.append(Order(product, sell_price, -current_sell_capacity))
            # logger.print(f"KELP MAKE SELL: Price={sell_price}, Qty={-current_sell_capacity}")


        return orders


    # Note: The original `clear_position_order` function was complex and its logic
    # has been simplified and integrated directly into `resin_orders` and `kelp_orders`.
    # If needed, it could be refactored back into a separate function.


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main method called by the trading engine for each time step.
        """
        # Initialize results
        result: dict[Symbol, list[Order]] = {}
        conversions = 0 # No conversions used in this strategy

        # Load persistent state variables from traderData
        self.load_trader_data(state.traderData)

        # Log current positions and PnL (optional)
        # logger.print(f"Timestamp: {state.timestamp}")
        # for product, pos in state.position.items():
        #      logger.print(f"Position {product}: {pos}")
        # Implement PnL calculation if needed


        # --- RAINFOREST_RESIN Strategy ---
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_od = state.order_depths["RAINFOREST_RESIN"]
            resin_pos = state.position.get("RAINFOREST_RESIN", 0)
            resin_limit = self.position_limits["RAINFOREST_RESIN"]
            # Calculate fair value first, as it might update internal state (trade history)
            resin_fv = self.resin_fair_value(state)
            # Generate orders based on fair value, order depth, position, and limits
            resin_orders = self.resin_orders(resin_od, resin_fv, resin_pos, resin_limit)
            result["RAINFOREST_RESIN"] = resin_orders
        else:
            # logger.print("No order depth data for RAINFOREST_RESIN.")
            pass # No orders if no data


        # --- KELP Strategy ---
        if "KELP" in state.order_depths:
            kelp_od = state.order_depths["KELP"]
            kelp_pos = state.position.get("KELP", 0)
            kelp_limit = self.position_limits["KELP"]
            # Calculate fair value first
            kelp_fv = self.kelp_fair_value(kelp_od, state)
            # Generate orders
            kelp_orders = self.kelp_orders(kelp_od, kelp_fv, kelp_pos, kelp_limit)
            result["KELP"] = kelp_orders
        else:
            # logger.print("No order depth data for KELP.")
            pass # No orders if no data


        # --- Prepare Trader Data for Persistence ---
        # Store relevant state variables to be passed to the next iteration
        trader_state_to_save = {
            "kelp_prices": self.kelp_prices, # Note: kelp_prices wasn't used in fair value, maybe remove?
            "kelp_vwap": self.kelp_vwap,     # Note: kelp_vwap wasn't used, maybe remove?
            "kelp_price_history": self.kelp_price_history,
            "resin_trade_prices": self.resin_trade_prices,
            # Add any other state variables you want to persist
        }
        # Use jsonpickle to serialize the state dictionary
        traderData = "" # Default empty string
        try:
             traderData = jsonpickle.encode(trader_state_to_save, unpicklable=False) # Make non-python specific
        except Exception as e:
             logger.print(f"Error encoding trader data with jsonpickle: {e}")
             # Fallback or handle error - maybe save less data or use simple json
             try:
                 traderData = json.dumps(trader_state_to_save)
             except Exception as json_e:
                 logger.print(f"Error encoding trader data with json.dumps: {json_e}")
                 traderData = "{}" # Empty JSON object as fallback


        # --- Log and Return ---
        # Flush logs, state, orders, etc. for the visualizer/runner
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData

# Note: Ensure that the TradingState object passed to `run` includes a `traderData`
# attribute (string) from the previous timestep for state persistence to work.
# The `ProsperityEncoder` and `datamodel` classes are assumed to be provided by the environment.