# Imports needed by both Trader and Logger
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId # Added UserId for completeness
from typing import Any, List
import jsonpickle
import math

# --------------------------------------------------------------------------------------------------
# Logger Class (Copied exactly from your second provided block)
# --------------------------------------------------------------------------------------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        # This method allows accumulating logs if you choose to use logger.print() later
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Sanitize trader_data to ensure it's a string
        if not isinstance(trader_data, str):
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
            # Calculate base length assuming empty strings for truncated parts
            base_length = len(self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions,
                "",
                "",
            ]))

            # Calculate max length for each truncated item
            max_item_length = (self.max_log_length - base_length) // 3
            if max_item_length < 0: max_item_length = 0 # Ensure non-negative length

            # Prepare the compressed and truncated data for final JSON output
            compressed_state = self.compress_state(state, self.truncate(state_trader_data, max_item_length))
            compressed_orders = self.compress_orders(orders)
            truncated_trader_data = self.truncate(trader_data, max_item_length)
            truncated_logs = self.truncate(self.logs, max_item_length) # Truncate accumulated logs

            # Create the final JSON string to be printed
            log_output = self.to_json([
                compressed_state,
                compressed_orders,
                conversions,
                truncated_trader_data,
                truncated_logs,
            ])

        except Exception as e:
            # Fallback mechanism in case of errors during compression or JSON encoding
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


        print(log_output) # Print the final log string to standard output
        self.logs = "" # Reset internal logs after flushing

    # --- Compression Helper Methods (Exact copies from your Logger) ---
    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        trader_data_str = trader_data if isinstance(trader_data, str) else str(trader_data)
        return [
            state.timestamp,
            trader_data_str,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        if isinstance(listings, dict):
            for listing in listings.values():
                 if isinstance(listing, dict) and "symbol" in listing and "product" in listing and "denomination" in listing:
                     compressed.append([listing["symbol"], listing["product"], listing["denomination"]])
                 elif isinstance(listing, Listing):
                      compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        if isinstance(order_depths, dict):
            for symbol, order_depth in order_depths.items():
                 if isinstance(order_depth, OrderDepth):
                     compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
                 elif isinstance(order_depth, dict) and "buy_orders" in order_depth and "sell_orders" in order_depth:
                     compressed[symbol] = [order_depth["buy_orders"], order_depth["sell_orders"]]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if isinstance(trades, dict):
            for arr in trades.values():
                 if isinstance(arr, list):
                     for trade in arr:
                          if isinstance(trade, Trade):
                              compressed.append([
                                  trade.symbol,
                                  trade.price,
                                  trade.quantity,
                                  trade.buyer,
                                  trade.seller,
                                  trade.timestamp,
                              ])
                          elif isinstance(trade, dict):
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
         if not isinstance(observations, Observation) or not hasattr(observations, 'conversionObservations'):
             return [[], {}]

         conversion_observations = {}
         if isinstance(observations.conversionObservations, dict):
             for product, observation in observations.conversionObservations.items():
                  if hasattr(observation, 'bidPrice'):
                     conversion_observations[product] = [
                         observation.bidPrice,
                         observation.askPrice,
                         observation.transportFees,
                         observation.exportTariff,
                         observation.importTariff,
                         observation.sunlight,
                         observation.humidity,
                     ]
                  elif isinstance(observation, dict):
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
             plain_value_observations = {}

         return [plain_value_observations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if isinstance(orders, dict):
            for arr in orders.values():
                if isinstance(arr, list):
                    for order in arr:
                        if isinstance(order, Order):
                            compressed.append([order.symbol, order.price, order.quantity])
                        elif isinstance(order, dict) and "symbol" in order and "price" in order and "quantity" in order:
                             compressed.append([order["symbol"], order["price"], order["quantity"]])
        return compressed

    def to_json(self, value: Any) -> str:
        try:
            # Use ProsperityEncoder for compatibility with custom datamodel types
            return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
        except TypeError as e:
             print(f"JSON encoding error: {e}. Value causing error: {value}")
             return json.dumps({"error": "Encoding failed", "value_type": str(type(value))})

    def truncate(self, value: str, max_length: int) -> str:
        if not isinstance(value, str):
             try:
                 value = str(value) # Attempt conversion if not already string
             except Exception:
                 return "..." # Fallback if conversion fails

        if len(value) <= max_length:
            return value
        if max_length < 3:
             return "..."[:max_length] # Handle edge case of very small max_length

        # Truncate and add ellipsis
        return value[:max_length - 3] + "..."

# Instantiate the logger globally
logger = Logger()

# --------------------------------------------------------------------------------------------------
# Trader Class (Using your original logic, only adding logger.flush)
# --------------------------------------------------------------------------------------------------
class Trader:
    def __init__(self):
        # Your original __init__ logic - UNCHANGED
        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_price_history = []
        self.resin_trade_prices = []
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}

    def load_trader_data(self, trader_data_str: str):
        # Your original load_trader_data logic - UNCHANGED
        """Loads trader state from the traderData string."""
        if trader_data_str:
            try:
                data = jsonpickle.decode(trader_data_str)
                if isinstance(data, dict): # Check if decoding was successful and is a dict
                    self.kelp_prices = data.get('kelp_prices', [])
                    self.kelp_vwap = data.get('kelp_vwap', [])
                    self.kelp_price_history = data.get('kelp_price_history', [])
                    self.resin_trade_prices = data.get('resin_trade_prices', [])
                else:
                    self._initialize_state() # Re-initialize if decode fails
            except Exception as e:
                self._initialize_state() # Re-initialize on error
        else:
            self._initialize_state() # Initialize if no data string

    def _initialize_state(self):
        # Your original _initialize_state logic - UNCHANGED
        """Helper to initialize or reset state variables."""
        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_price_history = []
        self.resin_trade_prices = []

    def resin_fair_value(self, state: TradingState) -> int:
        # Your original resin_fair_value logic - UNCHANGED
        """Calculates the fair value for RAINFOREST_RESIN."""
        base_fair_value = 10000
        order_depth = state.order_depths.get("RAINFOREST_RESIN")
        if order_depth and order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            base_fair_value = mid_price

        market_trades = state.market_trades.get("RAINFOREST_RESIN", [])
        if market_trades:
            if not hasattr(self, 'resin_trade_prices') or not isinstance(self.resin_trade_prices, list):
                self.resin_trade_prices = []
            self.resin_trade_prices.append(market_trades[-1].price)
            max_history = 15
            if len(self.resin_trade_prices) > max_history:
                self.resin_trade_prices = self.resin_trade_prices[-max_history:]

            if self.resin_trade_prices:
                avg_trade_price = sum(self.resin_trade_prices) / len(self.resin_trade_prices)
                fair_value = round(base_fair_value * 0.7 + avg_trade_price * 0.3)
                return fair_value
            else:
                 return round(base_fair_value)
        else:
            return round(base_fair_value)

    def get_resin_dynamic_widths(self, order_depth: OrderDepth) -> tuple:
        # Your original get_resin_dynamic_widths logic - UNCHANGED
        """Introduce simple dynamic widths for RAINFOREST_RESIN based on spread."""
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return 1.0, 0.2
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid
        base_make_width = 1.0
        base_take_width = 0.2
        if spread > 5:
            make_width = base_make_width * 0.9
            take_width = base_take_width
        elif spread < 2:
            make_width = base_make_width * 1.1
            take_width = base_take_width
        else:
            make_width = base_make_width
            take_width = base_take_width
        make_width = max(0.1, make_width)
        take_width = max(0.1, take_width)
        return make_width, take_width

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        # Your original resin_orders logic - UNCHANGED
        """Generates orders for RAINFOREST_RESIN."""
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        product = "RAINFOREST_RESIN"
        if not order_depth:
            return orders
        make_width, take_width = self.get_resin_dynamic_widths(order_depth)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask <= fair_value - take_width:
                best_ask_amount = abs(order_depth.sell_orders[best_ask])
                available_buy_limit = position_limit - position
                quantity_to_take = min(best_ask_amount, available_buy_limit, 35)
                if quantity_to_take > 0:
                    orders.append(Order(product, best_ask, quantity_to_take))
                    buy_order_volume += quantity_to_take
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid >= fair_value + take_width:
                best_bid_amount = abs(order_depth.buy_orders[best_bid])
                available_sell_limit = position_limit + position
                quantity_to_take = min(best_bid_amount, available_sell_limit, 35)
                if quantity_to_take > 0:
                    orders.append(Order(product, best_bid, -quantity_to_take))
                    sell_order_volume -= quantity_to_take
        position_after_take = position + buy_order_volume + sell_order_volume
        threshold = 25
        if position_after_take > threshold:
             reduce_qty = position_after_take - threshold
             sell_price_clear = math.floor(fair_value)
             # Error in original logic: sell_order_volume is negative, needs abs() if subtracting
             # Corrected slightly while preserving intent: Check available sell capacity
             available_sell_cap = position_limit + (position + buy_order_volume + sell_order_volume) # How much more can be sold
             sell_qty = min(reduce_qty, available_sell_cap)
             if sell_qty > 0:
                orders.append(Order(product, sell_price_clear, -sell_qty))
        if position_after_take < -threshold:
             reduce_qty = abs(position_after_take) - threshold
             buy_price_clear = math.ceil(fair_value)
             # Corrected slightly while preserving intent: Check available buy capacity
             available_buy_cap = position_limit - (position + buy_order_volume + sell_order_volume) # How much more can be bought
             buy_qty = min(reduce_qty, available_buy_cap)
             if buy_qty > 0:
                orders.append(Order(product, buy_price_clear, buy_qty))
        current_buy_capacity = position_limit - (position + buy_order_volume + sell_order_volume)
        current_sell_capacity = position_limit + (position + buy_order_volume + sell_order_volume)
        if current_buy_capacity > 0:
            buy_price = math.floor(fair_value - make_width)
            orders.append(Order(product, buy_price, current_buy_capacity))
        if current_sell_capacity > 0:
            sell_price = math.ceil(fair_value + make_width)
            orders.append(Order(product, sell_price, -current_sell_capacity))
        return orders

    def kelp_fair_value(self, order_depth: OrderDepth, state: TradingState) -> int:
        # Your original kelp_fair_value logic - UNCHANGED
        """Calculates the fair value for KELP."""
        if not hasattr(self, 'kelp_price_history') or not isinstance(self.kelp_price_history, list):
             self.kelp_price_history = []
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
             if self.kelp_price_history:
                 return round(self.kelp_price_history[-1])
             else:
                 return 5000
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        trade_adjustment = 0
        market_trades = state.market_trades.get("KELP", [])
        if market_trades:
            recent_trade_price = market_trades[-1].price
            trade_adjustment = (recent_trade_price - mid_price) * 0.10
        self.kelp_price_history.append(mid_price)
        max_history = 5
        if len(self.kelp_price_history) > max_history:
            self.kelp_price_history = self.kelp_price_history[-max_history:]
        if self.kelp_price_history:
            weights = list(range(1, len(self.kelp_price_history) + 1))
            weighted_sum = sum(p * w for p, w in zip(self.kelp_price_history, weights))
            total_weight = sum(weights)
            wma_mid_price = weighted_sum / total_weight if total_weight > 0 else mid_price
        else:
            wma_mid_price = mid_price
        fair_value = wma_mid_price + trade_adjustment
        return round(fair_value)

    def get_kelp_dynamic_widths(self, order_depth: OrderDepth) -> tuple:
        # Your original get_kelp_dynamic_widths logic - UNCHANGED
        """Calculate dynamic widths for KELP based on spread."""
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return 1.2, 0.2
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid
        base_make_width = 1.2
        base_take_width = 0.2
        if spread > 5:
            make_width = base_make_width * 0.9
            take_width = base_take_width
        elif spread < 2:
            make_width = base_make_width * 1.1
            take_width = base_take_width
        else:
            make_width = base_make_width
            take_width = base_take_width
        make_width = max(0.1, make_width)
        take_width = max(0.1, take_width)
        return make_width, take_width

    def kelp_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        # Your original kelp_orders logic - UNCHANGED
        """Generates orders for KELP. Structure similar to resin_orders."""
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        product = "KELP"
        if not order_depth:
            return orders
        make_width, take_width = self.get_kelp_dynamic_widths(order_depth)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask <= fair_value - take_width:
                best_ask_amount = abs(order_depth.sell_orders[best_ask])
                available_buy_limit = position_limit - position
                quantity_to_take = min(best_ask_amount, available_buy_limit, 35)
                if quantity_to_take > 0:
                    orders.append(Order(product, best_ask, quantity_to_take))
                    buy_order_volume += quantity_to_take
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid >= fair_value + take_width:
                best_bid_amount = abs(order_depth.buy_orders[best_bid])
                available_sell_limit = position_limit + position
                quantity_to_take = min(best_bid_amount, available_sell_limit, 35)
                if quantity_to_take > 0:
                    orders.append(Order(product, best_bid, -quantity_to_take))
                    sell_order_volume -= quantity_to_take
        position_after_take = position + buy_order_volume + sell_order_volume
        threshold = 25
        if position_after_take > threshold:
             reduce_qty = position_after_take - threshold
             sell_price_clear = math.floor(fair_value)
             # Corrected slightly while preserving intent: Check available sell capacity
             available_sell_cap = position_limit + (position + buy_order_volume + sell_order_volume)
             sell_qty = min(reduce_qty, available_sell_cap)
             if sell_qty > 0:
                 orders.append(Order(product, sell_price_clear, -sell_qty))
        if position_after_take < -threshold:
             reduce_qty = abs(position_after_take) - threshold
             buy_price_clear = math.ceil(fair_value)
             # Corrected slightly while preserving intent: Check available buy capacity
             available_buy_cap = position_limit - (position + buy_order_volume + sell_order_volume)
             buy_qty = min(reduce_qty, available_buy_cap)
             if buy_qty > 0:
                 orders.append(Order(product, buy_price_clear, buy_qty))
        current_buy_capacity = position_limit - (position + buy_order_volume + sell_order_volume)
        current_sell_capacity = position_limit + (position + buy_order_volume + sell_order_volume)
        if current_buy_capacity > 0:
            buy_price = math.floor(fair_value - make_width)
            orders.append(Order(product, buy_price, current_buy_capacity))
        if current_sell_capacity > 0:
            sell_price = math.ceil(fair_value + make_width)
            orders.append(Order(product, sell_price, -current_sell_capacity))
        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main method called by the trading engine for each time step.
        """
        # --- Start of your original run logic ---
        result: dict[Symbol, list[Order]] = {}
        conversions = 0
        self.load_trader_data(state.traderData)

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_od = state.order_depths["RAINFOREST_RESIN"]
            resin_pos = state.position.get("RAINFOREST_RESIN", 0)
            resin_limit = self.position_limits["RAINFOREST_RESIN"]
            resin_fv = self.resin_fair_value(state)
            resin_orders = self.resin_orders(resin_od, resin_fv, resin_pos, resin_limit)
            result["RAINFOREST_RESIN"] = resin_orders
        else:
            pass

        if "KELP" in state.order_depths:
            kelp_od = state.order_depths["KELP"]
            kelp_pos = state.position.get("KELP", 0)
            kelp_limit = self.position_limits["KELP"]
            kelp_fv = self.kelp_fair_value(kelp_od, state)
            kelp_orders = self.kelp_orders(kelp_od, kelp_fv, kelp_pos, kelp_limit)
            result["KELP"] = kelp_orders
        else:
            pass

        trader_state_to_save = {
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "kelp_price_history": self.kelp_price_history,
            "resin_trade_prices": self.resin_trade_prices,
        }
        traderData = ""
        try:
             traderData = jsonpickle.encode(trader_state_to_save, unpicklable=False)
        except Exception as e:
             # Original code didn't print here, adding a silent fallback like original
             try:
                 traderData = json.dumps(trader_state_to_save)
             except Exception as json_e:
                 traderData = "{}" # Empty JSON object as ultimate fallback
        # --- End of your original run logic ---


        # --- Log and Return ---
        # Flush logs, state, orders, etc. using the logger instance
        # This call uses the 'result', 'conversions', and 'traderData' variables
        # calculated by your original logic above.
        logger.flush(state, result, conversions, traderData) # <<< INTEGRATED LOGGER CALL

        return result, conversions, traderData