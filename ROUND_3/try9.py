# Add necessary imports at the top
import math
import numpy as np
from scipy.stats import norm  # For Black-Scholes N(d1), N(d2)
import json, jsonpickle # Added json for safety
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Listing, Trade, Observation, ConversionObservation
from typing import List, Dict, Tuple, Any

# --- Logger Class (Keep as is) ---
class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def flush(self, state: TradingState, orders: dict[str, list[Order]], conversions: int, trader_data: str):
        try:
            # Log critical state info first
            limited_log = f"Pos:{json.dumps(state.position, cls=ProsperityEncoder, separators=(',', ':'))} "
            limited_log += f"IVs:{trader_data[:100]}" # Log start of trader data (IVs)

            # Append trader logs, ensuring not to overflow significantly
            max_extra_log = self.max_log_length - len(limited_log) - 200 # Reserve space for base structure
            if len(self.logs) > max_extra_log:
                limited_log += self.logs[:max_extra_log] + "..."
            else:
                limited_log += self.logs

            # Basic structure logging (empty for brevity in actual logs)
            print(json.dumps({
                "state": [state.timestamp], # Minimal state info
                "orders": self.compress_orders(orders),
                "conversions": conversions,
                "traderData": self.truncate(trader_data, (self.max_log_length // 3) - 50),
                "logs": self.truncate(limited_log, (self.max_log_length // 2)) # Give logs more space
            }, cls=ProsperityEncoder, separators=(',', ':')))

        except Exception as e:
            print(f"Error during logging: {e}") # Print error directly if logging fails

        self.logs = "" # Clear logs after flushing


    # Keep compress methods, simplify state/obs compression if needed
    def compress_state(self, state: TradingState, trader_data: str): # Keep original compress methods if they work
       # Simplified version for brevity if needed
       return [state.timestamp, trader_data, {}, self.compress_order_depths(state.order_depths), {}, {}, state.position, {}]

    def compress_listings(self, listings: dict[str, Listing]):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[str, OrderDepth]):
         # Limit depth to save space? Optional.
         return {s: [list(od.buy_orders.items())[:2], list(od.sell_orders.items())[:2]] for s, od in order_depths.items()} # Log only top 2 levels


    def compress_trades(self, trades: dict[str, list[Trade]]):
         # Limit number of trades logged? Optional.
         return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in trades.values() for t in arr[-5:]] # Log last 5 trades


    def compress_observations(self, obs: Observation):
        co = {}
        if hasattr(obs, 'conversionObservations'):
             co = {p:[o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sunlight, o.humidity] # Adjusted fields based on potential Observation structure
                  for p, o in obs.conversionObservations.items()}
        pv_obs = {}
        if hasattr(obs, 'plainValueObservations'):
             pv_obs = obs.plainValueObservations # Assuming it's already a dict-like structure
        return [pv_obs, co]


    def compress_orders(self, orders: dict[str, list[Order]]):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value): # Keep original to_json
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int): # Keep original truncate
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

# --- Product Class Definition (Corrected and Consolidated) ---
class Product:
    # Original Products
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1" # Assumed based on PARAMS key
    DJEMBES = "DJEMBES"
    CROISSANT = "CROISSANTS" # Note: Your PARAMS uses "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET2 = "PICNIC_BASKET2" # Assumed based on PARAMS key

    # Added Products (from PARAMS/logic)
    SPREAD1 = "SPREAD1" # Used as a key in PARAMS and traderData
    SPREAD2 = "SPREAD2" # Used as a key in PARAMS and traderData
    ARTIFICAL1 = "ARTIFICAL1" # Used in execute_spreads
    ARTIFICAL2 = "ARTIFICAL2" # Not used currently, but good to have if logic expands

    # Volcanic Products
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

# --- Configuration (PARAMS, Weights, Volcanic List) ---
PARAMS = {
    # Original Params (ensure product names match Product class)
    Product.RAINFOREST_RESIN: {"fair_value": 10000, "take_width": 1, "clear_width": 0, "disregard_edge": 1, "join_edge": 2, "default_edge": 1, "soft_position_limit": 50},
    Product.KELP: {"take_width": 2, "clear_width": 0, "prevent_adverse": False, "adverse_volume": 15, "reversion_beta": -0.18, "disregard_edge": 2, "join_edge": 0, "default_edge": 1, "ink_adjustment_factor": 0.05},
    Product.SQUID_INK: {"take_width": 2, "clear_width": 1, "prevent_adverse": False, "adverse_volume": 15, "reversion_beta": -0.228, "disregard_edge": 2, "join_edge": 0, "default_edge": 1, "spike_lb": 3, "spike_ub": 5.6, "offset": 2, "reversion_window": 55, "reversion_weight": 0.12},
    Product.SPREAD1: {"default_spread_mean": 48.777856, "default_spread_std": 85.119723, "spread_window": 55, "zscore_threshold": 4, "target_position": 100},
    Product.SPREAD2: {"default_spread_mean": 30.2336, "default_spread_std": 59.8536, "spread_window": 59, "zscore_threshold": 6, "target_position": 100},
    Product.PICNIC_BASKET1: {"b2_adjustment_factor": 0.05}, # Keep if used, otherwise remove

    # Volcanic Params
    Product.VOLCANIC_ROCK_VOUCHER_9500: {"strike": 9500.0, "iv_ema_alpha": 0.3, "base_spread": 1.0, "inventory_skew": 0.1, "position_limit": 100},
    Product.VOLCANIC_ROCK_VOUCHER_9750: {"strike": 9750.0, "iv_ema_alpha": 0.3, "base_spread": 1.0, "inventory_skew": 0.1, "position_limit": 100},
    Product.VOLCANIC_ROCK_VOUCHER_10000: {"strike": 10000.0, "iv_ema_alpha": 0.3, "base_spread": 1.0, "inventory_skew": 0.1, "position_limit": 100},
    Product.VOLCANIC_ROCK_VOUCHER_10250: {"strike": 10250.0, "iv_ema_alpha": 0.3, "base_spread": 1.0, "inventory_skew": 0.1, "position_limit": 100},
    Product.VOLCANIC_ROCK_VOUCHER_10500: {"strike": 10500.0, "iv_ema_alpha": 0.3, "base_spread": 1.0, "inventory_skew": 0.1, "position_limit": 100},

    # Default IVs (Adjust based on analysis)
    "DEFAULT_IV": {
        9500.0: 0.205, # From your EDA summary
        9750.0: 0.185, # From your EDA summary
        10000.0: 0.164, # From your EDA summary
        10250.0: 0.153, # From your EDA summary
        10500.0: 0.156, # From your EDA summary
    }
}

PICNIC1_WEIGHTS = {Product.DJEMBES: 1, Product.CROISSANT: 6, Product.JAMS: 3}
PICNIC2_WEIGHTS = {Product.CROISSANT: 4, Product.JAMS: 2}

VOLCANIC_PRODUCTS = [
    Product.VOLCANIC_ROCK_VOUCHER_9500,
    Product.VOLCANIC_ROCK_VOUCHER_9750,
    Product.VOLCANIC_ROCK_VOUCHER_10000,
    Product.VOLCANIC_ROCK_VOUCHER_10250,
    Product.VOLCANIC_ROCK_VOUCHER_10500,
]

# --- Trader Class ---
class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        # Corrected PRODUCT_LIMIT using the defined Product class attributes
        self.PRODUCT_LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANT: 250, # Ensure keys match PARAMS and Product class
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            # Add voucher limits from PARAMS
            **{p: self.params[p]["position_limit"] for p in VOLCANIC_PRODUCTS}
        }
        # Default traderData structure
        self.traderData = {
            "volcanic_ivs": {prod: None for prod in VOLCANIC_PRODUCTS},
            Product.SPREAD1: {"spread_history": [], "prev_zscore": 0, "clear_flag": False, "curr_avg": 0},
            Product.SPREAD2: {"spread_history": [], "prev_zscore": 0, "clear_flag": False, "curr_avg": 0},
            "kelp_last_price": None,
            "ink_last_price": None,
            "ink_price_history": [],
            "currentSpike": False,
            "recoveryValue": None,
        }
        self.logger = Logger() # Instantiate logger inside trader

    # --- Helper Functions (WMP, TTE) ---
    def calculate_wmp(self, order_depth: OrderDepth) -> float | None:
        """Calculates Weighted Mid-Price"""
        # Ensure keys exist before accessing
        buy_keys = list(order_depth.buy_orders.keys())
        sell_keys = list(order_depth.sell_orders.keys())

        if not buy_keys or not sell_keys:
            return None

        best_bid = max(buy_keys)
        best_ask = min(sell_keys)

        # Handle potential KeyError if keys disappear between checks (though unlikely)
        try:
            bid_vol = order_depth.buy_orders[best_bid]
            ask_vol = abs(order_depth.sell_orders[best_ask])
        except KeyError:
            return None # Or recalculate best bid/ask

        if bid_vol + ask_vol == 0:
            return (best_bid + best_ask) / 2 # Fallback to simple mid
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)


    def calculate_tte(self, state: TradingState) -> float:
        """Calculates Time To Expiry as a fraction of the total round duration."""
        total_duration = 3 * 1_000_000 # 3 days * 1M timestamps/day
        # Day starts from 0, add 1 to get number of full days passed
        time_elapsed = state.day * 1_000_000 + state.timestamp
        time_left = total_duration - time_elapsed
        # Use fraction of *trading* days remaining (e.g., 1/3, 2/3, 3/3) - adjust if needed
        # TTE is typically expressed in years, but for short expiries, using fraction of total duration is fine for BS consistency
        return max(time_left / total_duration, 1e-9) # Ensure positive


    # --- Black-Scholes Implementation (BS Price, IV) ---
    def bs_price(self, S, K, T, sigma, r=0.0) -> float:
        """Calculates Black-Scholes price for a European call option."""
        if sigma <= 1e-6 or T <= 1e-9: # Use small thresholds
            return max(0.0, S - K * math.exp(-r * T)) # Intrinsic value approx for zero time/vol

        # Protect against math domain errors
        if S <= 0 or K <= 0:
             return max(0.0, S - K * math.exp(-r * T)) # Or return 0? Max ensures non-negativity

        try:
            d1_num = math.log(S / K) + (r + 0.5 * sigma**2) * T
            d1_den = sigma * math.sqrt(T)
            if abs(d1_den) < 1e-9: # Avoid division by zero
                 return max(0.0, S - K * math.exp(-r * T)) # Return intrinsic if denominator is zero
            d1 = d1_num / d1_den
            d2 = d1 - d1_den # Reuse denominator calculation

            price = (S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
            return max(price, 0.0) # Price cannot be negative
        except (ValueError, OverflowError) as e:
             self.logger.logs += f" WARN: bs_price math error S={S} K={K} T={T} sig={sigma} - {e}"
             return max(0.0, S - K * math.exp(-r * T)) # Fallback to intrinsic


    def implied_volatility(self, C, S, K, T, r=0.0, initial_guess=0.5, tolerance=1e-5, max_iterations=50) -> float | None:
        """
        Calculates Implied Volatility using Newton-Raphson method.
        Returns None if calculation fails or price is outside bounds.
        """
        # Basic input validation
        if C <= 0 or S <= 0 or K <= 0 or T <= 0:
            #self.logger.logs += f" IV FAIL: Non-positive input C={C:.2f} S={S:.2f} K={K} T={T:.4f} "
            return None

        # Check for arbitrage violation (price < intrinsic value, allowing for small tolerance)
        intrinsic = max(0.0, S - K * math.exp(-r * T))
        if C < intrinsic - tolerance:
            #self.logger.logs += f" IV FAIL: Price {C:.2f} < Intrinsic {intrinsic:.2f} "
            return None
        # Check if price is too high (can happen with bad data)
        if C > S: # Call price cannot exceed stock price
            #self.logger.logs += f" IV FAIL: Price {C:.2f} > Stock {S:.2f} "
            return None


        sigma = max(initial_guess, 0.01) # Ensure initial guess is positive

        for i in range(max_iterations):
             if sigma <= 1e-6: # Sigma too small
                  #self.logger.logs += f" IV FAIL: Sigma too small {sigma:.4g} iter {i} "
                  return None

             # Calculate BS price and Vega using current sigma
             try:
                 # Reuse d1/d2 logic from bs_price for consistency
                 d1_num = math.log(S / K) + (r + 0.5 * sigma**2) * T
                 d1_den = sigma * math.sqrt(T)
                 if abs(d1_den) < 1e-9: # Vega would be zero or unstable
                      #self.logger.logs += f" IV FAIL: d1_den near zero iter {i} "
                      return None
                 d1 = d1_num / d1_den

                 bs_val = self.bs_price(S, K, T, sigma, r) # Call BS function
                 # Vega calculation: S * N'(d1) * sqrt(T)
                 vega = S * norm.pdf(d1) * math.sqrt(T)

             except (ValueError, OverflowError) as e:
                  #self.logger.logs += f" IV FAIL: BS/Vega Math Error {e} sigma={sigma:.4f} iter {i} "
                  # Try reducing sigma maybe? Or just fail.
                  sigma *= 0.5 # Heuristic: If math error, try lower vol
                  continue # Skip this iteration's update

             # Check for convergence
             diff = bs_val - C
             if abs(diff) < tolerance:
                 # Return sigma, ensuring it's within reasonable bounds
                 return max(1e-5, min(sigma, 5.0)) # Clamp to 0.01% - 500%

             # Check if Vega is too small (Newton-Raphson step becomes huge/unstable)
             if abs(vega) < tolerance:
                  #self.logger.logs += f" IV FAIL: Vega too small {vega:.4g} iter {i} "
                  return None

             # Newton-Raphson step
             sigma = sigma - diff / vega

             # Clamp sigma within reasonable bounds during iteration
             sigma = max(1e-5, min(sigma, 5.0))


        #self.logger.logs += f" IV FAIL: Max iterations ({max_iterations}) reached for K={K} C={C:.2f} S={S:.2f} T={T:.4f}"
        return None # Failed to converge


    # --- Volcanic Voucher Trading Logic ---
    def trade_volcanic_vouchers(self, state: TradingState, traderData: dict) -> Dict[str, List[Order]]:
        """Generates orders for Volcanic Rock Vouchers based on BS model."""
        orders = {prod: [] for prod in VOLCANIC_PRODUCTS}
        positions = state.position
        order_depths = state.order_depths

        # 1. Get Underlying Price (S) and Time to Expiry (T)
        if Product.VOLCANIC_ROCK not in order_depths:
            self.logger.logs += " WARN: No VOLCANIC_ROCK order depth. Skipping voucher trading."
            return orders
        S = self.calculate_wmp(order_depths[Product.VOLCANIC_ROCK])
        if S is None:
            self.logger.logs += " WARN: Could not calculate VOLCANIC_ROCK WMP. Skipping voucher trading."
            # Optionally, use last known price or mid-price as fallback? For now, skip.
            return orders
        T = self.calculate_tte(state)

        # 2. Iterate through each voucher
        for product in VOLCANIC_PRODUCTS:
            if product not in order_depths or not order_depths[product].buy_orders or not order_depths[product].sell_orders:
                self.logger.logs += f" {product}: Skipping due to missing order depth. "
                continue # Skip if no data or empty book for this voucher

            strike = self.params[product]["strike"]
            position_limit = self.params[product]["position_limit"]
            current_position = positions.get(product, 0)
            voucher_wmp = self.calculate_wmp(order_depths[product])

            if voucher_wmp is None:
                self.logger.logs += f" WARN: Could not calculate WMP for {product}. "
                continue # Skip if cannot calculate voucher price

            # 3. Calculate and Smooth Implied Volatility (IV)
            # Ensure the structure exists in traderData
            if "volcanic_ivs" not in traderData: traderData["volcanic_ivs"] = {}
            if product not in traderData["volcanic_ivs"]: traderData["volcanic_ivs"][product] = None

            last_smoothed_iv = traderData["volcanic_ivs"].get(product)
            default_iv = self.params["DEFAULT_IV"].get(strike, 0.25) # Use default if no history or param

            # Use last smoothed IV as initial guess if available, otherwise use default
            initial_iv_guess = last_smoothed_iv if last_smoothed_iv is not None else default_iv
            initial_iv_guess = max(0.01, min(initial_iv_guess, 4.0)) # Keep guess reasonable

            current_iv = self.implied_volatility(voucher_wmp, S, strike, T, initial_guess=initial_iv_guess)

            smoothed_iv = last_smoothed_iv # Start with the previous value

            if current_iv is not None:
                # Update smoothed IV using EMA
                alpha = self.params[product]["iv_ema_alpha"]
                if last_smoothed_iv is None:
                    smoothed_iv = current_iv # Initialize
                else:
                    # Ensure last_smoothed_iv is a float before EMA calculation
                    if isinstance(last_smoothed_iv, (float, int)):
                         smoothed_iv = alpha * current_iv + (1 - alpha) * last_smoothed_iv
                    else:
                         smoothed_iv = current_iv # Fallback initialization if last_smoothed_iv was bad data
                traderData["volcanic_ivs"][product] = smoothed_iv # Store updated IV
            elif smoothed_iv is None:
                # If calculation failed and no history exists, use default
                smoothed_iv = default_iv
                traderData["volcanic_ivs"][product] = smoothed_iv # Store default
                self.logger.logs += f" {product}: IV calc failed, using default {smoothed_iv:.4f}. "
            else:
                # If calculation failed but we have a previous value, reuse it
                # Ensure smoothed_iv is valid before reusing
                if isinstance(smoothed_iv, (float, int)):
                     self.logger.logs += f" {product}: IV calc failed, reusing last smoothed {smoothed_iv:.4f}. "
                     pass # Keep the existing valid smoothed_iv
                else:
                     # If previous smoothed_iv was also invalid, reset to default
                     smoothed_iv = default_iv
                     traderData["volcanic_ivs"][product] = smoothed_iv
                     self.logger.logs += f" {product}: IV calc failed & last smoothed invalid, using default {smoothed_iv:.4f}. "


            # Ensure smoothed_iv is valid for BS calculation
            if not isinstance(smoothed_iv, (float, int)) or smoothed_iv <= 0:
                 self.logger.logs += f" {product}: Invalid smoothed_iv {smoothed_iv}, using default {default_iv}. "
                 smoothed_iv = default_iv


            # 4. Calculate BS Fair Value using Smoothed IV
            bs_fair_value = self.bs_price(S, strike, T, smoothed_iv)

            # 5. Determine Bid/Ask Prices for Market Making
            base_spread = self.params[product]["base_spread"]
            inventory_skew_param = self.params[product]["inventory_skew"] # Renamed to avoid conflict

            # Skew spread based on inventory: lower ask/raise bid if long, raise ask/lower bid if short
            # Ensure position_limit is not zero
            limit_for_skew = position_limit if position_limit != 0 else 1
            skew_amount = (current_position / limit_for_skew) * inventory_skew_param * (base_spread / 2) # Scale skew by spread

            # Calculate raw bid/ask based on fair value and base spread
            raw_bid = bs_fair_value - base_spread / 2
            raw_ask = bs_fair_value + base_spread / 2

            # Apply skew
            skewed_bid = raw_bid - skew_amount
            skewed_ask = raw_ask - skew_amount # Apply same skew amount to maintain spread width initially

            # Round to nearest tick (assuming integer ticks)
            final_bid = math.floor(skewed_bid)
            final_ask = math.ceil(skewed_ask)

            # Ensure minimum spread of 1 tick
            if final_ask <= final_bid:
                final_ask = final_bid + 1

            # 6. Place Market Making Orders respecting limits
            buy_capacity = position_limit - current_position
            if buy_capacity > 0:
                # Place order slightly smaller than capacity to avoid hitting limit exactly? Optional.
                order_size = buy_capacity # Or max(1, buy_capacity - 1) etc.
                orders[product].append(Order(product, final_bid, order_size))

            sell_capacity = position_limit + current_position # position is negative if short
            if sell_capacity > 0:
                 order_size = sell_capacity
                 orders[product].append(Order(product, final_ask, -order_size))

            # Optional: Log key values for debugging
            # self.logger.logs += (f" {product[-5:]}: S={S:.1f} K={strike} T={T:.4f} V_WMP={voucher_wmp:.2f} "
            #                      f"IV={current_iv if current_iv else -1:.3f} SmIV={smoothed_iv:.3f} FV={bs_fair_value:.2f} "
            #                      f"Bid={final_bid} Ask={final_ask} Pos={current_position} |")

        return orders

    # --- Existing Strategy Methods (Unchanged) ---
    # (Keep existing methods: take_best_orders, market_make, clear_position_order,
    #  kelp_fair_value, ink_fair_value, take_orders, clear_orders, make_orders,
    #  artifical_order_depth, convert_orders, execute_spreads, spread_orders,
    #  trade_resin)
    # Make sure these methods correctly use self.params and self.PRODUCT_LIMIT

    def take_best_orders(self,product:str,fair_value:float,take_width:float,orders:List[Order],order_depth:OrderDepth,position:int,buy_order_volume:int,sell_order_volume:int,prevent_adverse:bool=False,adverse_volume:int=0,traderObject:dict=None):
      position_limit=self.PRODUCT_LIMIT[product]
      # SQUID_INK specific logic (ensure traderObject is passed correctly)
      if product=="SQUID_INK" and traderObject is not None: # Check if traderObject exists
        if "currentSpike" not in traderObject: traderObject["currentSpike"]=False
        prev_price = traderObject.get("ink_last_price", fair_value) # Get last price or use current fair
        if traderObject["currentSpike"]:
            if abs(fair_value - prev_price) < self.params[Product.SQUID_INK]["spike_lb"]:
                traderObject["currentSpike"] = False # End spike state
            else: # Still in spike recovery
                if fair_value < traderObject["recoveryValue"]: # Trying to recover downwards, look to buy back cheaper
                    if order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_amount = -order_depth.sell_orders[best_ask] # Positive volume
                        quantity = min(best_ask_amount, position_limit - position)
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            # Don't modify order_depth here, let the main loop handle fills
                    return buy_order_volume, 0 # Return only buy volume change
                else: # Trying to recover upwards, look to sell higher
                    if order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_amount = order_depth.buy_orders[best_bid]
                        quantity = min(best_bid_amount, position_limit + position) # position_limit + current_pos
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -quantity))
                            sell_order_volume += quantity
                    return 0, sell_order_volume # Return only sell volume change

        # Check for new spike
        if abs(fair_value - prev_price) > self.params[Product.SQUID_INK]["spike_ub"]:
            traderObject["currentSpike"] = True
            traderObject["recoveryValue"] = prev_price + self.params[Product.SQUID_INK]["offset"] if fair_value > prev_price else prev_price - self.params[Product.SQUID_INK]["offset"]
            if fair_value > prev_price: # Price spiked up, sell immediately
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                return 0, sell_order_volume
            else: # Price spiked down, buy immediately
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = -order_depth.sell_orders[best_ask]
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                return buy_order_volume, 0


      # Original take_best_orders logic (outside of spike)
      if len(order_depth.sell_orders)!=0:
        best_ask=min(order_depth.sell_orders.keys())
        best_ask_amount=-order_depth.sell_orders[best_ask]
        # Check adverse volume condition if enabled
        is_adverse = prevent_adverse and abs(best_ask_amount) > adverse_volume
        if not is_adverse:
          if best_ask<=fair_value-take_width:
            quantity=min(best_ask_amount, position_limit-position)
            if quantity>0:
                orders.append(Order(product,best_ask,quantity))
                buy_order_volume+=quantity
                # Don't modify order_depth here

      if len(order_depth.buy_orders)!=0:
        best_bid=max(order_depth.buy_orders.keys())
        best_bid_amount=order_depth.buy_orders[best_bid]
        # Check adverse volume condition if enabled
        is_adverse = prevent_adverse and abs(best_bid_amount) > adverse_volume
        if not is_adverse:
          if best_bid>=fair_value+take_width:
            quantity=min(best_bid_amount, position_limit+position)
            if quantity>0:
                orders.append(Order(product,best_bid,-quantity))
                sell_order_volume+=quantity
                # Don't modify order_depth here

      return buy_order_volume,sell_order_volume

    def market_make(self,product:str,orders:List[Order],bid:int,ask:int,position:int,buy_order_volume:int,sell_order_volume:int):
        position_limit = self.PRODUCT_LIMIT.get(product, 0) # Get limit for the product
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, math.floor(bid), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, math.ceil(ask), -sell_quantity))
        # This function modifies the 'orders' list directly, returns updated volumes
        return buy_order_volume, sell_order_volume # Volumes aren't actually changed here, just passed through

    def clear_position_order(self,product:str,fair_value:float,width:int,orders:List[Order],order_depth:OrderDepth,position:int,buy_order_volume:int,sell_order_volume:int):
      position_limit = self.PRODUCT_LIMIT.get(product, 0) # Get limit for the product
      position_after_take = position + buy_order_volume - sell_order_volume

      # Use integers for price comparison
      fair_for_bid = math.floor(fair_value - width)
      fair_for_ask = math.ceil(fair_value + width)

      buy_capacity = position_limit - (position + buy_order_volume) # How much more we CAN buy
      sell_capacity = position_limit + (position - sell_order_volume) # How much more we CAN sell

      # If position is positive (long), try to sell down towards zero
      if position_after_take > 0:
        # Find volume available to hit at or above the target ask price
        clear_quantity_available = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
        # How much we WANT to sell (either the full position or what's available)
        quantity_to_clear = min(position_after_take, clear_quantity_available)
        # How much we CAN sell (limited by sell capacity)
        sent_quantity = min(sell_capacity, quantity_to_clear)

        if sent_quantity > 0:
          orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
          sell_order_volume += abs(sent_quantity)

      # If position is negative (short), try to buy back towards zero
      if position_after_take < 0:
        # Find volume available to hit at or below the target bid price
        clear_quantity_available = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
         # How much we WANT to buy (either the full position size or what's available)
        quantity_to_clear = min(abs(position_after_take), clear_quantity_available)
        # How much we CAN buy (limited by buy capacity)
        sent_quantity = min(buy_capacity, quantity_to_clear)

        if sent_quantity > 0:
          orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
          buy_order_volume += abs(sent_quantity)

      return buy_order_volume, sell_order_volume


    def kelp_fair_value(self,order_depth:OrderDepth,traderObject,ink_order_depth:OrderDepth):
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return traderObject.get('kelp_last_price', None) # Return last price if no book

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # Calculate mid-price based on 'adverse_volume' weighted participants if possible
        kelp_params = self.params.get(Product.KELP, {})
        adverse_vol = kelp_params.get("adverse_volume", 15)

        valid_ask_prices = [price for price, volume in order_depth.sell_orders.items() if abs(volume) >= adverse_vol]
        valid_buy_prices = [price for price, volume in order_depth.buy_orders.items() if abs(volume) >= adverse_vol]

        mm_ask = min(valid_ask_prices) if valid_ask_prices else best_ask
        mm_bid = max(valid_buy_prices) if valid_buy_prices else best_bid

        mmmid_price = (mm_ask + mm_bid) / 2

        # Reversion Component
        last_price = traderObject.get("kelp_last_price")
        fair = mmmid_price
        if last_price is not None and last_price != 0:
            reversion_beta = kelp_params.get("reversion_beta", -0.18)
            price_return = (mmmid_price - last_price) / last_price
            fair = mmmid_price + (mmmid_price * price_return * reversion_beta)

        # Ink Adjustment Component (ensure ink book exists)
        if ink_order_depth and ink_order_depth.sell_orders and ink_order_depth.buy_orders and "ink_last_price" in traderObject:
            ink_params = self.params.get(Product.SQUID_INK, {})
            ink_adverse_vol = ink_params.get("adverse_volume", 15)
            old_ink_price = traderObject["ink_last_price"]

            ink_valid_asks = [p for p, v in ink_order_depth.sell_orders.items() if abs(v) >= ink_adverse_vol]
            ink_valid_buys = [p for p, v in ink_order_depth.buy_orders.items() if abs(v) >= ink_adverse_vol]

            ink_mm_ask = min(ink_valid_asks) if ink_valid_asks else min(ink_order_depth.sell_orders.keys())
            ink_mm_bid = max(ink_valid_buys) if ink_valid_buys else max(ink_order_depth.buy_orders.keys())
            new_ink_mid = (ink_mm_ask + ink_mm_bid) / 2

            if old_ink_price != 0:
                 ink_return = (new_ink_mid - old_ink_price) / old_ink_price
                 ink_adj_factor = kelp_params.get("ink_adjustment_factor", 0.05)
                 fair = fair - (ink_adj_factor * ink_return * mmmid_price) # Adjust based on ink return


        traderObject["kelp_last_price"] = mmmid_price # Store the *unadjusted* mid price for next calculation
        return fair


    def ink_fair_value(self,order_depth:OrderDepth,traderObject):
        if not order_depth.sell_orders or not order_depth.buy_orders:
             return traderObject.get('ink_last_price', None) # Return last price if no book

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # Calculate mid-price based on 'adverse_volume'
        ink_params = self.params.get(Product.SQUID_INK, {})
        adverse_vol = ink_params.get("adverse_volume", 15)

        valid_ask_prices = [price for price, volume in order_depth.sell_orders.items() if abs(volume) >= adverse_vol]
        valid_buy_prices = [price for price, volume in order_depth.buy_orders.items() if abs(volume) >= adverse_vol]

        mm_ask = min(valid_ask_prices) if valid_ask_prices else best_ask
        mm_bid = max(valid_buy_prices) if valid_buy_prices else best_bid
        mmmid_price = (mm_ask + mm_bid) / 2

        # Update price history
        if traderObject.get('ink_price_history', None) is None: traderObject['ink_price_history'] = []
        traderObject['ink_price_history'].append(mmmid_price)
        reversion_window = ink_params.get("reversion_window", 55)
        if len(traderObject['ink_price_history']) > reversion_window:
            traderObject['ink_price_history'] = traderObject['ink_price_history'][-reversion_window:]

        # Calculate adaptive beta
        default_beta = ink_params.get("reversion_beta", -0.228)
        adaptive_beta = default_beta # Default value
        if len(traderObject['ink_price_history']) >= max(2, reversion_window // 2): # Need enough data points
            prices = np.array(traderObject['ink_price_history'])
            if np.all(prices > 0): # Ensure prices are positive
                returns = (prices[1:] - prices[:-1]) / prices[:-1]
                if len(returns) > 1:
                    X = returns[:-1]
                    Y = returns[1:]
                    # Simple linear regression for beta (could use statsmodels for robustness)
                    X_mean, Y_mean = np.mean(X), np.mean(Y)
                    numerator = np.sum((X - X_mean) * (Y - Y_mean))
                    denominator = np.sum((X - X_mean)**2)
                    if denominator != 0:
                         estimated_beta = numerator / denominator
                         # Ensure beta is negative (mean-reverting)
                         estimated_beta = min(0, estimated_beta) # Cap at 0
                         reversion_weight = ink_params.get("reversion_weight", 0.12)
                         adaptive_beta = (reversion_weight * estimated_beta + (1 - reversion_weight) * default_beta)


        # Calculate fair value with reversion
        last_price = traderObject.get("ink_last_price")
        fair = mmmid_price
        if last_price is not None and last_price != 0:
            price_return = (mmmid_price - last_price) / last_price
            fair = mmmid_price + (mmmid_price * price_return * adaptive_beta)

        traderObject["ink_last_price"] = mmmid_price # Store the *unadjusted* mid price
        return fair

    # --- Generic Order Placement Wrappers ---
    def take_orders(self,product:str,order_depth:OrderDepth,fair_value:float,take_width:float,position:int,prevent_adverse:bool=False,adverse_volume:int=0,traderObject:dict=None):
        orders: List[Order] = []
        # Pass traderObject if it's SQUID_INK
        td_obj = traderObject if product == Product.SQUID_INK else None
        buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, take_width, orders, order_depth, position, 0, 0, prevent_adverse, adverse_volume, td_obj)
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self,product:str,order_depth:OrderDepth,fair_value:float,clear_width:int,position:int,buy_order_volume:int,sell_order_volume:int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume)
        return orders, buy_order_volume, sell_order_volume

    def make_orders(self,product,order_depth:OrderDepth,fair_value:float,position:int,buy_order_volume:int,sell_order_volume:int,disregard_edge:float,join_edge:float,default_edge:float,manage_position:bool=False,soft_position_limit:int=0):
        orders: List[Order] = []
        prod_params = self.params.get(product, {})

        # Get edges/widths from params for this product
        disregard_edge = prod_params.get('disregard_edge', disregard_edge) # Use specific or default
        join_edge = prod_params.get('join_edge', join_edge)
        default_edge = prod_params.get('default_edge', default_edge)

        # --- Calculate Bid/Ask based on edges ---
        best_ask_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else fair_value + default_edge
        best_bid_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else fair_value - default_edge

        # Determine Ask Price
        ask_price = fair_value + default_edge # Default
        if best_ask_price > fair_value + disregard_edge: # Competitor far away?
             if best_ask_price <= fair_value + join_edge: # Far but joinable?
                 ask_price = best_ask_price - 1 # Undercut slightly
             else: # Too far, place at default or closer edge
                 ask_price = fair_value + default_edge # Or potentially fair_value + disregard_edge + 1?

        # Determine Bid Price
        bid_price = fair_value - default_edge # Default
        if best_bid_price < fair_value - disregard_edge: # Competitor far away?
             if best_bid_price >= fair_value - join_edge: # Far but joinable?
                 bid_price = best_bid_price + 1 # Overbid slightly
             else: # Too far, place at default or closer edge
                 bid_price = fair_value - default_edge # Or potentially fair_value - disregard_edge - 1?

        # Ensure ask > bid
        bid_price = math.floor(bid_price)
        ask_price = math.ceil(ask_price)
        if ask_price <= bid_price: ask_price = bid_price + 1

        # Apply position management skew if enabled
        if manage_position and soft_position_limit > 0:
            limit = self.PRODUCT_LIMIT.get(product, 0)
            if position > soft_position_limit:
                # Long, skew down to encourage selling
                ask_price -= 1 # Make ask more attractive
                # bid_price -= 1 # Optionally skew bid down too
            elif position < -soft_position_limit:
                # Short, skew up to encourage buying
                bid_price += 1 # Make bid more attractive
                # ask_price += 1 # Optionally skew ask up too

        # Recalculate to ensure ask > bid after skew
        if ask_price <= bid_price: ask_price = bid_price + 1

        # Place orders using market_make helper
        buy_order_volume, sell_order_volume = self.market_make(product, orders, bid_price, ask_price, position, buy_order_volume, sell_order_volume)
        return orders, buy_order_volume, sell_order_volume

    # --- Spread Trading Methods (Unchanged) ---
    def artifical_order_depth(self, order_depths:Dict[str,OrderDepth], picnic1:bool=True) -> OrderDepth:
        artifical_order_price = OrderDepth()

        # Get component order depths safely
        croissant_od = order_depths.get(Product.CROISSANT)
        jams_od = order_depths.get(Product.JAMS)
        djembes_od = order_depths.get(Product.DJEMBES) if picnic1 else None

        # Check if all necessary components exist
        if not croissant_od or not jams_od or (picnic1 and not djembes_od):
             return artifical_order_price # Return empty if components missing

        croissant_buy_orders = croissant_od.buy_orders
        croissant_sell_orders = croissant_od.sell_orders
        jams_buy_orders = jams_od.buy_orders
        jams_sell_orders = jams_od.sell_orders
        if picnic1:
            djembes_buy_orders = djembes_od.buy_orders
            djembes_sell_orders = djembes_od.sell_orders

        # Determine weights
        if picnic1:
            DJEMBES_W, CROISSANT_W, JAM_W = PICNIC1_WEIGHTS[Product.DJEMBES], PICNIC1_WEIGHTS[Product.CROISSANT], PICNIC1_WEIGHTS[Product.JAMS]
        else:
            CROISSANT_W, JAM_W = PICNIC2_WEIGHTS[Product.CROISSANT], PICNIC2_WEIGHTS[Product.JAMS]
            DJEMBES_W = 0 # For clarity in calculation

        # Calculate Artificial Bid
        croissant_best_bid = max(croissant_buy_orders.keys()) if croissant_buy_orders else 0
        jams_best_bid = max(jams_buy_orders.keys()) if jams_buy_orders else 0
        djembes_best_bid = max(djembes_buy_orders.keys()) if picnic1 and djembes_buy_orders else 0

        # Only calculate if all best bids are valid
        if croissant_best_bid > 0 and jams_best_bid > 0 and (not picnic1 or djembes_best_bid > 0):
             art_bid_price = (djembes_best_bid * DJEMBES_W +
                              croissant_best_bid * CROISSANT_W +
                              jams_best_bid * JAM_W)

             # Calculate volume based on minimum available component volume at best bid
             croissant_bid_vol = croissant_buy_orders[croissant_best_bid] // CROISSANT_W if CROISSANT_W else float('inf')
             jams_bid_vol = jams_buy_orders[jams_best_bid] // JAM_W if JAM_W else float('inf')
             djembes_bid_vol = djembes_buy_orders[djembes_best_bid] // DJEMBES_W if picnic1 and DJEMBES_W else float('inf')

             artifical_bid_volume = min(djembes_bid_vol, croissant_bid_vol, jams_bid_vol)
             if artifical_bid_volume > 0:
                 artifical_order_price.buy_orders[art_bid_price] = artifical_bid_volume

        # Calculate Artificial Ask
        croissant_best_ask = min(croissant_sell_orders.keys()) if croissant_sell_orders else float('inf')
        jams_best_ask = min(jams_sell_orders.keys()) if jams_sell_orders else float('inf')
        djembes_best_ask = min(djembes_sell_orders.keys()) if picnic1 and djembes_sell_orders else float('inf')

        # Only calculate if all best asks are valid
        if croissant_best_ask < float('inf') and jams_best_ask < float('inf') and (not picnic1 or djembes_best_ask < float('inf')):
             art_ask_price = (djembes_best_ask * DJEMBES_W +
                              croissant_best_ask * CROISSANT_W +
                              jams_best_ask * JAM_W)

             # Calculate volume based on minimum available component volume at best ask
             croissant_ask_vol = abs(croissant_sell_orders[croissant_best_ask]) // CROISSANT_W if CROISSANT_W else float('inf')
             jams_ask_vol = abs(jams_sell_orders[jams_best_ask]) // JAM_W if JAM_W else float('inf')
             djembes_ask_vol = abs(djembes_sell_orders[djembes_best_ask]) // DJEMBES_W if picnic1 and DJEMBES_W else float('inf')

             artifical_ask_volume = min(djembes_ask_vol, croissant_ask_vol, jams_ask_vol)
             if artifical_ask_volume > 0:
                 artifical_order_price.sell_orders[art_ask_price] = -artifical_ask_volume # Negative sign for sell orders

        return artifical_order_price


    def convert_orders(self, artifical_orders: List[Order], order_depths: Dict[str, OrderDepth], picnic1: bool = True) -> Dict[str, List[Order]]:
        component_products = [Product.DJEMBES, Product.CROISSANT, Product.JAMS] if picnic1 else [Product.CROISSANT, Product.JAMS]
        component_orders: Dict[str, List[Order]] = {prod: [] for prod in component_products}

        # Get component order depths safely
        component_ods = {prod: order_depths.get(prod) for prod in component_products}
        if any(od is None for od in component_ods.values()):
            self.logger.logs += " WARN: Missing component OD in convert_orders."
            return component_orders # Return empty if any component OD is missing

        artfical_order_depth = self.artifical_order_depth(order_depths, picnic1) # Recalculate for current state
        art_best_bid = max(artfical_order_depth.buy_orders.keys()) if artfical_order_depth.buy_orders else 0
        art_best_ask = min(artfical_order_depth.sell_orders.keys()) if artfical_order_depth.sell_orders else float('inf')

        for order in artifical_orders:
            price = order.price
            quantity = order.quantity
            component_prices = {}

            # Determine component prices based on order direction
            if quantity > 0 and price >= art_best_ask: # Buying the artificial -> Buying components at their asks
                for prod in component_products:
                    if component_ods[prod].sell_orders:
                        component_prices[prod] = min(component_ods[prod].sell_orders.keys())
                    else:
                        component_prices = {}; break # Cannot execute if any component ask is missing
            elif quantity < 0 and price <= art_best_bid: # Selling the artificial -> Selling components at their bids
                 for prod in component_products:
                    if component_ods[prod].buy_orders:
                         component_prices[prod] = max(component_ods[prod].buy_orders.keys())
                    else:
                         component_prices = {}; break # Cannot execute if any component bid is missing
            else:
                continue # Order is not executable at current market prices

            # If we found prices for all components, create the orders
            if len(component_prices) == len(component_products):
                weights = PICNIC1_WEIGHTS if picnic1 else PICNIC2_WEIGHTS
                for prod in component_products:
                    comp_quantity = quantity * weights[prod] # Positive for buy, negative for sell
                    component_orders[prod].append(Order(prod, component_prices[prod], comp_quantity))

        return component_orders

    def execute_spreads(self,target_position:int,picnic_position:int,order_depths:Dict[str,OrderDepth],picnic1:bool=True) -> Dict[str, List[Order]] | None:
        if target_position == picnic_position: return None # Already at target

        target_quantity = abs(target_position - picnic_position)
        if target_quantity == 0: return None

        picnic_product = Product.PICNIC_BASKET1 if picnic1 else Product.PICNIC_BASKET2
        picnic_order_depth = order_depths.get(picnic_product)
        artifical_order_depth = self.artifical_order_depth(order_depths, picnic1) # Recalculate

        if not picnic_order_depth or (not picnic_order_depth.buy_orders and not picnic_order_depth.sell_orders) or \
           (not artifical_order_depth.buy_orders and not artifical_order_depth.sell_orders):
            # self.logger.logs += f" SPREAD EXEC: Missing book for {picnic_product} or artificial. "
            return None # Cannot execute if books are empty

        aggregate_orders: Dict[str, List[Order]] = {}
        execute_volume = 0

        if target_position > picnic_position: # Need to BUY picnic, SELL artificial
            picnic_ask_price = min(picnic_order_depth.sell_orders.keys()) if picnic_order_depth.sell_orders else float('inf')
            artifical_bid_price = max(artifical_order_depth.buy_orders.keys()) if artifical_order_depth.buy_orders else 0

            if picnic_ask_price == float('inf') or artifical_bid_price == 0:
                 # self.logger.logs += " SPREAD EXEC BUY: Cannot find picnic ask or artif bid. "
                 return None # Cannot execute

            picnic_ask_vol = abs(picnic_order_depth.sell_orders[picnic_ask_price])
            artifical_bid_vol = abs(artifical_order_depth.buy_orders[artifical_bid_price])

            orderbook_volume = min(picnic_ask_vol, artifical_bid_vol)
            execute_volume = min(orderbook_volume, target_quantity)

            if execute_volume > 0:
                 picnic_orders = [Order(picnic_product, picnic_ask_price, execute_volume)]
                 artifical_orders_to_convert = [Order(Product.ARTIFICAL1, artifical_bid_price, -execute_volume)] # Sell artificial
                 aggregate_orders = self.convert_orders(artifical_orders_to_convert, order_depths, picnic1)
                 aggregate_orders[picnic_product] = picnic_orders # Add the picnic leg
                 # self.logger.logs += f" SPREAD EXEC BUY: Vol={execute_volume} @ PicnicAsk={picnic_ask_price}, ArtifBid={artifical_bid_price}"


        else: # Need to SELL picnic, BUY artificial (target_position < picnic_position)
            picnic_bid_price = max(picnic_order_depth.buy_orders.keys()) if picnic_order_depth.buy_orders else 0
            artifical_ask_price = min(artifical_order_depth.sell_orders.keys()) if artifical_order_depth.sell_orders else float('inf')

            if picnic_bid_price == 0 or artifical_ask_price == float('inf'):
                # self.logger.logs += " SPREAD EXEC SELL: Cannot find picnic bid or artif ask. "
                return None # Cannot execute

            picnic_bid_vol = abs(picnic_order_depth.buy_orders[picnic_bid_price])
            artifical_ask_vol = abs(artifical_order_depth.sell_orders[artifical_ask_price])

            orderbook_volume = min(picnic_bid_vol, artifical_ask_vol)
            execute_volume = min(orderbook_volume, target_quantity)

            if execute_volume > 0:
                picnic_orders = [Order(picnic_product, picnic_bid_price, -execute_volume)]
                artifical_orders_to_convert = [Order(Product.ARTIFICAL1, artifical_ask_price, execute_volume)] # Buy artificial
                aggregate_orders = self.convert_orders(artifical_orders_to_convert, order_depths, picnic1)
                aggregate_orders[picnic_product] = picnic_orders # Add the picnic leg
                # self.logger.logs += f" SPREAD EXEC SELL: Vol={execute_volume} @ PicnicBid={picnic_bid_price}, ArtifAsk={artifical_ask_price}"


        return aggregate_orders if execute_volume > 0 else None


    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: str, picnic_position: int, spread_data: Dict[str, Any], SPREAD: str, picnic1: bool = True) -> Dict[str, List[Order]] | None:
        picnic_product = Product.PICNIC_BASKET1 if picnic1 else Product.PICNIC_BASKET2

        # Check if necessary data exists
        if picnic_product not in order_depths or not order_depths.get(picnic_product):
             # self.logger.logs += f" {SPREAD}: Missing {picnic_product} OD. "
             return None
        if SPREAD not in self.params:
             self.logger.logs += f" {SPREAD}: Params not found. "
             return None

        picnic_order_depth = order_depths[picnic_product]
        artifical_order_depth = self.artifical_order_depth(order_depths, picnic1) # Recalculate

        # Calculate Picnic WMP
        picnic_wmp = self.calculate_wmp(picnic_order_depth)
        if picnic_wmp is None:
             # self.logger.logs += f" {SPREAD}: Could not calculate Picnic WMP for {picnic_product}. "
             return None

        # Calculate Artificial WMP
        artifical_wmp = self.calculate_wmp(artifical_order_depth)
        if artifical_wmp is None:
             # self.logger.logs += f" {SPREAD}: Could not calculate Artificial WMP. "
             return None

        # Calculate current spread
        spread = picnic_wmp - artifical_wmp

        # Update spread history
        if "spread_history" not in spread_data: spread_data["spread_history"] = []
        spread_data["spread_history"].append(spread)
        spread_window = self.params[SPREAD].get("spread_window", 50) # Default window
        if len(spread_data["spread_history"]) > spread_window:
             spread_data["spread_history"].pop(0) # Keep window size

        # Calculate Z-score (only if enough history)
        if len(spread_data["spread_history"]) < spread_window // 2 : # Need reasonable amount of data
             # self.logger.logs += f" {SPREAD}: Not enough history ({len(spread_data['spread_history'])}/{spread_window}). "
             return None

        spread_history_np = np.array(spread_data["spread_history"])
        spread_mean = np.mean(spread_history_np) # Use rolling mean
        spread_std = np.std(spread_history_np)
        default_std = self.params[SPREAD].get("default_spread_std", 1.0) # Use default if calculated is too small

        # Use robust std dev or default if calculated is near zero
        effective_std = max(spread_std, default_std * 0.1) # Prevent division by zero/tiny std
        if effective_std == 0:
             # self.logger.logs += f" {SPREAD}: Effective std is zero. "
             return None

        zscore = (spread - spread_mean) / effective_std
        # self.logger.logs += f" {SPREAD}: Spread={spread:.2f} Mean={spread_mean:.2f} Std={effective_std:.2f} Z={zscore:.2f} Pos={picnic_position} "


        # Trading Logic based on Z-score
        zscore_threshold = self.params[SPREAD].get("zscore_threshold", 2.0)
        target_position_size = self.params[SPREAD].get("target_position", 100)

        # Condition to SELL spread (Picnic low, Artificial high) -> Go SHORT Picnic
        if zscore <= -zscore_threshold:
             target = -target_position_size
             if picnic_position > target: # Only execute if not already short enough
                 # self.logger.logs += f" {SPREAD}: Z <= {-zscore_threshold:.2f}, Target: {target}, Current: {picnic_position}. Executing Short."
                 return self.execute_spreads(target, picnic_position, order_depths, picnic1)

        # Condition to BUY spread (Picnic high, Artificial low) -> Go LONG Picnic
        elif zscore >= zscore_threshold:
             target = target_position_size
             if picnic_position < target: # Only execute if not already long enough
                 # self.logger.logs += f" {SPREAD}: Z >= {zscore_threshold:.2f}, Target: {target}, Current: {picnic_position}. Executing Long."
                 return self.execute_spreads(target, picnic_position, order_depths, picnic1)

        # Condition to close position (optional - if Z-score reverts towards zero)
        # Example: Close if |zscore| < 0.5 and current position is non-zero
        close_threshold = 0.5
        if abs(zscore) < close_threshold and picnic_position != 0:
             target = 0
             # self.logger.logs += f" {SPREAD}: Z near zero ({zscore:.2f}), Target: {target}, Current: {picnic_position}. Closing."
             return self.execute_spreads(target, picnic_position, order_depths, picnic1)


        # No execution condition met
        spread_data["prev_zscore"] = zscore # Store zscore for potential future logic
        return None


    def trade_resin(self,state):
        # --- RAINFOREST_RESIN Strategy ---
        # Simple Market Making around perceived fair value (10000)
        # Uses logic from the original submission (sniper + MM)
        product = Product.RAINFOREST_RESIN
        params = self.params.get(product, {})
        fair_value = params.get("fair_value", 10000) # Get fair value from params
        position_limit = self.PRODUCT_LIMIT.get(product, 50)
        current_position = state.position.get(product, 0)

        orders: List[Order] = []
        order_depth = state.order_depths.get(product)

        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return orders # Cannot trade if book is empty

        # --- Sniper Logic (Take aggressive orders crossing fair value) ---
        buy_capacity = position_limit - current_position
        sell_capacity = position_limit + current_position

        # Snipe asks below fair value
        asks_to_snipe = {p: v for p, v in order_depth.sell_orders.items() if p < fair_value}
        for price, volume in sorted(asks_to_snipe.items()): # Take lowest asks first
             if buy_capacity <= 0: break
             take_vol = min(abs(volume), buy_capacity)
             orders.append(Order(product, price, take_vol))
             buy_capacity -= take_vol
             # self.logger.logs += f" RESIN SNIPE BUY: {take_vol}@{price} "

        # Snipe bids above fair value
        bids_to_snipe = {p: v for p, v in order_depth.buy_orders.items() if p > fair_value}
        for price, volume in sorted(bids_to_snipe.items(), reverse=True): # Take highest bids first
             if sell_capacity <= 0: break
             take_vol = min(abs(volume), sell_capacity)
             orders.append(Order(product, price, -take_vol))
             sell_capacity -= take_vol
             # self.logger.logs += f" RESIN SNIPE SELL: {take_vol}@{price} "


        # --- Market Making Logic ---
        # Update position based on snipe orders placed (approximate)
        current_position = state.position.get(product, 0) + (position_limit - current_position - buy_capacity) - (position_limit + state.position.get(product, 0) - sell_capacity)

        # Calculate remaining capacity
        buy_capacity = position_limit - current_position
        sell_capacity = position_limit + current_position

        # Determine MM bid/ask (simple fixed spread around fair value)
        mm_bid_price = fair_value - 1
        mm_ask_price = fair_value + 1

        # Place MM orders if capacity allows
        if buy_capacity > 0:
             orders.append(Order(product, mm_bid_price, buy_capacity))
        if sell_capacity > 0:
             orders.append(Order(product, mm_ask_price, -sell_capacity))
             
        # self.logger.logs += f" RESIN MM: Bid {buy_capacity}@{mm_bid_price}, Ask {-sell_capacity}@{mm_ask_price} | Pos: {current_position}"

        return orders


    # --- Main Run Method ---
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # Load traderData safely
        currentTraderData = self.traderData.copy() # Start with default structure
        if state.traderData is not None and state.traderData != "":
            try:
                loaded_data = jsonpickle.decode(state.traderData)
                # Deep merge loaded data into default structure (handle nested dicts)
                def merge_dicts(default, loaded):
                    merged = default.copy()
                    for key, value in loaded.items():
                        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                            merged[key] = merge_dicts(merged[key], value)
                        else:
                            merged[key] = value # Overwrite or add new key
                    return merged

                currentTraderData = merge_dicts(self.traderData, loaded_data)

            except Exception as e:
                self.logger.logs += f" ERROR decoding traderData: {e}. Resetting to default."
                # currentTraderData remains the default copy

        result: Dict[str, List[Order]] = {} # Stores orders for all products


        # --- Execute strategies for each product group ---

        # 1. Rainforest Resin
        try:
            if Product.RAINFOREST_RESIN in state.order_depths:
                 resin_orders = self.trade_resin(state)
                 if resin_orders: result[Product.RAINFOREST_RESIN] = resin_orders
        except Exception as e:
            self.logger.logs += f" ERR {Product.RAINFOREST_RESIN}: {e} "
            # import traceback; self.logger.logs += traceback.format_exc()


        # 2. Kelp & Squid Ink
        kelp_ink_orders = {Product.KELP: [], Product.SQUID_INK: []}
        try:
            if Product.KELP in self.params and Product.KELP in state.order_depths and Product.SQUID_INK in state.order_depths:
                kelp_position = state.position.get(Product.KELP, 0)
                ink_od = state.order_depths.get(Product.SQUID_INK) # Pass ink OD to kelp FV calc
                kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], currentTraderData, ink_od)

                if kelp_fair_value is not None:
                    kelp_params = self.params[Product.KELP]
                    kelp_take, buy_vol, sell_vol = self.take_orders(Product.KELP, state.order_depths[Product.KELP], kelp_fair_value, kelp_params['take_width'], kelp_position, kelp_params['prevent_adverse'], kelp_params['adverse_volume'], currentTraderData)
                    kelp_clear, buy_vol, sell_vol = self.clear_orders(Product.KELP, state.order_depths[Product.KELP], kelp_fair_value, kelp_params['clear_width'], kelp_position, buy_vol, sell_vol)
                    kelp_make, _, _ = self.make_orders(Product.KELP, state.order_depths[Product.KELP], kelp_fair_value, kelp_position, buy_vol, sell_vol, kelp_params['disregard_edge'], kelp_params['join_edge'], kelp_params['default_edge'])
                    kelp_ink_orders[Product.KELP].extend(kelp_take + kelp_clear + kelp_make)
                else:
                    self.logger.logs += f" {Product.KELP}: FV calc failed. "
        except Exception as e:
             self.logger.logs += f" ERR {Product.KELP}: {e} "
             # import traceback; self.logger.logs += traceback.format_exc()

        try:
             if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
                ink_position = state.position.get(Product.SQUID_INK, 0)
                ink_fair_value = self.ink_fair_value(state.order_depths[Product.SQUID_INK], currentTraderData)

                if ink_fair_value is not None:
                    ink_params = self.params[Product.SQUID_INK]
                    # Pass currentTraderData for SQUID_INK specific logic inside take_orders
                    ink_take, buy_vol, sell_vol = self.take_orders(Product.SQUID_INK, state.order_depths[Product.SQUID_INK], ink_fair_value, ink_params['take_width'], ink_position, ink_params['prevent_adverse'], ink_params['adverse_volume'], currentTraderData)
                    ink_clear, buy_vol, sell_vol = self.clear_orders(Product.SQUID_INK, state.order_depths[Product.SQUID_INK], ink_fair_value, ink_params['clear_width'], ink_position, buy_vol, sell_vol)
                    ink_make, _, _ = self.make_orders(Product.SQUID_INK, state.order_depths[Product.SQUID_INK], ink_fair_value, ink_position, buy_vol, sell_vol, ink_params['disregard_edge'], ink_params['join_edge'], ink_params['default_edge'])
                    kelp_ink_orders[Product.SQUID_INK].extend(ink_take + ink_clear + ink_make)
                else:
                    self.logger.logs += f" {Product.SQUID_INK}: FV calc failed. "
        except Exception as e:
             self.logger.logs += f" ERR {Product.SQUID_INK}: {e} "
             # import traceback; self.logger.logs += traceback.format_exc()

        # Add non-empty kelp/ink orders to result
        if kelp_ink_orders[Product.KELP]: result[Product.KELP] = kelp_ink_orders[Product.KELP]
        if kelp_ink_orders[Product.SQUID_INK]: result[Product.SQUID_INK] = kelp_ink_orders[Product.SQUID_INK]


        # 3. Spread Trading
        spread1_exec_orders = None
        try:
            picnic1_position = state.position.get(Product.PICNIC_BASKET1, 0)
            # Pass the specific sub-dict for spread1 data
            spread1_exec_orders = self.spread_orders(state.order_depths, Product.PICNIC_BASKET1, picnic1_position, currentTraderData.setdefault(Product.SPREAD1, {"spread_history": [], "prev_zscore": 0}), SPREAD=Product.SPREAD1, picnic1=True)
        except Exception as e:
            self.logger.logs += f" ERR {Product.SPREAD1}: {e} "
            # import traceback; self.logger.logs += traceback.format_exc()

        spread2_exec_orders = None
        try:
            picnic2_position = state.position.get(Product.PICNIC_BASKET2, 0)
            # Pass the specific sub-dict for spread2 data
            spread2_exec_orders = self.spread_orders(state.order_depths, Product.PICNIC_BASKET2, picnic2_position, currentTraderData.setdefault(Product.SPREAD2, {"spread_history": [], "prev_zscore": 0}), SPREAD=Product.SPREAD2, picnic1=False)
        except Exception as e:
            self.logger.logs += f" ERR {Product.SPREAD2}: {e} "
            # import traceback; self.logger.logs += traceback.format_exc()

        # Merge spread execution orders carefully (they contain multiple products)
        for spread_orders in [spread1_exec_orders, spread2_exec_orders]:
            if spread_orders:
                for product, orders in spread_orders.items():
                    if orders: # Ensure there are orders
                         if product in result:
                             result[product].extend(orders)
                         else:
                             result[product] = orders


        # 4. Volcanic Voucher Strategy
        try:
            volcanic_orders = self.trade_volcanic_vouchers(state, currentTraderData)
            # Merge volcanic orders
            for product, orders in volcanic_orders.items():
                if orders: # Only add if there are orders for this product
                    if product in result:
                        result[product].extend(orders)
                    else:
                        result[product] = orders
        except Exception as e:
            self.logger.logs += f" ERR VOLCANIC: {e} "
            import traceback # Keep traceback for debugging complex issues
            self.logger.logs += traceback.format_exc()


        # --- Final Steps ---
        # Serialize the potentially modified traderData
        # Use default=str to handle potential non-serializable types if needed
        traderData_encoded = jsonpickle.encode(currentTraderData, unpicklable=False)

        conversions = 0 # Set conversions if needed

        self.logger.flush(state, result, conversions, traderData_encoded)

        return result, conversions, traderData_encoded