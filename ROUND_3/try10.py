import json
import jsonpickle
import numpy as np
import math
import statistics
from typing import Any, List, Dict, Tuple
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


# --------------------------
# Constants and Parameters
# --------------------------
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
STRIKES = [9500, 9750, 10000, 10250, 10500]
VOUCHER_SYMBOLS = [f"{VOUCHER_PREFIX}{K}" for K in STRIKES]
ALL_SYMBOLS = [VOLCANIC_ROCK] + VOUCHER_SYMBOLS

POSITION_LIMITS = {
    VOLCANIC_ROCK: 400,
    **{symbol: 200 for symbol in VOUCHER_SYMBOLS}
}

# Volatility Smile Parameters (Quadratic: a*m_t^2 + b*m_t + c)
# Coeffs: [a, b, c] -> [-1.12938253, -0.0421697, 0.00938588]
SMILE_COEFFS = [-1.12938253, -0.0421697, 0.00938588]
BASE_IV = 0.009385877054057754 # Corresponds to c, IV at m_t=0

RISK_FREE_RATE = 0 # Assume 0 risk-free rate as none is provided
DAYS_PER_YEAR = 365.0
TOTAL_DURATION_DAYS = 8 # Starts at 8 days, expires after day 7 ends

# Trading Parameters
MISPRICING_THRESHOLD_STD_DEV = 1.0 # Number of std deviations from smile to trigger trade
ORDER_QUANTITY = 10             # Quantity per order
IV_HISTORY_LENGTH = 100        # Length of base IV history to keep for analysis


# --------------------------
# Black-Scholes Helper Functions
# --------------------------
def norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    """Probability density function for the standard normal distribution."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the price of a European call option using the Black-Scholes formula.
    S: Underlying price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate
    sigma: Volatility
    """
    if T <= 0 or sigma <= 0:
        # Handle expired or zero volatility cases: return intrinsic value
        return max(0.0, S - K)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = (S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))
    return call_price

def vega(S, K, T, r, sigma):
    """Calculates the Vega of a European option."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    vega_value = S * norm_pdf(d1) * math.sqrt(T)
    return vega_value

def implied_volatility(target_price, S, K, T, r, initial_guess=0.1, max_iterations=100, tolerance=1e-6):
    """
    Calculates the implied volatility using the Newton-Raphson method.
    target_price: Market price of the option
    S, K, T, r: Black-Scholes parameters
    initial_guess: Starting volatility for the iteration
    """
    sigma = initial_guess

    # Check for arbitrage bounds
    intrinsic_value = max(0.0, S - K * math.exp(-r * T))
    if target_price < intrinsic_value - tolerance: # Option price below intrinsic value
        return 0.0 # Or handle as error / invalid
    if target_price < 0:
        return 0.0 # Price cannot be negative

    for _ in range(max_iterations):
        price = black_scholes_call(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)

        diff = price - target_price

        if abs(diff) < tolerance:
            return sigma

        if v < 1e-10: # Vega is too small, cannot converge
            # Fallback: try slightly different sigma or return error/last guess
            # Try perturbing sigma slightly if vega is zero, e.g., if at intrinsic value boundary
            if sigma < 0.01: sigma = 0.01
            price_perturbed = black_scholes_call(S, K, T, r, sigma*1.01)
            v_perturbed = vega(S, K, T, r, sigma*1.01)
            if v_perturbed < 1e-10: break # Give up if still no vega
            sigma = sigma * 1.01 # Perturb
            continue # Retry iteration with perturbed sigma

        sigma = sigma - diff / v

        # Ensure sigma stays positive and reasonable
        if sigma <= 0:
            sigma = tolerance # Reset to a small positive value if it goes non-positive
        elif sigma > 5.0: # Cap volatility at a high value (e.g., 500%)
             sigma = 5.0

    # If max iterations reached without convergence
    # Check if final price is close enough anyway
    final_price = black_scholes_call(S, K, T, r, sigma)
    if abs(final_price - target_price) < tolerance * 10: # Looser tolerance on exit
        return sigma

    # print(f"Warning: Implied volatility did not converge for K={K}, T={T}, S={S}, Price={target_price}. Returning last estimate: {sigma}")
    return sigma # Return last estimate, might be inaccurate


# --------------------------
# Trader Class
# --------------------------
class Trader:

    def __init__(self, params = None):
        self.position_limits = POSITION_LIMITS
        self.strikes = {symbol: k for symbol, k in zip(VOUCHER_SYMBOLS, STRIKES)}
        self.smile_coeffs = SMILE_COEFFS
        self.risk_free_rate = RISK_FREE_RATE
        self.order_quantity = ORDER_QUANTITY
        self.iv_history_length = IV_HISTORY_LENGTH
        self.mispricing_threshold_std_dev = MISPRICING_THRESHOLD_STD_DEV

    def get_mid_price(self, symbol: str, state: TradingState) -> float | None:
        """Calculates the mid-price from the order book."""
        order_depth = state.order_depths.get(symbol)
        if not order_depth:
            # print(f"Warning: No order depth found for {symbol}")
            return None

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            return best_bid
        elif best_ask is not None:
            return best_ask
        else:
            # print(f"Warning: No bids or asks found for {symbol}")
            return None # No liquidity

    def get_weighted_mid_price(self, symbol: str, state: TradingState) -> float | None:
        """Calculates the volume-weighted mid-price (like example's get_swmid)."""
        order_depth = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            # Fallback to simple mid if weighted is not possible
            # print(f"Warning: Cannot compute weighted mid for {symbol}, using simple mid.")
            return self.get_mid_price(symbol, state)

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])

        if best_bid_vol + best_ask_vol == 0: # Avoid division by zero
             return (best_bid + best_ask) / 2.0

        weighted_mid = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
        return weighted_mid


    def calculate_tte(self, timestamp: int) -> float:
        """
        Calculates the Time To Expiry (TTE) in years, assuming linear decay within each day.
        Day 0 (t=0 to 9900): TTE decays from 8/365 to 7/365
        Day 1 (t=10000 to 19900): TTE decays from 7/365 to 6/365
        ...
        Day 7 (t=70000 to 79900): TTE decays from 1/365 to 0/365
        """
        if timestamp < 0: timestamp = 0 # Should not happen

        day_number = math.floor(timestamp / 10000) # Day 0, 1, 2...
        time_within_day = timestamp % 10000
        steps_per_day = 10000 # 1M total timestamp range, 100 steps per unit

        start_days_remaining = TOTAL_DURATION_DAYS - day_number
        end_days_remaining = TOTAL_DURATION_DAYS - day_number - 1

        # Linear interpolation of days remaining
        current_days_remaining = start_days_remaining - (time_within_day + 1) / steps_per_day

        # Clamp TTE to be non-negative
        tte_years = max(0.0, current_days_remaining / DAYS_PER_YEAR)

        # print(f"Timestamp: {timestamp}, Day: {day_number}, Time in Day: {time_within_day}, TTE (days): {current_days_remaining:.4f}, TTE (years): {tte_years:.6f}")
        return tte_years

    def calculate_m_t(self, S: float, K: int, TTE: float) -> float | None:
        """Calculates the moneyness parameter m_t = log(K/S) / sqrt(TTE)."""
        if S <= 0 or K <= 0 or TTE <= 0:
            # print(f"Warning: Invalid input for m_t calculation (S={S}, K={K}, TTE={TTE})")
            return None
        try:
            m_t = math.log(K / S) / math.sqrt(TTE)
            return m_t
        except ValueError as e:
            # print(f"Error calculating m_t (S={S}, K={K}, TTE={TTE}): {e}")
            return None

    def get_smile_volatility(self, m_t: float) -> float:
        """Calculates the theoretical volatility from the fitted parabolic smile."""
        a, b, c = self.smile_coeffs
        smile_iv = a * (m_t ** 2) + b * m_t + c
        # Ensure volatility isn't negative (can happen with parabola at extreme m_t)
        return max(0.0001, smile_iv) # Return a very small positive number if negative

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main trading logic function.
        """
        # print(f"\n--- Timestamp: {state.timestamp} ---")
        # Load trader data from previous state
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception as e:
                print(f"Error decoding traderData: {e}")
                traderObject = {} # Reset if decoding fails

        # Initialize state variables if not present
        traderObject.setdefault("base_iv_history", [])

        result: Dict[str, List[Order]] = {symbol: [] for symbol in ALL_SYMBOLS}
        conversions = 0

    
        # --- Calculations ---
        current_time = state.timestamp
        TTE = self.calculate_tte(current_time)

        # Get underlying price
        S = self.get_weighted_mid_price(VOLCANIC_ROCK, state)
        if S is None:
            # print("Warning: Could not get underlying price. Skipping this tick.")
            # Persist state without trading
            traderData = jsonpickle.encode(traderObject, unpicklable=False)
            return result, conversions, traderData

        # --- Volatility Analysis and Trading Logic ---
        calculated_ivs = {} # Store IVs calculated in this step for potential analysis

        for symbol in VOUCHER_SYMBOLS:
            K = self.strikes[symbol]
            limit = self.position_limits[symbol]
            current_pos = state.position.get(symbol, 0)

            # Get option market price
            voucher_price = self.get_weighted_mid_price(symbol, state)
            if voucher_price is None or TTE <= 0:
                # print(f"Warning: Skipping {symbol} due to missing price or TTE=0.")
                continue # Cannot price or calculate IV without market price or time

            # Calculate m_t
            m_t = self.calculate_m_t(S, K, TTE)
            if m_t is None:
                # print(f"Warning: Skipping {symbol} due to m_t calculation error.")
                continue

            # Calculate Market Implied Volatility
            # Use base IV or last known IV for this strike as initial guess if available
            last_iv = traderObject.get("last_ivs", {}).get(symbol, BASE_IV)
            iv_market = implied_volatility(voucher_price, S, K, TTE, self.risk_free_rate, initial_guess=last_iv)

            if iv_market <= 0: # Handle cases where IV calculation failed or gave non-positive result
                # print(f"Warning: Invalid market IV ({iv_market:.4f}) calculated for {symbol}. Skipping.")
                continue

            calculated_ivs[symbol] = iv_market # Store successfully calculated IV

            # Calculate Theoretical Volatility from Smile
            iv_smile = self.get_smile_volatility(m_t)

            # Calculate Mispricing
            mispricing = iv_market - iv_smile
            # print(f"{symbol} (K={K}): S={S:.2f}, P={voucher_price:.2f}, TTE={TTE:.4f}, m_t={m_t:.4f} -> IV_mkt={iv_market:.4f}, IV_smile={iv_smile:.4f}, Diff={mispricing:.4f}")


            # --- Decision Logic ---
            # Use a dynamic threshold based on recent volatility? For now, fixed threshold.
            # A simple threshold based on the difference:
            threshold = 0.001 # Example: Trade if IV difference > 0.1%
            # Or use the provided BASE_IV standard deviation idea (needs std dev calculation)
            # For now, let's use a simpler absolute threshold, can refine later.

            # If market IV is significantly higher than smile -> Sell opportunity
            if mispricing > threshold:
                volume_to_sell = self.order_quantity
                # Check position limit: can we sell 'volume_to_sell'?
                available_sell_limit = limit - (-current_pos) # Space available to sell
                actual_sell_volume = min(volume_to_sell, available_sell_limit)

                if actual_sell_volume > 0:
                    # Place sell order slightly below best ask or at best bid for higher fill chance
                    order_depth = state.order_depths.get(symbol)
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    sell_price = best_bid if best_bid is not None else int(math.floor(voucher_price)) # Fallback if no bids

                    if sell_price > 0 : # Ensure valid price
                        # print(f"  -> SELL Signal: {symbol}. Placing order for {-actual_sell_volume} @ {sell_price}")
                        result[symbol].append(Order(symbol, sell_price, -actual_sell_volume))


            # If market IV is significantly lower than smile -> Buy opportunity
            elif mispricing < -threshold:
                volume_to_buy = self.order_quantity
                 # Check position limit: can we buy 'volume_to_buy'?
                available_buy_limit = limit - current_pos # Space available to buy
                actual_buy_volume = min(volume_to_buy, available_buy_limit)

                if actual_buy_volume > 0:
                     # Place buy order slightly above best bid or at best ask for higher fill chance
                    order_depth = state.order_depths.get(symbol)
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                    buy_price = best_ask if best_ask is not None else int(math.ceil(voucher_price)) # Fallback if no asks

                    if buy_price > 0: # Ensure valid price
                        # print(f"  -> BUY Signal: {symbol}. Placing order for {actual_buy_volume} @ {buy_price}")
                        result[symbol].append(Order(symbol, buy_price, actual_buy_volume))


        # --- Update State ---
        # Store last calculated IVs for next iteration's initial guess
        traderObject["last_ivs"] = calculated_ivs

        # Base IV analysis (using ATM option K=10000)
        atm_symbol = f"{VOUCHER_PREFIX}10000"
        if atm_symbol in calculated_ivs:
            current_base_iv = calculated_ivs[atm_symbol]
            traderObject["base_iv_history"].append(current_base_iv)
            # Keep history length limited
            if len(traderObject["base_iv_history"]) > self.iv_history_length:
                traderObject["base_iv_history"] = traderObject["base_iv_history"][-self.iv_history_length:]

            # Potential future use: Analyze base_iv_history (mean, std dev, trend)
            # to adjust the smile curve or trading thresholds dynamically.
            # Example: if statistics.mean(traderObject["base_iv_history"]) > BASE_IV * 1.1:
            #    print("Base IV trending higher") # Adjust strategy?

        # --- Output ---
        traderData = jsonpickle.encode(traderObject, unpicklable=False) # Use unpicklable=False for simple dicts/lists

        # print("Orders:", {k:v for k,v in result.items() if v})
        # print(f"Positions: {state.position}")
        # print(f"TraderData Keys: {list(traderObject.keys())}")
        # print(f"--- End Tick {state.timestamp} ---")

        return result, conversions, traderData