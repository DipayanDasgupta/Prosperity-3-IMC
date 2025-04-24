import json
import jsonpickle
import numpy as np
import math
import statistics
from typing import Any, List, Dict, Tuple
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque # Added for potential future use, not strictly needed now

# --------------------------
# Constants and Parameters
# --------------------------
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
STRIKES = [9500, 9750, 10000, 10250, 10500]
VOUCHER_SYMBOLS = [f"{VOUCHER_PREFIX}{K}" for K in STRIKES]
ALL_SYMBOLS = [VOLCANIC_ROCK] + VOUCHER_SYMBOLS

POSITION_LIMITS = {
    VOLCANIC_ROCK: 400, # Allow hedging
    **{symbol: 200 for symbol in VOUCHER_SYMBOLS}
}

# Volatility Smile Parameters (Using original - consider recalculating for R3 later)
SMILE_COEFFS = [-1.12938253, -0.0421697, 0.00938588]

RISK_FREE_RATE = 0 # Assume 0 risk-free rate
DAYS_PER_YEAR = 365.0 # Standard convention
TOTAL_DURATION_DAYS = 8 # Total duration of the option's life from start
TICKS_PER_DAY = 10000 # 100 timestamps per unit, 100 units per day

# Trading Parameters
# REVERTED THRESHOLD to be more sensitive
MISPRICING_THRESHOLD_ABS = 0.001 # Trade if IV difference > 0.1% (absolute IV)
# *** ADDED MINIMUM HEDGE VOLUME ***
MIN_HEDGE_TRADE_VOLUME = 5       # Only hedge if required trade volume is >= this amount
ORDER_QUANTITY = 10             # Quantity per voucher order
IV_HISTORY_LENGTH = 100        # Length of base IV history to keep


# --------------------------
# Black-Scholes Helper Functions (Keep existing norm_cdf, norm_pdf)
# --------------------------
def norm_cdf(x):
    # Clip input to avoid overflow in erf for very large/small x
    x = np.clip(x, -10, 10)
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    # Clip input to avoid overflow in exp for very large/small x
    x = np.clip(x, -10, 10)
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def black_scholes_call(S, K, T, r, sigma):
    if T <= 1e-9 or sigma <= 1e-9:
        return max(0.0, S - K * math.exp(-r * T))

    try:
        # Check for potential domain issues before calculation
        if S <= 0 or K <= 0: return max(0.0, S - K * math.exp(-r * T)) # Treat as intrinsic if S or K invalid

        d1_num = math.log(S / K) + (r + 0.5 * sigma ** 2) * T
        d1_den = sigma * math.sqrt(T)
        if abs(d1_den) < 1e-9: # Avoid division by zero
             # If vol or T is near zero, fallback to intrinsic value logic
            return max(0.0, S - K * math.exp(-r * T))
        d1 = d1_num / d1_den
        d2 = d1 - d1_den # d1 - sigma * sqrt(T)

        call_price = (S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))
        # Ensure price is not negative due to floating point errors
        return max(0.0, call_price)
    except (ValueError, OverflowError):
         return max(0.0, S - K * math.exp(-r * T))


def black_scholes_delta(S, K, T, r, sigma):
    if T <= 1e-9 or sigma <= 1e-9:
        if S > K: return 1.0
        elif S < K: return 0.0
        else: return 0.5

    try:
        # Check for potential domain issues before calculation
        if S <= 0 or K <= 0:
            if S > K: return 1.0
            elif S < K: return 0.0
            else: return 0.5 # Estimate if S or K invalid

        d1_num = math.log(S / K) + (r + 0.5 * sigma ** 2) * T
        d1_den = sigma * math.sqrt(T)
        if abs(d1_den) < 1e-9: # Avoid division by zero
            if S > K: return 1.0
            elif S < K: return 0.0
            else: return 0.5 # Estimate if vol*sqrt(T) is zero

        d1 = d1_num / d1_den
        delta = norm_cdf(d1)
        return delta
    except (ValueError, OverflowError):
        # Estimate delta based on moneyness if calculation fails
        if S > K: return 1.0
        elif S < K: return 0.0
        else: return 0.5

def vega(S, K, T, r, sigma):
    if T <= 1e-9 or sigma <= 1e-9: return 0.0
    try:
        if S <= 0 or K <= 0: return 0.0 # Vega is 0 if S or K invalid

        d1_num = math.log(S / K) + (r + 0.5 * sigma ** 2) * T
        d1_den = sigma * math.sqrt(T)
        if abs(d1_den) < 1e-9: return 0.0 # Vega is 0 if vol*sqrt(T) is zero

        d1 = d1_num / d1_den
        vega_value = S * norm_pdf(d1) * math.sqrt(T)
        return max(0.0, vega_value) # Ensure non-negative
    except (ValueError, OverflowError):
        return 0.0

def implied_volatility(target_price, S, K, T, r, initial_guess=0.1, max_iterations=100, tolerance=1e-6):
    sigma = max(initial_guess, 0.01) # Ensure initial guess is positive
    min_sigma = 1e-4
    max_sigma = 10.0 # Cap max IV

    # Ensure S, K, T are valid before proceeding
    if S <= 0 or K <= 0 or T <= 1e-9: return min_sigma

    try:
        intrinsic_value = max(0.0, S - K * math.exp(-r * T))
        max_price = S # Approximation, can be slightly higher with interest rates

        # Handle edge cases where target price is outside practical bounds
        if target_price < intrinsic_value - tolerance : return min_sigma
        # If price is very high, it might imply extremely high IV or arbitrage
        if target_price > max_price * 1.1 : sigma = max_sigma # Start high if price > 110% of S

    except (ValueError, OverflowError):
        return min_sigma # Cannot calculate bounds, return min_sigma

    for i in range(max_iterations):
        try:
            price = black_scholes_call(S, K, T, r, sigma)
            v = vega(S, K, T, r, sigma)
        except (ValueError, OverflowError):
            return max(min_sigma, sigma) # Return current sigma if BS calculation fails

        diff = price - target_price

        if abs(diff) < tolerance: return max(min_sigma, sigma)

        # If vega is near zero, Newton-Raphson fails. Check if price is close to bounds.
        if abs(v) < 1e-10:
            if abs(target_price - intrinsic_value) < tolerance * 10: return min_sigma
            if abs(target_price - max_price) < tolerance * 10: return max_sigma # High IV if near max price
            # Cannot converge if vega is zero away from bounds
            return max(min_sigma, sigma) # Return current best guess

        # Newton-Raphson step
        sigma = sigma - diff / v

        # Clamp sigma to reasonable bounds
        sigma = max(min_sigma, min(max_sigma, sigma))

    # After max iterations, check if final price is close enough
    try:
        final_price = black_scholes_call(S, K, T, r, sigma)
        if abs(final_price - target_price) < tolerance * 10: # Looser tolerance
            return max(min_sigma, sigma)
    except (ValueError, OverflowError):
        pass # Ignore error here, will return last sigma below

    # Return last sigma if convergence not reached but within bounds
    return max(min_sigma, sigma)


# --------------------------
# Trader Class
# --------------------------
class Trader:

    def __init__(self):
        self.position_limits = POSITION_LIMITS
        self.strikes = {symbol: k for symbol, k in zip(VOUCHER_SYMBOLS, STRIKES)}
        self.smile_coeffs = SMILE_COEFFS
        self.risk_free_rate = RISK_FREE_RATE
        self.order_quantity = ORDER_QUANTITY
        self.iv_history_length = IV_HISTORY_LENGTH
        # USE THE SMALLER THRESHOLD
        self.mispricing_threshold = MISPRICING_THRESHOLD_ABS
        # USE THE MIN HEDGE VOLUME
        self.min_hedge_volume = MIN_HEDGE_TRADE_VOLUME

    def get_weighted_mid_price(self, symbol: str, state: TradingState) -> float | None:
        order_depth = state.order_depths.get(symbol)
        if not order_depth: return None

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None and best_ask is None: return None
        if best_bid is None: return best_ask
        if best_ask is None: return best_bid

        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        total_vol = best_bid_vol + best_ask_vol

        if total_vol < 1e-9: return (best_bid + best_ask) / 2.0 # Avoid division by zero

        weighted_mid = (best_bid * best_ask_vol + best_ask * best_bid_vol) / total_vol
        return weighted_mid


    def calculate_tte(self, timestamp: int) -> float:
        max_timestamp = TOTAL_DURATION_DAYS * TICKS_PER_DAY # = 80000
        # Ensure timestamp doesn't exceed max_timestamp (can happen in backtesting edge cases)
        current_timestamp = min(timestamp, max_timestamp)
        ticks_remaining = max_timestamp - current_timestamp
        tte_years = ticks_remaining / (TICKS_PER_DAY * DAYS_PER_YEAR)
        return max(0.0, tte_years) # Ensure non-negative


    def calculate_m_t(self, S: float, K: int, TTE: float) -> float | None:
        if S <= 0 or K <= 0 or TTE <= 1e-9: return None
        try:
            sqrt_TTE = math.sqrt(TTE + 1e-9) # Add epsilon for stability near T=0
            log_moneyness = math.log(K / S)
            m_t = log_moneyness / sqrt_TTE
            # Clamp m_t to avoid extreme values causing issues in polynomial evaluation
            m_t_clamp = 10.0
            return np.clip(m_t, -m_t_clamp, m_t_clamp)
        except (ValueError, OverflowError): return None

    def get_smile_volatility(self, m_t: float) -> float:
        a, b, c = self.smile_coeffs
        # Ensure m_t is a float for calculation
        m_t = float(m_t)
        try:
            smile_iv = a * (m_t ** 2) + b * m_t + c
            # Apply a floor to the volatility
            return max(0.0001, smile_iv) # Min IV of 0.01%
        except OverflowError:
            # If polynomial calculation overflows (e.g., extreme m_t despite clamp), return base IV
            return max(0.0001, c)


    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData:
            try: traderObject = jsonpickle.decode(state.traderData)
            except Exception: traderObject = {}
        traderObject.setdefault("last_ivs", {})

        result: Dict[str, List[Order]] = {symbol: [] for symbol in ALL_SYMBOLS}
        conversions = 0
        net_option_delta_trade = 0.0

        current_time = state.timestamp
        TTE = self.calculate_tte(current_time)

        S = self.get_weighted_mid_price(VOLCANIC_ROCK, state)
        # If underlying price invalid or TTE zero, cannot proceed
        if S is None or S <= 0 or TTE <= 1e-9:
            return result, conversions, jsonpickle.encode(traderObject, unpicklable=False)

        calculated_ivs_this_tick = {}

        # --- Voucher Trading Loop ---
        for symbol in VOUCHER_SYMBOLS:
            K = self.strikes[symbol]
            limit = self.position_limits[symbol]
            current_pos = state.position.get(symbol, 0)

            voucher_price = self.get_weighted_mid_price(symbol, state)
            if voucher_price is None or voucher_price <= 0: continue # Need valid market price

            m_t = self.calculate_m_t(S, K, TTE)
            if m_t is None: continue

            last_iv_guess = traderObject.get("last_ivs", {}).get(symbol, self.get_smile_volatility(m_t))
            # Ensure initial guess is reasonable
            iv_market = implied_volatility(voucher_price, S, K, TTE, self.risk_free_rate, initial_guess=max(0.01, min(last_iv_guess, 5.0)))

            # Store calculated market IV if valid
            if iv_market > 1e-4: # Store only if IV calculation seems reasonable
                calculated_ivs_this_tick[symbol] = iv_market
            else:
                continue # Skip trading this option if IV calculation failed/unrealistic

            iv_smile = self.get_smile_volatility(m_t)
            mispricing = iv_market - iv_smile

            actual_trade_volume = 0
            trade_price = 0
            order_depth_voucher = state.order_depths.get(symbol)

            # --- Decision Logic (Buy/Sell Vouchers) ---
            if mispricing > self.mispricing_threshold: # Market IV > Smile IV -> Sell Voucher
                available_sell = limit + current_pos # How many more can we short?
                volume_to_sell = min(self.order_quantity, available_sell)
                if volume_to_sell > 0:
                    actual_trade_volume = -volume_to_sell
                    best_bid = max(order_depth_voucher.buy_orders.keys()) if order_depth_voucher and order_depth_voucher.buy_orders else None
                    # Price aggressively: Sell at best bid, or slightly below mid as fallback
                    trade_price = best_bid if best_bid is not None else int(math.floor(voucher_price - 1))

            elif mispricing < -self.mispricing_threshold: # Market IV < Smile IV -> Buy Voucher
                available_buy = limit - current_pos # How many more can we long?
                volume_to_buy = min(self.order_quantity, available_buy)
                if volume_to_buy > 0:
                    actual_trade_volume = volume_to_buy
                    best_ask = min(order_depth_voucher.sell_orders.keys()) if order_depth_voucher and order_depth_voucher.sell_orders else None
                     # Price aggressively: Buy at best ask, or slightly above mid as fallback
                    trade_price = best_ask if best_ask is not None else int(math.ceil(voucher_price + 1))

            # --- Record Voucher Trade and Delta ---
            if actual_trade_volume != 0 and trade_price > 0:
                result[symbol].append(Order(symbol, trade_price, actual_trade_volume))
                # Use smile IV for theoretical delta, as we trade deviations from it
                trade_delta = black_scholes_delta(S, K, TTE, self.risk_free_rate, iv_smile)
                if trade_delta is not None: # Check if delta calculation was successful
                    net_option_delta_trade += actual_trade_volume * trade_delta


        # --- Delta Hedging Logic ---
        existing_portfolio_delta = 0.0
        # 1. Calculate delta of existing option positions
        for symbol in VOUCHER_SYMBOLS:
            current_pos = state.position.get(symbol, 0)
            if current_pos != 0:
                K = self.strikes[symbol]
                # Recalculate m_t and iv_smile based on current S, TTE
                m_t = self.calculate_m_t(S, K, TTE)
                if m_t is not None:
                    iv_smile = self.get_smile_volatility(m_t)
                    delta_per_option = black_scholes_delta(S, K, TTE, self.risk_free_rate, iv_smile)
                    if delta_per_option is not None:
                        existing_portfolio_delta += current_pos * delta_per_option

        # 2. Calculate total delta and required hedge
        total_target_delta = existing_portfolio_delta + net_option_delta_trade
        required_hedge_shares = -total_target_delta
        target_rock_position = round(required_hedge_shares) # Target integer position

        # 3. Calculate trade volume needed for hedge
        current_rock_pos = state.position.get(VOLCANIC_ROCK, 0)
        rock_trade_volume = target_rock_position - current_rock_pos

        # 4. Apply position limits to the hedge trade
        rock_limit = self.position_limits[VOLCANIC_ROCK]
        potential_new_pos = current_rock_pos + rock_trade_volume
        if potential_new_pos > rock_limit:
            rock_trade_volume -= (potential_new_pos - rock_limit) # Reduce buy volume
        elif potential_new_pos < -rock_limit:
            rock_trade_volume += (-potential_new_pos - rock_limit) # Reduce sell volume (make it less negative)


        # *** APPLY MINIMUM HEDGE VOLUME CHECK ***
        # 5. Place hedge order only if volume is significant
        if abs(rock_trade_volume) >= self.min_hedge_volume:
            order_depth_rock = state.order_depths.get(VOLCANIC_ROCK)
            if order_depth_rock is not None:
                hedge_price = 0
                # Use best bid/ask for pricing the hedge
                if rock_trade_volume > 0: # Buying ROCK to hedge
                    if order_depth_rock.sell_orders:
                         hedge_price = min(order_depth_rock.sell_orders.keys())
                    else: # Fallback if no asks
                        rock_mid_price = self.get_weighted_mid_price(VOLCANIC_ROCK, state)
                        if rock_mid_price is not None: hedge_price = int(math.ceil(rock_mid_price + 1))

                else: # Selling ROCK to hedge
                    if order_depth_rock.buy_orders:
                        hedge_price = max(order_depth_rock.buy_orders.keys())
                    else: # Fallback if no bids
                        rock_mid_price = self.get_weighted_mid_price(VOLCANIC_ROCK, state)
                        if rock_mid_price is not None: hedge_price = int(math.floor(rock_mid_price - 1))

                # Place the order if a valid price was determined
                if hedge_price > 0:
                     result[VOLCANIC_ROCK].append(Order(VOLCANIC_ROCK, hedge_price, rock_trade_volume))
                     # print(f" -> HEDGE: Placing order for {rock_trade_volume} {VOLCANIC_ROCK} @ {hedge_price}")


        # --- Update State and Return ---
        traderObject["last_ivs"] = calculated_ivs_this_tick # Store IVs calculated now
        traderData = jsonpickle.encode(traderObject, unpicklable=False)

        return result, conversions, traderData