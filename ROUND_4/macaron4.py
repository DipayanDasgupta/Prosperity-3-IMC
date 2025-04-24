# r4_macarons_conversion_focused_v2.py
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Observation
from typing import Dict, List, Tuple, Any
import numpy as np
import json
import math

# --- Constants ---
PRODUCT = "MAGNIFICENT_MACARONS"
POSITION_LIMIT = 75
SOFT_LIMIT = 50
STORAGE_COST_PER_UNIT_PER_TICK = 0.1
CONVERSION_LIMIT_PER_SIDE = 15

# --- Parameters (Slightly Relaxed for Initial Testing) ---
PARAMS = {
    "conv_buy_market_edge": 1.0,  # Lowered from 1.25
    "aggressive_market_buy_edge": 2.0, # Lowered from 2.5
    "aggressive_market_sell_profit_target": 1.5, # Lowered from 2.0
    "hedge_sell_offset": 0.75 + STORAGE_COST_PER_UNIT_PER_TICK * 8, # Approx 1.55
    "passive_buy_offset_from_conv_buy": 3.5, # Lowered from 4.0
    "mm_base_qty": 6, # Slightly increased base size
    "inventory_skew_factor": 0.06,
}

class Trader:
    def __init__(self):
        pass # No complex state needed for this version

    # ---------- helpers ----------
    def get_best_bid(self, od: OrderDepth) -> float | None:
        return max(od.buy_orders.keys()) if od.buy_orders else None

    def get_best_ask(self, od: OrderDepth) -> float | None:
        return min(od.sell_orders.keys()) if od.sell_orders else None

    def calculate_conversion_prices(self, obs: Any) -> Tuple[float | None, float | None]:
        #DEBUG: Uncomment to see raw observation data
        print(f"DEBUG: Raw obs: {obs}")
        if not obs:
            print("DEBUG: Obs is None")
            return None, None
        try:
            # Ensure all necessary fields exist and are numeric
            ask_price = float(obs.askPrice)
            bid_price = float(obs.bidPrice)
            transport = float(obs.transportFees)
            import_tariff = float(obs.importTariff)
            export_tariff = float(obs.exportTariff)

            buy_eff = ask_price + transport + import_tariff
            sell_eff = bid_price - transport - export_tariff
            # DEBUG: Uncomment to see calculated conversion prices
            print(f"DEBUG: Calculated eff_buy={buy_eff:.2f}, eff_sell={sell_eff:.2f}")
            return buy_eff, sell_eff
        except (TypeError, ValueError, AttributeError, KeyError) as e:
             # print(f"ERROR calculating conversion prices: {e}, Obs: {obs}")
             return None, None

    # ---------- main per‑tick ----------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        orders: List[Order] = []
        conversion_request = 0

        # --- Get Data ---
        od = state.order_depths.get(PRODUCT, OrderDepth())
        pos = state.position.get(PRODUCT, 0)
        obs = state.observations.conversionObservations.get(PRODUCT) if state.observations else None

        eff_buy, eff_sell = self.calculate_conversion_prices(obs)
        best_bid = self.get_best_bid(od)
        best_ask = self.get_best_ask(od)

        # DEBUG: Uncomment to see state at each tick
        print(f"DEBUG T={state.timestamp}: eff_buy={eff_buy}, eff_sell={eff_sell}, best_bid={best_bid}, best_ask={best_ask}, pos={pos}")

        # --- 1. Conversion Buy Strategy ---
        print("DEBUG: Checking Conversion Buy...") # Uncomment for debug
        if eff_buy is not None and best_bid is not None:
             # print(f"DEBUG: eff_buy={eff_buy:.2f}, compare_to={best_bid - PARAMS['conv_buy_market_edge']:.2f}") # Uncomment for debug
             if eff_buy < best_bid - PARAMS["conv_buy_market_edge"] and pos < POSITION_LIMIT:
                qty_to_convert = min(CONVERSION_LIMIT_PER_SIDE, POSITION_LIMIT - pos)
                if qty_to_convert > 0:
                    conversion_request = qty_to_convert
                    print(f"INFO: Requesting CONV BUY {qty_to_convert}") # Uncomment for debug

        # --- 2. Aggressive Market Order Strategy ---
        print("DEBUG: Checking Aggressive Market Orders...") # Uncomment for debug
        if conversion_request == 0: # Only if not converting
            # Hit market ASK
            if best_ask is not None and eff_buy is not None:
                print(f"DEBUG: Aggro Buy Check: best_ask={best_ask:.2f}, compare_to={eff_buy - PARAMS['aggressive_market_buy_edge']:.2f}") # Uncomment for debug
                if best_ask < eff_buy - PARAMS["aggressive_market_buy_edge"] and pos < POSITION_LIMIT:
                    vol = abs(od.sell_orders[best_ask])
                    qty_to_buy = min(vol, POSITION_LIMIT - pos, PARAMS["mm_base_qty"] * 2) # Limit aggressive take size
                    if qty_to_buy > 0:
                        orders.append(Order(PRODUCT, best_ask, qty_to_buy))
                        pos += qty_to_buy
                        print(f"INFO: Aggressive BUY {qty_to_buy} @ {best_ask:.2f}") # Uncomment for debug

            # Hit market BID
            if best_bid is not None and eff_buy is not None:
                 print(f"DEBUG: Aggro Sell Check: best_bid={best_bid:.2f}, compare_to={eff_buy + PARAMS['aggressive_market_sell_profit_target']:.2f}") # Uncomment for debug
                 if best_bid > eff_buy + PARAMS["aggressive_market_sell_profit_target"] and pos > -POSITION_LIMIT:
                    vol = abs(od.buy_orders[best_bid])
                    qty_to_sell = min(vol, POSITION_LIMIT + pos, PARAMS["mm_base_qty"] * 2) # Limit aggressive take size
                    if qty_to_sell > 0:
                        orders.append(Order(PRODUCT, best_bid, -qty_to_sell))
                        pos -= qty_to_sell
                        print(f"INFO: Aggressive SELL {qty_to_sell} @ {best_bid:.2f}") # Uncomment for debug

        # --- 3. Passive Market Making / Hedging (Simplified & Safer) ---
        print("DEBUG: Checking Passive MM...") # Uncomment for debug
        if eff_buy is not None: # Base quoting around the reliable conversion buy cost
            inv_skew = pos * PARAMS["inventory_skew_factor"]

            # Calculate base bid/ask around eff_buy + hedge offset, skewed by inventory
            base_ask_price = math.ceil(eff_buy + PARAMS["hedge_sell_offset"] - inv_skew)
            base_buy_price = math.floor(eff_buy - PARAMS["passive_buy_offset_from_conv_buy"] - inv_skew)

            # Ensure buy price is strictly less than ask price
            if base_buy_price >= base_ask_price:
                base_buy_price = base_ask_price - 1 # Force at least 1 tick spread

            final_buy_price = base_buy_price
            final_sell_price = base_ask_price

            # Optional: Clip prices to be closer to current market if they stray too far
            if best_ask is not None:
                final_sell_price = min(final_sell_price, best_ask + 2) # Don't quote too far above market ask
                final_sell_price = max(final_sell_price, final_buy_price + 1) # Re-ensure sell > buy
            if best_bid is not None:
                final_buy_price = max(final_buy_price, best_bid - 2) # Don't quote too far below market bid
                final_buy_price = min(final_buy_price, final_sell_price - 1) # Re-ensure buy < sell


            # Place Orders if within position limits
            print(f"DEBUG: Passive Quoting - Buy @ {final_buy_price}, Sell @ {final_sell_price}") # Uncomment for debug

            if pos < SOFT_LIMIT: # Place sell if below soft limit
                qty_to_sell = min(PARAMS["mm_base_qty"], POSITION_LIMIT + pos) # Max sellable qty respecting hard limit
                if qty_to_sell > 0:
                    orders.append(Order(PRODUCT, final_sell_price, -qty_to_sell))


            if pos > -SOFT_LIMIT: # Place buy if above -soft limit
                qty_to_buy = min(PARAMS["mm_base_qty"], POSITION_LIMIT - pos) # Max buyable qty respecting hard limit
                if qty_to_buy > 0:
                    # Safety check: don't place buy order above a sell order we just placed
                    current_sell_prices = {o.price for o in orders if o.quantity < 0}
                    if not current_sell_prices or final_buy_price < min(current_sell_prices):
                         orders.append(Order(PRODUCT, final_buy_price, qty_to_buy))


        # --- Prepare Output ---
        final_orders = {PRODUCT: orders}
        final_conversion = int(conversion_request) # Ensure integer
        trader_data = "" # Not using trader_data in this version

        #DEBUG: Uncomment to see final actions
        if orders or final_conversion != 0:
           print(f"FINAL T={state.timestamp}: Orders: {orders}, Conversion: {final_conversion}, Pos: {pos}")

        return final_orders, final_conversion, trader_data

# Instantiate trader
_trader = Trader()

# Entrypoint for Prosperity
def run(state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
    return _trader.run(state)