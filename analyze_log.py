#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyzes prosperity3bt log files to calculate performance metrics.

Prompts the user for the log file path and outputs metrics to the console.
"""

import json
import math
import argparse # Keep for potential future use
import os
from collections import deque # Might be needed if decoding trader data later

# --- Attempt to import required libraries ---
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm # Progress bar
except ImportError as e:
    # Provide specific instructions based on the missing library
    missing_module = str(e).split("'")[1] if "'" in str(e) else "a required library"
    print(f"Error: Could not import '{missing_module}'.")
    print("Please ensure pandas, numpy, matplotlib, and tqdm are installed in your environment.")
    print("Activate your virtual environment (e.g., 'source venv/bin/activate') and run:")
    print(f"pip install pandas numpy matplotlib tqdm")
    exit(1) # Exit if libraries are missing


# ==============================================================================
# Helper Functions
# ==============================================================================

def calculate_mid_price_from_log(depth_data):
    """
    Calculates mid-price from the compressed [buys, asks] dict structure
    found in the prosperity3bt logs. Includes robust error handling.
    """
    if not isinstance(depth_data, (list, tuple)) or len(depth_data) != 2:
        # Handles cases where depth_data is not a list/tuple of length 2
        return None
    buy_orders, sell_orders = depth_data
    if not isinstance(buy_orders, dict) or not isinstance(sell_orders, dict) or not buy_orders or not sell_orders:
        # Handles cases where buy/sell orders are not dicts or are empty
        return None

    try:
        # Extract keys, convert to int, handle potential non-numeric keys gracefully
        valid_buy_prices = [int(p) for p in buy_orders.keys() if p.lstrip('-').isdigit()]
        valid_sell_prices = [int(p) for p in sell_orders.keys() if p.lstrip('-').isdigit()]

        if not valid_buy_prices or not valid_sell_prices:
            # Handles cases where no valid numeric prices were found
            return None

        best_bid = max(valid_buy_prices)
        best_ask = min(valid_sell_prices)

        # Optional: Check for crossed book, though calculating midpoint might still be desired
        # if best_ask <= best_bid:
        #     print(f"Warning: Crossed book detected (bid {best_bid} >= ask {best_ask}).")

        return (best_bid + best_ask) / 2.0

    except (ValueError, TypeError, KeyError) as e:
        # Catch potential errors during conversion, max/min, or if keys somehow aren't strings
        print(f"Error calculating mid-price from log data segment: {e}")
        return None

def parse_log_file(file_path):
    """Reads and parses the prosperity3bt log file line by line."""
    parsed_data = []
    print(f"Reading and parsing log file: {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()

        for i, line in enumerate(tqdm(log_lines, desc="Parsing Logs")):
            line = line.strip()
            if not line: # Skip empty lines
                continue
            try:
                # Each line is expected to be a JSON array
                data_point = json.loads(line)
                # --- Validation: Check outer list and inner state list structure ---
                # Adjust indices based on your specific, consistent log format if it differs
                # Assuming outer list has >= 7 elements, and first element (state) is a list with >= 7 elements
                if (isinstance(data_point, (list, tuple)) and len(data_point) >= 5 # Check outer list minimum length (flexible)
                    and isinstance(data_point[0], (list, tuple)) and len(data_point[0]) >= 7): # Check inner state list minimum length
                      parsed_data.append(data_point)
                # else: # Optional: Be more verbose about skipped lines
                #      print(f"Warning: Skipping line {i+1} due to unexpected structure: {line[:100]}...")
            except json.JSONDecodeError:
                # Non-JSON lines might exist (like the sandbox log messages) - skip them
                # print(f"Warning: Skipping line {i+1} due to JSON decode error: {line[:100]}...")
                pass
            except Exception as e:
                 print(f"Warning: Skipping line {i+1} due to unexpected error: {e} - Line: {line[:100]}...")

        if not parsed_data:
            print("Warning: No valid data points parsed. Log file might be empty, malformed, or have an unexpected structure.")
            return None # Return None if no data parsed

        print(f"\nSuccessfully parsed {len(parsed_data)} data points (timestamps).")
        return parsed_data

    except FileNotFoundError:
        print(f"ERROR: Log file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"ERROR: Failed to read or process log file: {e}")
        return None


def calculate_portfolio_history(parsed_data):
    """Calculates approximate portfolio value over time from parsed log data."""
    portfolio_history = []
    last_mid_prices = {} # Store last known mid-price per product

    if not parsed_data:
        print("No parsed data available to calculate portfolio value.")
        return pd.DataFrame() # Return empty DataFrame

    print("Calculating portfolio value history...")
    for data_point in tqdm(parsed_data, desc="Processing Timestamps"):
        try:
            # --- Careful Indexing Based on Observed Log Structure ---
            # Outer: [state, orders, conversions, trader_data_out, log_string]
            # State: [timestamp, trader_data_in, listings, depths, own_trades, mkt_trades, pos, obs]
            state_details = data_point[0] # State is the first element
            timestamp = state_details[0]
            # Check lengths before indexing to prevent errors
            order_depths = state_details[3] if len(state_details) > 3 and isinstance(state_details[3], dict) else {}
            positions = state_details[6] if len(state_details) > 6 and isinstance(state_details[6], dict) else {}
            # ---------------------------------------------------------

            current_portfolio_value = 0.0

            # Iterate through products with non-zero positions
            for product, size in positions.items():
                if size == 0:
                    continue

                mid_price = None
                if product in order_depths:
                    # Pass the specific order depth data for the product
                    product_depth_data = order_depths.get(product)
                    if product_depth_data:
                        mid_price = calculate_mid_price_from_log(product_depth_data)

                # Use last known mid-price if current calculation fails
                if mid_price is None:
                    mid_price = last_mid_prices.get(product)

                if mid_price is not None:
                    current_portfolio_value += size * mid_price
                    last_mid_prices[product] = mid_price # Update last known
                # else: # Optional: Warning for consistently missing prices
                #     if timestamp > 0: # Avoid warning at timestamp 0
                #          print(f"Warning: Missing mid-price for {product} at ts {timestamp}.")

            portfolio_history.append({'timestamp': timestamp, 'portfolio_value': current_portfolio_value})

        except (IndexError, TypeError, KeyError, ValueError) as e:
            # Try to get timestamp for error message, default if fails
            error_ts = '_unknown_'
            try: error_ts = data_point[0][0]
            except Exception: pass
            print(f"Error processing data point around ts {error_ts}: {e}. Skipping.")


    # Convert history to DataFrame
    if not portfolio_history:
        print("\nFailed to create portfolio history (no valid data points processed?).")
        return pd.DataFrame()

    portfolio_df = pd.DataFrame(portfolio_history)

    # Ensure timestamps are numeric before dropping duplicates
    portfolio_df['timestamp'] = pd.to_numeric(portfolio_df['timestamp'], errors='coerce')
    portfolio_df.dropna(subset=['timestamp'], inplace=True)
    portfolio_df['timestamp'] = portfolio_df['timestamp'].astype(int)

    # Drop duplicates, keeping the last entry for a given timestamp
    portfolio_df = portfolio_df.drop_duplicates(subset=['timestamp'], keep='last')
    portfolio_df = portfolio_df.set_index('timestamp').sort_index()

    if portfolio_df.empty:
         print("\nPortfolio history is empty after processing and cleaning.")
         return pd.DataFrame()

    print(f"\nPortfolio history calculated. Shape: {portfolio_df.shape}")
    return portfolio_df

def calculate_metrics(portfolio_df):
    """Calculates performance metrics from the portfolio value DataFrame."""
    metrics = {}
    if portfolio_df.empty or 'portfolio_value' not in portfolio_df.columns:
        print("Cannot calculate metrics: Portfolio DataFrame is empty or missing 'portfolio_value' column.")
        return metrics, portfolio_df

    # Ensure portfolio values are numeric and handle NaNs
    portfolio_df['portfolio_value'] = pd.to_numeric(portfolio_df['portfolio_value'], errors='coerce')
    portfolio_df.dropna(subset=['portfolio_value'], inplace=True) # Drop rows where value couldn't be parsed

    if portfolio_df.empty:
        print("Cannot calculate metrics: No valid portfolio values after cleaning.")
        return metrics, portfolio_df

    print("\nCalculating performance metrics...")

    # Calculate Tick Returns
    portfolio_df['returns'] = portfolio_df['portfolio_value'].diff()
    # Fill the first NaN return; if only one row exists, return is 0.
    portfolio_df['returns'].fillna(0, inplace=True)


    # Total PnL (Approximation)
    # Check if DataFrame is still non-empty after potential drops/diff
    if portfolio_df.empty:
        print("Cannot calculate metrics: DataFrame became empty during return calculation.")
        return metrics, portfolio_df

    start_value = portfolio_df['portfolio_value'].iloc[0]
    end_value = portfolio_df['portfolio_value'].iloc[-1]
    total_pnl_approx = end_value - start_value
    metrics['Total PnL (Approx.)'] = total_pnl_approx

    # Volatility
    # Ensure there's more than one return value to calculate std dev
    tick_volatility = portfolio_df['returns'].std() if len(portfolio_df['returns']) > 1 else 0.0
    metrics['Tick Volatility (Std Dev of Returns)'] = tick_volatility if pd.notna(tick_volatility) else 0.0

    # Sharpe Ratio (Rf=0)
    # Ensure there's more than one return value for mean calculation
    mean_tick_return = portfolio_df['returns'].mean() if len(portfolio_df['returns']) > 0 else 0.0
    sharpe_ratio_raw = np.nan # Default
    current_volatility = metrics['Tick Volatility (Std Dev of Returns)'] # Use calculated volatility
    if current_volatility > 0 and pd.notna(mean_tick_return):
        sharpe_ratio_raw = mean_tick_return / current_volatility
    elif pd.notna(mean_tick_return) and mean_tick_return == 0: # Handle zero return, zero vol case
         sharpe_ratio_raw = 0.0
    metrics['Sharpe Ratio (Raw, Tick-Based)'] = sharpe_ratio_raw

    # Sortino Ratio (Target=0)
    negative_returns = portfolio_df['returns'][portfolio_df['returns'] < 0]
    sortino_ratio_raw = np.nan # Default
    if pd.notna(mean_tick_return):
        if not negative_returns.empty:
            # Ensure more than one negative return for std dev calculation
            downside_deviation = negative_returns.std() if len(negative_returns) > 1 else 0.0
            if pd.notna(downside_deviation) and downside_deviation > 0:
                sortino_ratio_raw = mean_tick_return / downside_deviation
            elif pd.notna(downside_deviation) and downside_deviation == 0: # Zero deviation means all neg returns were same (or only one)
                sortino_ratio_raw = np.inf if mean_tick_return > 0 else (0.0 if mean_tick_return == 0 else -np.inf)
        else: # No negative returns
            sortino_ratio_raw = np.inf if mean_tick_return > 0 else (0.0 if mean_tick_return == 0 else np.nan) # Should not be neg mean with no neg returns
    metrics['Sortino Ratio (Raw, Tick-Based)'] = sortino_ratio_raw

    # Maximum Drawdown
    portfolio_df['cumulative_pnl'] = portfolio_df['portfolio_value'] - start_value
    portfolio_df['running_max_pnl'] = portfolio_df['cumulative_pnl'].cummax()
    portfolio_df['drawdown'] = portfolio_df['running_max_pnl'] - portfolio_df['cumulative_pnl']
    max_drawdown_value = portfolio_df['drawdown'].max() if pd.notna(portfolio_df['drawdown'].max()) else 0.0

    max_drawdown_pct = 0.0
    if max_drawdown_value > 0:
        try:
            mdd_peak_pnl = portfolio_df.loc[portfolio_df['drawdown'].idxmax(), 'running_max_pnl']
            # Calculate peak portfolio value = start + peak pnl
            mdd_peak_portfolio_value = start_value + mdd_peak_pnl
            if mdd_peak_portfolio_value > 0: # Avoid division by zero if peak was at or below start
                 max_drawdown_pct = (max_drawdown_value / mdd_peak_portfolio_value) * 100
            # Alternative: % relative to PnL gain (can be huge if peak pnl is small)
            # elif mdd_peak_pnl > 0:
            #      max_drawdown_pct = (max_drawdown_value / mdd_peak_pnl) * 100
        except KeyError:
            pass # Handle cases where idxmax might fail (e.g., all drawdowns are NaN)

    metrics['Max Drawdown (Value)'] = max_drawdown_value
    metrics['Max Drawdown (Percent of Peak Value)'] = max_drawdown_pct

    return metrics, portfolio_df # Return df with calculations

def print_metrics(metrics):
    """Prints the calculated metrics in a formatted way."""
    print("\n\n" + "="*30 + " Performance Metrics Summary " + "="*30)
    if metrics:
        for key, value in metrics.items():
            # Format numbers nicely, handle NaN/inf
            if isinstance(value, (float, np.floating)):
                if np.isnan(value): formatted_value = "NaN"
                elif np.isinf(value): formatted_value = "inf" if value > 0 else "-inf"
                else: formatted_value = f"{value:,.4f}" # Format normal floats
            else: formatted_value = str(value) # Keep non-floats as strings
            print(f"- {key}: {formatted_value}")
    else:
        print("No metrics calculated.")
    print("="*70)
    print("\nNotes:")
    print("- PnL is approximate, based on portfolio value changes (using mid-prices).")
    print("- Sharpe & Sortino Ratios are raw (tick-based) and assume Risk-Free Rate = 0.")
    print("- Max Drawdown % is relative to the peak portfolio value achieved before the drawdown.")

def plot_pnl(portfolio_df_with_calcs):
    """Plots the approximate PnL curve and annotates Max Drawdown."""
    if portfolio_df_with_calcs is None or portfolio_df_with_calcs.empty or 'cumulative_pnl' not in portfolio_df_with_calcs.columns:
        print("\nCannot plot PnL curve due to missing data.")
        return

    print("\nGenerating Approximate Cumulative PnL Plot...")
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_df_with_calcs.index, portfolio_df_with_calcs['cumulative_pnl'], label='Approx. Cumulative PnL')
        plt.title('Approximate Cumulative PnL Over Time (Based on Portfolio Value)')
        plt.xlabel('Timestamp')
        plt.ylabel('Cumulative PnL (Approx.)')
        plt.grid(True)

        # Annotate Max Drawdown
        max_drawdown_value = portfolio_df_with_calcs['drawdown'].max()
        if pd.notna(max_drawdown_value) and max_drawdown_value > 0:
            mdd_time = portfolio_df_with_calcs['drawdown'].idxmax()
            # Find the peak time associated with this drawdown period
            peak_time = portfolio_df_with_calcs.loc[:mdd_time, 'running_max_pnl'].idxmax() # Peak PnL time up to Mdd point
            peak_val = portfolio_df_with_calcs.loc[peak_time, 'cumulative_pnl']
            mdd_val = portfolio_df_with_calcs.loc[mdd_time, 'cumulative_pnl']

            # Add points for peak and trough
            plt.scatter([peak_time, mdd_time], [peak_val, mdd_val],
                        color='red', s=50, zorder=5, label=f'Max Drawdown ({max_drawdown_value:,.0f})')
            # Optional: Add line/arrow for drawdown
            plt.annotate('', xy=(mdd_time, mdd_val), xytext=(peak_time, peak_val),
                         arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

        plt.legend()
        plt.tight_layout()
        # Check if running in an environment that supports showing plots
        try:
            plt.show()
        except Exception:
            print("Plot generated, but cannot display automatically in this environment.")
            print("Consider saving the plot: plt.savefig('pnl_curve.png')")

    except Exception as e:
        print(f"Error generating plot: {e}")

# ==============================================================================
# Main Execution Logic
# ==============================================================================
def main():
    """Main function to run the analysis."""
    print("Prosperity Backtester Log Analyzer")
    print("-" * 30)

    # Get log file path from user
    while True:
        log_file_path = input("Enter the path to the .log file: ").strip()
        if os.path.isfile(log_file_path): # Use isfile for better check
            break
        else:
             print(f"Error: File not found at '{log_file_path}'. Please check the path and try again.")

    # --- Run Analysis Steps ---
    parsed_log_data = parse_log_file(log_file_path)

    if parsed_log_data:
        portfolio_df = calculate_portfolio_history(parsed_log_data)

        if portfolio_df is not None and not portfolio_df.empty:
            # Pass a copy to calculate_metrics to avoid SettingWithCopyWarning if modifying df
            metrics, portfolio_df_with_calcs = calculate_metrics(portfolio_df.copy())
            print_metrics(metrics)
            plot_pnl(portfolio_df_with_calcs) # Plot the df with calculations
        else:
            print("Could not calculate metrics due to issues with portfolio history.")
    else:
         print("Log parsing failed. Cannot continue analysis.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()