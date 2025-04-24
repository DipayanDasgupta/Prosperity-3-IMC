import json
import numpy as np
from typing import Dict, List, Tuple, Any
from datamodel import Order, OrderDepth, TradingState, ProsperityEncoder

class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[str, list[Order]], conversions: int, trader_data: str):
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            ""
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length)
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str):
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations)
        ]

    def compress_listings(self, listings: dict[str, Any]):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[str, OrderDepth]):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[str, list[Any]]):
        compressed = []
        seen = set()
        for arr in trades.values():
            for trade in arr:
                trade_tuple = (
                    trade.symbol,
                    int(trade.price),
                    int(trade.quantity),
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                )
                if trade_tuple not in seen:
                    seen.add(trade_tuple)
                    compressed.append([
                        trade.symbol,
                        int(trade.price),
                        int(trade.quantity),
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ])
        return compressed

    def compress_observations(self, obs: Any):
        co = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
              for p, o in obs.conversionObservations.items()}
        return [obs.plainValueObservations, co]

    def compress_orders(self, orders: dict[str, list[Order]]):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        """Initialize the trader with parameters for VOLCANIC_ROCK trading."""
        self.history = []  # Store mid-price history
        self.position_limit = 400  # Max position limit for VOLCANIC_ROCK
        self.window_size = 50  # Rolling window for z-score calculation
        self.zscore_threshold = 2  # Z-score threshold for trading signals

    def update_volcanic_rock_history(self, state: TradingState) -> List[Order]:
        """Generate trading orders for VOLCANIC_ROCK based on z-score strategy."""
        orders = []
        rock_depth = state.order_depths.get("VOLCANIC_ROCK")

        # Check if order depth is valid
        if not rock_depth or not rock_depth.buy_orders or not rock_depth.sell_orders:
            logger.print("No valid order depth for VOLCANIC_ROCK")
            return orders

        # Calculate mid-price
        rock_bid = max(rock_depth.buy_orders.keys())
        rock_ask = min(rock_depth.sell_orders.keys())
        rock_mid = (rock_bid + rock_ask) / 2
        self.history.append(rock_mid)

        # Calculate z-score if enough data is available
        rock_prices = np.array(self.history)
        if len(rock_prices) > self.window_size:
            recent = rock_prices[-self.window_size:]
            mean = np.mean(recent)
            std = np.std(recent)
            z_score = (rock_prices[-1] - mean) / std if std > 0 else 0

            position = state.position.get("VOLCANIC_ROCK", 0)

            # Buy if price is significantly undervalued (z-score < -threshold)
            if z_score < -self.zscore_threshold and rock_depth.sell_orders:
                best_ask = min(rock_depth.sell_orders.keys())
                qty = -rock_depth.sell_orders[best_ask]  # Available volume
                buy_qty = min(qty, self.position_limit - position)
                if buy_qty > 0:
                    orders.append(Order("VOLCANIC_ROCK", best_ask, buy_qty))
                    logger.print(f"Buy VOLCANIC_ROCK: {buy_qty} @ {best_ask}, z-score: {z_score:.2f}")

            # Sell if price is significantly overvalued (z-score > threshold)
            elif z_score > self.zscore_threshold and rock_depth.buy_orders:
                best_bid = max(rock_depth.buy_orders.keys())
                qty = rock_depth.buy_orders[best_bid]  # Available volume
                sell_qty = min(qty, self.position_limit + position)
                if sell_qty > 0:
                    orders.append(Order("VOLCANIC_ROCK", best_bid, -sell_qty))
                    logger.print(f"Sell VOLCANIC_ROCK: {sell_qty} @ {best_bid}, z-score: {z_score:.2f}")

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main trading logic for the backtester.
        Returns: (orders, conversions, trader_data)
        """
        result = {"VOLCANIC_ROCK": []}
        conversions = 0

        # Update history and generate orders for VOLCANIC_ROCK
        result["VOLCANIC_ROCK"].extend(self.update_volcanic_rock_history(state))

        # Serialize trader data (price history) for persistence
        trader_data = json.dumps({
            "history": self.history[-self.window_size:] if len(self.history) > self.window_size else self.history
        })

        # Filter out empty order lists
        final_result = {k: v for k, v in result.items() if v}

        # Log state and orders
        logger.flush(state, final_result, conversions, trader_data)

        return final_result, conversions, trader_data