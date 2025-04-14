import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias
import statistics

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# Logger class (unchanged)
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[:max_length - 3] + "..."

logger = Logger()

# Base Strategy class (unchanged)
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.average_buy_price = None  # Track average buy price for stop-loss
        self.average_sell_price = None  # Track average sell price for stop-loss

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        if quantity > 0:
            if self.average_buy_price is None:
                self.average_buy_price = price
            else:
                total_quantity = quantity + (self.limit - quantity)
                self.average_buy_price = (self.average_buy_price * (self.limit - quantity) + price * quantity) / total_quantity
            self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        if quantity > 0:
            if self.average_sell_price is None:
                self.average_sell_price = price
            else:
                total_quantity = quantity + (self.limit - quantity)
                self.average_sell_price = (self.average_sell_price * (self.limit - quantity) + price * quantity) / total_quantity
            self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return {"average_buy_price": self.average_buy_price, "average_sell_price": self.average_sell_price}

    def load(self, data: JSON) -> None:
        self.average_buy_price = data.get("average_buy_price", None) if data else None
        self.average_sell_price = data.get("average_sell_price", None) if data else None

# MarketMakingStrategy (unchanged)
class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Stop-loss: Exit position if loss exceeds 5% of average price
        if position > 0 and self.average_sell_price and buy_orders:
            best_bid = buy_orders[0][0]
            if best_bid < self.average_sell_price * 0.95:  # 5% loss
                self.sell(best_bid, position)
                return
        elif position < 0 and self.average_buy_price and sell_orders:
            best_ask = sell_orders[0][0]
            if best_ask > self.average_buy_price * 1.05:  # 5% loss
                self.buy(best_ask, -position)
                return

        # Cap trade size to 25% of limit to avoid overexposure
        to_buy = min(to_buy, self.limit // 4)
        to_sell = min(to_sell, self.limit // 4)

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        base_save = super().save()
        base_save["window"] = list(self.window)
        return base_save

    def load(self, data: JSON) -> None:
        super().load(data)
        self.window = deque(data.get("window", []))

# InkMeanReversionTrader (reverted to less aggressive thresholds)
class InkMeanReversionTrader(Strategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.prices = deque(maxlen=300)
        self.window = 300

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        if not buy_orders or not sell_orders:
            return

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        mid_price = (best_bid + best_ask) / 2
        self.prices.append(mid_price)

        if len(self.prices) < self.window:
            return

        mean = statistics.mean(self.prices)
        std_dev = statistics.stdev(self.prices)
        position = state.position.get(self.symbol, 0)

        # Stop-loss: Exit position if loss exceeds 5% of average price
        if position > 0 and self.average_sell_price and buy_orders:
            best_bid = buy_orders[0][0]
            if best_bid < self.average_sell_price * 0.95:  # 5% loss
                self.sell(best_bid, position)
                return
        elif position < 0 and self.average_buy_price and sell_orders:
            best_ask = sell_orders[0][0]
            if best_ask > self.average_buy_price * 1.05:  # 5% loss
                self.buy(best_ask, -position)
                return

        # Less aggressive thresholds (reverted to r2strat11.py style)
        if mid_price < mean - 1 * std_dev and position < self.limit:
            quantity = min(self.limit - position, self.limit // 4)  # Cap trade size
            self.buy(best_ask, quantity)
        elif mid_price > mean + 1 * std_dev and position > -self.limit:
            quantity = min(self.limit + position, self.limit // 4)  # Cap trade size
            self.sell(best_bid, quantity)
        elif abs(mid_price - mean) <= 0.25:
            if position > 0:
                self.sell(best_bid, position)
            elif position < 0:
                self.buy(best_ask, -position)

    def save(self) -> JSON:
        base_save = super().save()
        base_save["prices"] = list(self.prices)
        return base_save

    def load(self, data: JSON) -> None:
        super().load(data)
        self.prices = deque(data.get("prices", []), maxlen=300)

# Basket1Trader (unchanged)
class Basket1Trader(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)

    def act(self, state: TradingState) -> None:
        required_symbols = ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]
        for sym in required_symbols:
            if sym not in state.order_depths:
                return

        basket_od = state.order_depths["PICNIC_BASKET1"]
        croissants_od = state.order_depths["CROISSANTS"]
        jams_od = state.order_depths["JAMS"]
        djembe_od = state.order_depths["DJEMBES"]

        basket_bids = sorted(basket_od.buy_orders.items(), reverse=True)
        basket_asks = sorted(basket_od.sell_orders.items())
        if not basket_bids or not basket_asks:
            return
        basket_best_bid = basket_bids[0][0]
        basket_best_ask = basket_asks[0][0]
        basket_mid = (basket_best_bid + basket_best_ask) / 2

        def get_mid(order_depth):
            bids = sorted(order_depth.buy_orders.items(), reverse=True)
            asks = sorted(order_depth.sell_orders.items())
            if not bids or not asks:
                return None
            return (bids[0][0] + asks[0][0]) / 2

        mid_croissants = get_mid(croissants_od)
        mid_jams = get_mid(jams_od)
        mid_djembe = get_mid(djembe_od)
        if mid_croissants is None or mid_jams is None or mid_djembe is None:
            return

        composite_value = 6 * mid_croissants + 3 * mid_jams + mid_djembe
        fair_value = composite_value
        deviation = basket_mid - fair_value
        pos = state.position.get("PICNIC_BASKET1", 0)
        threshold = 1

        if deviation < -threshold and pos < self.limit:
            quantity = min(self.limit - pos, self.limit // 4)  # Cap trade size
            self.buy(basket_best_bid, quantity)
        elif deviation > threshold and pos > -self.limit:
            quantity = min(self.limit + pos, self.limit // 4)  # Cap trade size
            self.sell(basket_best_ask, quantity)
        elif abs(deviation) < threshold and pos != 0:
            if pos > 0:
                self.sell(basket_best_ask, pos)
            elif pos < 0:
                self.buy(basket_best_bid, -pos)

# AdvancedMarketMakingStrategy (tuned parameters)
class AdvancedMarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, params: dict) -> None:
        super().__init__(symbol, limit)
        self.params = params

    def dynamic_fair_value(self, state: TradingState, traderObject: dict) -> float:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return self.params["fair_value"]
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        current_mid = (best_ask + best_bid) / 2
        last_price = traderObject.get(self.symbol + "_last_price", current_mid)
        returns = (current_mid - last_price) / last_price if last_price != 0 else 0
        adjustment = returns * self.params.get("reversion_beta", 0)
        new_fair_value = current_mid + (current_mid * adjustment)
        traderObject[self.symbol + "_last_price"] = current_mid
        return new_fair_value

    def compute_risk_factor(self, position: int) -> float:
        soft_limit = self.params.get("soft_position_limit", 40)
        factor = max(0.2, (soft_limit - abs(position)) / soft_limit)
        return factor

    def take_best_orders(self, state: TradingState, fair_value: float, orders: list[Order], position: int) -> tuple[int, int]:
        order_depth = state.order_depths[self.symbol]
        buy_order_volume = 0
        sell_order_volume = 0
        risk_factor = self.compute_risk_factor(position)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_qty = -order_depth.sell_orders[best_ask]
            prevent_adverse = self.params.get("prevent_adverse", False)
            adverse_volume = self.params.get("adverse_volume", 0)
            if (not prevent_adverse or best_ask_qty <= adverse_volume) and best_ask <= fair_value - self.params["take_width"]:
                quantity = min(int(best_ask_qty * risk_factor), self.limit - position, self.limit // 4)  # Cap trade size
                if quantity > 0:
                    orders.append(Order(self.symbol, best_ask, quantity))
                    buy_order_volume += quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_qty = order_depth.buy_orders[best_bid]
            prevent_adverse = self.params.get("prevent_adverse", False)
            adverse_volume = self.params.get("adverse_volume", 0)
            if (not prevent_adverse or best_bid_qty <= adverse_volume) and best_bid >= fair_value + self.params["take_width"]:
                quantity = min(int(best_bid_qty * risk_factor), self.limit + position, self.limit // 4)  # Cap trade size
                if quantity > 0:
                    orders.append(Order(self.symbol, best_bid, -quantity))
                    sell_order_volume += quantity
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, state: TradingState, fair_value: float, orders: list[Order], position: int, buy_order_volume: int, sell_order_volume: int) -> tuple[int, int]:
        order_depth = state.order_depths[self.symbol]
        position_after = position + buy_order_volume - sell_order_volume
        fair_bid = round(fair_value - self.params["clear_width"])
        fair_ask = round(fair_value + self.params["clear_width"])
        buy_qty = self.limit - (position + buy_order_volume)
        sell_qty = self.limit + (position - sell_order_volume)
        risk_factor = self.compute_risk_factor(position)
        if position_after > 0:
            clear_vol = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_ask)
            clear_vol = min(clear_vol, position_after)
            send_qty = min(int(sell_qty * risk_factor), clear_vol, self.limit // 4)  # Cap trade size
            if send_qty > 0:
                orders.append(Order(self.symbol, fair_ask, -abs(send_qty)))
                sell_order_volume += abs(send_qty)
        if position_after < 0:
            clear_vol = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_bid)
            clear_vol = min(clear_vol, abs(position_after))
            send_qty = min(int(buy_qty * risk_factor), clear_vol, self.limit // 4)  # Cap trade size
            if send_qty > 0:
                orders.append(Order(self.symbol, fair_bid, abs(send_qty)))
                buy_order_volume += abs(send_qty)
        return buy_order_volume, sell_order_volume

    def make_orders(self, state: TradingState, fair_value: float, position: int, buy_order_volume: int, sell_order_volume: int) -> tuple[list[Order], int, int]:
        orders = []
        order_depth = state.order_depths[self.symbol]
        disregard_edge = self.params["disregard_edge"]
        join_edge = self.params["join_edge"]
        default_edge = self.params["default_edge"]
        soft_position_limit = self.params.get("soft_position_limit", 40)
        asks_above = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        ask = round(fair_value + default_edge)
        if asks_above:
            best_ask = min(asks_above)
            if abs(best_ask - fair_value) <= join_edge:
                ask = best_ask
            else:
                ask = best_ask - 1
        bid = round(fair_value - default_edge)
        if bids_below:
            best_bid = max(bids_below)
            if abs(fair_value - best_bid) <= join_edge:
                bid = best_bid
            else:
                bid = best_bid + 1
        if position > soft_position_limit:
            ask -= 1
        elif position < -soft_position_limit:
            bid += 1
        risk_factor = self.compute_risk_factor(position)
        buy_qty = int((self.limit - (position + buy_order_volume)) * risk_factor)
        buy_qty = min(buy_qty, self.limit // 4)  # Cap trade size
        if buy_qty > 0:
            orders.append(Order(self.symbol, round(bid), buy_qty))
        sell_qty = int((self.limit + (position - sell_order_volume)) * risk_factor)
        sell_qty = min(sell_qty, self.limit // 4)  # Cap trade size
        if sell_qty > 0:
            orders.append(Order(self.symbol, round(ask), -sell_qty))
        return orders, buy_order_volume, sell_order_volume

    def act(self, state: TradingState) -> None:
        traderObject = state.traderData.get(self.symbol, {}) if isinstance(state.traderData, dict) else {}
        fair_value = self.dynamic_fair_value(state, traderObject) if self.symbol == "SQUID_INK" else self.params["fair_value"]
        position = state.position.get(self.symbol, 0)

        # Stop-loss: Exit position if loss exceeds 5% of average price
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        if position > 0 and self.average_sell_price and buy_orders:
            best_bid = buy_orders[0][0]
            if best_bid < self.average_sell_price * 0.95:  # 5% loss
                self.sell(best_bid, position)
                return
        elif position < 0 and self.average_buy_price and sell_orders:
            best_ask = sell_orders[0][0]
            if best_ask > self.average_buy_price * 1.05:  # 5% loss
                self.buy(best_ask, -position)
                return

        buy_order_volume, sell_order_volume = self.take_best_orders(state, fair_value, self.orders, position)
        buy_order_volume, sell_order_volume = self.clear_position_order(state, fair_value, self.orders, position, buy_order_volume, sell_order_volume)
        make_orders, buy_order_volume, sell_order_volume = self.make_orders(state, fair_value, position, buy_order_volume, sell_order_volume)
        self.orders.extend(make_orders)

    def save(self) -> JSON:
        return super().save()

    def load(self, data: JSON) -> None:
        super().load(data)

# KelpStrategy (reverted from r2strat11.py)
class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return round((popular_buy_price + popular_sell_price) / 2)

# TrendVolatilityStrategy (improved with trend-following)
class TrendVolatilityStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int, analysis_params: dict) -> None:
        super().__init__(symbol, limit)
        self.analysis_params = analysis_params
        self.prices = deque(maxlen=100)
        self.volatility_window = 50
        self.short_ma = deque(maxlen=20)  # Short-term moving average for trend
        self.long_ma = deque(maxlen=50)  # Long-term moving average for trend

    def get_true_value(self, state: TradingState) -> int:
        params = self.analysis_params.get(self.symbol, {})
        base_value = params.get("median_price", 10000)
        trend = params.get("trend_slope", 0)
        volatility = params.get("volatility", 10)

        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return base_value

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else base_value
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else base_value
        market_mid = (best_bid + best_ask) // 2

        self.prices.append(market_mid)
        self.short_ma.append(market_mid)
        self.long_ma.append(market_mid)

        analysis_value = base_value + trend
        true_value = (analysis_value + market_mid) / 2

        # Trend-following adjustment
        if len(self.short_ma) >= 20 and len(self.long_ma) >= 50:
            short_ma = statistics.mean(self.short_ma)
            long_ma = statistics.mean(self.long_ma)
            if short_ma > long_ma:  # Uptrend
                true_value += 1
            elif short_ma < long_ma:  # Downtrend
                true_value -= 1

        # Mean-reversion adjustment
        if len(self.prices) >= self.volatility_window:
            mean = statistics.mean(self.prices)
            std_dev = statistics.stdev(self.prices)
            if market_mid < mean - 1.5 * std_dev:
                true_value += 2
            elif market_mid > mean + 1.5 * std_dev:
                true_value -= 2

        return round(true_value)

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Stop-loss: Exit position if loss exceeds 5% of average price
        if position > 0 and self.average_sell_price and buy_orders:
            best_bid = buy_orders[0][0]
            if best_bid < self.average_sell_price * 0.95:  # 5% loss
                self.sell(best_bid, position)
                return
        elif position < 0 and self.average_buy_price and sell_orders:
            best_ask = sell_orders[0][0]
            if best_ask > self.average_buy_price * 1.05:  # 5% loss
                self.buy(best_ask, -position)
                return

        # Cap trade size to 25% of limit
        to_buy = min(to_buy, self.limit // 4)
        to_sell = min(to_sell, self.limit // 4)

        volatility = self.analysis_params.get(self.symbol, {}).get("volatility", 10)
        confidence = max(0.3, 1 - (volatility / 50))
        to_buy = int(to_buy * confidence)
        to_sell = int(to_sell * confidence)

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        base_save = super().save()
        base_save.update({
            "window": list(self.window),
            "prices": list(self.prices),
            "short_ma": list(self.short_ma),
            "long_ma": list(self.long_ma)
        })
        return base_save

    def load(self, data: JSON) -> None:
        super().load(data)
        self.window = deque(data.get("window", []))
        self.prices = deque(data.get("prices", []), maxlen=100)
        self.short_ma = deque(data.get("short_ma", []), maxlen=20)
        self.long_ma = deque(data.get("long_ma", []), maxlen=50)

# Trader class (updated)
class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 50,  # Reduced from 250
            "JAMS": 50,  # Reduced from 350
            "DJEMBES": 50,
        }

        # Tuned analysis parameters for TrendVolatilityStrategy
        analysis_params = {
            "RAINFOREST_RESIN": {"median_price": 10000, "trend_slope": 0, "volatility": 5},
            "DJEMBES": {"median_price": 13400, "trend_slope": -5, "volatility": 20},
            "CROISSANTS": {"median_price": 4300, "trend_slope": 0, "volatility": 5},
            "JAMS": {"median_price": 6500, "trend_slope": 0, "volatility": 5},
        }

        # Tuned parameters for AdvancedMarketMakingStrategy
        advanced_params = {
            "SQUID_INK": {
                "fair_value": 1836,  # Adjusted based on second strategy's backtest
                "take_width": 2,  # Less aggressive
                "clear_width": 1,
                "prevent_adverse": True,
                "adverse_volume": 10,
                "reversion_beta": -0.1,  # Less aggressive
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 3,
                "soft_position_limit": 40,
            },
            "PICNIC_BASKET2": {
                "fair_value": 30254,  # Adjusted based on second strategy's backtest
                "take_width": 2,  # Less aggressive
                "clear_width": 1,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 3,
                "soft_position_limit": 80,
            },
        }

        self.strategies = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": lambda symbol, limit: TrendVolatilityStrategy(symbol, limit, analysis_params),
            "KELP": KelpStrategy,  # Reverted to KelpStrategy
            "SQUID_INK": lambda symbol, limit: AdvancedMarketMakingStrategy(symbol, limit, advanced_params[symbol]),
            "PICNIC_BASKET1": Basket1Trader,
            "PICNIC_BASKET2": lambda symbol, limit: AdvancedMarketMakingStrategy(symbol, limit, advanced_params[symbol]),
            "CROISSANTS": lambda symbol, limit: TrendVolatilityStrategy(symbol, limit, analysis_params),
            "JAMS": lambda symbol, limit: TrendVolatilityStrategy(symbol, limit, analysis_params),
            "DJEMBES": lambda symbol, limit: TrendVolatilityStrategy(symbol, limit, analysis_params),
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # Use conversions only for PICNIC_BASKET1 and PICNIC_BASKET2 when positions are non-zero
        conversions = 0
        position = state.position
        if ("PICNIC_BASKET1" in position and position["PICNIC_BASKET1"] != 0) or \
           ("PICNIC_BASKET2" in position and position["PICNIC_BASKET2"] != 0):
            conversions = 1

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))

            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data