import json
import jsonpickle
import math
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from math import erf, sqrt
from dataclasses import dataclass

# Define data models
@dataclass
class Order:
    symbol: str
    price: int
    quantity: int

@dataclass
class OrderDepth:
    buy_orders: Dict[int, int]
    sell_orders: Dict[int, int]

@dataclass
class ConversionObservation:
    bidPrice: float
    askPrice: float
    transportFees: float
    exportTariff: float
    importTariff: float
    sunlightIndex: float
    sugarPrice: float

@dataclass
class Observation:
    conversionObservations: Dict[str, ConversionObservation]

@dataclass
class TradingState:
    timestamp: int
    listings: Dict[str, Any]
    order_depths: Dict[str, OrderDepth]
    own_trades: Dict[str, List[Any]]
    market_trades: Dict[str, List[Any]]
    position: Dict[str, int]
    observations: Observation
    traderData: str

# Logger class
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[str, List[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
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

    def compress_listings(self, listings: Dict[str, Any]) -> List[List[Any]]:
        return [[symbol, "", ""] for symbol in listings.keys()]

    def compress_order_depths(self, order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Any]]:
        return {symbol: [od.buy_orders, od.sell_orders] for symbol, od in order_depths.items()}

    def compress_trades(self, trades: Dict[str, List[Any]]) -> List[List[Any]]:
        return []

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        plain_observations = {}
        if observations and observations.conversionObservations:
            for product, obs in observations.conversionObservations.items():
                conversion_observations[product] = [
                    obs.bidPrice,
                    obs.askPrice,
                    obs.transportFees,
                    obs.exportTariff,
                    obs.importTariff,
                    obs.sunlightIndex,
                    obs.sugarPrice,
                ]
        return [plain_observations, conversion_observations]

    def compress_orders(self, orders: Dict[str, List[Order]]) -> List[List[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[:max_length - 3] + "..."

logger = Logger()

# MacaronTrader class
class MacaronTrader:
    def __init__(self):
        self.position_limit = 75
        self.conversion_limit = 10
        self.storage_cost = 0.1
        self.min_edge = 0.5
        self.adaptive_step = 0.2
        self.trader_data = {
            "edge": 1.0,
            "volume_history": [],
            "sunlight_history": [],
            "sugar_history": [],
            "last_sugar": None,
            "last_sunlight": None,
            "forecast_below_csi": False,
            "below_csi_duration": 0,
            "position_clearing_started": False
        }
        self.position_entry_time = 0
        self.current_position = 0
        self.CSI = 55.0
        self.forecast_periods = 2
        self.max_position_normal = 50
        self.max_position_csi = 75
        self.start_clearing_timestamp = 998 * 1000
        self.must_clear_timestamp = 1000 * 1000

    def calculate_storage_costs(self, position: int, timestamp: int) -> float:
        if position <= 0 or self.position_entry_time == 0:
            return 0.0
        return position * self.storage_cost * (timestamp - self.position_entry_time) / 100.0

    def update_position_tracking(self, new_position: int, timestamp: int):
        if new_position != self.current_position:
            if new_position != 0 and self.current_position == 0:
                self.position_entry_time = timestamp
            elif new_position == 0:
                self.position_entry_time = 0
            self.current_position = new_position

    def update_edge(self, fill_rate: float, volatility: float) -> float:
        if fill_rate > 0.7:
            new_edge = min(1.5, self.trader_data["edge"] * 1.1)
        elif fill_rate < 0.3:
            new_edge = max(0.1, self.trader_data["edge"] * 0.9)
        else:
            new_edge = self.trader_data["edge"]
        volatility_boost = min(0.5, volatility * 0.2)
        return new_edge + volatility_boost

    def calculate_fill_rate(self, timestamp: int) -> float:
        if not hasattr(self, 'fill_history'):
            self.fill_history = []
            self.ema_fill_rate = 0.5
            self.last_timestamp = None
        if timestamp == self.last_timestamp:
            return self.ema_fill_rate
        filled_orders = [o for o in self.last_orders if o.quantity > 0]
        fill_rate = len(filled_orders) / max(1, len(self.last_orders))
        self.ema_fill_rate = 0.2 * fill_rate + 0.8 * self.ema_fill_rate
        self.last_timestamp = timestamp
        return self.ema_fill_rate

    def calculate_volatility(self, order_depth: OrderDepth) -> float:
        if not hasattr(self, 'price_history'):
            self.price_history = []
        if order_depth.buy_orders and order_depth.sell_orders:
            high = min(order_depth.sell_orders.keys())
            low = max(order_depth.buy_orders.keys())
            self.price_history.append((high, low))
        if len(self.price_history) > 20:
            self.price_history.pop(0)
        if len(self.price_history) >= 5:
            sum_sq = 0.0
            for h, l in self.price_history:
                sum_sq += (math.log(h / l)) ** 2
            return math.sqrt(sum_sq / (4 * len(self.price_history) * math.log(2)))
        return 1.0

    def update_sugar_sunlight_history(self, obs: ConversionObservation):
        if obs:
            self.trader_data["sunlight_history"].append(obs.sunlightIndex)
            if len(self.trader_data["sunlight_history"]) > 10:
                self.trader_data["sunlight_history"].pop(0)
            self.trader_data["sugar_history"].append(obs.sugarPrice)
            if len(self.trader_data["sugar_history"]) > 10:
                self.trader_data["sugar_history"].pop(0)
            self.trader_data["last_sugar"] = obs.sugarPrice
            self.trader_data["last_sunlight"] = obs.sunlightIndex

    def forecast_sunlight(self) -> bool:
        history = self.trader_data["sunlight_history"]
        if not history or len(history) < 3:
            return False
        if history[-1] >= self.CSI:
            self.trader_data["below_csi_duration"] = 0
            return False
        self.trader_data["below_csi_duration"] += 1
        if len(history) >= 3:
            recent_trend = (history[-1] - history[-3]) / 2
            predicted_values = []
            last_value = history[-1]
            for _ in range(self.forecast_periods):
                next_value = last_value + recent_trend
                predicted_values.append(next_value)
                last_value = next_value
            return all(val < self.CSI for val in predicted_values)
        return False

    def is_sugar_below_average(self) -> bool:
        if not self.trader_data["sugar_history"]:
            return False
        current_sugar = self.trader_data["last_sugar"]
        avg_sugar = sum(self.trader_data["sugar_history"]) / len(self.trader_data["sugar_history"])
        return current_sugar < avg_sugar

    def calculate_implied_prices(self, observation: ConversionObservation) -> Tuple[float, float]:
        implied_bid = observation.bidPrice - observation.transportFees - observation.exportTariff
        implied_ask = observation.askPrice + observation.transportFees + observation.importTariff
        return implied_bid, implied_ask

    def predict_price(self, obs: ConversionObservation) -> float:
        fair_value = (obs.bidPrice + obs.askPrice) / 2
        sugar_impact = 0.05 * (obs.sugarPrice - 200)
        sunlight_impact = 0.1 * (obs.sunlightIndex - 60)
        csi_premium = 0
        if obs.sunlightIndex < self.CSI and self.trader_data["forecast_below_csi"]:
            duration_factor = min(3, self.trader_data["below_csi_duration"] / 2)
            csi_premium = 5.0 * duration_factor
            if self.is_sugar_below_average():
                csi_premium += 3.0
        if obs.sunlightIndex < (self.CSI - 5):
            csi_distance = self.CSI - obs.sunlightIndex
            csi_premium += 0.5 * (csi_distance ** 1.5)
        return fair_value + sugar_impact + sunlight_impact + csi_premium

    def smart_order_router(self, product: str, target_price: float, quantity: int,
                          order_depth: OrderDepth, side: str) -> List[Order]:
        orders = []
        if side == "buy":
            prices = sorted([p for p in order_depth.sell_orders if p <= target_price])
            for price in prices:
                qty = min(quantity, -order_depth.sell_orders[price])
                if qty > 0:
                    orders.append(Order(product, price, qty))
                    quantity -= qty
                if quantity <= 0:
                    break
        else:
            prices = sorted([p for p in order_depth.buy_orders if p >= target_price], reverse=True)
            for price in prices:
                qty = min(quantity, order_depth.buy_orders[price])
                if qty > 0:
                    orders.append(Order(product, price, -qty))
                    quantity -= qty
                if quantity <= 0:
                    break
        return orders

    def manage_position_risk(self, position: int, timestamp: int, sunlight_index: float) -> float:
        self.update_position_tracking(position, timestamp)
        storage_cost = self.calculate_storage_costs(position, timestamp)
        if timestamp >= self.start_clearing_timestamp:
            self.trader_data["position_clearing_started"] = True
            clearing_progress = min(1.0, (timestamp - self.start_clearing_timestamp) /
                                   (self.must_clear_timestamp - self.start_clearing_timestamp))
            aggression_factor = min(1.5, 1.0 + clearing_progress)
            if position > 0:
                return min(0.6, 1.0 - clearing_progress)
            if position < 0:
                return min(0.6, 1.0 - clearing_progress)
        max_pos = self.max_position_csi if sunlight_index < self.CSI else self.max_position_normal
        position_pct = abs(position) / max_pos
        if position_pct > 0.8:
            return 0.6
        elif position_pct > 0.5:
            return 0.8
        elif position_pct < 0.2:
            return 1.2
        if sunlight_index < self.CSI and self.trader_data["forecast_below_csi"]:
            if position > 0:
                return 1.5
            else:
                return 0.7
        return 1.0

    def calculate_market_impact(self, order_depth: OrderDepth, quantity: int) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
        bid_vol = sum(order_depth.buy_orders.values())
        ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        lambda_val = (1 / (bid_vol + ask_vol)) * 100
        return lambda_val * quantity

    def local_arbitrage(self, state: TradingState) -> List[Order]:
        orders = []
        product = "MAGNIFICENT_MACARONS"
        order_depth = state.order_depths.get(product, None)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        position = state.position.get(product, 0)
        if best_bid > best_ask:
            buy_volume = -order_depth.sell_orders[best_ask]
            sell_volume = order_depth.buy_orders[best_bid]
            max_buy = self.position_limit - position
            max_sell = self.position_limit + position
            execute_volume = min(buy_volume, sell_volume, max_buy, max_sell)
            if execute_volume > 0:
                orders.append(Order(product, best_ask, execute_volume))
                orders.append(Order(product, best_bid, -execute_volume))
        return orders

    def cross_market_arbitrage(self, state: TradingState) -> Tuple[List[Order], int]:
        orders = []
        conversions = 0
        product = "MAGNIFICENT_MACARONS"
        position = state.position.get(product, 0)
        if product not in state.observations.conversionObservations:
            return orders, conversions
        obs = state.observations.conversionObservations[product]
        implied_bid, implied_ask = self.calculate_implied_prices(obs)
        edge_multiplier = 1.0
        if obs.sunlightIndex < self.CSI and self.trader_data["forecast_below_csi"]:
            edge_multiplier = 0.7
        if self.trader_data["position_clearing_started"]:
            clearing_progress = min(1.0, (state.timestamp - self.start_clearing_timestamp) /
                                   (self.must_clear_timestamp - self.start_clearing_timestamp))
            edge_multiplier *= max(0.3, 1.0 - clearing_progress)
        effective_edge = self.trader_data["edge"] * edge_multiplier
        if len(state.order_depths[product].buy_orders) > 0:
            local_bid = max(state.order_depths[product].buy_orders.keys())
            required_edge = effective_edge
            if self.trader_data["position_clearing_started"] and position > 0:
                clearing_progress = min(1.0, (state.timestamp - self.start_clearing_timestamp) /
                                       (self.must_clear_timestamp - self.start_clearing_timestamp))
                required_edge *= max(0.5, 1.0 - clearing_progress)
            if local_bid > implied_ask + required_edge:
                qty = min(
                    self.position_limit - position,
                    state.order_depths[product].buy_orders[local_bid],
                    self.conversion_limit
                )
                if qty > 0:
                    orders.append(Order(product, local_bid, -qty))
                    conversions = -qty
        if len(state.order_depths[product].sell_orders) > 0:
            local_ask = min(state.order_depths[product].sell_orders.keys())
            required_edge = effective_edge
            if self.trader_data["position_clearing_started"] and position < 0:
                clearing_progress = min(1.0, (state.timestamp - self.start_clearing_timestamp) /
                                       (self.must_clear_timestamp - self.start_clearing_timestamp))
                required_edge *= max(0.5, 1.0 - clearing_progress)
            if implied_bid > local_ask + required_edge:
                qty = min(
                    self.position_limit + position,
                    -state.order_depths[product].sell_orders[local_ask],
                    self.conversion_limit
                )
                if qty > 0:
                    orders.append(Order(product, local_ask, qty))
                    conversions = qty
        return orders, conversions

    def position_clearing_strategy(self, state: TradingState) -> List[Order]:
        product = "MAGNIFICENT_MACARONS"
        orders = []
        position = state.position.get(product, 0)
        if state.timestamp < self.start_clearing_timestamp or position == 0:
            return orders
        order_depth = state.order_depths.get(product, OrderDepth())
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
        clearing_progress = min(1.0, (state.timestamp - self.start_clearing_timestamp) /
                               (self.must_clear_timestamp - self.start_clearing_timestamp))
        aggression_factor = 1.0
        if clearing_progress > 0.8:
            aggression_factor = 3.0
        elif clearing_progress > 0.5:
            aggression_factor = 2.0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        if position > 0:
            target_cleared = int(position * clearing_progress * aggression_factor)
            price_discount = max(0, int(clearing_progress * 2.0))
            target_price = max(1, best_bid - price_discount)
            selling_qty = min(position, target_cleared)
            if selling_qty > 0:
                orders.append(Order(product, target_price, -selling_qty))
        elif position < 0:
            target_cleared = int(abs(position) * clearing_progress * aggression_factor)
            price_premium = max(0, int(clearing_progress * 2.0))
            target_price = best_ask + price_premium
            buying_qty = min(abs(position), target_cleared)
            if buying_qty > 0:
                orders.append(Order(product, target_price, buying_qty))
        return orders

    def csi_driven_strategy(self, state: TradingState) -> List[Order]:
        product = "MAGNIFICENT_MACARONS"
        orders = []
        if self.trader_data["position_clearing_started"]:
            return orders
        if product not in state.observations.conversionObservations:
            return orders
        obs = state.observations.conversionObservations[product]
        position = state.position.get(product, 0)
        order_depth = state.order_depths.get(product, OrderDepth())
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
        self.trader_data["forecast_below_csi"] = self.forecast_sunlight()
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        if obs.sunlightIndex < self.CSI and self.trader_data["forecast_below_csi"] and self.is_sugar_below_average():
            if not self.trader_data["position_clearing_started"]:
                target_buy_price = int(best_ask + 0.5)
                available_position = self.position_limit - position
                target_position = int(available_position * 0.8)
                if target_position > 0:
                    orders.append(Order(product, target_buy_price, target_position))
                    return orders
        prev_below_csi = len(self.trader_data["sunlight_history"]) >= 2 and self.trader_data["sunlight_history"][-2] < self.CSI
        current_above_csi = obs.sunlightIndex >= self.CSI
        if prev_below_csi and current_above_csi and position > 0:
            target_sell_price = int(best_bid - 0.5)
            orders.append(Order(product, target_sell_price, -position))
            return orders
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        product = "MAGNIFICENT_MACARONS"
        all_orders = []
        conversions = 0
        self.last_orders = []
        order_depth = state.order_depths.get(product, OrderDepth())
        obs = state.observations.conversionObservations.get(product)
        position = state.position.get(product, 0)
        if obs:
            self.update_sugar_sunlight_history(obs)
        volatility = self.calculate_volatility(order_depth)
        fill_rate = self.calculate_fill_rate(state.timestamp)
        if state.timestamp >= self.start_clearing_timestamp:
            self.trader_data["position_clearing_started"] = True
            clearing_orders = self.position_clearing_strategy(state)
            if clearing_orders:
                all_orders.extend(clearing_orders)
        sunlight_index = obs.sunlightIndex if obs else 60
        risk_multiplier = self.manage_position_risk(position, state.timestamp, sunlight_index)
        self.trader_data["edge"] = self.update_edge(fill_rate, volatility)
        effective_edge = self.trader_data["edge"] * risk_multiplier
        arb_orders = self.local_arbitrage(state)
        if arb_orders:
            all_orders.extend(arb_orders)
            return {product: all_orders}, 0, jsonpickle.encode(self.trader_data)
        cross_orders, cross_conversions = self.cross_market_arbitrage(state)
        if cross_orders:
            all_orders.extend(cross_orders)
            return {product: all_orders}, cross_conversions, jsonpickle.encode(self.trader_data)
        if self.trader_data["position_clearing_started"] and position != 0:
            clearing_progress = min(1.0, (state.timestamp - self.start_clearing_timestamp) /
                                   (self.must_clear_timestamp - self.start_clearing_timestamp))
            if clearing_progress > 0.9 and position != 0:
                if position > 0 and order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    price_discount = int(min(5.0, 10.0 * (clearing_progress - 0.9) * 10.0))
                    price = max(1, best_bid - price_discount)
                    all_orders.append(Order(product, price, -position))
                elif position < 0 and order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    price_premium = int(min(5.0, 10.0 * (clearing_progress - 0.9) * 10.0))
                    price = best_ask + price_premium
                    all_orders.append(Order(product, price, abs(position)))
                return {product: all_orders}, conversions, jsonpickle.encode(self.trader_data)
        if not self.trader_data["position_clearing_started"]:
            csi_orders = self.csi_driven_strategy(state)
            if csi_orders:
                all_orders.extend(csi_orders)
                return {product: all_orders}, 0, jsonpickle.encode(self.trader_data)
        if obs and len(order_depth.buy_orders) and len(order_depth.sell_orders):
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid
            fair_value = self.predict_price(obs)
            if spread > 2 * effective_edge:
                bid_price = int(fair_value - effective_edge)
                ask_price = int(fair_value + effective_edge)
                if self.trader_data["position_clearing_started"]:
                    clearing_progress = min(1.0, (state.timestamp - self.start_clearing_timestamp) /
                                           (self.must_clear_timestamp - self.start_clearing_timestamp))
                    size_multiplier = max(0.1, 1.0 - clearing_progress)
                    if position > 0:
                        buy_qty = int(min(self.position_limit - position, 2 * size_multiplier))
                        sell_qty = int(min(self.position_limit + position, 8 * size_multiplier))
                    elif position < 0:
                        buy_qty = int(min(self.position_limit - position, 8 * size_multiplier))
                        sell_qty = int(min(self.position_limit + position, 2 * size_multiplier))
                    else:
                        buy_qty = int(min(self.position_limit - position, 1 * size_multiplier))
                        sell_qty = int(min(self.position_limit + position, 1 * size_multiplier))
                elif obs.sunlightIndex < self.CSI and self.trader_data["forecast_below_csi"]:
                    buy_qty = min(self.position_limit - position, 10)
                    sell_qty = min(self.position_limit + position, 3)
                else:
                    buy_qty = min(self.position_limit - position, 5)
                    sell_qty = min(self.position_limit + position, 5)
                if buy_qty > 0:
                    all_orders.append(Order(product, bid_price, buy_qty))
                if sell_qty > 0:
                    all_orders.append(Order(product, ask_price, -sell_qty))
        return {product: all_orders}, conversions, jsonpickle.encode(self.trader_data)

# Simulated TradingState for testing
def create_test_state(timestamp: int, position: int = 0) -> TradingState:
    product = "MAGNIFICENT_MACARONS"
    return TradingState(
        timestamp=timestamp,
        listings={product: {"symbol": product, "product": product, "denomination": "SEASHELLS"}},
        order_depths={
            product: OrderDepth(
                buy_orders={100: 10, 99: 15},
                sell_orders={101: -10, 102: -15}
            )
        },
        own_trades={product: []},
        market_trades={product: []},
        position={product: position},
        observations=Observation(
            conversionObservations={
                product: ConversionObservation(
                    bidPrice=100.0,
                    askPrice=101.0,
                    transportFees=0.5,
                    exportTariff=0.2,
                    importTariff=0.3,
                    sunlightIndex=50.0,
                    sugarPrice=190.0
                )
            }
        ),
        traderData=""
    )

# Main execution
if __name__ == "__main__":
    trader = MacaronTrader()
    state = create_test_state(timestamp=500000, position=0)
    result, conversions, trader_data = trader.run(state)
    logger.flush(state, result, conversions, trader_data)
    print("Orders:", [(o.symbol, o.price, o.quantity) for o in result["MAGNIFICENT_MACARONS"]])
    print("Conversions:", conversions)