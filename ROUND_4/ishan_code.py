from datamodel import Order, OrderDepth, TradingState, ProsperityEncoder, Listing, Observation, Trade, Symbol, ConversionObservation
from typing import List, Any, Optional, Dict, Tuple
import jsonpickle
from collections import deque
import json
import statistics
from math import log, sqrt, erf
import math
import numpy as np

JSON = Dict[str, Any] | List[Any] | str | int | float | bool | None

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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
        for symbol, listing in listings.items():
            # Handle both dictionary and object formats
            if isinstance(listing, dict):
                # Dictionary format
                compressed.append([symbol, listing.get("product", ""), listing.get("denomination", "")])
            else:
                # Object format - try to access attributes
                try:
                    compressed.append([symbol, getattr(listing, "product", ""), getattr(listing, "denomination", "")])
                except:
                    # Fallback if all else fails
                    compressed.append([symbol, "", ""])

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
        plain_observations = {}
        
        # Handle the case when observations might be None
        if observations is None:
            return [plain_observations, conversion_observations]
        
        # Handle conversionObservations
        if hasattr(observations, 'conversionObservations'):
            try:
                for product, observation in observations.conversionObservations.items():
                    conversion_observations[product] = [
                        getattr(observation, 'bidPrice', 0),
                        getattr(observation, 'askPrice', 0),
                        getattr(observation, 'transportFees', 0),
                        getattr(observation, 'exportTariff', 0),
                        getattr(observation, 'importTariff', 0),
                        getattr(observation, 'sunlight', 0),
                        getattr(observation, 'humidity', 0),
                    ]
            except:
                # Fallback if something goes wrong
                pass

        # Handle plainValueObservations
        if hasattr(observations, 'plainValueObservations'):
            try:
                plain_observations = observations.plainValueObservations
            except:
                # Fallback if something goes wrong
                pass

        return [plain_observations, conversion_observations]

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

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
    STRIKES = [9500, 9750, 10000, 10250, 10500]

Product.VOUCHER_SYMBOLS = [f"{Product.VOUCHER_PREFIX}{K}" for K in Product.STRIKES]

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 25,
    },
    Product.KELP: {
        "take_width": 1.5,
        "clear_width": 0.5,
        "prevent_adverse": False,
        "adverse_volume": float("inf"),
        "reversion_beta": -1,
        "disregard_edge": 0,
        "join_edge": 4,
        "default_edge": 1,
        "strategy": "volume_weighted_mid_price"
    },
    Product.SQUID_INK: {
        "take_width": 2, 
        "clear_width": 1, 
        "prevent_adverse": False, 
        "adverse_volume": 15,
        "reversion_beta": -0.228, 
        "disregard_edge": 2, 
        "join_edge": 0, 
        "default_edge": 1,
        "spike_lb": 3, 
        "spike_ub": 5.6, 
        "offset": 2, 
        "reversion_window": 55,
        "reversion_weight": 0.16,
    },
    Product.JAMS: {
        "sma_short": 10,
        "sma_long": 50,
        "min_edge": 1,
        "volume_limit": 300,
        "order_volume": 50,
        "clear_threshold": 2,
        "buy_volume_reduction_bearish": 0,
        "sell_volume_increase_bearish": 1,
        "target_position_bearish": -300,
    },
    Product.CROISSANTS: {
        "fair_value": 10,
        "take_width": 0.5,
        "clear_width": 0.2,
        "disregard_edge": 0.5,
        "join_edge": 0.5,
        "default_edge": 1,
        "soft_position_limit": 125,
    },
    Product.DJEMBES: {
        "fair_value": 50,
        "take_width": 1,
        "clear_width": 0.5,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 2,
        "soft_position_limit": 30,
    },
    Product.PICNIC_BASKET1: {
        "trade_size": 5,
        "threshold": 10,
    },
    Product.PICNIC_BASKET2: {
        "lookback_window": 100,
        "k_entry": 1.83,
        "k_exit": 0.549,
        "min_std_dev": 5,
        "min_entry_threshold": 10,
        "max_entry_threshold": 90,
        "min_exit_threshold": 12,
        "max_exit_threshold": 32,
        "default_entry": 25,
        "default_exit": 21,
    },
}

POSITION_LIMITS = {
    Product.RAINFOREST_RESIN: 50,
    Product.KELP: 50,
    Product.SQUID_INK: 50,
    Product.JAMS: 350,
    Product.CROISSANTS: 250,
    Product.DJEMBES: 60,
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
}

PICNIC_BASKET1_WEIGHTS = {Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1}
PICNIC_BASKET2_WEIGHTS = {Product.CROISSANTS: 4, Product.JAMS: 2}

def get_best_bid(order_depth: OrderDepth) -> Optional[int]:
    if order_depth.buy_orders:
        return max(order_depth.buy_orders.keys())
    return None

def get_best_ask(order_depth: OrderDepth) -> Optional[int]:
    if order_depth.sell_orders:
        return min(order_depth.sell_orders.keys())
    return None

def get_volume_at_price(order_depth: OrderDepth, side: str, price: int) -> int:
    if price is None:
        return 0
    if side == "buy":
        return order_depth.buy_orders.get(price, 0)
    if side == "sell":
        return abs(order_depth.sell_orders.get(price, 0))
    return 0

class MacaronTrader:
    def __init__(self):
        self.position_limit = 75
        self.conversion_limit = 10
        self.storage_cost = 0.1 
        self.min_edge = 0.5 #0.5     
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
        self.start_clearing_timestamp = 998*1000 
        self.must_clear_timestamp = 1000*1000
        self.clearing_target_percentage = 0.0

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
        fill_rate = len(filled_orders)/max(1, len(self.last_orders))
        
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
                sum_sq += (math.log(h/l)) ** 2
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

    def calculate_implied_prices(self, observation: ConversionObservation) -> tuple:
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
                if quantity <= 0: break
        else:
            prices = sorted([p for p in order_depth.buy_orders if p >= target_price], reverse=True)
            for price in prices:
                qty = min(quantity, order_depth.buy_orders[price])
                if qty > 0:
                    orders.append(Order(product, price, -qty))
                    quantity -= qty
                if quantity <= 0: break
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
        
        lambda_val = (1/(bid_vol + ask_vol)) * 100
        return lambda_val * quantity

    def local_arbitrage(self, state: TradingState) -> List[Order]:
        orders = []
        product = Product.MAGNIFICENT_MACARONS
        order_depth = state.order_depths.get(product, None)
        
        if not order_depth or len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
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

    def cross_market_arbitrage(self, state: TradingState) -> tuple:
        orders = []
        conversions = 0
        product = Product.MAGNIFICENT_MACARONS
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
            
            if self.trader_data["position_clearing_started"] and position > 0:
                clearing_progress = min(1.0, (state.timestamp - self.start_clearing_timestamp) / 
                                  (self.must_clear_timestamp - self.start_clearing_timestamp))
                required_edge = effective_edge * max(0.5, 1.0 - clearing_progress)
            else:
                required_edge = effective_edge
                
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
            
            if self.trader_data["position_clearing_started"] and position < 0:
                clearing_progress = min(1.0, (state.timestamp - self.start_clearing_timestamp) / 
                                  (self.must_clear_timestamp - self.start_clearing_timestamp))
                required_edge = effective_edge * max(0.5, 1.0 - clearing_progress)
            else:
                required_edge = effective_edge
                
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
        product = Product.MAGNIFICENT_MACARONS
        orders = []
        position = state.position.get(product, 0)
        
        if state.timestamp < self.start_clearing_timestamp or position == 0:
            return orders
            
        order_depth = state.order_depths.get(product, OrderDepth())
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        clearing_progress = min(1.0, (state.timestamp - self.start_clearing_timestamp) / 
                               (self.must_clear_timestamp - self.start_clearing_timestamp))
        
        target_cleared_pct = clearing_progress
        
        aggression_factor = 1.0
        if clearing_progress > 0.8:
            aggression_factor = 3.0
        elif clearing_progress > 0.5:
            aggression_factor = 2.0
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        if position > 0:
            target_cleared = int(position * target_cleared_pct * aggression_factor)
            
            price_discount = max(0, int(clearing_progress * 2.0))
            target_price = max(1, best_bid - price_discount)
            
            selling_qty = min(position, target_cleared)
            if selling_qty > 0:
                orders.append(Order(product, target_price, -selling_qty))
        
        elif position < 0:
            target_cleared = int(abs(position) * target_cleared_pct * aggression_factor)
            
            price_premium = max(0, int(clearing_progress * 2.0))
            target_price = best_ask + price_premium
            
            buying_qty = min(abs(position), target_cleared)
            if buying_qty > 0:
                orders.append(Order(product, target_price, buying_qty))
                
        return orders

    def csi_driven_strategy(self, state: TradingState) -> List[Order]:
        product = Product.MAGNIFICENT_MACARONS
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
        mid_price = (best_bid + best_ask) / 2
        
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

    def run(self, state: TradingState) -> tuple:
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
                order_depth = state.order_depths.get(product, OrderDepth())
                
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

class ParabolaFitIVStrategy:
    def __init__(self, voucher: str, strike: int, adaptive: bool = False, absolute: bool = False):
        self.voucher = voucher
        self.strike = strike
        self.adaptive = adaptive
        self.absolute = absolute
        self.expiry_day = 5
        self.ticks_per_day = 1000
        self.window = 500
        self.position_limit = 200
        self.start_ts = None
        self.history = deque(maxlen=self.window)
        self.iv_cache = {}
        self.a = self.b = self.c = None

    def norm_cdf(self, x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    def bs_price(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0 or S <= 0:
            return max(S - K, 0)
        d1 = (log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * self.norm_cdf(d1) - K * self.norm_cdf(d2)

    def implied_vol(self, S, K, T, price, tol=1e-4, max_iter=50):
        key = (round(S, 1), round(K, 1), round(T, 5), round(price, 1))
        if key in self.iv_cache:
            return self.iv_cache[key]
        low, high = 1e-6, 5.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            val = self.bs_price(S, K, T, mid) - price
            if abs(val) < tol:
                self.iv_cache[key] = mid
                return mid
            if val > 0:
                high = mid
            else:
                low = mid
        return None

    def update_fit(self):
        m_vals = [m for m, v in self.history]
        v_vals = [v for m, v in self.history]
        self.a, self.b, self.c = np.polyfit(m_vals, v_vals, 2)

    def fitted_iv(self, m):
        return self.a * m ** 2 + self.b * m + self.c

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        ts = state.timestamp
        if self.start_ts is None:
            self.start_ts = ts

        depth = state.order_depths.get(self.voucher, OrderDepth())
        rock_depth = state.order_depths.get("VOLCANIC_ROCK", OrderDepth())
        if not depth.sell_orders or not depth.buy_orders:
            return {}, 0, ""
        if not rock_depth.sell_orders or not rock_depth.buy_orders:
            return {}, 0, ""

        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        rock_bid = max(rock_depth.buy_orders.keys())
        rock_ask = min(rock_depth.sell_orders.keys())
        spot_price = (rock_bid + rock_ask) / 2

        TTE = max(0.1, self.expiry_day - (ts - self.start_ts) / self.ticks_per_day)
        T = TTE / 365

        if self.absolute:
            iv_guess = 1.1
            fair_value = self.bs_price(spot_price, self.strike, T, iv_guess)
            mispricing = mid_price - fair_value

            position = state.position.get(self.voucher, 0)
            result = []

            if mispricing > 2 and position < self.position_limit:
                qty = min(20, self.position_limit - position)
                result.append(Order(self.voucher, best_ask, qty))
            elif mispricing < -2 and position > -self.position_limit:
                qty = min(20, self.position_limit + position)
                result.append(Order(self.voucher, best_bid, -qty))

            orders[self.voucher] = result
            return orders, 0, ""

        m_t = log(self.strike / spot_price) / sqrt(TTE)
        v_t = self.implied_vol(spot_price, self.strike, T, mid_price)
        if v_t is None or v_t < 0.5:
            return {}, 0, ""

        self.history.append((m_t, v_t))
        if len(self.history) < self.window:
            return {}, 0, ""

        self.update_fit()
        current_fit = self.fitted_iv(m_t)
        position = state.position.get(self.voucher, 0)
        result = []

        if v_t < current_fit - 0.019 and position < self.position_limit:
            qty = min(30, self.position_limit - position)
            result.append(Order(self.voucher, best_ask, qty))
        elif v_t > current_fit + 0.013 and position > -self.position_limit:
            qty = min(30, self.position_limit + position)
            result.append(Order(self.voucher, best_bid, -qty))

        conversions = 0
        orders[self.voucher] = result
        return orders, conversions, ""

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders = []

    def buy(self, price: float, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: float, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def run(self, state: TradingState) -> Tuple[List[Order], str]:
        self.orders = []
        self.act(state)
        return self.orders, ""

    def act(self, state: TradingState) -> None:
        pass

class VolcanicMarketMakingStrategy(Strategy):
    def _init_(self, symbol: Symbol, limit: int) -> None:
        super()._init_(symbol, limit)
        self.order_cap = 10
        self.min_spread = 0.5
        self.offset = 0.2
        self.max_loss = -10000

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            logger.print(f"No order depth for {self.symbol}")
            return
        depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        if not depth.buy_orders or not depth.sell_orders:
            logger.print(f"No valid bids/asks for {self.symbol}")
            return
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2.0
        logger.print(f"{self.symbol}: bid={best_bid}, ask={best_ask}, spread={spread}, position={position}")
        realized_pnl = sum(
            (trade.price - mid_price) * trade.quantity
            for trade in state.own_trades.get(self.symbol, [])
        )
        if realized_pnl < -10000:
            if position > 0:
                qty = min(position, depth.buy_orders.get(best_bid, 0), 10)
                if qty > 0:
                    self.orders.append(Order(self.symbol, best_bid, -qty))
                    logger.print(f"Liquidate {self.symbol}: SELL {qty} @ {best_bid}")
            elif position < 0:
                qty = min(-position, abs(depth.sell_orders.get(best_ask, 0)), 10)
                if qty > 0:
                    self.orders.append(Order(self.symbol, best_ask, qty))
                    logger.print(f"Liquidate {self.symbol}: BUY {qty} @ {best_ask}")
        else:
            if spread >= 0.5:
                if position < self.limit:
                    qty = min(10, self.limit - position, abs(depth.sell_orders.get(best_ask, 0)))
                    if qty > 0:
                        buy_price = int(best_bid + 0.2)
                        self.orders.append(Order(self.symbol, buy_price, qty))
                        logger.print(f"Market-make {self.symbol}: BUY {qty} @ {buy_price}")
                if position > -self.limit:
                    qty = min(10, self.limit + position, depth.buy_orders.get(best_bid, 0))
                    if qty > 0:
                        sell_price = int(best_ask - 0.2)
                        self.orders.append(Order(self.symbol, sell_price, -qty))
                        logger.print(f"Market-make {self.symbol}: SELL {qty} @ {sell_price}")

class Trader:
    def __init__(self):
        self.params = PARAMS
        self.position_limits = POSITION_LIMITS
        self.jams_mid_prices = deque(maxlen=self.params[Product.JAMS]["sma_long"])
        self.jams_sma_short = None
        self.jams_sma_long = None
        self.jams_last_mid_price = None
        self.jams_trend_history = deque(maxlen=100)
        self.kelp_state = {}
        self.basket2_raw_spreads = deque(maxlen=self.params[Product.PICNIC_BASKET2]["lookback_window"])

        self.strategy = [
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_10000", 10000),
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_9500", 9500, absolute=True),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_9750", 200),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_10250", 200),
            VolcanicMarketMakingStrategy("VOLCANIC_ROCK_VOUCHER_10500", 200),
        ]

        self.macaron_trader = MacaronTrader()

        self.position_limit={"VOLCANIC_ROCK":400}
        self.threshold = {"VOLCANIC_ROCK": 2.25}
        self.history = {"VOLCANIC_ROCK":[]}
        self.basket1_spread_history = []
        self.basket2_spread_history = []
        self.history["PICNIC_BASKET1"] = deque(maxlen=20)
        self.history["PICNIC_BASKET2"] = deque(maxlen=20)

    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None

    def calculate_volume_weighted_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        total_vol = 0
        total_val = 0
        for price, vol in order_depth.sell_orders.items():
            total_vol += abs(vol)
            total_val += price * abs(vol)
        for price, vol in order_depth.buy_orders.items():
            total_vol += abs(vol)
            total_val += price * abs(vol)
        if total_vol == 0:
            return None
        return total_val/total_vol

    def take_best_orders(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        position_limit: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amt = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amt, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amt = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amt, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        position_limit: int,
    ) -> (int, int):
        buy_qty = position_limit - (position + buy_order_volume)
        if buy_qty > 0 and bid is not None:
            orders.append(Order(product, bid, buy_qty))
        sell_qty = position_limit + (position - sell_order_volume)
        if sell_qty > 0 and ask is not None:
            orders.append(Order(product, ask, -sell_qty))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        position_limit: int,
    ) -> (int, int):
        pos_after = position + buy_order_volume - sell_order_volume
        bid_price = round(fair_value - width)
        ask_price = round(fair_value + width)
        buy_qty = position_limit - (position + buy_order_volume)
        sell_qty = position_limit + (position - sell_order_volume)
        if pos_after > 0:
            clear_qty = sum(volume for price, volume in order_depth.buy_orders.items() if price >= ask_price)
            clear_qty = min(clear_qty, pos_after)
            send_qty = min(sell_qty, clear_qty)
            if send_qty > 0:
                orders.append(Order(product, ask_price, -send_qty))
                sell_order_volume += send_qty
        if pos_after < 0:
            clear_qty = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= bid_price)
            clear_qty = min(clear_qty, abs(pos_after))
            send_qty = min(buy_qty, clear_qty)
            if send_qty > 0:
                orders.append(Order(product, bid_price, send_qty))
                buy_order_volume += send_qty
        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        position_limit: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume, position_limit, prevent_adverse, adverse_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        position_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume, position_limit
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        position_limit: int,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        asks = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        best_ask = min(asks) if asks else None
        best_bid = max(bids) if bids else None
        ask = round(fair_value + default_edge)
        if best_ask is not None:
            if abs(best_ask - fair_value) <= join_edge:
                ask = best_ask
            else:
                ask = best_ask - 1
        bid = round(fair_value - default_edge)
        if best_bid is not None:
            if abs(fair_value - best_bid) <= join_edge:
                bid = best_bid
            else:
                bid = best_bid + 1
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1
        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume, position_limit
        )
        return orders, buy_order_volume, sell_order_volume

    def compute_basket_implied_value(self, state: TradingState) -> Optional[float]:
        total = 0
        components = [
            (Product.CROISSANTS, PICNIC_BASKET1_WEIGHTS[Product.CROISSANTS]),
            (Product.JAMS, PICNIC_BASKET1_WEIGHTS[Product.JAMS]),
            (Product.DJEMBES, PICNIC_BASKET1_WEIGHTS[Product.DJEMBES])
        ]
        for prod, qty in components:
            if prod not in state.order_depths:
                return None
            mid = self.get_mid_price(state.order_depths[prod])
            if mid is None:
                return None
            total += mid * qty
        return total

    def rainforest_resin_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.RAINFOREST_RESIN) is not None:
            position = state.position.get(Product.RAINFOREST_RESIN, 0)
            order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            fair_value = self.params[Product.RAINFOREST_RESIN]["fair_value"]
            resin_take, bo, so = self.take_orders(
                Product.RAINFOREST_RESIN, order_depth, fair_value,
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                position, self.position_limits[Product.RAINFOREST_RESIN]
            )
            resin_clear, bo, so = self.clear_orders(
                Product.RAINFOREST_RESIN, order_depth, fair_value,
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                position, bo, so, self.position_limits[Product.RAINFOREST_RESIN]
            )
            resin_make, bo, so = self.make_orders(
                Product.RAINFOREST_RESIN, order_depth, fair_value, position, bo, so,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                self.position_limits[Product.RAINFOREST_RESIN], True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"]
            )
            orders = resin_take + resin_clear + resin_make
        return orders

    def kelp_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.KELP) is not None:
            position = state.position.get(Product.KELP, 0)
            order_depth = state.order_depths[Product.KELP]
            fair_value = self.kelp_fair_value(order_depth)
            if fair_value is None:
                logger.print("[KELP] No valid fair value, skipping trading.")
                return orders

            logger.print(f"[KELP] Fair Value: {fair_value:.2f}, Position: {position}")

            kelp_take, bo, so = self.take_orders(
                Product.KELP,
                order_depth,
                fair_value,
                self.params[Product.KELP]["take_width"],
                position,
                self.position_limits[Product.KELP],
                self.params[Product.KELP]["prevent_adverse"],
                self.params[Product.KELP]["adverse_volume"]
            )
            kelp_clear, bo, so = self.clear_orders(
                Product.KELP,
                order_depth,
                fair_value,
                self.params[Product.KELP]["clear_width"],
                position,
                bo,
                so,
                self.position_limits[Product.KELP]
            )
            kelp_make, bo, so = self.make_orders(
                Product.KELP,
                order_depth,
                fair_value,
                position,
                bo,
                so,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
                self.position_limits[Product.KELP]
            )
            orders = kelp_take + kelp_clear + kelp_make
        return orders
    
    def kelp_fair_value(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return self.kelp_state.get("last_fair_value", None)

        mid_price = self.calculate_volume_weighted_mid_price(order_depth)
        if mid_price is None:
            return self.kelp_state.get("last_fair_value", None)

        last_price = self.kelp_state.get("last_price", None)
        if last_price is not None:
            last_returns = (mid_price - last_price) / last_price
            pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
            fair_value = mid_price + (mid_price * pred_returns)
        else:
            fair_value = mid_price

        self.kelp_state["last_price"] = mid_price
        self.kelp_state["last_fair_value"] = fair_value
        return fair_value
    
    def ink_fair_value(self, order_depth: OrderDepth, traderObject):
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            valid_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
            valid_buy = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
            
            mn_ask = min(valid_ask) if len(valid_ask) > 0 else None
            mn_bid = max(valid_buy) if len(valid_buy) > 0 else None
            
            if valid_ask and valid_buy:
                mmid_price = (mn_ask + mn_bid) / 2
            
            else:
                mmid_price = (best_ask + best_bid) / 2 if traderObject.get("ink_last_price", None) is None else traderObject["ink_last_price"]
            
            if traderObject.get("ink_price_history", None) is None:
                traderObject["ink_price_history"] = []
            
            traderObject["ink_price_history"].append(mmid_price)
            
            if len(traderObject["ink_price_history"]) > self.params[Product.SQUID_INK]["reversion_window"]:
                traderObject["ink_price_history"] = traderObject["ink_price_history"][-self.params[Product.SQUID_INK]["reversion_window"]:]
                
            if len(traderObject["ink_price_history"]) >= self.params[Product.SQUID_INK]["reversion_window"]:
                prices = np.array(traderObject["ink_price_history"])
                returns = (prices[1:] - prices[:-1]) / prices[:-1]
                X = returns[:-1]
                Y = returns[1:]
                
                estimated_beta = -np.dot(X, Y) / np.dot(X, X) if np.dot(X, X) != 0 else self.params[Product.SQUID_INK]["reversion_beta"]
                adaptive_beta = (self.params[Product.SQUID_INK]["reversion_weight"] * estimated_beta +
                            (1 - self.params[Product.SQUID_INK]["reversion_weight"]) * self.params[Product.SQUID_INK]["reversion_beta"])
            
            else:
                adaptive_beta = self.params[Product.SQUID_INK]["reversion_beta"]
                
            fair = mmid_price if traderObject.get("ink_last_price", None) is None else mmid_price + (
                    mmid_price * ((mmid_price - traderObject["ink_last_price"]) / traderObject["ink_last_price"]) * adaptive_beta)
            
            traderObject["ink_last_price"] = mmid_price
            return fair
        
        return None

    def jams_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.JAMS) is not None:
            position = state.position.get(Product.JAMS, 0)
            order_depth = state.order_depths[Product.JAMS]
            mid_price = self.get_mid_price(order_depth)
            if mid_price is None:
                return []
            self.jams_mid_prices.append(mid_price)
            if len(self.jams_mid_prices) >= self.params[Product.JAMS]["sma_short"]:
                self.jams_sma_short = sum(list(self.jams_mid_prices)[-self.params[Product.JAMS]["sma_short"]:]) / self.params[Product.JAMS]["sma_short"]
            if len(self.jams_mid_prices) >= self.params[Product.JAMS]["sma_long"]:
                self.jams_sma_long = sum(self.jams_mid_prices) / len(self.jams_mid_prices)
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            if best_bid is None or best_ask is None:
                return []
            if self.jams_sma_short is None or self.jams_sma_long is None:
                return []
            trend = 0
            if self.jams_sma_short > self.jams_sma_long:
                trend = 1
            elif self.jams_sma_short < self.jams_sma_long:
                trend = -1
            self.jams_trend_history.append(trend)
            recent_trends = list(self.jams_trend_history)
            is_strongly_bearish = len(recent_trends) >= 50 and all(t == -1 for t in recent_trends[-50:])
            if position > 0 and trend == -1 and best_bid is not None:
                if self.jams_last_mid_price is not None and mid_price < self.jams_last_mid_price - self.params[Product.JAMS]["clear_threshold"]:
                    sell_volume = -position
                    orders.append(Order(Product.JAMS, best_bid, sell_volume))
            if position < 0 and trend == 1 and best_ask is not None:
                if self.jams_last_mid_price is not None and mid_price > self.jams_last_mid_price + self.params[Product.JAMS]["clear_threshold"]:
                    buy_volume = -position
                    orders.append(Order(Product.JAMS, best_ask, buy_volume))
            if is_strongly_bearish and position > self.params[Product.JAMS]["target_position_bearish"] and best_bid is not None:
                sell_volume = position - self.params[Product.JAMS]["target_position_bearish"]
                orders.append(Order(Product.JAMS, best_bid, -sell_volume))
            buy_price = round(mid_price - self.params[Product.JAMS]["min_edge"])
            sell_price = round(mid_price + self.params[Product.JAMS]["min_edge"])
            buy_volume = min(self.params[Product.JAMS]["order_volume"], self.params[Product.JAMS]["volume_limit"] - position)
            sell_volume = min(self.params[Product.JAMS]["order_volume"], self.params[Product.JAMS]["volume_limit"] + position)
            if trend == -1:
                buy_volume = int(buy_volume * self.params[Product.JAMS]["buy_volume_reduction_bearish"])
                sell_volume = int(sell_volume * self.params[Product.JAMS]["sell_volume_increase_bearish"])
            elif trend == 1:
                buy_volume = int(buy_volume * 1.5)
                sell_volume = int(sell_volume * 0.5)
            if trend == -1 and position > 0:
                buy_volume = 0
            if buy_volume > 0:
                orders.append(Order(Product.JAMS, buy_price, buy_volume))
            if sell_volume > 0:
                orders.append(Order(Product.JAMS, sell_price, -sell_volume))
            self.jams_last_mid_price = mid_price
        return orders

    def croissants_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.CROISSANTS) is not None:
            position = state.position.get(Product.CROISSANTS, 0)
            order_depth = state.order_depths[Product.CROISSANTS]
            fair_value = self.params[Product.CROISSANTS]["fair_value"]
            croissants_take, bo, so = self.take_orders(
                Product.CROISSANTS, order_depth, fair_value,
                self.params[Product.CROISSANTS]["take_width"],
                position, self.position_limits[Product.CROISSANTS]
            )
            croissants_clear, bo, so = self.clear_orders(
                Product.CROISSANTS, order_depth, fair_value,
                self.params[Product.CROISSANTS]["clear_width"],
                position, bo, so, self.position_limits[Product.CROISSANTS]
            )
            croissants_make, bo, so = self.make_orders(
                Product.CROISSANTS, order_depth, fair_value, position, bo, so,
                self.params[Product.CROISSANTS]["disregard_edge"],
                self.params[Product.CROISSANTS]["join_edge"],
                self.params[Product.CROISSANTS]["default_edge"],
                self.position_limits[Product.CROISSANTS], True,
                self.params[Product.CROISSANTS]["soft_position_limit"]
            )
            orders = croissants_take + croissants_clear + croissants_make
        return orders

    def djembes_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        if state.order_depths.get(Product.DJEMBES) is not None:
            position = state.position.get(Product.DJEMBES, 0)
            order_depth = state.order_depths[Product.DJEMBES]
            fair_value = self.params[Product.DJEMBES]["fair_value"]
            djembes_take, bo, so = self.take_orders(
                Product.DJEMBES, order_depth, fair_value,
                self.params[Product.DJEMBES]["take_width"],
                position, self.position_limits[Product.DJEMBES]
            )
            djembes_clear, bo, so = self.clear_orders(
                Product.DJEMBES, order_depth, fair_value,
                self.params[Product.DJEMBES]["clear_width"],
                position, bo, so, self.position_limits[Product.DJEMBES]
            )
            djembes_make, bo, so = self.make_orders(
                Product.DJEMBES, order_depth, fair_value, position, bo, so,
                self.params[Product.DJEMBES]["disregard_edge"],
                self.params[Product.DJEMBES]["join_edge"],
                self.params[Product.DJEMBES]["default_edge"],
                self.position_limits[Product.DJEMBES], True,
                self.params[Product.DJEMBES]["soft_position_limit"]
            )
            orders = djembes_take + djembes_clear + djembes_make
        return orders

    def picnic_basket1_strategy(self, state: TradingState) -> List[Order]:
        orders = []
        products_for_pricing = [Product.PICNIC_BASKET1, Product.CROISSANTS, Product.JAMS, Product.DJEMBES]
        all_prices_available = True
        order_depths = {}
        best_bids = {}
        best_asks = {}
        for prod in products_for_pricing:
            if prod not in state.order_depths:
                all_prices_available = False
                break
            depth = state.order_depths[prod]
            order_depths[prod] = depth
            bid = get_best_bid(depth)
            ask = get_best_ask(depth)
            best_bids[prod] = bid
            best_asks[prod] = ask
            if bid is None or ask is None:
                all_prices_available = False
                break
        if not all_prices_available:
            return orders
        basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
        cost_to_buy_components = sum(best_asks[comp] * PICNIC_BASKET1_WEIGHTS[comp] for comp in PICNIC_BASKET1_WEIGHTS)
        revenue_from_selling_components = sum(best_bids[comp] * PICNIC_BASKET1_WEIGHTS[comp] for comp in PICNIC_BASKET1_WEIGHTS)
        sell_basket_profit = best_bids[Product.PICNIC_BASKET1] - cost_to_buy_components
        buy_basket_profit = revenue_from_selling_components - best_asks[Product.PICNIC_BASKET1]
        ENTRY_THRESHOLD = 46
        if sell_basket_profit > ENTRY_THRESHOLD:
            max_size_by_limit = self.position_limits[Product.PICNIC_BASKET1] + basket_position
            max_size_by_volume = get_volume_at_price(order_depths[Product.PICNIC_BASKET1], "buy", best_bids[Product.PICNIC_BASKET1])
            max_size_by_limit = max(0, max_size_by_limit)
            max_size_by_volume = max(0, max_size_by_volume)
            trade_size = min(max_size_by_limit, max_size_by_volume)
            if trade_size > 0:
                orders.append(Order(Product.PICNIC_BASKET1, best_bids[Product.PICNIC_BASKET1], -trade_size))
        elif buy_basket_profit > ENTRY_THRESHOLD:
            max_size_by_limit = self.position_limits[Product.PICNIC_BASKET1] - basket_position
            max_size_by_volume = get_volume_at_price(order_depths[Product.PICNIC_BASKET1], "sell", best_asks[Product.PICNIC_BASKET1])
            max_size_by_limit = max(0, max_size_by_limit)
            max_size_by_volume = max(0, max_size_by_volume)
            trade_size = min(max_size_by_limit, max_size_by_volume)
            if trade_size > 0:
                orders.append(Order(Product.PICNIC_BASKET1, best_asks[Product.PICNIC_BASKET1], trade_size))
        return orders

    def picnic_basket2_strategy(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        products_for_pricing = [Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS]
        components = {Product.CROISSANTS: 4, Product.JAMS: 2}
        all_prices_available = True
        order_depths = {}
        best_bids = {}
        best_asks = {}
        mid_prices = {}
        for product in products_for_pricing:
            if product not in state.order_depths:
                all_prices_available = False
                break
            depth = state.order_depths[product]
            order_depths[product] = depth
            best_bids[product] = get_best_bid(depth)
            best_asks[product] = get_best_ask(depth)
            mid_prices[product] = self.get_mid_price(depth)
            if best_bids[product] is None or best_asks[product] is None or mid_prices[product] is None:
                all_prices_available = False
                break
        if not all_prices_available:
            return orders
        basket_position = state.position.get(Product.PICNIC_BASKET2, 0)
        cost_to_buy_components = sum(best_asks[comp] * ratio for comp, ratio in components.items())
        revenue_from_selling_components = sum(best_bids[comp] * ratio for comp, ratio in components.items())
        sell_basket_profit = best_bids[Product.PICNIC_BASKET2] - cost_to_buy_components
        buy_basket_profit = revenue_from_selling_components - best_asks[Product.PICNIC_BASKET2]
        components_mid_value = sum(mid_prices[comp] * ratio for comp, ratio in components.items())
        raw_spread = mid_prices[Product.PICNIC_BASKET2] - components_mid_value
        self.basket2_raw_spreads.append(raw_spread)
        params = self.params[Product.PICNIC_BASKET2]
        current_entry_threshold = params["default_entry"]
        current_exit_threshold = params["default_exit"]
        if len(self.basket2_raw_spreads) >= params["lookback_window"] // 2:
            try:
                mean_raw_spread = statistics.mean(self.basket2_raw_spreads)
                std_dev_raw_spread = statistics.stdev(self.basket2_raw_spreads)
                std_dev_raw_spread = max(params["min_std_dev"], std_dev_raw_spread)
                potential_exit = abs(mean_raw_spread) + params["k_exit"] * std_dev_raw_spread
                potential_entry = abs(mean_raw_spread) + params["k_entry"] * std_dev_raw_spread
                current_entry_threshold = max(params["min_entry_threshold"], min(potential_entry, params["max_entry_threshold"]))
                current_exit_threshold = max(params["min_exit_threshold"], min(potential_exit, params["max_exit_threshold"]))
                current_entry_threshold = max(current_entry_threshold, current_exit_threshold + 1)
            except statistics.StatisticsError:
                pass
        exit_order_placed = False
        position_limit = self.position_limits[Product.PICNIC_BASKET2]
        if basket_position > 0 and buy_basket_profit < current_exit_threshold:
            exit_size = basket_position
            volume_available = get_volume_at_price(order_depths[Product.PICNIC_BASKET2], "buy", best_bids[Product.PICNIC_BASKET2])
            exit_quantity = min(exit_size, volume_available)
            if exit_quantity > 0:
                orders.append(Order(Product.PICNIC_BASKET2, best_bids[Product.PICNIC_BASKET2], -exit_quantity))
                exit_order_placed = True
        elif basket_position < 0 and sell_basket_profit < current_exit_threshold:
            exit_size = abs(basket_position)
            volume_available = get_volume_at_price(order_depths[Product.PICNIC_BASKET2], "sell", best_asks[Product.PICNIC_BASKET2])
            exit_quantity = min(exit_size, volume_available)
            if exit_quantity > 0:
                orders.append(Order(Product.PICNIC_BASKET2, best_asks[Product.PICNIC_BASKET2], exit_quantity))
                exit_order_placed = True
        if not exit_order_placed:
            if sell_basket_profit > current_entry_threshold:
                max_size_by_limit = position_limit + basket_position
                max_size_by_volume = get_volume_at_price(order_depths[Product.PICNIC_BASKET2], "buy", best_bids[Product.PICNIC_BASKET2])
                trade_size = min(max(0, max_size_by_limit), max(0, max_size_by_volume))
                if trade_size > 0:
                    orders.append(Order(Product.PICNIC_BASKET2, best_bids[Product.PICNIC_BASKET2], -trade_size))
            elif buy_basket_profit > current_entry_threshold:
                max_size_by_limit = position_limit - basket_position
                max_size_by_volume = get_volume_at_price(order_depths[Product.PICNIC_BASKET2], "sell", best_asks[Product.PICNIC_BASKET2])
                trade_size = min(max(0, max_size_by_limit), max(0, max_size_by_volume))
                if trade_size > 0:
                    orders.append(Order(Product.PICNIC_BASKET2, best_asks[Product.PICNIC_BASKET2], trade_size))
        return orders

    def update_volcanic_rock_history(self, state: TradingState) -> List[Order]:
        orders = []
        rock_depth = state.order_depths.get("VOLCANIC_ROCK")
        if rock_depth and rock_depth.buy_orders and rock_depth.sell_orders:
            rock_bid = max(rock_depth.buy_orders)
            rock_ask = min(rock_depth.sell_orders)
            rock_mid = (rock_bid + rock_ask) / 2
            self.history["VOLCANIC_ROCK"].append(rock_mid)

        rock_prices = np.array(self.history["VOLCANIC_ROCK"])

        if len(rock_prices) >= 50:
            recent = rock_prices[-50:]
            mean = np.mean(recent)
            std = np.std(recent)
            self.z = (rock_prices[-1] - mean) / std if std > 0 else 0
            
            threshold = self.threshold["VOLCANIC_ROCK"]
            position = state.position.get("VOLCANIC_ROCK", 0)
            position_limit = self.position_limit["VOLCANIC_ROCK"]
            product = "VOLCANIC_ROCK"

            # Z-score low → buy (expecting VOLCANIC_ROCK to rebound)
            if self.z < -threshold and rock_depth.sell_orders:
                best_ask = min(rock_depth.sell_orders)
                qty = -rock_depth.sell_orders[best_ask]
                buy_qty = min(qty, position_limit - position)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
            # Z-score high → sell (expecting VOLCANIC_ROCK to fall)
            elif self.z > threshold and rock_depth.buy_orders:
                best_bid = max(rock_depth.buy_orders)
                qty = rock_depth.buy_orders[best_bid]
                sell_qty = min(qty, position + position_limit)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
        
        return orders

    def run(self, state: TradingState):
        result = {symbol: [] for symbol in state.listings.keys()} 
        trader_state = {}
        if state.traderData not in [None, ""]:
            trader_state = jsonpickle.decode(state.traderData)
        self.kelp_state = trader_state.get("kelp_state", {})
        self.jams_mid_prices = trader_state.get("jams_mid_prices", deque(maxlen=self.params[Product.JAMS]["sma_long"]))
        self.jams_sma_short = trader_state.get("jams_sma_short", None)
        self.jams_sma_long = trader_state.get("jams_sma_long", None)
        self.jams_last_mid_price = trader_state.get("jams_last_mid_price", None)
        self.jams_trend_history = trader_state.get("jams_trend_history", deque(maxlen=100))
        self.basket2_raw_spreads = trader_state.get("basket2_raw_spreads", deque(maxlen=self.params[Product.PICNIC_BASKET2]["lookback_window"]))
        strategies = [
            (Product.RAINFOREST_RESIN, self.rainforest_resin_strategy),
            (Product.KELP, self.kelp_strategy),
            (Product.JAMS, self.jams_strategy),
            #(Product.CROISSANTS, self.croissants_strategy),
            (Product.DJEMBES, self.djembes_strategy),
            (Product.PICNIC_BASKET1, self.picnic_basket1_strategy),
            (Product.PICNIC_BASKET2, self.picnic_basket2_strategy),
        ]
        for product, strategy_func in strategies:
            if product in state.order_depths:
                orders = strategy_func(state)
                if orders:
                    result[product] = orders

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            ink_position = state.position.get(Product.SQUID_INK, 0)
            ink_od_copy = OrderDepth()
            ink_od_copy.buy_orders = state.order_depths[Product.SQUID_INK].buy_orders.copy()
            ink_od_copy.sell_orders = state.order_depths[Product.SQUID_INK].sell_orders.copy()
            ink_fair_value = self.ink_fair_value(ink_od_copy, trader_state)
            if ink_fair_value is not None:
                ink_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.SQUID_INK, ink_od_copy, ink_fair_value,
                    self.params[Product.SQUID_INK]['take_width'], ink_position, self.position_limits[Product.SQUID_INK],
                    self.params[Product.SQUID_INK]['prevent_adverse'], self.params[Product.SQUID_INK]['adverse_volume']
                )
                ink_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.SQUID_INK, ink_od_copy, ink_fair_value,
                    self.params[Product.SQUID_INK]['clear_width'], ink_position, buy_order_volume, sell_order_volume,
                    self.position_limits[Product.SQUID_INK]
                )
                ink_make_orders, _, _ = self.make_orders(
                    Product.SQUID_INK, state.order_depths[Product.SQUID_INK], ink_fair_value, ink_position,
                    buy_order_volume, sell_order_volume, self.params[Product.SQUID_INK]['disregard_edge'],
                    self.params[Product.SQUID_INK]['join_edge'], self.params[Product.SQUID_INK]['default_edge'], self.position_limits[Product.SQUID_INK]
                )
                result[Product.SQUID_INK] = (ink_take_orders + ink_clear_orders + ink_make_orders)

        result["VOLCANIC_ROCK"]=self.update_volcanic_rock_history(state)
        if Product.VOLCANIC_ROCK in state.order_depths:
            for strategy in self.strategy:
                if isinstance(strategy, ParabolaFitIVStrategy):
                        orders, _, _ = strategy.run(state)
                        for symbol, order_list in orders.items():
                            result[symbol].extend(order_list)
                elif isinstance(strategy, VolcanicMarketMakingStrategy):
                    orders, _ = strategy.run(state)
                    result[strategy.symbol].extend(orders)

        trader_state["kelp_state"] = self.kelp_state
        trader_state["jams_mid_prices"] = self.jams_mid_prices
        trader_state["jams_sma_short"] = self.jams_sma_short
        trader_state["jams_sma_long"] = self.jams_sma_long
        trader_state["jams_last_mid_price"] = self.jams_last_mid_price
        trader_state["jams_trend_history"] = self.jams_trend_history
        trader_state["basket2_raw_spreads"] = self.basket2_raw_spreads

        macaron_conversions = 0
        try:
            # Call the macaron trader and catch any errors
            macaron_result, macaron_conversions, macaron_trader_state_encoded = self.macaron_trader.run(state)
            
            # Check if we got any orders back
            if macaron_result and Product.MAGNIFICENT_MACARONS in macaron_result:
                # If the result is a nested dictionary, we need to extract the actual orders
                if isinstance(macaron_result[Product.MAGNIFICENT_MACARONS], dict):
                    inner_orders = macaron_result[Product.MAGNIFICENT_MACARONS].get(Product.MAGNIFICENT_MACARONS, [])
                    # Convert tuple format to Order objects if needed
                    if inner_orders and isinstance(inner_orders[0], tuple):
                        result[Product.MAGNIFICENT_MACARONS] = [
                            Order(item[0], item[1], item[2]) for item in inner_orders
                        ]
                    else:
                        result[Product.MAGNIFICENT_MACARONS] = inner_orders
                else:
                    # If it's already a list of orders, use it directly
                    result[Product.MAGNIFICENT_MACARONS] = macaron_result[Product.MAGNIFICENT_MACARONS]
            
            # Decode the encoded trader state if it's a string
            if isinstance(macaron_trader_state_encoded, str):
                try:
                    macaron_trader_state = jsonpickle.decode(macaron_trader_state_encoded)
                    if isinstance(macaron_trader_state, dict):
                        for key, value in macaron_trader_state.items():
                            trader_state[key] = value
                except Exception as e:
                    logger.print(f"Error decoding MacaronTrader state: {e}")
        except Exception as e:
            # Handle the exception and optionally log it
            logger.print(f"Error in MacaronTrader: {e}")

        # Update conversions variable - use macaron conversions if available
        conversions = macaron_conversions


        traderData = jsonpickle.encode(trader_state)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

if __name__ == "__main__":
    trader = Trader()