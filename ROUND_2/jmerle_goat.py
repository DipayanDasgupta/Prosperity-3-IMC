import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias
import statistics


JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

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

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

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
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)

class InkMeanReversionTrader(Strategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.prices = deque(maxlen=300)
        self.window = 300

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return

        # --- Get Best Bid/Ask ---
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

        # === BUY ZONE ===

        if mid_price < mean - 1 * std_dev and position < self.limit:
            # 💧 Passive buy at best ask
            self.buy(best_bid, self.limit - position)

        elif mid_price > mean + 1 * std_dev and position > -self.limit:
            # 💧 Passive sell at best bid
            self.sell(best_ask, self.limit + position)

        # === EXIT ZONE ===
        elif abs(mid_price - mean) <= 0.25:
            if position > 0:
                self.sell(best_bid, position)
            elif position < 0:
                self.buy(best_ask, -position)

class Basket1Trader(Strategy):

    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
    
    def act(self, state: TradingState) -> None:
        # Ensure the order depth information is available for the basket and all components.
        required_symbols = ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]
        for sym in required_symbols:
            if sym not in state.order_depths:
                return
        
        # Get order depths for the basket and for each component.
        basket_od = state.order_depths["PICNIC_BASKET1"]
        croissants_od = state.order_depths["CROISSANTS"]
        jams_od = state.order_depths["JAMS"]
        djembe_od = state.order_depths["DJEMBES"]
        
        # Compute basket mid-price (average of best bid and best ask).
        basket_bids = sorted(basket_od.buy_orders.items(), reverse=True)
        basket_asks = sorted(basket_od.sell_orders.items())
        if not basket_bids or not basket_asks:
            return
        basket_best_bid = basket_bids[0][0]
        basket_best_ask = basket_asks[0][0]
        basket_mid = (basket_best_bid + basket_best_ask) / 2
        
        # Helper function: calculate mid-price for a component.
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
        
        # Composite value of the underlying components.
        composite_value = 6 * mid_croissants + 3 * mid_jams + mid_djembe
        
        # Calculate the fair value of the basket by adding the fixed mean spread of 50.
        fair_value = composite_value
        
        # Determine the deviation (spread difference).
        deviation = basket_mid - fair_value
        
        # Get current position for PICNIC_BASKET1.
        pos = state.position.get("PICNIC_BASKET1", 0)
        
        # Define a threshold below which we consider the deviation significant.
        threshold = 1  # Adjust this value as needed.
        
        # --- Trading logic based on the deviation ---
        if deviation < -threshold and pos < self.limit:
            # Basket is undervalued -> place a buy order.
            # Calculate order quantity (trade in chunks, e.g., up to 10 units).
            quantity = self.limit - pos
            self.buy(basket_best_bid, quantity)
        elif deviation > threshold and pos > -self.limit:
            # Basket is overvalued -> place a sell order.
            # Use available capacity on the sell side (note: pos is positive for a net long).
            quantity = self.limit + pos
            self.sell(basket_best_ask, quantity)
        elif abs(deviation) < threshold and pos != 0:
            # If price is near fair value, exit any open position.
            if pos > 0:
                self.sell(basket_best_ask, pos)
            elif pos < 0:
                self.buy(basket_best_bid, -pos)
    
    def save(self) -> JSON:
        # Optionally, you can implement state saving logic if needed.
        return None

    def load(self, data: JSON) -> None:
        # Optionally, implement loading of saved state if required.
        pass

class Basket2Trader(Strategy):

    def __init__(self, symbol: str, limit: int):
        # For this trader, symbol should be "PICNIC_BASKET2".
        super().__init__(symbol, limit)
    
    def act(self, state: TradingState) -> None:
        # Ensure order depth data is available for the basket and its components.
        required_symbols = ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]
        for sym in required_symbols:
            if sym not in state.order_depths:
                return
        
        # Retrieve order depths.
        basket_od = state.order_depths["PICNIC_BASKET2"]
        croissants_od = state.order_depths["CROISSANTS"]
        jams_od = state.order_depths["JAMS"]
        
        # Compute basket mid-price (average of best bid and best ask).
        basket_bids = sorted(basket_od.buy_orders.items(), reverse=True)
        basket_asks = sorted(basket_od.sell_orders.items())
        if not basket_bids or not basket_asks:
            return
        basket_best_bid = basket_bids[0][0]
        basket_best_ask = basket_asks[0][0]
        basket_mid = (basket_best_bid + basket_best_ask) / 2
        
        # Helper function: calculate mid-price for a component.
        def get_mid(order_depth):
            bids = sorted(order_depth.buy_orders.items(), reverse=True)
            asks = sorted(order_depth.sell_orders.items())
            if not bids or not asks:
                return None
            return (bids[0][0] + asks[0][0]) / 2
        
        mid_croissants = get_mid(croissants_od)
        mid_jams = get_mid(jams_od)
        if mid_croissants is None or mid_jams is None:
            return
        
        # Composite value of the underlying components.
        composite_value = 4 * mid_croissants + 2 * mid_jams
        
        # Calculate the basket's theoretical fair value.
        fair_value = composite_value - 12
        
        # Determine the deviation.
        deviation = basket_mid - fair_value
        
        # Retrieve current position for PICNIC_BASKET2.
        pos = state.position.get("PICNIC_BASKET2", 0)
        
        # Define a threshold for deviation significance.
        threshold = 1  # Adjust as needed.
        
        # --- Trading logic based on the deviation ---
        if deviation < -threshold and pos < self.limit:
            # Basket is undervalued -> place a buy order.
            quantity = min(self.limit - pos, 10)  # Trade in chunks up to 10 units.
            self.buy(basket_best_bid, quantity)
        elif deviation > threshold and pos > -self.limit:
            # Basket is overvalued -> place a sell order.
            quantity = min(self.limit + pos, 10)
            self.sell(basket_best_ask, quantity)
        elif abs(deviation) < threshold and pos != 0:
            # If the price is near fair value, exit any open position.
            if pos > 0:
                self.sell(basket_best_ask, pos)
            elif pos < 0:
                self.buy(basket_best_bid, -pos)
    
    def save(self) -> dict:
        return None

    def load(self, data: dict) -> None:
        pass

class JamMarketMaker(Strategy):
    """
    A market making strategy for JAMS.
    
    Overview:
      - It calculates the mid-price from the best bid and ask in JAMS' order book.
      - It then sets:
            buy_price  = mid_price - delta
            sell_price = mid_price + delta
      - Orders are placed using a fixed base order size, subject to remaining capacity
        given the position limit.
      
    Parameters:
      - delta: The fixed price offset from the mid-price for quoting orders.
      - base_order_size: The number of units to trade per order (subject to limit).
      - limit: The maximum absolute position allowed.
    """
    def __init__(self, symbol: str, limit: int):
        # symbol must be "JAMS" when instantiating this strategy.
        super().__init__(symbol, limit)
        self.base_order_size = 10  # Adjust this value as needed.
        self.delta = 1             # Price offset from the mid-price.

    def act(self, state: TradingState) -> None:
        # Verify that order depth data for JAMS is available.
        if self.symbol not in state.order_depths:
            return
        
        order_depth = state.order_depths[self.symbol]
        
        # Retrieve best bid and ask orders.
        bids = sorted(order_depth.buy_orders.items(), reverse=True)
        asks = sorted(order_depth.sell_orders.items())
        if not bids or not asks:
            return

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        
        # Calculate the mid-price as the average of the best bid and best ask.
        mid_price = (best_bid + best_ask) / 2
        
        # Determine the quoting prices using the fixed offset (delta).
        buy_price = mid_price - self.delta
        sell_price = mid_price + self.delta
        
        # Get the current position for JAMS.
        pos = state.position.get(self.symbol, 0)
        
        # Compute the remaining capacity for new orders:
        # For a buy order, new position = pos + quantity must be <= limit.
        allowed_buy = self.limit - pos
        
        # For a sell order, new position = pos - quantity must be >= -limit.
        allowed_sell = self.limit + pos
        
        # Determine the order size (trade in chunks up to base_order_size) without breaching limits.
        order_size_buy = min(self.base_order_size, allowed_buy) if allowed_buy > 0 else 0
        order_size_sell = min(self.base_order_size, allowed_sell) if allowed_sell > 0 else 0
        
        # Place a buy order if capacity exists.
        if order_size_buy > 0:
            self.buy(int(buy_price), order_size_buy)
        
        # Place a sell order if capacity exists.
        if order_size_sell > 0:
            self.sell(int(sell_price), order_size_sell)

    def save(self) -> dict:
        # This method can be used to save any internal state if needed.
        return None

    def load(self, data: dict) -> None:
        # This method can be used to load a previously saved state if needed.
        pass


class CroissantMarketMaker(Strategy):
    """
    A self-contained market-making strategy for CROISSANTS.
    
    Overview:
      - Computes the mid-price from the best bid and ask.
      - Places a buy order at (mid_price - delta) and a sell order at (mid_price + delta).
      - Adjusts order sizes based on current position to remain within risk limits.
    
    Tweakable Parameters:
      - delta: Price offset from the mid-price for quoting orders.
      - base_order_size: Maximum number of units traded per order.
      - limit: Position limit for CROISSANTS.
    """
    
    def __init__(self, symbol: str, limit: int):
        # The symbol should be "CROISSANTS" for this strategy.
        super().__init__(symbol, limit)
        self.base_order_size = 25  # Adjust the base order size as needed.
        self.delta = 1# Fixed price offset from mid-price.

    def act(self, state: TradingState) -> None:
        # Ensure that the order depth data for CROISSANTS is available.
        if self.symbol not in state.order_depths:
            return
        
        order_depth = state.order_depths[self.symbol]
        
        # Get best bid and ask orders.
        bids = sorted(order_depth.buy_orders.items(), reverse=True)
        asks = sorted(order_depth.sell_orders.items())
        if not bids or not asks:
            return
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        
        # Compute mid-price.
        mid_price = (best_bid + best_ask) / 2
        
        # Determine our quoting prices.
        buy_price = mid_price - self.delta
        sell_price = mid_price + self.delta
        
        # Get current CROISSANTS position.
        pos = state.position.get(self.symbol, 0)
        
        # Determine how many units we can buy/sell without breaching the position limits.
        # Assumption: Limit is symmetrical, e.g. valid positions range from -limit to +limit.
        allowed_buy = self.limit - pos      # How many more we can buy.
        allowed_sell = self.limit + pos      # How many more we can sell if pos is negative.
        
        # Calculate order sizes in chunks, while ensuring orders don't exceed allowed capacity.
        order_size_buy = min(self.base_order_size, allowed_buy) if allowed_buy > 0 else 0
        order_size_sell = min(self.base_order_size, allowed_sell) if allowed_sell > 0 else 0
        
        # Place a buy order if there's capacity to increase the position.
        if order_size_buy > 0:
            self.buy(int(buy_price), order_size_buy)
        
        # Place a sell order if there's capacity to reduce the position.
        if order_size_sell > 0:
            self.sell(int(sell_price), order_size_sell)

    def save(self) -> dict:
        # Save any internal state if needed (not used in this simple strategy).
        return None

    def load(self, data: dict) -> None:
        # Load any saved state if needed.
        pass


class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)

class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "PICNIC_BASKET1" :60,
            "PICNIC_BASKET2" :100,
            "CROISSANTS" :250,
            "JAMS" :350,
        }

        self.strategies = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": InkMeanReversionTrader,
            "PICNIC_BASKET1" :Basket1Trader,
            "PICNIC_BASKET2" :Basket2Trader,
            "CROISSANTS" :CroissantMarketMaker,
            "JAMS" :JamMarketMaker, 
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
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
