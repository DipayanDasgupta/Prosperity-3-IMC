import json
import math
import collections
import statistics
from typing import Any, List, Dict, Tuple

from datamodel import (
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    TradingState,
)

# =============================================================================
# Logger Class (unchanged)
# =============================================================================
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[str, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        payload = [
            self._compress_state(state, ""),
            self._compress_orders(orders),
            conversions,
            "",
            "",
        ]
        base_len = len(json.dumps(payload, cls=ProsperityEncoder, separators=(",",":")))
        max_item = (self.max_log_length - base_len) // 3
        output = [
            self._compress_state(state, self._truncate(state.traderData, max_item)),
            self._compress_orders(orders),
            conversions,
            self._truncate(trader_data, max_item),
            self._truncate(self.logs, max_item),
        ]
        print(json.dumps(output, cls=ProsperityEncoder, separators=(",",":")))
        self.logs = ""

    def _compress_state(self, state: TradingState, td: str) -> Any:
        return [
            state.timestamp,
            td,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {sym: [od.buy_orders, od.sell_orders] for sym, od in state.order_depths.items()},
            [],  # own_trades omitted
            [],  # market_trades omitted
            state.position,
            self._compress_observations(state.observations),
        ]

    def _compress_observations(self, obs: Observation) -> Any:
        if not obs:
            return [[], {}]
        conv = {
            p: [
                o.bidPrice,
                o.askPrice,
                o.transportFees,
                o.exportTariff,
                o.importTariff,
                o.sugarPrice,
                o.sunlightIndex,
            ]
            for p, o in obs.conversionObservations.items()
        }
        return [obs.plainValueObservations, conv]

    def _compress_orders(self, orders: Dict[str, List[Order]]) -> Any:
        return [[o.symbol, o.price, o.quantity] for lst in orders.values() for o in lst]

    def _truncate(self, s: str, max_len: int) -> str:
        lo, hi, best = 0, min(len(s), max_len), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = s[:mid] + ("..." if mid < len(s) else "")
            if len(json.dumps(cand)) <= max_len:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best

logger = Logger()

# =============================================================================
# Constants & Params
# =============================================================================
PRODUCT = "MAGNIFICENT_MACARONS"

PARAMS = {
    PRODUCT: {
        "base_value": 50,
        "factor_sugar": 0.1,
        "factor_sunlight": 0.2,
        "factor_tariff": 1.0,
        "factor_transport": 1.0,
        "storage_cost": 0.1,
        "soft_position_limit": 75,
        "conversion_limit": 10,
        "default_edge": 1,
        "conv_buy_threshold": 0.97,
        "conv_sell_threshold": 1.03,
        # add or tune other parameters here...
    }
}

# =============================================================================
# Helper
# =============================================================================
def dynamic_order_qty(desired: int, pos: int, limit: int) -> int:
    avail = limit - abs(pos)
    if avail <= 0:
        return 0
    scaled = int(desired * avail / limit)
    return max(1, min(avail, scaled))

# =============================================================================
# OrderManager (stubs – plug in your aggressive/clear logic)
# =============================================================================
class OrderManager:
    def __init__(self, limit: int) -> None:
        self.limit = limit

    def take_orders(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        width: float,
        pos: int,
    ) -> Tuple[List[Order], int, int]:
        # your aggressive‐entry logic here
        return [], 0, 0

    def clear_orders(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        width: float,
        pos: int,
        bo: int,
        so: int,
    ) -> Tuple[List[Order], int, int]:
        # your risk‐off exit logic here
        return [], bo, so

# =============================================================================
# Trader
# =============================================================================
class Trader:
    def __init__(self) -> None:
        cfg = PARAMS[PRODUCT]
        self.limit = cfg["soft_position_limit"]
        self.om = OrderManager(self.limit)

    def macarons_strategy(self, state: TradingState) -> List[Order]:
        depth = state.order_depths.get(PRODUCT)
        if not depth:
            return []
        pos = state.position.get(PRODUCT, 0)
        cfg = PARAMS[PRODUCT]

        # --- Fair value calc ---
        fair = cfg["base_value"]
        obs = state.observations.conversionObservations.get(PRODUCT) if state.observations else None
        if obs:
            mid = (obs.bidPrice + obs.askPrice) / 2
            fair += mid
            fair += obs.sugarPrice * cfg["factor_sugar"]
            fair += obs.sunlightIndex * cfg["factor_sunlight"]
            fair += (obs.exportTariff - obs.importTariff) * cfg["factor_tariff"]
            fair -= obs.transportFees * cfg["factor_transport"]
        if pos > 0:
            fair -= cfg["storage_cost"] * pos

        orders: List[Order] = []
        bo = so = 0

        # 1) Aggressive
        agg, bo, so = self.om.take_orders(PRODUCT, depth, fair, cfg["default_edge"], pos)
        orders.extend(agg)

        # 2) Clear
        clr, bo, so = self.om.clear_orders(PRODUCT, depth, fair, cfg["default_edge"], pos, bo, so)
        orders.extend(clr)

        # 3) Passive inside‐spread MM
        pa = pos + bo - so
        if bo == 0 and so == 0 and depth.buy_orders and depth.sell_orders:
            bb = max(depth.buy_orders)
            ba = min(depth.sell_orders)
            if ba - bb > 2:
                orders.append(Order(PRODUCT, bb + 1, 1))
                orders.append(Order(PRODUCT, ba - 1, -1))

        # 4) Conversion
        net = pa
        if obs and abs(net) < cfg["conversion_limit"]:
            buy_eff = obs.askPrice + obs.transportFees + obs.importTariff
            if buy_eff < fair * cfg["conv_buy_threshold"]:
                q = dynamic_order_qty(cfg["conversion_limit"], net, self.limit)
                if q:
                    orders.append(Order(PRODUCT, obs.askPrice, q))
                    net += q

            sell_eff = obs.bidPrice - obs.transportFees - obs.exportTariff
            if sell_eff > fair * cfg["conv_sell_threshold"]:
                q = dynamic_order_qty(cfg["conversion_limit"], net, self.limit)
                if q:
                    orders.append(Order(PRODUCT, obs.bidPrice, -q))
                    net -= q

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # 1) build orders dict
        mm_orders = self.macarons_strategy(state)
        orders_dict = {PRODUCT: mm_orders}
        # 2) no conversions in this example
        conversions = 0
        # 3) optional debug data
        trader_data = ""
        # 4) log for Prosperity’s debugger
        logger.flush(state, orders_dict, conversions, trader_data)
        # 5) return a tuple (orders, conversions, trader_data)
        return orders_dict, conversions, trader_data

# instantiate once
_trader = Trader()

# this is the entrypoint Prosperity3bt uses
def run(state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
    return _trader.run(state)
