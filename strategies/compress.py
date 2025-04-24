import json, jsonpickle, math, numpy as np
from abc import abstractmethod
from typing import Any, TypeAlias, List, Dict, Tuple
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
from collections import deque

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self): self.logs, self.max_len = "", 3750
    def print(self, *args, sep=" ", end="\n"): self.logs += sep.join(map(str, args)) + end
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conv: int, data: str):
        base_len = len(self.to_json([self.cmp_state(state, ""), self.cmp_orders(orders), conv, "", ""]))
        item_len = (self.max_len - base_len) // 3
        print(self.to_json([self.cmp_state(state, self.trunc(state.traderData, item_len)), self.cmp_orders(orders), conv, self.trunc(data, item_len), self.trunc(self.logs, item_len)]))
        self.logs = ""
    def cmp_state(self, state: TradingState, data: str): return [state.timestamp, data, [[l.symbol, l.product, l.denomination] for l in state.listings.values()], {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()}, self.cmp_trades(state.own_trades), self.cmp_trades(state.market_trades), state.position, [state.observations.plainValueObservations, {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex] for p, o in state.observations.conversionObservations.items()}]]
    def cmp_trades(self, trades: dict[str, list[object]]): seen, res = set(), []; [seen.add(t := (trade.symbol, int(trade.price), int(trade.quantity), trade.buyer, trade.seller, trade.timestamp)) or res.append([trade.symbol, int(trade.price), int(trade.quantity), trade.buyer, trade.seller, trade.timestamp]) for arr in trades.values() for trade in arr if t not in seen]; return res
    def cmp_orders(self, orders: dict[str, list[Order]]): return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]
    def to_json(self, value): return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
    def trunc(self, value: str, max_len: int): return value[:max_len - 3] + "..." if len(value) > max_len else value

logger = Logger()

class Product:
    RESIN, KELP, INK, BASKET1, DJEMBES, CROISSANTS, JAMS, BASKET2, ART1, ART2, SPRD1, SPRD2, ROCK, MACARONS = "RAINFOREST_RESIN", "KELP", "SQUID_INK", "PICNIC_BASKET1", "DJEMBES", "CROISSANTS", "JAMS", "PICNIC_BASKET2", "ARTIFICAL1", "ARTIFICAL2", "SPREAD1", "SPREAD2", "VOLCANIC_ROCK", "MAGNIFICENT_MACARONS"
    V_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
    STRIKES = [9500, 9750, 10000, 10250, 10500]
    VOUCHERS = [f"{V_PREFIX}{K}" for K in STRIKES] # type: ignore

PARAMS = {
    Product.RESIN: {"fair": 10000, "take_w": 1, "clear_w": 0, "dis_edge": 1, "join_edge": 2, "def_edge": 1, "soft_lim": 50},
    Product.KELP: {"take_w": 2, "clear_w": 0, "adv": False, "adv_vol": 15, "rev_beta": -0.18, "dis_edge": 2, "join_edge": 0, "def_edge": 1, "ink_adj": 0.05},
    Product.INK: {"take_w": 2, "clear_w": 1, "adv": False, "adv_vol": 15, "rev_beta": -0.228, "dis_edge": 2, "join_edge": 0, "def_edge": 1, "spike_lb": 3, "spike_ub": 5.6, "offset": 2, "rev_win": 55, "rev_wt": 0.12},
    Product.CROISSANTS: {"fair": 10, "take_w": 0.5, "clear_w": 0.2, "dis_edge": 0.5, "join_edge": 1, "def_edge": 1, "soft_lim": 125},
    Product.JAMS: {"fair": 15, "take_w": 0.5, "clear_w": 0.2, "dis_edge": 0.5, "join_edge": 1, "def_edge": 1, "soft_lim": 175},
    Product.DJEMBES: {"fair": 20, "take_w": 0.5, "clear_w": 0.2, "dis_edge": 0.5, "join_edge": 1, "def_edge": 1, "soft_lim": 30},
    Product.SPRD1: {"sp_mean": 48.777856, "sp_std": 85.119723, "sp_win": 55, "z_thr": 4, "tgt_pos": 60},
    Product.BASKET2: {"take_w": 2, "clear_w": 0, "adv": True, "adv_vol": 15, "dis_edge": 2, "join_edge": 0, "def_edge": 2, "syn_wt": 0.03, "vol_win": 10, "adv_vola": 0.1},
    Product.ROCK: {"win_size": 50, "z_thr": 2}
}

W1, W2 = {Product.DJEMBES: 1, Product.CROISSANTS: 6, Product.JAMS: 3}, {Product.CROISSANTS: 4, Product.JAMS: 2}

class Strategy:
    def __init__(self, sym: Symbol, lim: int): self.sym, self.lim, self.orders = sym, lim, []
    def buy(self, p: float, q: int): self.orders.append(Order(self.sym, p, q))
    def sell(self, p: float, q: int): self.orders.append(Order(self.sym, p, -q))
    def mid(self, state: TradingState, sym: str) -> float: od = state.order_depths.get(sym); return (max(od.buy_orders) + min(od.sell_orders)) / 2 if od and od.buy_orders and od.sell_orders else 0
    def run(self, state: TradingState) -> Tuple[List[Order], str]: self.orders = []; self.act(state); return self.orders, ""
    def act(self, state: TradingState): pass

class ParabolaIV:
    def __init__(self, v: str, k: int, adapt=False, abs=False):
        self.v, self.k, self.adapt, self.abs, self.exp, self.tpd, self.win, self.lim = v, k, adapt, abs, 7, 1000, 500, 200
        self.start, self.hist, self.ivc, self.a, self.b, self.c = None, deque(maxlen=self.win), {}, None, None, None
    def ncdf(self, x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    def bs(self, S, K, T, s): return max(S - K, 0) if T <= 0 or s <= 0 or S <= 0 else S * self.ncdf((math.log(S / K) + 0.5 * s**2 * T) / (s * math.sqrt(T))) - K * self.ncdf((math.log(S / K) + 0.5 * s**2 * T) / (s * math.sqrt(T)) - s * math.sqrt(T))
    def iv(self, S, K, T, p, tol=1e-4, max_i=50):
        key = (round(S, 1), round(K, 1), round(T, 5), round(p, 1))
        if key in self.ivc: return self.ivc[key]
        lo, hi = 1e-6, 5
        for _ in range(max_i):
            mid = (lo + hi) / 2; val = self.bs(S, K, T, mid) - p
            if abs(val) < tol: self.ivc[key] = mid; return mid
            hi, lo = mid, mid if val > 0 else lo, mid
        return None
    def fit(self): m, v = zip(*self.hist); self.a, self.b, self.c = np.polyfit(m, v, 2)
    def fiv(self, m): return self.a * m**2 + self.b * m + self.c
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        orders, ts = {}, state.timestamp
        if self.start is None: self.start = ts
        d, rd = state.order_depths.get(self.v, OrderDepth()), state.order_depths.get(Product.ROCK, OrderDepth())
        if not (d.sell_orders and d.buy_orders and rd.sell_orders and rd.buy_orders): return {}, 0, ""
        ask, bid = min(d.sell_orders), max(d.buy_orders); mp = (ask + bid) / 2
        rb, ra = max(rd.buy_orders), min(rd.sell_orders); sp = (rb + ra) / 2
        TTE, T = max(0.1, self.exp - (ts - self.start) / self.tpd), max(0.1, self.exp - (ts - self.start) / self.tpd) / 365
        if self.abs:
            iv, fv, mis = 1.1, self.bs(sp, self.k, T, 1.1), mp - self.bs(sp, self.k, T, 1.1); pos = state.position.get(self.v, 0); res = []
            if mis > 1 and pos < self.lim: res.append(Order(self.v, ask, min(20, self.lim - pos)))
            elif mis < -1 and pos > -self.lim: res.append(Order(self.v, bid, -min(20, self.lim + pos)))
            orders[self.v] = res; return orders, 0, ""
        m = math.log(self.k / sp) / math.sqrt(TTE); v = self.iv(sp, self.k, T, mp)
        if v is None or v < 0.5: return {}, 0, ""
        self.hist.append((m, v))
        if len(self.hist) < self.win: return {}, 0, ""
        self.fit(); fit, pos, res = self.fiv(m), state.position.get(self.v, 0), []
        if v < fit - 0.019 and pos < self.lim: res.append(Order(self.v, ask, min(35, self.lim - pos)))
        elif v > fit + 0.013 and pos > -self.lim: res.append(Order(self.v, bid, -min(40, self.lim + pos)))
        orders[self.v] = res; return orders, 0, ""

class VolcanicMM(Strategy):
    def __init__(self, sym: Symbol, lim: int): super().__init__(sym, lim); self.cap, self.min_sp, self.off, self.max_loss = 10, 0.5, 0.2, -10000
    def act(self, state: TradingState):
        if self.sym not in state.order_depths: return
        d, pos = state.order_depths[self.sym], state.position.get(self.sym, 0)
        if not (d.buy_orders and d.sell_orders): return
        bid, ask, sp, mp = max(d.buy_orders), min(d.sell_orders), min(d.sell_orders) - max(d.buy_orders), (max(d.buy_orders) + min(d.sell_orders)) / 2
        pnl = sum((t.price - mp) * t.quantity for t in state.own_trades.get(self.sym, []))
        if pnl < self.max_loss:
            if pos > 0:
                q = min(pos, d.buy_orders.get(bid, 0), self.cap)
                if q > 0: self.orders.append(Order(self.sym, bid, -q))
            elif pos < 0:
                q = min(-pos, abs(d.sell_orders.get(ask, 0)), self.cap)
                if q > 0: self.orders.append(Order(self.sym, ask, q))
        elif sp >= self.min_sp:
            if pos < self.lim:
                q = min(self.cap, self.lim - pos, abs(d.sell_orders.get(ask, 0)))
                if q > 0: self.orders.append(Order(self.sym, int(bid + self.off), q))
            if pos > -self.lim:
                q = min(self.cap, self.lim + pos, d.buy_orders.get(bid, 0))
                if q > 0: self.orders.append(Order(self.sym, int(ask - self.off), -q))

class Signal(Strategy):
    def long(self, state: TradingState): self.buy(min(state.order_depths[self.sym].sell_orders), self.lim - state.position.get(self.sym, 0))
    def short(self, state: TradingState): self.sell(max(state.order_depths[self.sym].buy_orders), self.lim + state.position.get(self.sym, 0))

class Picnic(Signal):
    def act(self, state: TradingState):
        if any(s not in state.order_depths for s in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"]): return
        diff = self.mid(state, "PICNIC_BASKET1") - 6 * self.mid(state, "CROISSANTS") - 3 * self.mid(state, "JAMS") - self.mid(state, "DJEMBES")
        lo, hi = {"CROISSANTS": (10, 80), "JAMS": (10, 80), "DJEMBES": (10, 80), "PICNIC_BASKET1": (10, 80)}[self.sym]
        if diff < lo: self.long(state)
        elif diff > hi: self.short(state)

class Macaron:
    def __init__(self):
        self.lim, self.conv_lim, self.cost, self.min_edge, self.step = 75, 10, 0.1, 0.5, 0.2
        self.data = {"edge": 1.0, "vol_hist": [], "sun_hist": [], "sug_hist": [], "last_sug": None, "last_sun": None, "fc_below_csi": False, "below_csi_dur": 0, "clr_start": False}
        self.entry, self.pos, self.CSI, self.fc_per, self.max_norm, self.max_csi, self.clr_start, self.clr_end, self.clr_tgt = 0, 0, 55.0, 2, 50, 75, 998000, 1000000, 0.0
    def calc_cost(self, pos: int, ts: int): return 0.0 if pos <= 0 or self.entry == 0 else pos * self.cost * (ts - self.entry) / 100
    def upd_pos(self, pos: int, ts: int):
        if pos != self.pos: self.entry = ts if pos and not self.pos else 0 if not pos else self.entry; self.pos = pos
    def upd_edge(self, fr: float, vol: float): return min(1.5, self.data["edge"] * 1.1 + min(0.5, vol * 0.2)) if fr > 0.7 else max(0.1, self.data["edge"] * 0.9 + min(0.5, vol * 0.2)) if fr < 0.3 else self.data["edge"] + min(0.5, vol * 0.2)
    def calc_fr(self, ts: int):
        if not hasattr(self, 'fill_hist'): self.fill_hist, self.ema_fr, self.last_ts = [], 0.5, None
        if ts == self.last_ts: return self.ema_fr
        fo = [o for o in self.last_orders if o.quantity > 0]; fr = len(fo) / max(1, len(self.last_orders)); self.ema_fr = 0.2 * fr + 0.8 * self.ema_fr; self.last_ts = ts; return self.ema_fr
    def calc_vol(self, od: OrderDepth):
        if not hasattr(self, 'p_hist'): self.p_hist = []
        if od.buy_orders and od.sell_orders: self.p_hist.append((min(od.sell_orders), max(od.buy_orders)))
        if len(self.p_hist) > 20: self.p_hist.pop(0)
        return math.sqrt(sum((math.log(h/l))**2 for h, l in self.p_hist) / (4 * len(self.p_hist) * math.log(2))) if len(self.p_hist) >= 5 else 1
    def upd_hist(self, obs: ConversionObservation):
        if obs:
            self.data["sun_hist"].append(obs.sunlightIndex); self.data["sug_hist"].append(obs.sugarPrice); self.data["last_sug"], self.data["last_sun"] = obs.sugarPrice, obs.sunlightIndex
            if len(self.data["sun_hist"]) > 10: self.data["sun_hist"].pop(0); self.data["sug_hist"].pop(0)
    def fc_sun(self):
        h = self.data["sun_hist"]
        if not h or len(h) < 3: return False
        if h[-1] >= self.CSI: self.data["below_csi_dur"] = 0; return False
        self.data["below_csi_dur"] += 1
        trend, val, pred = (h[-1] - h[-3]) / 2, h[-1], []
        for _ in range(self.fc_per): val += trend; pred.append(val)
        return all(v < self.CSI for v in pred)
    def sug_below(self): return self.data["last_sug"] < sum(self.data["sug_hist"]) / len(self.data["sug_hist"]) if self.data["sug_hist"] else False
    def imp_prices(self, obs: ConversionObservation): return obs.bidPrice - obs.transportFees - obs.exportTariff, obs.askPrice + obs.transportFees + obs.importTariff
    def pred_price(self, obs: ConversionObservation):
        fv, csi_p = (obs.bidPrice + obs.askPrice) / 2, 0
        sug_i, sun_i = 0.05 * (obs.sugarPrice - 200), 0.1 * (obs.sunlightIndex - 60)
        if obs.sunlightIndex < self.CSI and self.data["fc_below_csi"]:
            df = min(3, self.data["below_csi_dur"] / 2); csi_p = 5.0 * df + (3.0 if self.sug_below() else 0)
        if obs.sunlightIndex < (self.CSI - 5): csi_p += 0.5 * ((self.CSI - obs.sunlightIndex) ** 1.5)
        return fv + sug_i + sun_i + csi_p
    def route(self, prod: str, tp: float, q: int, od: OrderDepth, side: str):
        orders = []
        if side == "buy":
            for p in sorted(p for p in od.sell_orders if p <= tp):
                qty = min(q, -od.sell_orders[p]); orders.append(Order(prod, p, qty)); q -= qty
                if q <= 0: break
        else:
            for p in sorted((p for p in od.buy_orders if p >= tp), reverse=True):
                qty = min(q, od.buy_orders[p]); orders.append(Order(prod, p, -qty)); q -= qty
                if q <= 0: break
        return orders
    def risk(self, pos: int, ts: int, sun: float):
        self.upd_pos(pos, ts)
        if ts >= self.clr_start:
            self.data["clr_start"] = True; prog = min(1, (ts - self.clr_start) / (self.clr_end - self.clr_start)); agg = min(1.5, 1 + prog)
            return min(0.6, 1 - prog) if pos else 1
        max_pos = self.max_csi if sun < self.CSI else self.max_norm; pos_pct = abs(pos) / max_pos
        return 0.6 if pos_pct > 0.8 else 0.8 if pos_pct > 0.5 else 1.2 if pos_pct < 0.2 else 1.5 if sun < self.CSI and self.data["fc_below_csi"] and pos > 0 else 0.7 if sun < self.CSI and self.data["fc_below_csi"] else 1
    def impact(self, od: OrderDepth, q: int): return 0 if not (od.buy_orders and od.sell_orders) else (1 / (sum(od.buy_orders.values()) + sum(abs(v) for v in od.sell_orders.values()))) * 100 * q
    def arb(self, state: TradingState):
        p, od = Product.MACARONS, state.order_depths.get(Product.MACARONS)
        if not (od and od.buy_orders and od.sell_orders): return []
        bid, ask, pos = max(od.buy_orders), min(od.sell_orders), state.position.get(p, 0)
        if bid <= ask: return []
        bv, sv = -od.sell_orders[ask], od.buy_orders[bid]
        vol = min(bv, sv, self.lim - pos, self.lim + pos)
        return [Order(p, ask, vol), Order(p, bid, -vol)] if vol > 0 else []
    def cross_arb(self, state: TradingState):
        p, orders, conv, pos = Product.MACARONS, [], 0, state.position.get(p, 0)
        if p not in state.observations.conversionObservations: return orders, conv
        obs = state.observations.conversionObservations[p]; ib, ia = self.imp_prices(obs); em = 0.7 if obs.sunlightIndex < self.CSI and self.data["fc_below_csi"] else 1
        if self.data["clr_start"]: em *= max(0.3, 1 - min(1, (state.timestamp - self.clr_start) / (self.clr_end - self.clr_start)))
        edge = self.data["edge"] * em
        if state.order_depths[p].buy_orders:
            lb = max(state.order_depths[p].buy_orders); re = edge * max(0.5, 1 - min(1, (state.timestamp - self.clr_start) / (self.clr_end - self.clr_start))) if self.data["clr_start"] and pos > 0 else edge
            if lb > ia + re:
                q = min(self.lim - pos, state.order_depths[p].buy_orders[lb], self.conv_lim)
                if q > 0: orders.append(Order(p, lb, -q)); conv = -q
        if state.order_depths[p].sell_orders:
            la = min(state.order_depths[p].sell_orders); re = edge * max(0.5, 1 - min(1, (state.timestamp - self.clr_start) / (self.clr_end - self.clr_start))) if self.data["clr_start"] and pos < 0 else edge
            if ib > la + re:
                q = min(self.lim + pos, -state.order_depths[p].sell_orders[la], self.conv_lim)
                if q > 0: orders.append(Order(p, la, q)); conv = q
        return orders, conv
    def clr(self, state: TradingState):
        p, pos, od = Product.MACARONS, state.position.get(p, 0), state.order_depths.get(p, OrderDepth())
        if state.timestamp < self.clr_start or not pos or not (od.buy_orders and od.sell_orders): return []
        prog, agg = min(1, (state.timestamp - self.clr_start) / (self.clr_end - self.clr_start)), 3 if min(1, (state.timestamp - self.clr_start) / (self.clr_end - self.clr_start)) > 0.8 else 2 if min(1, (state.timestamp - self.clr_start) / (self.clr_end - self.clr_start)) > 0.5 else 1
        bid, ask = max(od.buy_orders), min(od.sell_orders)
        if pos > 0:
            tc, pd, tp = int(pos * prog * agg), max(0, int(prog * 2)), max(1, bid - max(0, int(prog * 2))); sq = min(pos, tc)
            return [Order(p, tp, -sq)] if sq > 0 else []
        elif pos < 0:
            tc, pp, tp = int(abs(pos) * prog * agg), max(0, int(prog * 2)), ask + max(0, int(prog * 2)); bq = min(abs(pos), tc)
            return [Order(p, tp, bq)] if bq > 0 else []
        return []
    def csi(self, state: TradingState):
        p, obs, pos, od = Product.MACARONS, state.observations.conversionObservations.get(p), state.position.get(p, 0), state.order_depths.get(p, OrderDepth())
        if self.data["clr_start"] or not (obs and od and od.buy_orders and od.sell_orders): return []
        self.data["fc_below_csi"] = self.fc_sun(); bid, ask = max(od.buy_orders), min(od.sell_orders)
        if obs.sunlightIndex < self.CSI and self.data["fc_below_csi"] and self.sug_below() and not self.data["clr_start"]:
            ap = self.lim - pos; tp = int(ap * 0.8)
            return [Order(p, int(ask + 0.5), tp)] if tp > 0 else []
        pb, ca = len(self.data["sun_hist"]) >= 2 and self.data["sun_hist"][-2] < self.CSI, obs.sunlightIndex >= self.CSI
        return [Order(p, int(bid - 0.5), -pos)] if pb and ca and pos > 0 else []
    def run(self, state: TradingState):
        p, orders, conv = Product.MACARONS, [], 0; self.last_orders = []
        od, obs, pos = state.order_depths.get(p, OrderDepth()), state.observations.conversionObservations.get(p), state.position.get(p, 0)
        if obs: self.upd_hist(obs)
        vol, fr = self.calc_vol(od), self.calc_fr(state.timestamp)
        if state.timestamp >= self.clr_start: self.data["clr_start"] = True; orders.extend(self.clr(state))
        sun, rm = obs.sunlightIndex if obs else 60, self.risk(pos, state.timestamp, obs.sunlightIndex if obs else 60); self.data["edge"] = self.upd_edge(fr, vol); edge = self.data["edge"] * rm
        if arb := self.arb(state): orders.extend(arb); return {p: orders}, 0, jsonpickle.encode(self.data)
        co, cc = self.cross_arb(state)
        if co: orders.extend(co); return {p: orders}, cc, jsonpickle.encode(self.data)
        if self.data["clr_start"] and pos and (prog := min(1, (state.timestamp - self.clr_start) / (self.clr_end - self.clr_start))) > 0.9:
            if pos > 0 and od.buy_orders: orders.append(Order(p, max(1, max(od.buy_orders) - int(min(5, 10 * (prog - 0.9) * 10))), -pos))
            elif pos < 0 and od.sell_orders: orders.append(Order(p, min(od.sell_orders) + int(min(5, 10 * (prog - 0.9) * 10)), abs(pos)))
            return {p: orders}, conv, jsonpickle.encode(self.data)
        if not self.data["clr_start"] and (csi := self.csi(state)): orders.extend(csi); return {p: orders}, 0, jsonpickle.encode(self.data)
        if obs and od.buy_orders and od.sell_orders:
            bid, ask, sp, fv = max(od.buy_orders), min(od.sell_orders), min(od.sell_orders) - max(od.buy_orders), self.pred_price(obs)
            if sp > 2 * edge:
                bp, ap = int(fv - edge), int(fv + edge)
                sm = max(0.1, 1 - min(1, (state.timestamp - self.clr_start) / (self.clr_end - self.clr_start))) if self.data["clr_start"] else 1
                bq, sq = (min(self.lim - pos, (2 if pos > 0 else 8 if pos < 0 else 1) * sm), min(self.lim + pos, (8 if pos > 0 else 2 if pos < 0 else 1) * sm)) if self.data["clr_start"] else (min(self.lim - pos, 10 if obs.sunlightIndex < self.CSI and self.data["fc_below_csi"] else 5), min(self.lim + pos, 3 if obs.sunlightIndex < self.CSI and self.data["fc_below_csi"] else 5))
                if bq > 0: orders.append(Order(p, bp, bq))
                if sq > 0: orders.append(Order(p, ap, -sq))
        return {p: orders}, conv, jsonpickle.encode(self.data)

class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.lims = {Product.RESIN: 50, Product.KELP: 50, Product.INK: 50, Product.CROISSANTS: 250, Product.JAMS: 350, Product.DJEMBES: 60, Product.BASKET1: 60, Product.BASKET2: 100, Product.ROCK: 400, **{s: 200 for s in Product.VOUCHERS}}
        self.strikes = {s: k for s, k in zip(Product.VOUCHERS, Product.STRIKES)}
        self.v_strats = [ParabolaIV(f"{Product.V_PREFIX}9500", 9500, abs=True), ParabolaIV(f"{Product.V_PREFIX}10000", 10000), VolcanicMM(f"{Product.V_PREFIX}9750", 200), VolcanicMM(f"{Product.V_PREFIX}10250", 200), VolcanicMM(f"{Product.V_PREFIX}10500", 200)]
        self.hist, self.thr, self.lim, self.z = {Product.ROCK: []}, {Product.ROCK: 0}, {Product.ROCK: 0}, {Product.ROCK: 0}
        self.croi_strat, self.b1_strat, self.mac = Picnic("CROISSANTS", 250), Picnic(Product.BASKET1, self.lims[Product.BASKET1]), Macaron()
    def ncdf(self, x): return (1 + math.erf(x / math.sqrt(2))) / 2
    def npdf(self, x): return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)
    def bs(self, S, K, T, r, s): return max(0, S - K) if T <= 0 or s <= 0 else S * self.ncdf((math.log(S / K) + (r + 0.5 * s**2) * T) / (s * math.sqrt(T))) - K * math.exp(-r * T) * self.ncdf((math.log(S / K) + (r + 0.5 * s**2) * T) / (s * math.sqrt(T)) - s * math.sqrt(T))
    def vega(self, S, K, T, r, s): return 0 if T <= 0 or s <= 0 else S * self.npdf((math.log(S / K) + (r + 0.5 * s**2) * T) / (s * math.sqrt(T))) * math.sqrt(T)
    def iv(self, tp, S, K, T, r, g=0.1, mi=100, tol=1e-6):
        s, iv = g, max(0, S - K * math.exp(-r * T))
        if tp < iv - tol or tp < 0: return 0
        for _ in range(mi):
            p, v = self.bs(S, K, T, r, s), self.vega(S, K, T, r, s); diff = p - tp
            if abs(diff) < tol: return s
            if v < 1e-10:
                if s < 0.01: s = 0.01; pp = self.bs(S, K, T, r, s * 1.01); vp = self.vega(S, K, T, r, s * 1.01)
                if vp < 1e-10: break
                s *= 1.01; continue
            s -= diff / v; s = tol if s <= 0 else 5 if s > 5 else s
        return s if abs(self.bs(S, K, T, r, s) - tp) < tol * 10 else s
    def mid(self, sym: str, state: TradingState):
        od = state.order_depths.get(sym); return (max(od.buy_orders) + min(od.sell_orders)) / 2 if od and od.buy_orders and od.sell_orders else max(od.buy_orders) if od and od.buy_orders else min(od.sell_orders) if od and od.sell_orders else None
    def wmid(self, sym: str, state: TradingState):
        od = state.order_depths.get(sym)
        if not (od and od.buy_orders and od.sell_orders): return self.mid(sym, state)
        bb, ba = max(od.buy_orders), min(od.sell_orders); bbv, bav = abs(od.buy_orders[bb]), abs(od.sell_orders[ba])
        return (bb * bav + ba * bbv) / (bbv + bav) if bbv + bav > 0 else (bb + ba) / 2
    def tte(self, ts: int): return max(0, (8 - math.floor(max(0, ts) / 10000) - (max(0, ts) % 10000 + 1) / 10000) / 365)
    def mt(self, S: float, K: int, TTE: float): return math.log(K / S) / math.sqrt(TTE) if S > 0 and K > 0 and TTE > 0 else None
    def smile(self, m: float): return max(0.0001, -1.12938253 * (m**2) - 0.0421697 * m + 0.00938588)
    def fmid(self, p: str, od: OrderDepth):
        if not (od.buy_orders and od.sell_orders): return None
        ba, bb = min(od.sell_orders), max(od.buy_orders); thr = self.params.get(p, {}).get("adv_vol", 0)
        va, vb = [p for p, v in od.sell_orders.items() if abs(v) >= thr], [p for p, v in od.buy_orders.items() if abs(v) >= thr]
        fa, fb = min(va) if va else None, max(vb) if vb else None
        return (fa + fb) / 2 if fa and fb else (ba + bb) / 2
    def b2_fv(self, bd: OrderDepth, cd: OrderDepth, jd: OrderDepth, pos: int, to: dict):
        m, cm, jm = self.fmid(Product.BASKET2, bd), self.fmid(Product.CROISSANTS, cd), self.fmid(Product.JAMS, jd)
        if m is None: return None
        if cm is None or jm is None: return m
        sm = cm * W2[Product.CROISSANTS] + jm * W2[Product.JAMS]; w = self.params[Product.BASKET2]["syn_wt"]
        if pos: w *= math.exp(-abs(pos) / self.lims[Product.BASKET2])
        return (1 - w) * m + w * sm
    def take(self, p: str, fv: float, tw: float, orders: List[Order], od: OrderDepth, pos: int, bov: int, sov: int, adv=False, av=0, to=None):
        pl = self.lims[p]
        if p == Product.INK:
            to.setdefault("spike", False); pp = to.get("ink_last_price", fv)
            if to["spike"]:
                if abs(fv - pp) < self.params[p]["spike_lb"]: to["spike"] = False
                elif fv < to["rv"]:
                    if ba := min(od.sell_orders) if od.sell_orders else None:
                        q = min(abs(od.sell_orders[ba]), pl - pos); orders.append(Order(p, ba, q)) if q > 0 else None; return bov + q, 0
                else:
                    if bb := max(od.buy_orders) if od.buy_orders else None:
                        q = min(od.buy_orders[bb], pl + pos); orders.append(Order(p, bb, -q)) if q > 0 else None; return 0, sov + q
            if abs(fv - pp) > self.params[p]["spike_ub"]:
                to["spike"], to["rv"] = True, pp + self.params[p]["offset"] if fv > pp else pp - self.params[p]["offset"]
                if fv > pp:
                    if bb := max(od.buy_orders) if od.buy_orders else None:
                        q = min(od.buy_orders[bb], pl + pos); orders.append(Order(p, bb, -q)) if q > 0 else None; return 0, sov + q
                else:
                    if ba := min(od.sell_orders) if od.sell_orders else None:
                        q = min(abs(od.sell_orders[ba]), pl - pos); orders.append(Order(p, ba, q)) if q > 0 else None; return bov + q, 0
        if od.sell_orders and (ba := min(od.sell_orders)) <= fv - tw and (not adv or abs(baa := -od.sell_orders[ba]) <= av):
            q = min(baa, pl - pos); orders.append(Order(p, ba, q)) if q > 0 else None; bov += q
        if od.buy_orders and (bb := max(od.buy_orders)) >= fv + tw and (not adv or abs(bba := od.buy_orders[bb]) <= av):
            q = min(bba, pl + pos); orders.append(Order(p, bb, -q)) if q > 0 else None; sov += q
        return bov, sov
    def mm(self, p: str, orders: List[Order], bid: int, ask: int, pos: int, bov: int, sov: int):
        bq = self.lims[p] - (pos + bov); sq = self.lims[p] + (pos - sov)
        orders.append(Order(p, math.floor(bid), bq)) if bq > 0 else None
        orders.append(Order(p, math.ceil(ask), -sq)) if sq > 0 else None
        return bov, sov
    def clr(self, p: str, fv: float, w: int, orders: List[Order], od: OrderDepth, pos: int, bov: int, sov: int):
        pa, fb, fa = pos + bov - sov, round(fv - w), round(fv + w); bq, sq = self.lims[p] - (pos + bov), self.lims[p] + (pos - sov)
        if pa > 0:
            cq = min(sum(v for p, v in od.buy_orders.items() if p >= fa), pa); sq = min(sq, cq)
            orders.append(Order(p, fa, -abs(sq))) if sq > 0 else None; sov += abs(sq)
        if pa < 0:
            cq = min(sum(abs(v) for p, v in od.sell_orders.items() if p <= fb), abs(pa)); bq = min(bq, cq)
            orders.append(Order(p, fb, abs(bq))) if bq > 0 else None; bov += abs(bq)
        return bov, sov
    def kfv(self, od: OrderDepth, to, iod: OrderDepth):
        if not (od.sell_orders and od.buy_orders): return None
        ba, bb = min(od.sell_orders), max(od.buy_orders)
        va, vb = [p for p in od.sell_orders if abs(od.sell_orders[p]) >= self.params[Product.KELP]["adv_vol"]], [p for p in od.buy_orders if abs(od.buy_orders[p]) >= self.params[Product.KELP]["adv_vol"]]
        mp = (min(va) + max(vb)) / 2 if va and vb else (ba + bb) / 2 if to.get('kelp_last_price') is None else to['kelp_last_price']
        f = mp if to.get('kelp_last_price') is None else mp + (mp * ((mp - to["kelp_last_price"]) / to["kelp_last_price"] * self.params[Product.KELP]["rev_beta"]))
        if to.get("ink_last_price"):
            oip = to["ink_last_price"]; via, vib = [p for p in iod.sell_orders if abs(iod.sell_orders[p]) >= self.params[Product.INK]["adv_vol"]], [p for p in iod.buy_orders if abs(iod.buy_orders[p]) >= self.params[Product.INK]["adv_vol"]]
            nim = (min(via) + max(vib)) / 2 if via and vib else (min(iod.sell_orders) + max(iod.buy_orders)) / 2
            f -= (self.params[Product.KELP].get("ink_adj", 0.5) * ((nim - oip) / oip) * mp)
        to["kelp_last_price"] = mp
        return f
    def ifv(self, od: OrderDepth, to):
        if not (od.sell_orders and od.buy_orders): return None
        ba, bb = min(od.sell_orders), max(od.buy_orders)
        va, vb = [p for p in od.sell_orders if abs(od.sell_orders[p]) >= self.params[Product.INK]["adv_vol"]], [p for p in od.buy_orders if abs(od.buy_orders[p]) >= self.params[Product.INK]["adv_vol"]]
        mp = (min(va) + max(vb)) / 2 if va and vb else (ba + bb) / 2 if to.get('ink_last_price') is None else to['ink_last_price']
        to.setdefault('ink_price_hist', []).append(mp)
        if len(to['ink_price_hist']) > self.params[Product.INK]["rev_win"]: to['ink_price_hist'] = to['ink_price_hist'][-self.params[Product.INK]["rev_win"]:]
        ab = self.params[Product.INK]["rev_beta"]
        if len(to['ink_price_hist']) >= self.params[Product.INK]["rev_win"]:
            pr = np.array(to['ink_price_hist']); ret = (pr[1:] - pr[:-1]) / pr[:-1]; X, Y = ret[:-1], ret[1:]
            eb = -np.dot(X, Y) / np.dot(X, X) if np.dot(X, X) != 0 else self.params[Product.INK]["rev_beta"]
            ab = self.params[Product.INK]['rev_wt'] * eb + (1 - self.params[Product.INK]['rev_wt']) * self.params[Product.INK]["rev_beta"]
        f = mp if to.get('ink_last_price') is None else mp + (mp * ((mp - to["ink_last_price"]) / to["ink_last_price"] * ab))
        to["ink_last_price"] = mp
        return f
    def to(self, p: str, od: OrderDepth, fv: float, tw: float, pos: int, adv=False, av=0, to=None):
        orders = []; bov, sov = self.take(p, fv, tw, orders, od, pos, 0, 0, adv, av, to)
        return orders, bov, sov
    def co(self, p: str, od: OrderDepth, fv: float, cw: int, pos: int, bov: int, sov: int):
        orders = []; bov, sov = self.clr(p, fv, cw, orders, od, pos, bov, sov)
        return orders, bov, sov
    def mo(self, p, od: OrderDepth, fv: float, pos: int, bov: int, sov: int, de: float, je: float, dfe: float, mp=False, sl=0):
        adj = round(4 * ((sum(od.buy_orders.values()) - sum(abs(v) for v in od.sell_orders.values())) / (sum(od.buy_orders.values()) + sum(abs(v) for v in od.sell_orders.values())))) if p == Product.RESIN and (sum(od.buy_orders.values()) + sum(abs(v) for v in od.sell_orders.values())) > 0 else 0
        orders, aa, bb = [], [p for p in od.sell_orders if p > fv + de], [p for p in od.buy_orders if p < fv - de]
        baa, bbb = min(aa) if aa else None, max(bb) if bb else None
        ask = round(fv + dfe) if not baa else baa + 1 if abs(baa - fv) <= je else baa
        bid = round(fv - dfe) if not bbb else bbb if abs(fv - bbb) <= je else bbb + 1
        if mp: ask -= 1 if pos > sl else 0; bid += 1 if pos < -sl else 0
        bov, sov = self.mm(p, orders, bid, ask, pos, bov, sov)
        return orders, bov, sov
    def croi(self, state: TradingState): orders, _ = self.croi_strat.run(state); return orders
    def jams(self, state: TradingState):
        if not state.order_depths.get(Product.JAMS): return []
        pos, od, fv = state.position.get(Product.JAMS, 0), state.order_depths[Product.JAMS], self.params[Product.JAMS]["fair"]
        t, bo, so = self.to(Product.JAMS, od, fv, self.params[Product.JAMS]["take_w"], pos)
        c, bo, so = self.co(Product.JAMS, od, fv, self.params[Product.JAMS]["clear_w"], pos, bo, so)
        m, _, _ = self.mo(Product.JAMS, od, fv, pos, bo, so, self.params[Product.JAMS]["dis_edge"], self.params[Product.JAMS]["join_edge"], self.params[Product.JAMS]["def_edge"], True, self.params[Product.JAMS]["soft_lim"])
        return t + c + m
    def djembes(self, state: TradingState):
        if not state.order_depths.get(Product.DJEMBES): return []
        pos, od, fv = state.position.get(Product.DJEMBES, 0), state.order_depths[Product.DJEMBES], self.params[Product.DJEMBES]["fair"]
        t, bo, so = self.to(Product.DJEMBES, od, fv, self.params[Product.DJEMBES]["take_w"], pos)
        c, bo, so = self.co(Product.DJEMBES, od, fv, self.params[Product.DJEMBES]["clear_w"], pos, bo, so)
        m, _, _ = self.mo(Product.DJEMBES, od, fv, pos, bo, so, self.params[Product.DJEMBES]["dis_edge"], self.params[Product.DJEMBES]["join_edge"], self.params[Product.DJEMBES]["def_edge"], True, self.params[Product.DJEMBES]["soft_lim"])
        return t + c + m
    def art_od(self, ods: Dict[str, OrderDepth], p1=True):
        w = W1 if p1 else W2; aop = OrderDepth()
        cb, ca = max(ods[Product.CROISSANTS].buy_orders) if ods[Product.CROISSANTS].buy_orders else 0, min(ods[Product.CROISSANTS].sell_orders) if ods[Product.CROISSANTS].sell_orders else float("inf")
        jb, ja = max(ods[Product.JAMS].buy_orders) if ods[Product.JAMS].buy_orders else 0, min(ods[Product.JAMS].sell_orders) if ods[Product.JAMS].sell_orders else float("inf")
        if p1:
            db, da = max(ods[Product.DJEMBES].buy_orders) if ods[Product.DJEMBES].buy_orders else 0, min(ods[Product.DJEMBES].sell_orders) if ods[Product.DJEMBES].sell_orders else float("inf")
            ab, aa = db * w[Product.DJEMBES] + cb * w[Product.CROISSANTS] + jb * w[Product.JAMS], da * w[Product.DJEMBES] + ca * w[Product.CROISSANTS] + ja * w[Product.JAMS]
        else:
            ab, aa = cb * w[Product.CROISSANTS] + jb * w[Product.JAMS], ca * w[Product.CROISSANTS] + ja * w[Product.JAMS]
        if ab > 0:
            cbv, jbv = ods[Product.CROISSANTS].buy_orders.get(cb, 0) // w[Product.CROISSANTS], ods[Product.JAMS].buy_orders.get(jb, 0) // w[Product.JAMS]
            abv = min(cbv, jbv, ods[Product.DJEMBES].buy_orders.get(db, 0) // w[Product.DJEMBES]) if p1 else min(cbv, jbv)
            aop.buy_orders[ab] = abv
        if aa < float("inf"):
            cav, jav = abs(ods[Product.CROISSANTS].sell_orders.get(ca, 0)) // w[Product.CROISSANTS], abs(ods[Product.JAMS].sell_orders.get(ja, 0)) // w[Product.JAMS]
            aav = min(cav, jav, abs(ods[Product.DJEMBES].sell_orders.get(da, 0)) // w[Product.DJEMBES]) if p1 else min(cav, jav)
            aop.sell_orders[aa] = -aav
        return aop
    def conv_o(self, ao: List[Order], ods: Dict[str, OrderDepth], p1=True):
        w = W1 if p1 else W2; co = {Product.DJEMBES: [], Product.CROISSANTS: [], Product.JAMS: []} if p1 else {Product.CROISSANTS: [], Product.JAMS: []}
        aod = self.art_od(ods, p1); bb, ba = max(aod.buy_orders) if aod.buy_orders else 0, min(aod.sell_orders) if aod.sell_orders else float("inf")
        for o in ao:
            p, q = o.price, o.quantity
            if q > 0 and p >= ba and all(ods[p].sell_orders for p in w):
                cp, jp = min(ods[Product.CROISSANTS].sell_orders), min(ods[Product.JAMS].sell_orders)
                dp = min(ods[Product.DJEMBES].sell_orders) if p1 else None
            elif q < 0 and p <= bb and all(ods[p].buy_orders for p in w):
                cp, jp = max(ods[Product.CROISSANTS].buy_orders), max(ods[Product.JAMS].buy_orders)
                dp = max(ods[Product.DJEMBES].buy_orders) if p1 else None
            else: continue
            co[Product.CROISSANTS].append(Order(Product.CROISSANTS, cp, q * w[Product.CROISSANTS])); co[Product.JAMS].append(Order(Product.JAMS, jp, q * w[Product.JAMS]))
            if p1: co[Product.DJEMBES].append(Order(Product.DJEMBES, dp, q * w[Product.DJEMBES]))
        return co
    def exec_sp(self, tp: int, pp: int, ods: Dict[str, OrderDepth], p1=True):
        if tp == pp: return None
        tq = abs(tp - pp); pod = ods[Product.BASKET1 if p1 else Product.BASKET2]; aod = self.art_od(ods, p1)
        if not all([pod.sell_orders, aod.buy_orders, pod.buy_orders, aod.sell_orders]): return None
        if tp > pp:
            pap, pav = min(pod.sell_orders), abs(pod.sell_orders[min(pod.sell_orders)]); abp, abv = max(aod.buy_orders), abs(aod.buy_orders[max(aod.buy_orders)])
            if pav == 0 or abv == 0: return None
            ev = min(pav, abv, tq)
            if ev <= 0: return None
            po, ao = [Order(Product.BASKET1 if p1 else Product.BASKET2, pap, ev)], [Order(Product.ART1, abp, -ev)]
            agg = self.conv_o(ao, ods, p1); agg[Product.BASKET1 if p1 else Product.BASKET2] = po
            return agg
        pbp, pbv = max(pod.buy_orders), abs(pod.buy_orders[max(pod.buy_orders)]); aap, aav = min(aod.sell_orders), abs(aod.sell_orders[min(aod.sell_orders)])
        if pbv == 0 or aav == 0: return None
        ev = min(pbv, aav, tq)
        if ev <= 0: return None
        po, ao = [Order(Product.BASKET1 if p1 else Product.BASKET2, pbp, -ev)], [Order(Product.ART1, aap, ev)]
        agg = self.conv_o(ao, ods, p1); agg[Product.BASKET1 if p1 else Product.BASKET2] = po
        return agg
    def sp_o(self, ods: Dict[str, OrderDepth], p: str, pp: int, sd: Dict[str, object], SPRD, p1=True):
        rp = [Product.BASKET1, Product.DJEMBES, Product.CROISSANTS, Product.JAMS] if p1 else [Product.BASKET2, Product.CROISSANTS, Product.JAMS]
        if not all(p in ods for p in rp): return None
        pod, aod = ods[Product.BASKET1 if p1 else Product.BASKET2], self.art_od(ods, p1)
        if not all([pod.buy_orders, pod.sell_orders, aod.buy_orders, aod.sell_orders]): return None
        pm = sum(p * abs(pod.sell_orders[p]) for p in pod.buy_orders) + sum(p * abs(pod.buy_orders[p]) for p in pod.sell_orders) / (sum(abs(pod.buy_orders[p]) for p in pod.buy_orders) + sum(abs(pod.sell_orders[p]) for p in pod.sell_orders)) if pod.buy_orders and pod.sell_orders else 0
        am = sum(p * abs(aod.sell_orders[p]) for p in aod.buy_orders) + sum(p * abs(aod.buy_orders[p]) for p in aod.sell_orders) / (sum(abs(aod.buy_orders[p]) for p in aod.buy_orders) + sum(abs(aod.sell_orders[p]) for p in aod.sell_orders)) if aod.buy_orders and aod.sell_orders else 0
        sd["spread_hist"].append(pm - am)
        if len(sd["spread_hist"]) < self.params[SPRD]["sp_win"]: return None
        if len(sd["spread_hist"]) > self.params[SPRD]["sp_win"]: sd["spread_hist"].pop(0)
        ss = np.std(sd["spread_hist"]) or 1e-10;
        if ss == 0: return None
        z = (pm - am - self.params[SPRD]["sp_mean"]) / ss
        if z >= self.params[SPRD]["z_thr"] and pp != -self.params[SPRD]["tgt_pos"]: return self.exec_sp(-self.params[SPRD]["tgt_pos"], pp, ods, p1)
        if z <= -self.params[SPRD]["z_thr"] and pp != self.params[SPRD]["tgt_pos"]: return self.exec_sp(self.params[SPRD]["tgt_pos"], pp, ods, p1)
        sd["prev_zscore"] = z; return None
    def upd_rock(self, state: TradingState):
        orders = []; rd = state.order_depths.get(Product.ROCK)
        if rd and rd.buy_orders and rd.sell_orders: self.hist[Product.ROCK].append((max(rd.buy_orders) + min(rd.sell_orders)) / 2)
        if len(self.hist[Product.ROCK]) > 50:
            r = np.array(self.hist[Product.ROCK][-50:]); self.z[Product.ROCK] = (r[-1] - np.mean(r)) / np.std(r) if np.std(r) > 0 else 0
        thr, pos, pl = self.params[Product.ROCK]["z_thr"], state.position.get(Product.ROCK, 0), self.lims[Product.ROCK]
        if self.z[Product.ROCK] < -thr and rd.sell_orders: orders.append(Order(Product.ROCK, min(rd.sell_orders), min(-rd.sell_orders[min(rd.sell_orders)], pl - pos)))
        elif self.z[Product.ROCK] > thr and rd.buy_orders: orders.append(Order(Product.ROCK, max(rd.buy_orders), -min(rd.buy_orders[max(rd.buy_orders)], pl + pos)))
        return orders
    def run(self, state: TradingState):
        to = jsonpickle.decode(state.traderData) if state.traderData else {}; to.setdefault("base_iv_hist", []); to.setdefault(Product.SPRD1, {"spread_hist": [], "prev_zscore": 0, "clear_flag": False, "curr_avg": 0}); to.setdefault(Product.BASKET2, {"spread_hist": [], "prev_zscore": 0, "clear_flag": False, "curr_avg": 0}); to.setdefault("last_ivs", {})
        res = {s: [] for s in state.listings}
        if Product.RESIN in state.order_depths:
            pl, fv, pos = self.lims[Product.RESIN], self.params[Product.RESIN]["fair"], state.position.get(Product.RESIN, 0); bs, ss, orders, od = pl - pos, pl + pos, [], state.order_depths[Product.RESIN]
            if od.sell_orders and (ba := min(od.sell_orders)) < fv: f = min(abs(od.sell_orders[ba]), bs); orders.append(Order(Product.RESIN, ba, f)) if f > 0 else None; bs -= f; pos += f
            if od.buy_orders and (bb := max(od.buy_orders)) > fv: f = min(od.buy_orders[bb], ss); orders.append(Order(Product.RESIN, bb, -f)) if f > 0 else None; ss -= f; pos -= f
            orders, _, _ = self.mo(Product.RESIN, od, fv, pos, bs, ss, self.params[Product.RESIN]["dis_edge"], self.params[Product.RESIN]["join_edge"], self.params[Product.RESIN]["def_edge"], True, self.params[Product.RESIN]["soft_lim"])
            res[Product.RESIN] = orders
        if Product.KELP in state.order_depths:
            pos, od, iod = state.position.get(Product.KELP, 0), state.order_depths[Product.KELP], state.order_depths.get(Product.INK, OrderDepth())
            fv = self.kfv(od, to, iod)
            if fv is not None:
                t, bo, so = self.to(Product.KELP, od, fv, self.params[Product.KELP]["take_w"], pos, self.params[Product.KELP]["adv"], self.params[Product.KELP]["adv_vol"], to)
                c, bo, so = self.co(Product.KELP, od, fv, self.params[Product.KELP]["clear_w"], pos, bo, so)
                m, _, _ = self.mo(Product.KELP, od, fv, pos, bo, so, self.params[Product.KELP]["dis_edge"], self.params[Product.KELP]["join_edge"], self.params[Product.KELP]["def_edge"])
                res[Product.KELP] = t + c + m
        if Product.INK in state.order_depths:
            pos, od = state.position.get(Product.INK, 0), state.order_depths[Product.INK]
            fv = self.ifv(od, to)
            if fv is not None:
                t, bo, so = self.to(Product.INK, od, fv, self.params[Product.INK]["take_w"], pos, self.params[Product.INK]["adv"], self.params[Product.INK]["adv_vol"], to)
                c, bo, so = self.co(Product.INK, od, fv, self.params[Product.INK]["clear_w"], pos, bo, so)
                m, _, _ = self.mo(Product.INK, od, fv, pos, bo, so, self.params[Product.INK]["dis_edge"], self.params[Product.INK]["join_edge"], self.params[Product.INK]["def_edge"])
                res[Product.INK] = t + c + m
        res[Product.CROISSANTS] = self.croi(state)
        res[Product.JAMS] = self.jams(state)
        res[Product.DJEMBES] = self.djembes(state)
        if Product.BASKET1 in state.order_depths:
            pos, orders, _ = state.position.get(Product.BASKET1, 0), self.b1_strat.run(state)
            res[Product.BASKET1] = orders[0]
            sp = self.sp_o(state.order_depths, Product.BASKET1, pos, to[Product.SPRD1], Product.SPRD1, True)
            if sp:
                for p in sp: res[p] += sp[p]
        if Product.BASKET2 in state.order_depths:
            pos, od, cd, jd = state.position.get(Product.BASKET2, 0), state.order_depths[Product.BASKET2], state.order_depths.get(Product.CROISSANTS, OrderDepth()), state.order_depths.get(Product.JAMS, OrderDepth())
            fv = self.b2_fv(od, cd, jd, pos, to)
            if fv is not None:
                t, bo, so = self.to(Product.BASKET2, od, fv, self.params[Product.BASKET2]["take_w"], pos, self.params[Product.BASKET2]["adv"], self.params[Product.BASKET2]["adv_vol"], to)
                c, bo, so = self.co(Product.BASKET2, od, fv, self.params[Product.BASKET2]["clear_w"], pos, bo, so)
                m, _, _ = self.mo(Product.BASKET2, od, fv, pos, bo, so, self.params[Product.BASKET2]["dis_edge"], self.params[Product.BASKET2]["join_edge"], self.params[Product.BASKET2]["def_edge"])
                res[Product.BASKET2] = t + c + m
            sp = self.sp_o(state.order_depths, Product.BASKET2, pos, to[Product.BASKET2], Product.SPRD2, False)
            if sp:
                for p in sp: res[p] += sp[p]
        res[Product.ROCK] = self.upd_rock(state)
        for s in self.v_strats:
            orders, _, _ = s.run(state)
            for sym in orders: res[sym] += orders[sym]
        mac_res, conv, new_data = self.mac.run(state)
        for sym in mac_res: res[sym] += mac_res[sym]
        logger.flush(state, res, conv, new_data)
        return res, conv, new_data