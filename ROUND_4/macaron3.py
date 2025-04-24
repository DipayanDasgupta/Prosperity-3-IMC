# r4_macarons_factor_mm.py
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder
from typing import Dict, List, Tuple
import numpy as np, json, jsonpickle, math

PRODUCT = "MAGNIFICENT_MACARONS"
POS_LIMIT = 75          # hard exchange limit
SOFT_LIMIT = 40         # where we start skewing quotes
WIN_SIZE   = 500        # ticks to fit regression at start of day
ENTRY_LONG = 0.98       # import trigger
ENTRY_SHORT= 1.02       # export trigger
STORAGE    = 0.1        # cost per long unit per tick

class LinReg:
    """very small OLS helper updated once"""
    def __init__(self): self.x,self.y=None,None
    def fit(self,X,Y):
        # X : n×2  (sugar, sunlight)
        A = np.column_stack((np.ones(len(X)),X))
        beta = np.linalg.lstsq(A,Y,rcond=None)[0]   # α,β_s,β_l
        self.x,self.y=beta[:3],None
        self.alpha,self.b_s,self.b_l=beta
    def predict(self,sugar,light): return self.alpha+self.b_s*sugar+self.b_l*light

class Trader:
    def __init__(self):
        self.reg   : Dict[int,LinReg] = {}     # per‑day model
        self.obs_buf: Dict[int,List[Tuple[float,float,float]]] = {}  # day→[(sugar,light,mid)]
    # ---------- helpers ----------
    def mid_book(self,od:OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders)+min(od.sell_orders))/2
    # ---------- main per‑tick ----------
    def run(self,state:TradingState) -> Tuple[Dict[str,List[Order]],int,str]:
        day = state.timestamp//10000+1                      # 1‑based day number
        od  = state.order_depths.get(PRODUCT)
        orders: List[Order]=[]
        if not od: return {},0,""
        mid = self.mid_book(od)
        # --- collect obs & build model ---
        obs = state.observations.conversionObservations.get(PRODUCT)
        if obs is None:
              
    # fallback: simple MM if spread exists
                    if od.buy_orders and od.sell_orders:
                        best_bid = max(od.buy_orders)
                        best_ask = min(od.sell_orders)
                        spread = best_ask - best_bid
                        if spread >= 2:
                            bid = best_bid + 1
                            ask = best_ask - 1
                            lot = 5
                            orders.append(Order(PRODUCT, bid,  lot))
                            orders.append(Order(PRODUCT, ask, -lot))
                            return {PRODUCT: orders}, 0, "passive MM fallback"
                    return {}, 0, "no obs, no spread"

        sugar, light = obs.sugarPrice, obs.sunlightIndex
  # always present
        if day not in self.obs_buf: self.obs_buf[day]=[]
        if mid: self.obs_buf[day].append((sugar,light,mid))

        if day not in self.reg and len(self.obs_buf[day])>=WIN_SIZE:
            X=np.array([(s,l) for s,l,_ in self.obs_buf[day]])
            Y=np.array([m for *_,m in self.obs_buf[day]])
            self.reg[day]=LinReg(); self.reg[day].fit(X,Y)

        if day not in self.reg:   # model not ready yet → only make‑spread MM
            fair = mid
        else:
            fair = self.reg[day].predict(sugar,light) - STORAGE*max(state.position.get(PRODUCT,0),0)

        pos   = state.position.get(PRODUCT,0)
        # ---------- 1) conversion opportunities ----------
        eff_buy  = obs.askPrice + obs.transportFees + obs.importTariff
        eff_sell = obs.bidPrice - obs.transportFees - obs.exportTariff
        conv=0
        if eff_buy < ENTRY_LONG*fair and pos < POS_LIMIT:
            qty = min(POS_LIMIT-pos,10)
            conv += qty          # +ve means buy/import
        elif eff_sell > ENTRY_SHORT*fair and pos>-POS_LIMIT:
            qty = min(POS_LIMIT+pos,10)
            conv -= qty          # -ve means sell/export

        # ---------- 2) aggressive take if book offers edge ----------
        take_width = 1.5
        if od.sell_orders:
            best_ask=min(od.sell_orders); vol=-od.sell_orders[best_ask]
            if best_ask<fair-take_width and pos<POS_LIMIT:
                qty=min(vol,POS_LIMIT-pos); orders.append(Order(PRODUCT,best_ask,qty)); pos+=qty
        if od.buy_orders:
            best_bid=max(od.buy_orders); vol=od.buy_orders[best_bid]
            if best_bid>fair+take_width and pos>-POS_LIMIT:
                qty=min(vol,POS_LIMIT+pos); orders.append(Order(PRODUCT,best_bid,-qty)); pos-=qty
        # ---------- 3) passive MM ----------
        if od.buy_orders and od.sell_orders:
            spread=min(od.sell_orders)-max(od.buy_orders)
            if spread>=2:
                edge=1 if abs(pos)<=SOFT_LIMIT else 2
                bid=max(od.buy_orders)+edge
                ask=min(od.sell_orders)-edge
                lot=5
                if pos<POS_LIMIT: orders.append(Order(PRODUCT,bid, lot))
                if pos>-POS_LIMIT: orders.append(Order(PRODUCT,ask,-lot))

        return {PRODUCT:orders}, {PRODUCT:conv}, json.dumps({"fair":fair,"pos":pos})

# single instance for the back‑tester
_trader=Trader()
def run(state:TradingState): return _trader.run(state)
