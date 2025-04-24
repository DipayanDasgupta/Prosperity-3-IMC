from datamodel import OrderDepth,TradingState,Order,ProsperityEncoder
from typing import List,Dict
import json,jsonpickle,math,numpy as np

class Logger:
 def __init__(self):self.logs="";self.max_log_length=3750
 def flush(self,state:TradingState,orders:dict[str,list[Order]],conversions:int,trader_data:str):
  base_length=len(self.to_json([self.compress_state(state,""),self.compress_orders(orders),conversions,"",""]))
  max_item_length=(self.max_log_length-base_length)//3
  print(self.to_json([self.compress_state(state,self.truncate(state.traderData,max_item_length)),self.compress_orders(orders),conversions,self.truncate(trader_data,max_item_length),self.truncate(self.logs,max_item_length)]))
  self.logs=""
 def compress_state(self,state:TradingState,trader_data:str):return[state.timestamp,trader_data,self.compress_listings(state.listings),self.compress_order_depths(state.order_depths),self.compress_trades(state.own_trades),self.compress_trades(state.market_trades),state.position,self.compress_observations(state.observations)]
 def compress_listings(self,listings:dict[str,object]):return[[l.symbol,l.product,l.denomination]for l in listings.values()]
 def compress_order_depths(self,order_depths:dict[str,OrderDepth]):return{s:[od.buy_orders,od.sell_orders]for s,od in order_depths.items()}
 def compress_trades(self,trades:dict[str,list[object]]):return[[t.symbol,t.price,t.quantity,t.buyer,t.seller,t.timestamp]for arr in trades.values()for t in arr]
 def compress_observations(self,obs:object):co={p:[o.bidPrice,o.askPrice,o.transportFees,o.exportTariff,o.importTariff,o.sugarPrice,o.sunlightIndex]for p,o in obs.conversionObservations.items()};return[obs.plainValueObservations,co]
 def compress_orders(self,orders:dict[str,list[Order]]):return[[o.symbol,o.price,o.quantity]for arr in orders.values()for o in arr]
 def to_json(self,value):return json.dumps(value,cls=ProsperityEncoder,separators=(",",":"))
 def truncate(self,value:str,max_length:int):return value if len(value)<=max_length else value[:max_length-3]+"..."

logger=Logger()

class Product:
 RAINFOREST_RESIN="RAINFOREST_RESIN"
 KELP="KELP"
 SQUID_INK="SQUID_INK"
 PICNIC_BASKET1="PICNIC_BASKET1"
 DJEMBES="DJEMBES"
 CROISSANT="CROISSANTS"
 JAMS="JAMS"
 PICNIC_BASKET2="PICNIC_BASKET2"
 ARTIFICAL1="ARTIFICAL1"
 SPREAD1="SPREAD1"
 SPREAD2="SPREAD2"

PARAMS={
 Product.RAINFOREST_RESIN:{"fair_value":10000,"spread_pct":0.02,"mispricing_threshold":0.02,"max_position_pct":0.7},
 Product.KELP:{"fair_value":2023,"spread_pct":0.02,"mispricing_threshold":0.02,"max_position_pct":0.7,"take_width":2,"reversion_beta":-0.18},
 Product.SQUID_INK:{"fair_value":1972,"spread_pct":0.02,"mispricing_threshold":0.02,"max_position_pct":0.7,"take_width":2,"reversion_beta":-0.228},
 Product.SPREAD1:{"default_spread_mean":48.777856,"default_spread_std":85.119723,"spread_window":55,"zscore_threshold":4,"target_position":100},
 Product.SPREAD2:{"default_spread_mean":30.2336,"default_spread_std":59.8536,"spread_window":59,"zscore_threshold":6,"target_position":100},
 Product.PICNIC_BASKET1:{"b2_adjustment_factor":0.05}
}

PICNIC1_WEIGHTS={Product.DJEMBES:1,Product.CROISSANT:6,Product.JAMS:3}
PICNIC2_WEIGHTS={Product.CROISSANT:4,Product.JAMS:2}

class Trader:
 def __init__(self):
  self.params=PARAMS
  self.PRODUCT_LIMIT={Product.RAINFOREST_RESIN:50,Product.KELP:50,Product.SQUID_INK:50,Product.CROISSANT:250,Product.JAMS:350,Product.DJEMBES:60,Product.PICNIC_BASKET1:60,Product.PICNIC_BASKET2:100}
  self.history={p:[]for p in[Product.RAINFOREST_RESIN,Product.KELP,Product.SQUID_INK,Product.CROISSANT,Product.JAMS,Product.DJEMBES,Product.PICNIC_BASKET1,Product.PICNIC_BASKET2]}
  self.round=0;self.cash=0;self.trader_data={}
 def get_mid_price(self,product,state):
  default_price=self.params.get(product,{}).get("fair_value",500)
  if product not in state.order_depths:return default_price
  market_bids=state.order_depths[product].buy_orders
  market_asks=state.order_depths[product].sell_orders
  if not market_bids or not market_asks:return default_price
  return(max(market_bids)+min(market_asks))/2
 def mm_strategy(self,product,state):
  pos=state.position.get(product,0);od=state.order_depths[product];pl=self.PRODUCT_LIMIT[product]
  mid=self.get_mid_price(product,state);self.history[product].append(mid)
  fair_value=sum(self.history[product][-20:])/min(len(self.history[product]),20) if self.history[product] else self.params[product]["fair_value"]
  sp=self.params[product]["spread_pct"];bc=pl-pos;sc=pl+pos;orders=[]
  if od.buy_orders and od.sell_orders:
   bb=max(od.buy_orders.keys());ba=min(od.sell_orders.keys());spread=ba-bb
   if spread>2:
    ob=bb+1;oa=ba-1
    if ob>=oa:ob=bb;oa=ba
    typical_vol=od.buy_orders.get(bb,0)
    order_size=min(max(1,int(typical_vol*0.3) if typical_vol else 5),20)
    bs=min(order_size,bc);as_=min(order_size,sc)
    if bc>0 and ob>0:orders.append(Order(product,ob,bs))
    if sc>0:orders.append(Order(product,oa,-as_))
  if pos>pl*0.3 and od.buy_orders:
   bb=max(od.buy_orders.keys());bv=od.buy_orders[bb];ss=min(abs(pos),abs(bv))
   orders.append(Order(product,bb,-ss))
  elif pos<-pl*0.3 and od.sell_orders:
   ba=min(od.sell_orders.keys());av=od.sell_orders[ba];bs=min(abs(pos),abs(av))
   orders.append(Order(product,ba,bs))
  if len(orders)<2:
   bp=int(fair_value*(1-sp));ap=int(fair_value*(1+sp))
   if not any(o.price==bp and o.quantity>0 for o in orders) and bc>0:orders.append(Order(product,bp,min(5,bc)))
   if not any(o.price==ap and o.quantity<0 for o in orders) and sc>0:orders.append(Order(product,ap,-min(5,sc)))
  print(f"[{product}] pos={pos}, mid={mid:.1f}, fair={fair_value:.1f}, orders={[(o.price,o.quantity) for o in orders]}")
  return orders
 def arbitrage_strategy(self,product,state):
  pos=state.position.get(product,0);od=state.order_depths[product];pl=self.PRODUCT_LIMIT[product]
  mid=self.get_mid_price(product,state);fair_value=sum(self.history[product][-20:])/min(len(self.history[product]),20) if self.history[product] else self.params[product]["fair_value"]
  mp=self.params[product]["mispricing_threshold"];bc=pl-pos;sc=pl+pos;orders=[]
  if od.buy_orders and od.sell_orders:
   bb=max(od.buy_orders.keys());ba=min(od.sell_orders.keys())
   mp_pct=(mid/fair_value-1) if fair_value>0 else 0
   if abs(mp_pct)>mp:
    if mp_pct<0 and bc>0:
     order_size=min(5,bc,od.sell_orders.get(ba,0))
     if order_size>0:orders.append(Order(product,ba,order_size))
    elif mp_pct>0 and sc>0:
     order_size=min(5,sc,od.buy_orders.get(bb,0))
     if order_size>0:orders.append(Order(product,bb,-order_size))
  print(f"[ARB {product}] pos={pos}, mid={mid:.1f}, fair={fair_value:.1f}, mp_pct={mp_pct*100:.2f}%, orders={[(o.price,o.quantity) for o in orders]}")
  return orders
 def artifical_order_depth(self,od:Dict[str,OrderDepth],p1:bool=True):
  if p1:dj,c,j=PICNIC1_WEIGHTS[Product.DJEMBES],PICNIC1_WEIGHTS[Product.CROISSANT],PICNIC1_WEIGHTS[Product.JAMS]
  else:c,j=PICNIC2_WEIGHTS[Product.CROISSANT],PICNIC2_WEIGHTS[Product.JAMS]
  aod=OrderDepth()
  cb=max(od[Product.CROISSANT].buy_orders.keys(),default=0)
  ca=min(od[Product.CROISSANT].sell_orders.keys(),default=float("inf"))
  jb=max(od[Product.JAMS].buy_orders.keys(),default=0)
  ja=min(od[Product.JAMS].sell_orders.keys(),default=float("inf"))
  if p1:
   db=max(od[Product.DJEMBES].buy_orders.keys(),default=0)
   da=min(od[Product.DJEMBES].sell_orders.keys(),default=float("inf"))
   ab=db*dj+cb*c+jb*j
   aa=da*dj+ca*c+ja*j
  else:
   ab=cb*c+jb*j
   aa=ca*c+ja*j
  if ab>0:
   cv=od[Product.CROISSANT].buy_orders.get(cb,0)//c
   jv=od[Product.JAMS].buy_orders.get(jb,0)//j
   if p1:dv=od[Product.DJEMBES].buy_orders.get(db,0)//dj;av=min(dv,cv,jv)
   else:av=min(cv,jv)
   aod.buy_orders[ab]=av
  if aa<float("inf"):
   cv=-od[Product.CROISSANT].sell_orders.get(ca,0)//c
   jv=-od[Product.JAMS].sell_orders.get(ja,0)//j
   if p1:dv=-od[Product.DJEMBES].sell_orders.get(da,0)//dj;av_=min(dv,cv,jv)
   else:av_=min(cv,jv)
   aod.sell_orders[aa]=-av_
  return aod
 def convert_orders(self,ao:List[Order],od:Dict[str,OrderDepth],p1:bool=True):
  co={Product.DJEMBES:[],Product.CROISSANT:[],Product.JAMS:[]}
  if p1:dj,c,j=PICNIC1_WEIGHTS[Product.DJEMBES],PICNIC1_WEIGHTS[Product.CROISSANT],PICNIC1_WEIGHTS[Product.JAMS]
  else:c,j=PICNIC2_WEIGHTS[Product.CROISSANT],PICNIC2_WEIGHTS[Product.JAMS]
  for o in ao:
   q=abs(o.quantity);s=1 if o.quantity>0 else-1
   if p1:
    if s>0:
     db=max(od[Product.DJEMBES].buy_orders.keys(),default=0)
     cb=max(od[Product.CROISSANT].buy_orders.keys(),default=0)
     jb=max(od[Product.JAMS].buy_orders.keys(),default=0)
     if db:co[Product.DJEMBES].append(Order(Product.DJEMBES,db,q*dj*s))
     if cb:co[Product.CROISSANT].append(Order(Product.CROISSANT,cb,q*c*s))
     if jb:co[Product.JAMS].append(Order(Product.JAMS,jb,q*j*s))
    else:
     da=min(od[Product.DJEMBES].sell_orders.keys(),default=float("inf"))
     ca=min(od[Product.CROISSANT].sell_orders.keys(),default=float("inf"))
     ja=min(od[Product.JAMS].sell_orders.keys(),default=float("inf"))
     if da<float("inf"):co[Product.DJEMBES].append(Order(Product.DJEMBES,da,q*dj*s))
     if ca<float("inf"):co[Product.CROISSANT].append(Order(Product.CROISSANT,ca,q*c*s))
     if ja<float("inf"):co[Product.JAMS].append(Order(Product.JAMS,ja,q*j*s))
   else:
    if s>0:
     cb=max(od[Product.CROISSANT].buy_orders.keys(),default=0)
     jb=max(od[Product.JAMS].buy_orders.keys(),default=0)
     if cb:co[Product.CROISSANT].append(Order(Product.CROISSANT,cb,q*c*s))
     if jb:co[Product.JAMS].append(Order(Product.JAMS,jb,q*j*s))
    else:
     ca=min(od[Product.CROISSANT].sell_orders.keys(),default=float("inf"))
     ja=min(od[Product.JAMS].sell_orders.keys(),default=float("inf"))
     if ca<float("inf"):co[Product.CROISSANT].append(Order(Product.CROISSANT,ca,q*c*s))
     if ja<float("inf"):co[Product.JAMS].append(Order(Product.JAMS,ja,q*j*s))
  return co
 def execute_spreads(self,po:List[Order],co:Dict[str,List[Order]],state:TradingState,p1:bool=True):
  od=state.order_depths;orders={Product.DJEMBES:[],Product.CROISSANT:[],Product.JAMS:[]}
  if p1:p=Product.PICNIC_BASKET1
  else:p=Product.PICNIC_BASKET2
  for o in po:
   q=abs(o.quantity);s=1 if o.quantity>0 else-1
   if s>0 and o.price<=min(od[p].sell_orders.keys(),default=float("inf")):
    av=od[p].sell_orders.get(o.price,0)
    eq=min(q,abs(av));orders[p].append(Order(p,o.price,eq*s))
   elif s<0 and o.price>=max(od[p].buy_orders.keys(),default=0):
    av=od[p].buy_orders.get(o.price,0)
    eq=min(q,abs(av));orders[p].append(Order(p,o.price,eq*s))
  for prod in co:
   for o in co[prod]:
    q=abs(o.quantity);s=1 if o.quantity>0 else-1
    if s>0 and o.price<=min(od[prod].sell_orders.keys(),default=float("inf")):
     av=od[prod].sell_orders.get(o.price,0)
     eq=min(q,abs(av));orders[prod].append(Order(prod,o.price,eq*s))
    elif s<0 and o.price>=max(od[prod].buy_orders.keys(),default=0):
     av=od[prod].buy_orders.get(o.price,0)
     eq=min(q,abs(av));orders[prod].append(Order(prod,o.price,eq*s))
  return orders
 def spread_orders(self,state:TradingState,p1:bool=True):
  od=state.order_depths
  if p1:p=Product.PICNIC_BASKET1;sp=Product.SPREAD1
  else:p=Product.PICNIC_BASKET2;sp=Product.SPREAD2
  if p not in od or Product.CROISSANT not in od or Product.JAMS not in od or (p1 and Product.DJEMBES not in od):return{}
  pos=state.position.get(p,0);aod=self.artifical_order_depth(od,p1)
  sm=self.params[sp]["default_spread_mean"];ss=self.params[sp]["default_spread_std"]
  sw=self.params[sp]["spread_window"];zt=self.params[sp]["zscore_threshold"]
  spreads=[self.history[p][-i]-self.history[Product.ARTIFICAL1][-i] for i in range(1,min(sw,len(self.history[p]))+1) if len(self.history[Product.ARTIFICAL1])>=i]
  ms=sum(spreads)/len(spreads) if spreads else sm
  sd=np.std(spreads) if len(spreads)>1 else ss
  pb=max(od[p].buy_orders.keys(),default=0);pa=min(od[p].sell_orders.keys(),default=float("inf"))
  ab=max(aod.buy_orders.keys(),default=0);aa=min(aod.sell_orders.keys(),default=float("inf"))
  orders={p:[]}
  if ab>0 and aa<float("inf"):
   spr=pb-aa if pb>0 else float("inf")
   z=(spr-ms)/sd if sd>0 else 0
   if z>zt and pos<self.params[sp]["target_position"]:
    q=min(5,self.PRODUCT_LIMIT[p]-pos,abs(aod.sell_orders.get(aa,0)))
    if q>0:orders[p].append(Order(p,pb,q))
   spr_=pa-ab if pa<float("inf") else float("inf")
   z_=(spr_-ms)/sd if sd>0 else 0
   if z_<zt and pos>-self.PRODUCT_LIMIT[p]:
    q=min(5,self.PRODUCT_LIMIT[p]+pos,abs(aod.buy_orders.get(ab,0)))
    if q>0:orders[p].append(Order(p,pa,-q))
  co=self.convert_orders(orders[p],od,p1)
  orders.update(self.execute_spreads(orders[p],co,state,p1))
  return orders
 def run(self,state:TradingState):
  self.round+=1;self.cash=0
  for p in state.own_trades:
   for t in state.own_trades[p]:
    if t.buyer=="SUBMISSION":self.cash-=t.quantity*t.price
    if t.seller=="SUBMISSION":self.cash+=t.quantity*t.price
  result={}
  try:result[Product.RAINFOREST_RESIN]=self.mm_strategy(Product.RAINFOREST_RESIN,state)+self.arbitrage_strategy(Product.RAINFOREST_RESIN,state)
  except Exception as e:print(f"Error RESIN: {e}")
  try:result[Product.KELP]=self.mm_strategy(Product.KELP,state)+self.arbitrage_strategy(Product.KELP,state)
  except Exception as e:print(f"Error KELP: {e}")
  try:result[Product.SQUID_INK]=self.mm_strategy(Product.SQUID_INK,state)+self.arbitrage_strategy(Product.SQUID_INK,state)
  except Exception as e:print(f"Error INK: {e}")
  try:result.update(self.spread_orders(state,True))
  except Exception as e:print(f"Error PICNIC1: {e}")
  try:result.update(self.spread_orders(state,False))
  except Exception as e:print(f"Error PICNIC2: {e}")
  td=jsonpickle.encode(self.trader_data)
  logger.flush(state,result,0,td)
  return result,0,td