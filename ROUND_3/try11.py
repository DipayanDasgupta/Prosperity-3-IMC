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
 RAINFOREST_RESIN="RAINFOREST_RESIN";KELP="KELP";SQUID_INK="SQUID_INK";PICNIC_BASKET1="PICNIC_BASKET1";DJEMBES="DJEMBES";CROISSANT="CROISSANTS";JAMS="JAMS";PICNIC_BASKET2="PICNIC_BASKET2"
 ARTIFICAL1="ARTIFICAL1";SPREAD1="SPREAD1";SPREAD2="SPREAD2"
 VOLCANIC_ROCK="VOLCANIC_ROCK";V_VOUCHER_9500="VOLCANIC_ROCK_VOUCHER_9500";V_VOUCHER_9750="VOLCANIC_ROCK_VOUCHER_9750";V_VOUCHER_10000="VOLCANIC_ROCK_VOUCHER_10000"
 V_VOUCHER_10250="VOLCANIC_ROCK_VOUCHER_10250";V_VOUCHER_10500="VOLCANIC_ROCK_VOUCHER_10500"

PARAMS={
 Product.RAINFOREST_RESIN:{"fair_value":10000,"take_width":1,"clear_width":0,"disregard_edge":1,"join_edge":2,"default_edge":1,"soft_position_limit":50},
 Product.KELP:{"take_width":2,"clear_width":0,"prevent_adverse":False,"adverse_volume":15,"reversion_beta":-0.18,"disregard_edge":2,"join_edge":0,"default_edge":1,"ink_adjustment_factor":0.05},
 Product.SQUID_INK:{"take_width":2,"clear_width":1,"prevent_adverse":False,"adverse_volume":15,"reversion_beta":-0.228,"disregard_edge":2,"join_edge":0,"default_edge":1,"spike_lb":3,"spike_ub":5.6,"offset":2,"reversion_window":55,"reversion_weight":0.12},
 Product.SPREAD1:{"default_spread_mean":48.777856,"default_spread_std":85.119723,"spread_window":55,"zscore_threshold":4,"target_position":100},
 Product.SPREAD2:{"default_spread_mean":30.2336,"default_spread_std":59.8536,"spread_window":59,"zscore_threshold":6,"target_position":100},
 Product.PICNIC_BASKET1:{"b2_adjustment_factor":0.05},
 Product.VOLCANIC_ROCK:{"fair_value":10414,"spread_base":1.5,"iv_factor":0.05,"max_position_pct":0.7},
 Product.V_VOUCHER_9500:{"strike":9500,"avg_iv":0.2774,"iv_threshold":0.29,"time_value":0.32},
 Product.V_VOUCHER_9750:{"strike":9750,"avg_iv":0.2936,"iv_threshold":0.30,"time_value":1.33},
 Product.V_VOUCHER_10000:{"strike":10000,"avg_iv":0.2696,"iv_threshold":0.27,"time_value":10.96},
 Product.V_VOUCHER_10250:{"strike":10250,"avg_iv":0.2471,"iv_threshold":0.245,"time_value":34.53},
 Product.V_VOUCHER_10500:{"strike":10500,"avg_iv":0.2449,"iv_threshold":0.245,"time_value":39.06}
}

PICNIC1_WEIGHTS={Product.DJEMBES:1,Product.CROISSANT:6,Product.JAMS:3}
PICNIC2_WEIGHTS={Product.CROISSANT:4,Product.JAMS:2}

class Trader:
 def __init__(self):
  self.params=PARAMS
  self.PRODUCT_LIMIT={Product.RAINFOREST_RESIN:50,Product.KELP:50,Product.SQUID_INK:50,Product.CROISSANT:250,Product.JAMS:350,Product.DJEMBES:60,Product.PICNIC_BASKET1:60,Product.PICNIC_BASKET2:100,Product.VOLCANIC_ROCK:50,Product.V_VOUCHER_9500:50,Product.V_VOUCHER_9750:50,Product.V_VOUCHER_10000:50,Product.V_VOUCHER_10250:50,Product.V_VOUCHER_10500:50}
  self.history={p:[] for p in [Product.VOLCANIC_ROCK,Product.V_VOUCHER_9500,Product.V_VOUCHER_9750,Product.V_VOUCHER_10000,Product.V_VOUCHER_10250,Product.V_VOUCHER_10500]}
 def get_mid_price(self,product,state):
  default_price=self.params.get(product,{}).get("fair_value",self.params.get(product,{}).get("strike",500))
  if product not in state.order_depths:return default_price
  od=state.order_depths[product];market_bids=od.buy_orders;market_asks=od.sell_orders
  if not market_bids or not market_asks:return default_price
  return(max(market_bids.keys())+min(market_asks.keys()))/2
 def volcanic_mm_strategy(self,product,state):
  pos=state.position.get(product,0);od=state.order_depths[product];pl=self.PRODUCT_LIMIT[product]
  mid=self.get_mid_price(product,state);self.history[product].append(mid)
  fair_value=sum(self.history[product][-20:])/min(len(self.history[product]),20) if self.history[product] else self.params[product]["fair_value"]
  time_idx=state.timestamp/1e6;iv_adj=0.20+0.10*(time_idx/2) if time_idx>1.5 else 0.20
  spread=self.params[product]["spread_base"]*(1+self.params[product]["iv_factor"]*iv_adj)
  bc=pl-pos;sc=pl+pos;orders=[]
  if od.buy_orders and od.sell_orders:
   bb=max(od.buy_orders.keys());ba=min(od.sell_orders.keys());spread_market=ba-bb
   if spread_market>spread:
    ob=bb+1;oa=ba-1
    if ob>=oa:ob=bb;oa=ba
    typical_vol=od.buy_orders.get(bb,0)
    order_size=min(max(1,int(typical_vol*0.3) if typical_vol else 5),10)
    bs=min(order_size,bc);as_=min(order_size,sc)
    if bc>0 and ob>0:orders.append(Order(product,ob,bs))
    if sc>0:orders.append(Order(product,oa,-as_))
  if pos>pl*0.7 and od.buy_orders:
   bb=max(od.buy_orders.keys());bv=od.buy_orders[bb];ss=min(abs(pos),abs(bv))
   orders.append(Order(product,bb,-ss))
  elif pos<-pl*0.7 and od.sell_orders:
   ba=min(od.sell_orders.keys());av=od.sell_orders[ba];bs=min(abs(pos),abs(av))
   orders.append(Order(product,ba,bs))
  if len(orders)<2:
   bp=int(fair_value-spread);ap=int(fair_value+spread)
   if not any(o.price==bp and o.quantity>0 for o in orders) and bc>0:orders.append(Order(product,bp,min(5,bc)))
   if not any(o.price==ap and o.quantity<0 for o in orders) and sc>0:orders.append(Order(product,ap,-min(5,sc)))
  print(f"[{product}] pos={pos}, mid={mid:.1f}, fair={fair_value:.1f}, orders={[(o.price,o.quantity) for o in orders]}")
  return orders
 def voucher_arbitrage_strategy(self,voucher,state):
  pos=state.position.get(voucher,0);od=state.order_depths[voucher];pl=self.PRODUCT_LIMIT[voucher]
  rock_mid=self.get_mid_price(Product.VOLCANIC_ROCK,state);mid=self.get_mid_price(voucher,state)
  strike=self.params[voucher]["strike"];time_value=self.params[voucher]["time_value"]
  intrinsic_value=max(0,rock_mid-strike);theoretical_price=intrinsic_value+time_value
  mispricing_threshold=0.02;bc=pl-pos;sc=pl+pos;orders=[]
  if od.buy_orders and od.sell_orders:
   bb=max(od.buy_orders.keys());ba=min(od.sell_orders.keys())
   mp_pct=(mid/theoretical_price-1) if theoretical_price>0 else 0
   if mp_pct<-mispricing_threshold and bc>0:
    order_size=min(10,bc,od.sell_orders.get(ba,0))
    if order_size>0:orders.append(Order(voucher,ba,order_size))
   elif mp_pct>mispricing_threshold and sc>0:
    order_size=min(10,sc,od.buy_orders.get(bb,0))
    if order_size>0:orders.append(Order(voucher,bb,-order_size))
  print(f"[ARB {voucher}] pos={pos}, mid={mid:.1f}, theo={theoretical_price:.1f}, mp_pct={mp_pct*100:.2f}%, orders={[(o.price,o.quantity) for o in orders]}")
  return orders
 def voucher_vol_smile_strategy(self,voucher,state):
  pos=state.position.get(voucher,0);od=state.order_depths[voucher];pl=self.PRODUCT_LIMIT[voucher]
  mid=self.get_mid_price(voucher,state);self.history[voucher].append(mid)
  avg_iv=self.params[voucher]["avg_iv"];iv_threshold=self.params[voucher]["iv_threshold"]
  time_idx=state.timestamp/1e6;current_iv=avg_iv*(1+0.5*(time_idx/2-1.5)) if time_idx>1.5 else avg_iv
  bc=pl-pos;sc=pl+pos;orders=[]
  if od.buy_orders and od.sell_orders:
   bb=max(od.buy_orders.keys());ba=min(od.sell_orders.keys())
   if current_iv>iv_threshold and sc>0:
    order_size=min(10,sc,od.buy_orders.get(bb,0))
    if order_size>0:orders.append(Order(voucher,bb,-order_size))
   elif current_iv<iv_threshold and bc>0:
    order_size=min(10,bc,od.sell_orders.get(ba,0))
    if order_size>0:orders.append(Order(voucher,ba,order_size))
  print(f"[VOL {voucher}] pos={pos}, mid={mid:.1f}, iv={current_iv:.4f}, thresh={iv_threshold:.4f}, orders={[(o.price,o.quantity) for o in orders]}")
  return orders
 def take_best_orders(self,product,fair_value,take_width,orders,order_depth,position,buy_order_volume,sell_order_volume,prevent_adverse=False,adverse_volume=0,traderObject=None):pass
 def market_make(self,product,orders,bid,ask,position,buy_order_volume,sell_order_volume):pass
 def clear_position_order(self,product,fair_value,width,orders,order_depth,position,buy_order_volume,sell_order_volume):pass
 def kelp_fair_value(self,order_depth,traderObject,ink_order_depth):pass
 def ink_fair_value(self,order_depth,traderObject):pass
 def take_orders(self,product,order_depth,fair_value,take_width,position,prevent_adverse=False,adverse_volume=0,traderObject=None):pass
 def clear_orders(self,product,order_depth,fair_value,clear_width,position,buy_order_volume,sell_order_volume):pass
 def make_orders(self,product,order_depth,fair_value,position,buy_order_volume,sell_order_volume,disregard_edge,join_edge,default_edge,manage_position=False,soft_position_limit=0):pass
 def artifical_order_depth(self,order_depths,picnic1=True):pass
 def convert_orders(self,artifical_orders,order_depths,picnic1=True):pass
 def execute_spreads(self,target_position,picnic_position,order_depths,picnic1=True):pass
 def spread_orders(self,order_depths,product,picnic_position,spread_data,SPREAD,picnic1=True):pass
 def trade_resin(self,state):pass
 def run(self,state:TradingState):
  traderObject={}
  result={}
  if state.traderData and state.traderData!="":traderObject=jsonpickle.decode(state.traderData)
  result[Product.RAINFOREST_RESIN]=self.trade_resin(state)
  if Product.KELP in self.params and Product.KELP in state.order_depths:
   kelp_position=state.position.get(Product.KELP,0)
   kelp_fair_value=self.kelp_fair_value(state.order_depths[Product.KELP],traderObject,state.order_depths[Product.SQUID_INK])
   kelp_take_orders,buy_order_volume,sell_order_volume=self.take_orders(Product.KELP,state.order_depths[Product.KELP],kelp_fair_value,self.params[Product.KELP]['take_width'],kelp_position,self.params[Product.KELP]['prevent_adverse'],self.params[Product.KELP]['adverse_volume'],traderObject)
   kelp_clear_orders,buy_order_volume,sell_order_volume=self.clear_orders(Product.KELP,state.order_depths[Product.KELP],kelp_fair_value,self.params[Product.KELP]['clear_width'],kelp_position,buy_order_volume,sell_order_volume)
   kelp_make_orders,_,_=self.make_orders(Product.KELP,state.order_depths[Product.KELP],kelp_fair_value,kelp_position,buy_order_volume,sell_order_volume,self.params[Product.KELP]['disregard_edge'],self.params[Product.KELP]['join_edge'],self.params[Product.KELP]['default_edge'])
   result[Product.KELP]=kelp_take_orders+kelp_clear_orders+kelp_make_orders
  if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
   ink_position=state.position.get(Product.SQUID_INK,0)
   ink_fair_value=self.ink_fair_value(state.order_depths[Product.SQUID_INK],traderObject)
   ink_take_orders,buy_order_volume,sell_order_volume=self.take_orders(Product.SQUID_INK,state.order_depths[Product.SQUID_INK],ink_fair_value,self.params[Product.SQUID_INK]['take_width'],ink_position,self.params[Product.SQUID_INK]['prevent_adverse'],self.params[Product.SQUID_INK]['adverse_volume'],traderObject)
   ink_clear_orders,buy_order_volume,sell_order_volume=self.clear_orders(Product.SQUID_INK,state.order_depths[Product.SQUID_INK],ink_fair_value,self.params[Product.SQUID_INK]['clear_width'],ink_position,buy_order_volume,sell_order_volume)
   ink_make_orders,_,_=self.make_orders(Product.SQUID_INK,state.order_depths[Product.SQUID_INK],ink_fair_value,ink_position,buy_order_volume,sell_order_volume,self.params[Product.SQUID_INK]['disregard_edge'],self.params[Product.SQUID_INK]['join_edge'],self.params[Product.SQUID_INK]['default_edge'])
   result[Product.SQUID_INK]=ink_take_orders+ink_clear_orders+ink_make_orders
  if Product.SPREAD1 not in traderObject:traderObject[Product.SPREAD1]={"spread_history":[],"prev_zscore":0,"clear_flag":False,"curr_avg":0}
  picnic1_position=state.position.get(Product.PICNIC_BASKET1,0)
  spread1_orders=self.spread_orders(state.order_depths,Product.PICNIC_BASKET1,picnic1_position,traderObject[Product.SPREAD1],SPREAD=Product.SPREAD1,picnic1=True)
  if spread1_orders:result[Product.DJEMBES]=spread1_orders[Product.DJEMBES];result[Product.CROISSANT]=spread1_orders[Product.CROISSANT];result[Product.JAMS]=spread1_orders[Product.JAMS];result[Product.PICNIC_BASKET1]=spread1_orders[Product.PICNIC_BASKET1]
  if Product.SPREAD2 not in traderObject:traderObject[Product.SPREAD2]={"spread_history":[],"prev_zscore":0,"clear_flag":False,"curr_avg":0}
  picnic2_position=state.position.get(Product.PICNIC_BASKET2,0)
  spread2_orders=self.spread_orders(state.order_depths,Product.PICNIC_BASKET2,picnic2_position,traderObject[Product.SPREAD2],SPREAD=Product.SPREAD2,picnic1=False)
  if spread2_orders:result[Product.CROISSANT]=spread2_orders[Product.CROISSANT];result[Product.JAMS]=spread2_orders[Product.JAMS];result[Product.PICNIC_BASKET2]=spread2_orders[Product.PICNIC_BASKET2]
  if Product.VOLCANIC_ROCK in state.order_depths:
   result[Product.VOLCANIC_ROCK]=self.volcanic_mm_strategy(Product.VOLCANIC_ROCK,state)
  for voucher in [Product.V_VOUCHER_9500,Product.V_VOUCHER_9750,Product.V_VOUCHER_10000,Product.V_VOUCHER_10250,Product.V_VOUCHER_10500]:
   if voucher in state.order_depths:
    result[voucher]=self.voucher_arbitrage_strategy(voucher,state)+self.voucher_vol_smile_strategy(voucher,state)
  traderData=jsonpickle.encode(traderObject)
  conversions=1
  logger.flush(state,result,conversions,traderData)
  return result,conversions,traderData