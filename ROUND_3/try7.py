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
 ARTIFICAL2="ARTIFICAL2"
 SPREAD1="SPREAD1"
 SPREAD2="SPREAD2"

PARAMS={
 Product.RAINFOREST_RESIN:{"fair_value":10000,"take_width":1,"clear_width":0,"disregard_edge":1,"join_edge":2,"default_edge":1,"soft_position_limit":50},
 Product.KELP:{"take_width":2,"clear_width":0,"prevent_adverse":False,"adverse_volume":15,"reversion_beta":-0.18,"disregard_edge":2,"join_edge":0,"default_edge":1,"ink_adjustment_factor":0.05},
 Product.SQUID_INK:{"take_width":2,"clear_width":1,"prevent_adverse":False,"adverse_volume":15,"reversion_beta":-0.228,"disregard_edge":2,"join_edge":0,"default_edge":1,"spike_lb":3,"spike_ub":5.6,"offset":2,"reversion_window":55,"reversion_weight":0.12},
 Product.SPREAD1:{"default_spread_mean":48.777856,"default_spread_std":85.119723,"spread_window":55,"zscore_threshold":4,"target_position":100},
 Product.SPREAD2:{"default_spread_mean":30.2336,"default_spread_std":59.8536,"spread_window":59,"zscore_threshold":6,"target_position":100},
 Product.PICNIC_BASKET1:{"b2_adjustment_factor":0.05}
}

PICNIC1_WEIGHTS={Product.DJEMBES:1,Product.CROISSANT:6,Product.JAMS:3}
PICNIC2_WEIGHTS={Product.CROISSANT:4,Product.JAMS:2}

class Trader:
 def __init__(self,params=None):
  self.params=params or PARAMS
  self.PRODUCT_LIMIT={Product.RAINFOREST_RESIN:50,Product.KELP:50,Product.SQUID_INK:50,Product.CROISSANT:250,Product.JAMS:350,Product.DJEMBES:60,Product.PICNIC_BASKET1:60,Product.PICNIC_BASKET2:100}
 def take_best_orders(self,product:str,fair_value:float,take_width:float,orders:List[Order],order_depth:OrderDepth,position:int,buy_order_volume:int,sell_order_volume:int,prevent_adverse:bool=False,adverse_volume:int=0,traderObject:dict=None):
  position_limit=self.PRODUCT_LIMIT[product]
  if product=="SQUID_INK":
   if "currentSpike" not in traderObject:traderObject["currentSpike"]=False
   prev_price=traderObject.get("ink_last_price",fair_value)
   if traderObject["currentSpike"]:
    if abs(fair_value-prev_price)<self.params[Product.SQUID_INK]["spike_lb"]:traderObject["currentSpike"]=False
    else:
     if fair_value<traderObject["recoveryValue"]:
      best_ask=min(order_depth.sell_orders.keys())
      best_ask_amount=order_depth.sell_orders[best_ask]
      quantity=min(abs(best_ask_amount),position_limit-position)
      if quantity>0:orders.append(Order(product,best_ask,quantity));buy_order_volume+=quantity;order_depth.sell_orders[best_ask]+=quantity
      if order_depth.sell_orders[best_ask]==0:del order_depth.sell_orders[best_ask]
      return buy_order_volume,0
     else:
      best_bid=max(order_depth.buy_orders.keys())
      best_bid_amount=order_depth.buy_orders[best_bid]
      quantity=min(best_bid_amount,position_limit+position)
      if quantity>0:orders.append(Order(product,best_bid,-quantity));sell_order_volume+=quantity;order_depth.buy_orders[best_bid]-=quantity
      if order_depth.buy_orders[best_bid]==0:del order_depth.buy_orders[best_bid]
      return 0,sell_order_volume
   if abs(fair_value-prev_price)>self.params[Product.SQUID_INK]["spike_ub"]:
    traderObject["currentSpike"]=True
    traderObject["recoveryValue"]=prev_price+self.params[Product.SQUID_INK]["offset"] if fair_value>prev_price else prev_price-self.params[Product.SQUID_INK]["offset"]
    if fair_value>prev_price:
     best_bid=max(order_depth.buy_orders.keys())
     best_bid_amount=order_depth.buy_orders[best_bid]
     quantity=min(best_bid_amount,position_limit+position)
     if quantity>0:orders.append(Order(product,best_bid,-quantity));sell_order_volume+=quantity;order_depth.buy_orders[best_bid]-=quantity
     if order_depth.buy_orders[best_bid]==0:del order_depth.buy_orders[best_bid]
     return 0,sell_order_volume
    else:
     best_ask=min(order_depth.sell_orders.keys())
     best_ask_amount=order_depth.sell_orders[best_ask]
     quantity=min(abs(best_ask_amount),position_limit-position)
     if quantity>0:orders.append(Order(product,best_ask,quantity));buy_order_volume+=quantity;order_depth.sell_orders[best_ask]+=quantity
     if order_depth.sell_orders[best_ask]==0:del order_depth.sell_orders[best_ask]
     return buy_order_volume,0
  if len(order_depth.sell_orders)!=0:
   best_ask=min(order_depth.sell_orders.keys())
   best_ask_amount=-order_depth.sell_orders[best_ask]
   if not prevent_adverse or abs(best_ask_amount)<=adverse_volume:
    if best_ask<=fair_value-take_width:
     quantity=min(best_ask_amount,position_limit-position)
     if quantity>0:orders.append(Order(product,best_ask,quantity));buy_order_volume+=quantity;order_depth.sell_orders[best_ask]+=quantity
     if order_depth.sell_orders[best_ask]==0:del order_depth.sell_orders[best_ask]
  if len(order_depth.buy_orders)!=0:
   best_bid=max(order_depth.buy_orders.keys())
   best_bid_amount=order_depth.buy_orders[best_bid]
   if not prevent_adverse or abs(best_bid_amount)<=adverse_volume:
    if best_bid>=fair_value+take_width:
     quantity=min(best_bid_amount,position_limit+position)
     if quantity>0:orders.append(Order(product,best_bid,-quantity));sell_order_volume+=quantity;order_depth.buy_orders[best_bid]-=quantity
     if order_depth.buy_orders[best_bid]==0:del order_depth.buy_orders[best_bid]
  return buy_order_volume,sell_order_volume
 def market_make(self,product:str,orders:List[Order],bid:int,ask:int,position:int,buy_order_volume:int,sell_order_volume:int):
  buy_quantity=self.PRODUCT_LIMIT[product]-(position+buy_order_volume)
  if buy_quantity>0:orders.append(Order(product,math.floor(bid),buy_quantity))
  sell_quantity=self.PRODUCT_LIMIT[product]+(position-sell_order_volume)
  if sell_quantity>0:orders.append(Order(product,math.ceil(ask),-sell_quantity))
  return buy_order_volume,sell_order_volume
 def clear_position_order(self,product:str,fair_value:float,width:int,orders:List[Order],order_depth:OrderDepth,position:int,buy_order_volume:int,sell_order_volume:int):
  position_after_take=position+buy_order_volume-sell_order_volume
  fair_for_bid=round(fair_value-width)
  fair_for_ask=round(fair_value+width)
  buy_quantity=self.PRODUCT_LIMIT[product]-(position+buy_order_volume)
  sell_quantity=self.PRODUCT_LIMIT[product]+(position-sell_order_volume)
  if position_after_take>0:
   clear_quantity=sum(volume for price,volume in order_depth.buy_orders.items()if price>=fair_for_ask)
   clear_quantity=min(clear_quantity,position_after_take)
   sent_quantity=min(sell_quantity,clear_quantity)
   if sent_quantity>0:orders.append(Order(product,fair_for_ask,-abs(sent_quantity)));sell_order_volume+=abs(sent_quantity)
  if position_after_take<0:
   clear_quantity=sum(abs(volume)for price,volume in order_depth.sell_orders.items()if price<=fair_for_bid)
   clear_quantity=min(clear_quantity,abs(position_after_take))
   sent_quantity=min(buy_quantity,clear_quantity)
   if sent_quantity>0:orders.append(Order(product,fair_for_bid,abs(sent_quantity)));buy_order_volume+=abs(sent_quantity)
  return buy_order_volume,sell_order_volume
 def kelp_fair_value(self,order_depth:OrderDepth,traderObject,ink_order_depth:OrderDepth):
  if len(order_depth.sell_orders)!=0 and len(order_depth.buy_orders)!=0:
   best_ask=min(order_depth.sell_orders.keys())
   best_bid=max(order_depth.buy_orders.keys())
   valid_ask=[price for price in order_depth.sell_orders.keys()if abs(order_depth.sell_orders[price])>=self.params[Product.KELP]["adverse_volume"]]
   valid_buy=[price for price in order_depth.buy_orders.keys()if abs(order_depth.buy_orders[price])>=self.params[Product.KELP]["adverse_volume"]]
   mm_ask=min(valid_ask)if len(valid_ask)>0 else None
   mm_bid=max(valid_buy)if len(valid_buy)>0 else None
   if valid_ask and valid_buy:mmmid_price=(mm_ask+mm_bid)/2
   else:mmmid_price=(best_ask+best_bid)/2 if traderObject.get('kelp_last_price',None)==None else traderObject['kelp_last_price']
   fair=mmmid_price if traderObject.get('kelp_last_price',None)is None else mmmid_price+(mmmid_price*((mmmid_price-traderObject["kelp_last_price"])/traderObject["kelp_last_price"]*self.params[Product.KELP]["reversion_beta"]))
   if traderObject.get("ink_last_price",None)is not None:
    old_ink_price=traderObject["ink_last_price"]
    valid_ask_ink=[price for price in ink_order_depth.sell_orders.keys()if abs(ink_order_depth.sell_orders[price])>=self.params[Product.SQUID_INK]["adverse_volume"]]
    valid_buy_ink=[price for price in ink_order_depth.buy_orders.keys()if abs(ink_order_depth.buy_orders[price])>=self.params[Product.SQUID_INK]["adverse_volume"]]
    new_ink_mid=(min(valid_ask_ink)+max(valid_buy_ink))/2 if valid_ask_ink and valid_buy_ink else (min(ink_order_depth.sell_orders.keys())+max(ink_order_depth.buy_orders.keys()))/2
    ink_return=(new_ink_mid-old_ink_price)/old_ink_price
    fair=fair-(self.params[Product.KELP].get("ink_adjustment_factor",0.5)*ink_return*mmmid_price)
   traderObject["kelp_last_price"]=mmmid_price
   return fair
  return None
 def ink_fair_value(self,order_depth:OrderDepth,traderObject):
  if len(order_depth.sell_orders)!=0 and len(order_depth.buy_orders)!=0:
   best_ask=min(order_depth.sell_orders.keys())
   best_bid=max(order_depth.buy_orders.keys())
   valid_ask=[price for price in order_depth.sell_orders.keys()if abs(order_depth.sell_orders[price])>=self.params[Product.SQUID_INK]["adverse_volume"]]
   valid_buy=[price for price in order_depth.buy_orders.keys()if abs(order_depth.buy_orders[price])>=self.params[Product.SQUID_INK]["adverse_volume"]]
   mm_ask=min(valid_ask)if len(valid_ask)>0 else None
   mm_bid=max(valid_buy)if len(valid_buy)>0 else None
   if valid_ask and valid_buy:mmmid_price=(mm_ask+mm_bid)/2
   else:mmmid_price=(best_ask+best_bid)/2 if traderObject.get('ink_last_price',None)==None else traderObject['ink_last_price']
   if traderObject.get('ink_price_history',None)is None:traderObject['ink_price_history']=[]
   traderObject['ink_price_history'].append(mmmid_price)
   if len(traderObject['ink_price_history'])>self.params[Product.SQUID_INK]["reversion_window"]:traderObject['ink_price_history']=traderObject['ink_price_history'][-self.params[Product.SQUID_INK]["reversion_window"]:]
   if len(traderObject['ink_price_history'])>=self.params[Product.SQUID_INK]["reversion_window"]:
    prices=np.array(traderObject['ink_price_history'])
    returns=(prices[1:]-prices[:-1])/prices[:-1]
    X=returns[:-1]
    Y=returns[1:]
    estimated_beta=-np.dot(X,Y)/np.dot(X,X)if np.dot(X,X)!=0 else self.params[Product.SQUID_INK]["reversion_beta"]
    adaptive_beta=(self.params[Product.SQUID_INK]['reversion_weight']*estimated_beta+(1-self.params[Product.SQUID_INK]['reversion_weight'])*self.params[Product.SQUID_INK]["reversion_beta"])
   else:adaptive_beta=self.params[Product.SQUID_INK]["reversion_beta"]
   fair=mmmid_price if traderObject.get('ink_last_price',None)is None else mmmid_price+(mmmid_price*((mmmid_price-traderObject["ink_last_price"])/traderObject["ink_last_price"]*adaptive_beta))
   traderObject["ink_last_price"]=mmmid_price
   return fair
  return None
 def take_orders(self,product:str,order_depth:OrderDepth,fair_value:float,take_width:float,position:int,prevent_adverse:bool=False,adverse_volume:int=0,traderObject:dict=None):
  orders=[]
  buy_order_volume,sell_order_volume=self.take_best_orders(product,fair_value,take_width,orders,order_depth,position,0,0,prevent_adverse,adverse_volume,traderObject)
  return orders,buy_order_volume,sell_order_volume
 def clear_orders(self,product:str,order_depth:OrderDepth,fair_value:float,clear_width:int,position:int,buy_order_volume:int,sell_order_volume:int):
  orders=[]
  buy_order_volume,sell_order_volume=self.clear_position_order(product,fair_value,clear_width,orders,order_depth,position,buy_order_volume,sell_order_volume)
  return orders,buy_order_volume,sell_order_volume
 def make_orders(self,product,order_depth:OrderDepth,fair_value:float,position:int,buy_order_volume:int,sell_order_volume:int,disregard_edge:float,join_edge:float,default_edge:float,manage_position:bool=False,soft_position_limit:int=0):
  adjustment=0
  if product==Product.RAINFOREST_RESIN:
   total_buy_volume=sum(order_depth.buy_orders.values())if order_depth.buy_orders else 0
   total_sell_volume=sum(abs(v)for v in order_depth.sell_orders.values())if order_depth.sell_orders else 0
   total_volume=total_buy_volume+total_sell_volume if(total_buy_volume+total_sell_volume)>0 else 1
   imbalance_ratio=(total_buy_volume-total_sell_volume)/total_volume
   adjustment=round(4.0*imbalance_ratio)
  orders=[]
  asks_above_fair=[price for price in order_depth.sell_orders.keys()if price>fair_value+disregard_edge]
  bids_below_fair=[price for price in order_depth.buy_orders.keys()if price<fair_value-disregard_edge]
  best_ask_above_fair=min(asks_above_fair)if len(asks_above_fair)>0 else None
  best_bid_below_fair=max(bids_below_fair)if len(bids_below_fair)>0 else None
  ask=round(fair_value+default_edge)
  if best_ask_above_fair:
   if abs(best_ask_above_fair-fair_value)<=join_edge:ask=best_ask_above_fair+1
   else:ask=best_ask_above_fair
  bid=round(fair_value-default_edge)
  if best_bid_below_fair:
   if abs(fair_value-best_bid_below_fair)<=join_edge:bid=best_bid_below_fair
   else:bid=best_bid_below_fair+1
  if manage_position:
   if position>soft_position_limit:ask-=1
   elif position<-soft_position_limit:bid+=1
  buy_order_volume,sell_order_volume=self.market_make(product,orders,bid,ask,position,buy_order_volume,sell_order_volume)
  return orders,buy_order_volume,sell_order_volume
 def artifical_order_depth(self,order_depths:Dict[str,OrderDepth],picnic1:bool=True):
  if picnic1:DJEMBES_PER_PICNIC,CROISSANT_PER_PICNIC,JAM_PER_PICNIC=PICNIC1_WEIGHTS[Product.DJEMBES],PICNIC1_WEIGHTS[Product.CROISSANT],PICNIC1_WEIGHTS[Product.JAMS]
  else:CROISSANT_PER_PICNIC,JAM_PER_PICNIC=PICNIC2_WEIGHTS[Product.CROISSANT],PICNIC2_WEIGHTS[Product.JAMS]
  artifical_order_price=OrderDepth()
  croissant_best_bid=max(order_depths[Product.CROISSANT].buy_orders.keys())if order_depths[Product.CROISSANT].buy_orders else 0
  croissant_best_ask=min(order_depths[Product.CROISSANT].sell_orders.keys())if order_depths[Product.CROISSANT].sell_orders else float("inf")
  jams_best_bid=max(order_depths[Product.JAMS].buy_orders.keys())if order_depths[Product.JAMS].buy_orders else 0
  jams_best_ask=min(order_depths[Product.JAMS].sell_orders.keys())if order_depths[Product.JAMS].sell_orders else float("inf")
  if picnic1:
   djembes_best_bid=max(order_depths[Product.DJEMBES].buy_orders.keys())if order_depths[Product.DJEMBES].buy_orders else 0
   djembes_best_ask=min(order_depths[Product.DJEMBES].sell_orders.keys())if order_depths[Product.DJEMBES].sell_orders else float("inf")
   art_bid=djembes_best_bid*DJEMBES_PER_PICNIC+croissant_best_bid*CROISSANT_PER_PICNIC+jams_best_bid*JAM_PER_PICNIC
   art_ask=djembes_best_ask*DJEMBES_PER_PICNIC+croissant_best_ask*CROISSANT_PER_PICNIC+jams_best_ask*JAM_PER_PICNIC
  else:
   art_bid=croissant_best_bid*CROISSANT_PER_PICNIC+jams_best_bid*JAM_PER_PICNIC
   art_ask=croissant_best_ask*CROISSANT_PER_PICNIC+jams_best_ask*JAM_PER_PICNIC
  if art_bid>0:
   croissant_bid_volume=order_depths[Product.CROISSANT].buy_orders[croissant_best_bid]//CROISSANT_PER_PICNIC
   jams_bid_volume=order_depths[Product.JAMS].buy_orders[jams_best_bid]//JAM_PER_PICNIC
   if picnic1:djembes_bid_volume=order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]//DJEMBES_PER_PICNIC;artifical_bid_volume=min(djembes_bid_volume,croissant_bid_volume,jams_bid_volume)
   else:artifical_bid_volume=min(croissant_bid_volume,jams_bid_volume)
   artifical_order_price.buy_orders[art_bid]=artifical_bid_volume
  if art_ask<float("inf"):
   croissant_ask_volume=-order_depths[Product.CROISSANT].sell_orders[croissant_best_ask]//CROISSANT_PER_PICNIC
   jams_ask_volume=-order_depths[Product.JAMS].sell_orders[jams_best_ask]//JAM_PER_PICNIC
   if picnic1:djembes_ask_volume=-order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]//DJEMBES_PER_PICNIC;artifical_ask_volume=min(djembes_ask_volume,croissant_ask_volume,jams_ask_volume)
   else:artifical_ask_volume=min(croissant_ask_volume,jams_ask_volume)
   artifical_order_price.sell_orders[art_ask]=-artifical_ask_volume
  return artifical_order_price
 def convert_orders(self,artifical_orders:List[Order],order_depths:Dict[str,OrderDepth],picnic1:bool=True):
  component_orders={Product.DJEMBES:[],Product.CROISSANT:[],Product.JAMS:[]}if picnic1 else{Product.CROISSANT:[],Product.JAMS:[]}
  artfical_order_depth=self.artifical_order_depth(order_depths,picnic1)
  best_bid=max(artfical_order_depth.buy_orders.keys())if artfical_order_depth.buy_orders else 0
  best_ask=min(artfical_order_depth.sell_orders.keys())if artfical_order_depth.sell_orders else float("inf")
  for order in artifical_orders:
   price=order.price;quantity=order.quantity
   if quantity>0 and price>=best_ask:
    croissant_price=min(order_depths[Product.CROISSANT].sell_orders.keys())
    jams_price=min(order_depths[Product.JAMS].sell_orders.keys())
    if picnic1:djembes_price=min(order_depths[Product.DJEMBES].sell_orders.keys())
   elif quantity<0 and price<=best_bid:
    croissant_price=max(order_depths[Product.CROISSANT].buy_orders.keys())
    jams_price=max(order_depths[Product.JAMS].buy_orders.keys())
    if picnic1:djembes_price=max(order_depths[Product.DJEMBES].buy_orders.keys())
   else:continue
   croissaint_order=Order(Product.CROISSANT,croissant_price,quantity*PICNIC1_WEIGHTS[Product.CROISSANT]if picnic1 else quantity*PICNIC2_WEIGHTS[Product.CROISSANT])
   jams_order=Order(Product.JAMS,jams_price,quantity*PICNIC1_WEIGHTS[Product.JAMS]if picnic1 else quantity*PICNIC2_WEIGHTS[Product.JAMS])
   if picnic1:component_orders[Product.DJEMBES].append(Order(Product.DJEMBES,djembes_price,quantity*PICNIC1_WEIGHTS[Product.DJEMBES]))
   component_orders[Product.CROISSANT].append(croissaint_order);component_orders[Product.JAMS].append(jams_order)
  return component_orders
 def execute_spreads(self,target_position:int,picnic_position:int,order_depths:Dict[str,OrderDepth],picnic1:bool=True):
  if target_position==picnic_position:return None
  target_quantity=abs(target_position-picnic_position)
  picnic_order_depth=order_depths[Product.PICNIC_BASKET1]if picnic1 else order_depths[Product.PICNIC_BASKET2]
  artifical_order_depth=self.artifical_order_depth(order_depths,picnic1)
  if target_position>picnic_position:
   picnic_ask_price=min(picnic_order_depth.sell_orders.keys())
   picnic_ask_vol=abs(picnic_order_depth.sell_orders[picnic_ask_price])
   artifical_bid_price=max(artifical_order_depth.buy_orders.keys())
   artifical_bid_vol=abs(artifical_order_depth.buy_orders[artifical_bid_price])
   orderbook_volume=min(picnic_ask_vol,artifical_bid_vol)
   execute_volume=min(orderbook_volume,target_quantity)
   picnic_orders=[Order(Product.PICNIC_BASKET1,picnic_ask_price,execute_volume)if picnic1 else Order(Product.PICNIC_BASKET2,picnic_ask_price,execute_volume)]
   artifical_orders=[Order(Product.ARTIFICAL1,artifical_bid_price,-execute_volume)]
   aggregate_orders=self.convert_orders(artifical_orders,order_depths,picnic1)
   if picnic1:aggregate_orders[Product.PICNIC_BASKET1]=picnic_orders
   else:aggregate_orders[Product.PICNIC_BASKET2]=picnic_orders
   return aggregate_orders
  else:
   picnic_bid_price=max(picnic_order_depth.buy_orders.keys())
   picnic_bid_vol=abs(picnic_order_depth.buy_orders[picnic_bid_price])
   artifical_ask_price=min(artifical_order_depth.sell_orders.keys())
   artifical_ask_vol=abs(artifical_order_depth.sell_orders[artifical_ask_price])
   orderbook_volume=min(picnic_bid_vol,artifical_ask_vol)
   execute_volume=min(orderbook_volume,target_quantity)
   picnic_orders=[Order(Product.PICNIC_BASKET1,picnic_bid_price,-execute_volume)if picnic1 else Order(Product.PICNIC_BASKET2,picnic_bid_price,-execute_volume)]
   artifical_orders=[Order(Product.ARTIFICAL1,artifical_ask_price,execute_volume)]
   aggregate_orders=self.convert_orders(artifical_orders,order_depths,picnic1)
   if picnic1:aggregate_orders[Product.PICNIC_BASKET1]=picnic_orders
   else:aggregate_orders[Product.PICNIC_BASKET2]=picnic_orders
   return aggregate_orders
 def spread_orders(self,order_depths:Dict[str,OrderDepth],product:Product,picnic_position:int,spread_data:Dict[str,object],SPREAD,picnic1:bool=True):
  if Product.PICNIC_BASKET1 not in order_depths.keys()or Product.PICNIC_BASKET2 not in order_depths.keys():return None
  picnic_order_depth=order_depths[Product.PICNIC_BASKET1]if picnic1 else order_depths[Product.PICNIC_BASKET2]
  artifical_order_depth=self.artifical_order_depth(order_depths,picnic1)
  best_bid=picnic_order_depth.buy_orders and max(picnic_order_depth.buy_orders.keys())
  best_bid_vol=abs(picnic_order_depth.buy_orders[best_bid])if best_bid else 0
  best_ask=picnic_order_depth.sell_orders and min(picnic_order_depth.sell_orders.keys())
  best_ask_vol=abs(picnic_order_depth.sell_orders[best_ask])if best_ask else 0
  picnic_mprice=(best_bid*best_ask_vol+best_ask*best_bid_vol)/(best_bid_vol+best_ask_vol)if best_bid_vol+best_ask_vol>0 else 0
  best_bid=artifical_order_depth.buy_orders and max(artifical_order_depth.buy_orders.keys())
  best_bid_vol=abs(artifical_order_depth.buy_orders[best_bid])if best_bid else 0
  best_ask=artifical_order_depth.sell_orders and min(artifical_order_depth.sell_orders.keys())
  best_ask_vol=abs(artifical_order_depth.sell_orders[best_ask])if best_ask else 0
  artifical_mprice=(best_bid*best_ask_vol+best_ask*best_bid_vol)/(best_bid_vol+best_ask_vol)if best_bid_vol+best_ask_vol>0 else 0
  spread=picnic_mprice-artifical_mprice
  spread_data["spread_history"].append(spread)
  if len(spread_data["spread_history"])<self.params[SPREAD]["spread_window"]:return None
  elif len(spread_data["spread_history"])>self.params[SPREAD]["spread_window"]:spread_data["spread_history"].pop(0)
  spread_std=np.std(spread_data["spread_history"])
  zscore=(spread-self.params[SPREAD]["default_spread_mean"])/spread_std
  if zscore>=self.params[SPREAD]["zscore_threshold"]:
   if picnic_position!=-self.params[SPREAD]["target_position"]:return self.execute_spreads(-self.params[SPREAD]["target_position"],picnic_position,order_depths,picnic1)
  if zscore<=-self.params[SPREAD]["zscore_threshold"]:
   if picnic_position!=self.params[SPREAD]["target_position"]:return self.execute_spreads(self.params[SPREAD]["target_position"],picnic_position,order_depths,picnic1)
  spread_data["prev_zscore"]=zscore
  return None
 def trade_resin(self,state):
  product="RAINFOREST_RESIN"
  end_pos=state.position.get(product,0)
  buy_sum=50-end_pos
  sell_sum=50+end_pos
  orders=[]
  order_depth=state.order_depths[product]
  bids=order_depth.buy_orders
  asks=order_depth.sell_orders
  bid_prices=list(bids.keys())
  bid_volumes=list(bids.values())
  ask_prices=list(asks.keys())
  ask_volumes=list(asks.values())
  if sell_sum>0:
   for i in range(0,len(bid_prices)):
    if bid_prices[i]>10000:
     fill=min(bid_volumes[i],sell_sum)
     orders.append(Order(product,bid_prices[i],-fill))
     sell_sum-=fill;end_pos-=fill;bid_volumes[i]-=fill
  bid_prices,bid_volumes=zip(*[(ai,bi)for ai,bi in zip(bid_prices,bid_volumes)if bi!=0])
  bid_prices=list(bid_prices);bid_volumes=list(bid_volumes)
  if buy_sum>0:
   for i in range(0,len(ask_prices)):
    if ask_prices[i]<10000:
     fill=min(-ask_volumes[i],buy_sum)
     orders.append(Order(product,ask_prices[i],fill))
     buy_sum-=fill;end_pos+=fill;ask_volumes[i]+=fill
  ask_prices,ask_volumes=zip(*[(ai,bi)for ai,bi in zip(ask_prices,ask_volumes)if bi!=0])
  ask_prices=list(ask_prices);ask_volumes=list(ask_volumes)
  orders.append(Order(product,max(ask_prices[0]-1,10000+1),-min(14,sell_sum))if abs(ask_volumes[0])>1 else Order(product,max(10000+1,ask_prices[0]),-min(14,sell_sum)))
  sell_sum-=min(14,sell_sum)
  orders.append(Order(product,min(bid_prices[0]+1,10000-1),min(14,buy_sum))if bid_volumes[0]>1 else Order(product,min(10000-1,bid_prices[0]),min(14,buy_sum)))
  buy_sum-=min(14,buy_sum)
  if end_pos>0:
   for i in range(0,len(bid_prices)):
    if bid_prices[i]==10000:
     fill=min(bid_volumes[i],sell_sum)
     orders.append(Order(product,bid_prices[i],-fill))
     sell_sum-=fill;end_pos-=fill
  if end_pos<0:
   for i in range(0,len(ask_prices)):
    if ask_prices[i]==10000:
     fill=min(-ask_volumes[i],buy_sum)
     orders.append(Order(product,ask_prices[i],fill))
     buy_sum-=fill;end_pos+=fill
  return orders
 def run(self,state:TradingState):
  traderObject={}
  result={}
  end_pos=state.position.get("RAINFOREST_RESIN",0)
  order_depth=state.order_depths["RAINFOREST_RESIN"]
  bids,asks=order_depth.buy_orders,order_depth.sell_orders
  mm_bid=max(bids.items(),key=lambda tup:tup[1])[0]
  mm_ask=min(asks.items(),key=lambda tup:tup[1])[0]
  if state.traderData!=None and state.traderData!="":traderObject=jsonpickle.decode(state.traderData)
  result["RAINFOREST_RESIN"]=self.trade_resin(state)
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
  traderData=jsonpickle.encode(traderObject)
  conversions=1
  logger.flush(state,result,conversions,traderData)
  return result,conversions,traderData