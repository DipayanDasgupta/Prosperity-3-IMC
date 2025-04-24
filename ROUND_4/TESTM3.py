from datamodel import OrderDepth, TradingState, Order
import jsonpickle
from typing import List, Dict
import statistics

class Trader:
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0

        # Decode traderData
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)
        else:
            trader_data = {
                "buy_price_history": [],
                "position_history": {"MAGNIFICENT_MACARONS": []},
                "acquisition_price": 0,
                "fee_history": []
            }

        product = "MAGNIFICENT_MACARONS"
        order_depth: OrderDepth = state.order_depths.get(product, OrderDepth())
        position = state.position.get(product, 0)
        position_limit = 75
        conversion_limit = 10

        # Get ConversionObservation
        conv_obs = state.observations.conversionObservations.get(product)
        if not conv_obs:
            return result, conversions, jsonpickle.encode(trader_data)

        bid_price = conv_obs.bidPrice
        ask_price = conv_obs.askPrice
        transport_fees = conv_obs.transportFees
        import_tariff = conv_obs.importTariff
        export_tariff = conv_obs.exportTariff

        # Calculate effective prices
        buy_price = ask_price + transport_fees + import_tariff
        sell_price_conv = bid_price - transport_fees - export_tariff
        trader_data["buy_price_history"].append(buy_price)

        # Log fees
        trader_data["fee_history"].append({
            "timestamp": state.timestamp,
            "transport_fees": transport_fees,
            "import_tariff": import_tariff,
            "export_tariff": export_tariff
        })

        # Local market analysis
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        bid_quantity = order_depth.buy_orders.get(best_bid, 0)

        orders: List[Order] = []

        # Conversion: Import to close short position
        recent_prices = trader_data["buy_price_history"][-20:] if len(trader_data["buy_price_history"]) >= 20 else trader_data["buy_price_history"]
        price_stable = buy_price <= (statistics.mean(recent_prices) + statistics.stdev(recent_prices)) if len(recent_prices) > 1 else True
        if position < 0 and trader_data.get("last_sell_price", float('inf')) > buy_price + 0.5 and price_stable and best_bid > buy_price + 1.0:
            conversions = min(-position, conversion_limit, bid_quantity)
            trader_data["acquisition_price"] = buy_price
            print(f"CONVERSION REQUEST: Import {conversions} units at {buy_price}")

        # Short selling
        max_short = min(bid_quantity, position_limit + position, 10)
        if max_short > 0 and bid_quantity >= 5:  # Ensure sufficient liquidity
            base_sell_price = max(int(bid_price - 0.5), int(buy_price + 1.5))
            sell_price = best_bid if best_bid >= base_sell_price else base_sell_price
            if sell_price > best_bid + 0.5:
                sell_price = int(best_bid + 0.5) if best_bid > 0 else base_sell_price
            quantity = -max_short
            orders.append(Order(product, sell_price, quantity))
            trader_data["last_sell_price"] = sell_price
            print(f"SELL {product} {-quantity}x at {sell_price}")

        # Avoid long positions
        if position > 0:
            acquisition_price = trader_data.get("acquisition_price", buy_price)
            storage_cost = position * 0.1
            if best_bid >= acquisition_price + storage_cost + 0.5:
                quantity = -position
                orders.append(Order(product, best_bid, quantity))
                print(f"SELL {product} {-quantity}x at {best_bid} to avoid storage")
            elif best_bid == 0 and sell_price_conv > acquisition_price + storage_cost + 0.5:
                conversions = -min(position, conversion_limit)
                print(f"CONVERSION REQUEST: Export {conversions} units at {sell_price_conv}")

        # Log state
        print(f"Timestamp: {state.timestamp}, Position: {position}, Best Bid: {best_bid}, Buy Price: {buy_price}, Conversions: {conversions}, Orders: {orders}")

        trader_data["position_history"][product].append(position)
        result[product] = orders
        trader_data_str = jsonpickle.encode(trader_data)

        return result, conversions, trader_data_str