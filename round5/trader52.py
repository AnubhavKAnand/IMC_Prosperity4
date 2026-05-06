from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import math

class Trader:
    def __init__(self):
        # Memory to store the Exponential Moving Average (EMA) for fair value
        self.emas: Dict[str, float] = {}
        # Fast alpha to track current market reality closely
        self.alpha = 0.2

    def run(self, state: TradingState):
        """
        Pure Market Making Strategy.
        Provides liquidity on all 50 goods to constantly harvest the Bid-Ask spread.
        """
        result = {}
        
        # We can hold a max of 10 for any product
        POSITION_LIMIT = 10
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Skip if the order book is empty
            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue
                
            # Find the absolute edges of the current market
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0
            
            # 1. Update our Fair Value estimate
            if product not in self.emas:
                self.emas[product] = mid_price
            else:
                self.emas[product] = self.alpha * mid_price + (1 - self.alpha) * self.emas[product]
                
            fair_value = self.emas[product]
            
            # 2. Inventory Risk Management (Skew)
            current_position = state.position.get(product, 0)
            
            # If we are long (positive pos), we drop our target price to encourage selling.
            # If we are short (negative pos), we raise our target price to encourage buying.
            inventory_skew = current_position * 0.5
            adjusted_fair = fair_value - inventory_skew
            
            # 3. Determine our ideal quote prices
            # We demand a margin of at least 2 ticks from our fair value
            margin = 2
            my_bid = math.floor(adjusted_fair - margin)
            my_ask = math.ceil(adjusted_fair + margin)
            
            # 4. Penny the Market (Competitive Quoting)
            # We want to be at the top of the order book to get filled, so we improve 
            # the current market by 1 tick if our margin allows it.
            if my_bid > best_bid:
                my_bid = best_bid + 1
            if my_ask < best_ask:
                my_ask = best_ask - 1
                
            # 5. Strict Liquidity Provision Safeguard
            # NEVER cross the spread. We want to BE the spread.
            my_bid = min(my_bid, best_ask - 1)
            my_ask = max(my_ask, best_bid + 1)
            
            # Ensure our own quotes don't overlap
            if my_bid >= my_ask:
                my_bid = my_ask - 1
            
            # 6. Calculate how much we can legally trade
            buy_qty = POSITION_LIMIT - current_position
            sell_qty = -POSITION_LIMIT - current_position
            
            # 7. Post the Limit Orders to the Exchange
            if buy_qty > 0:
                orders.append(Order(product, my_bid, buy_qty))
            if sell_qty < 0:
                orders.append(Order(product, my_ask, sell_qty))
                
            result[product] = orders
            
        conversions = 0
        traderData = "SPREAD_HARVESTER_MARKET_MAKER"
        
        return result, conversions, traderData