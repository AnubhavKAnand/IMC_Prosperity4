from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import math

class Trader:
    def __init__(self):
        # Memory to store the Exponential Moving Average (EMA) for baseline fair value
        self.emas: Dict[str, float] = {}
        # Alpha controls the speed of the EMA
        self.alpha = 0.2

    def run(self, state: TradingState):
        """
        Microprice & Dynamic Spread Market Maker.
        """
        result = {}
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
            
            # Volume at the best bid and ask (used for Microprice calculation)
            bid_vol = order_depth.buy_orders[best_bid]
            ask_vol = -order_depth.sell_orders[best_ask] # Negative integer in datamodel
            
            total_vol = bid_vol + ask_vol
            
            if total_vol == 0:
                continue
            
            # 1. Microprice Imbalance Calculation
            # If bid_vol is huge, imbalance approaches 1.0 (Upward pressure)
            # If ask_vol is huge, imbalance approaches 0.0 (Downward pressure)
            imbalance = bid_vol / total_vol
            
            # Standard mid price
            mid_price = (best_bid + best_ask) / 2.0
            
            # 2. Update EMA Baseline
            if product not in self.emas:
                self.emas[product] = mid_price
            else:
                self.emas[product] = self.alpha * mid_price + (1 - self.alpha) * self.emas[product]
                
            # 3. Compute our ultra-accurate Fair Value
            # We blend our EMA with the Microprice Imbalance to front-run the trend
            # (imbalance - 0.5) shifts the price up or down by max 1.5 ticks based on volume pressure
            adjusted_fair_value = self.emas[product] + ((imbalance - 0.5) * 3)
            
            # 4. Inventory Risk Skew
            current_position = state.position.get(product, 0)
            inventory_skew = current_position * 0.6  # Slightly more aggressive dumping than last time
            skewed_fair_value = adjusted_fair_value - inventory_skew
            
            # 5. Dynamic Margin Requirement
            # Calculate the current width of the market spread
            current_spread = best_ask - best_bid
            
            # Demand roughly 1/3rd of the spread as our profit margin, minimum 1 tick
            dynamic_margin = max(1.0, current_spread / 3.0)
            
            my_bid = math.floor(skewed_fair_value - dynamic_margin)
            my_ask = math.ceil(skewed_fair_value + dynamic_margin)
            
            # 6. Pennying & Strict Liquidity Provision
            if my_bid > best_bid:
                my_bid = best_bid + 1
            if my_ask < best_ask:
                my_ask = best_ask - 1
                
            my_bid = min(my_bid, best_ask - 1)
            my_ask = max(my_ask, best_bid + 1)
            
            if my_bid >= my_ask:
                my_bid = my_ask - 1
            
            # 7. Asymmetric Toxic Flow Shutoff
            buy_qty = POSITION_LIMIT - current_position
            sell_qty = -POSITION_LIMIT - current_position
            
            # Only quote a bid if we actually have room to hold the inventory.
            # If we are at +10, buy_qty is 0, so the Order is skipped. No falling knife traps!
            if buy_qty > 0:
                orders.append(Order(product, my_bid, buy_qty))
                
            if sell_qty < 0:
                orders.append(Order(product, my_ask, sell_qty))
                
            result[product] = orders
            
        conversions = 0
        traderData = "FLOW_TOXIC_MM_V3"
        
        return result, conversions, traderData