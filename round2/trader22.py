"""
Prosperity Round 2 — Trading Algorithm V2 (Aggressive Market Making)
=====================================================================
Key improvements over V1:
1. Osmium: Uses fixed fair value of 10,000 (extremely tight mean reversion)
2. Penny-in quoting — always post INSIDE the market for queue priority
3. Much more aggressive liquidity taking
4. Pepper root: buys aggressively AND market-makes within the trend
5. Proper position management with dynamic sizing

Target: 150-200k+ PnL per day
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState
)
from typing import Any
import json


class Trader:

    POSITION_LIMIT = 80
    OSM_FAIR = 10000  # Fixed fair value — osmium mean-reverts tightly

    def __init__(self):
        # Rolling state for pepper root trend model
        self.pepper_mids = []    # recent mid prices
        self.pepper_ts = []      # corresponding timestamps
        self.pepper_init_price = None
        self.pepper_init_ts = None

    def bid(self) -> int:
        """Market Access Fee bid. 3000 balances cost vs top-50% likelihood."""
        return 3000

    def run(self, state: TradingState):
        result = {}

        # Restore persisted state
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                self.pepper_mids = td.get("pepper_mids", [])
                self.pepper_ts = td.get("pepper_ts", [])
                self.pepper_init_price = td.get("pepper_init_price", None)
                self.pepper_init_ts = td.get("pepper_init_ts", None)
            except Exception:
                pass

        for product in state.order_depths:
            if product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_osmium(state, product)
            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.trade_pepper(state, product)

        # Persist state (keep memory bounded)
        self.pepper_mids = self.pepper_mids[-50:]
        self.pepper_ts = self.pepper_ts[-50:]
        trader_data = json.dumps({
            "pepper_mids": self.pepper_mids,
            "pepper_ts": self.pepper_ts,
            "pepper_init_price": self.pepper_init_price,
            "pepper_init_ts": self.pepper_init_ts,
        })

        return result, 0, trader_data

    # ─────────────────────────────────────────────────────────────
    # ASH_COATED_OSMIUM — Tight mean-reversion market making
    # ─────────────────────────────────────────────────────────────
    def trade_osmium(self, state, product):
        orders = []
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        fair = self.OSM_FAIR
        limit = self.POSITION_LIMIT

        # Track capacity consumed by active orders
        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # ── PHASE 1: AGGRESSIVE LIQUIDITY TAKING ──
        # Snap up any ask at or below fair - 1 (we're buying below fair = profit)
        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price <= fair - 1 and buy_capacity > 0:
                    avail = abs(od.sell_orders[ask_price])
                    take = min(avail, buy_capacity)
                    if take > 0:
                        orders.append(Order(product, ask_price, take))
                        buy_capacity -= take
                else:
                    break

        # Hit any bid at or above fair + 1 (we're selling above fair = profit)
        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price >= fair + 1 and sell_capacity > 0:
                    avail = od.buy_orders[bid_price]
                    take = min(avail, sell_capacity)
                    if take > 0:
                        orders.append(Order(product, bid_price, -take))
                        sell_capacity -= take
                else:
                    break

        # ── PHASE 2: PENNY-IN PASSIVE QUOTES ──
        # Determine best bid/ask (AFTER removing orders we just took)
        remaining_bids = {p: v for p, v in od.buy_orders.items() if p < fair + 1}
        remaining_asks = {p: v for p, v in od.sell_orders.items() if p > fair - 1}

        best_bid = max(remaining_bids.keys()) if remaining_bids else fair - 5
        best_ask = min(remaining_asks.keys()) if remaining_asks else fair + 5

        # Post quotes 1 tick inside the market, but never cross fair value
        our_bid = min(best_bid + 1, fair - 1)   # up to 9999
        our_ask = max(best_ask - 1, fair + 1)   # down to 10001

        # Inventory skew: if we're already long, be less eager to buy more
        if pos > 30:
            our_bid = min(our_bid, fair - 2)   # 9998
            our_ask = min(our_ask, fair + 1)   # more eager to sell
        elif pos > 50:
            our_bid = min(our_bid, fair - 3)   # 9997
        elif pos < -30:
            our_ask = max(our_ask, fair + 2)   # 10002
            our_bid = max(our_bid, fair - 1)   # more eager to buy
        elif pos < -50:
            our_ask = max(our_ask, fair + 3)   # 10003

        # Place quotes with full available capacity
        if buy_capacity > 0:
            orders.append(Order(product, our_bid, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(product, our_ask, -sell_capacity))

        return orders

    # ─────────────────────────────────────────────────────────────
    # INTARIAN_PEPPER_ROOT — Trend-following + market-making
    # ─────────────────────────────────────────────────────────────
    def trade_pepper(self, state, product):
        orders = []
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMIT
        ts = state.timestamp

        # Compute current mid
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is None and best_ask is None:
            return orders

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
        else:
            mid = best_bid if best_bid is not None else best_ask

        # Initialize trend model on first observation
        if self.pepper_init_price is None:
            self.pepper_init_price = mid
            self.pepper_init_ts = ts

        self.pepper_mids.append(mid)
        self.pepper_ts.append(ts)

        # ── Estimate fair value using trend model ──
        # Linear trend assumption: ~100 points per 100k timestamps (day 1 data)
        # But start conservative and let actual prices drive estimate
        if len(self.pepper_mids) >= 3:
            # Use recent mid as fair, slightly biased upward (trend)
            recent_avg = sum(self.pepper_mids[-5:]) / min(5, len(self.pepper_mids))
            fair = recent_avg + 1   # small upward bias reflecting trend
        else:
            fair = mid

        fair = round(fair)

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # ── PHASE 1: AGGRESSIVE BUYING ──
        # Since price trends up, ANY ask at or below fair+2 is a buy opportunity
        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                # Accept paying up to 3 ticks above current fair (trend will close the gap)
                if ask_price <= fair + 3 and buy_capacity > 0:
                    avail = abs(od.sell_orders[ask_price])
                    take = min(avail, buy_capacity)
                    if take > 0:
                        orders.append(Order(product, ask_price, take))
                        buy_capacity -= take
                else:
                    break

        # ── PHASE 2: SELECTIVE SELLING (only at premium) ──
        # Only sell if someone is bidding 6+ ticks above fair (unusually high)
        if od.buy_orders and pos > 20:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price >= fair + 6 and sell_capacity > 0:
                    avail = od.buy_orders[bid_price]
                    # Only sell excess inventory (keep base long position)
                    excess = max(0, pos - 40)
                    take = min(avail, sell_capacity, excess)
                    if take > 0:
                        orders.append(Order(product, bid_price, -take))
                        sell_capacity -= take
                        pos -= take
                else:
                    break

        # ── PHASE 3: PASSIVE PENNY-IN QUOTES ──
        # Aggressive buy quote to fill regularly
        if buy_capacity > 0 and best_bid is not None:
            our_bid = min(best_bid + 1, fair + 1)   # willing to pay fair+1
            orders.append(Order(product, our_bid, buy_capacity))

        # Sell quote only if we have excess inventory beyond base position
        excess_inv = max(0, pos - 50)
        if excess_inv > 0 and best_ask is not None:
            our_ask = max(best_ask - 1, fair + 5)
            sell_size = min(excess_inv, sell_capacity)
            if sell_size > 0:
                orders.append(Order(product, our_ask, -sell_size))

        return orders