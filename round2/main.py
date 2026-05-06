from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState
)
from typing import Any
import json


class Trader:
    POSITION_LIMIT = 80
    OSM_FAIR = 10000  # Fixed fair value for osmium

    def __init__(self):
        # Pepper trend state
        self.pepper_mids = []
        self.pepper_ts = []
        self.pepper_init_price = None
        self.pepper_init_ts = None
        self.pepper_peak_mid = None

    def bid(self) -> int:
        return 3000

    def _slope(self, values):
        if len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        num = 0.0
        den = 0.0
        for i, p in enumerate(values):
            dx = i - x_mean
            num += dx * (p - y_mean)
            den += dx * dx
        return num / den if den != 0 else 0.0

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
                self.pepper_peak_mid = td.get("pepper_peak_mid", None)
            except Exception:
                pass

        for product in state.order_depths:
            if product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_osmium(state, product)
            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.trade_pepper(state, product)

        # Persist state
        self.pepper_mids = self.pepper_mids[-50:]
        self.pepper_ts = self.pepper_ts[-50:]
        trader_data = json.dumps({
            "pepper_mids": self.pepper_mids,
            "pepper_ts": self.pepper_ts,
            "pepper_init_price": self.pepper_init_price,
            "pepper_init_ts": self.pepper_init_ts,
            "pepper_peak_mid": self.pepper_peak_mid,
        })

        return result, 0, trader_data

    # ─────────────────────────────────────────────────────────────
    # ASH_COATED_OSMIUM — tight fixed-fair mean reversion
    # ─────────────────────────────────────────────────────────────
    def trade_osmium(self, state, product):
        orders = []
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMIT
        fair = self.OSM_FAIR

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # Take obvious edge aggressively
        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price <= fair - 2 and buy_capacity > 0:
                    avail = abs(od.sell_orders[ask_price])
                    take = min(avail, buy_capacity)
                    if take > 0:
                        orders.append(Order(product, ask_price, take))
                        buy_capacity -= take
                else:
                    break

        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price >= fair + 2 and sell_capacity > 0:
                    avail = od.buy_orders[bid_price]
                    take = min(avail, sell_capacity)
                    if take > 0:
                        orders.append(Order(product, bid_price, -take))
                        sell_capacity -= take
                else:
                    break

        # Passive penny-in quoting
        if best_bid is None:
            best_bid = fair - 5
        if best_ask is None:
            best_ask = fair + 5

        our_bid = min(best_bid + 1, fair - 1)
        our_ask = max(best_ask - 1, fair + 1)

        # Mild inventory skew
        if pos > 40:
            our_bid = min(our_bid, fair - 2)
            our_ask = min(our_ask, fair + 1)
        elif pos > 60:
            our_bid = min(our_bid, fair - 3)
        elif pos < -40:
            our_ask = max(our_ask, fair + 2)
            our_bid = max(our_bid, fair - 1)
        elif pos < -60:
            our_ask = max(our_ask, fair + 3)

        if buy_capacity > 0:
            orders.append(Order(product, our_bid, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(product, our_ask, -sell_capacity))

        return orders

    # ─────────────────────────────────────────────────────────────
    # INTARIAN_PEPPER_ROOT — breakout + trailing-hold trend logic
    # ─────────────────────────────────────────────────────────────
    def trade_pepper(self, state, product):
        orders = []
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMIT
        ts = state.timestamp

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is None and best_ask is None:
            return orders

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
        else:
            mid = best_bid if best_bid is not None else best_ask

        if self.pepper_init_price is None:
            self.pepper_init_price = mid
            self.pepper_init_ts = ts

        self.pepper_mids.append(mid)
        self.pepper_ts.append(ts)
        self.pepper_mids = self.pepper_mids[-50:]
        self.pepper_ts = self.pepper_ts[-50:]

        recent = self.pepper_mids[-10:] if len(self.pepper_mids) >= 10 else self.pepper_mids[:]
        slope = self._slope(recent)
        recent_high = max(recent) if recent else mid

        if self.pepper_peak_mid is None:
            self.pepper_peak_mid = mid
        else:
            self.pepper_peak_mid = max(self.pepper_peak_mid, mid)

        drawdown_from_peak = self.pepper_peak_mid - mid

        # Trend fair: extrapolate from recent slope
        if len(self.pepper_mids) >= 10:
            fair = round(recent[-1] + 6 * slope + 1)
        else:
            fair = round(mid + 1)

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # Strong breakout buying: follow strength and keep pushing long
        buy_cutoff = fair + 6 if slope > 0 else fair + 4
        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price <= buy_cutoff:
                    avail = abs(od.sell_orders[ask_price])
                    take = min(avail, buy_capacity)
                    if take > 0:
                        orders.append(Order(product, ask_price, take))
                        buy_capacity -= take
                else:
                    break

        # If trend is strong, slam into remaining capacity
        if buy_capacity > 0 and best_ask is not None and len(recent) >= 5:
            if slope > 0.35 and mid >= recent_high - 1:
                take = buy_capacity
                orders.append(Order(product, best_ask, take))
                buy_capacity = 0

        # Trailing-stop style selling: only after a real pullback from peak
        if od.buy_orders and pos > 60:
            excess = max(0, pos - 65)
            if excess > 0 and (drawdown_from_peak >= 3 or slope < 0):
                for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_price >= fair + 10:
                        avail = od.buy_orders[bid_price]
                        take = min(avail, sell_capacity, excess)
                        if take > 0:
                            orders.append(Order(product, bid_price, -take))
                            sell_capacity -= take
                            excess -= take
                    else:
                        break

        # Passive buy remains aggressive
        if buy_capacity > 0 and best_bid is not None:
            our_bid = min(best_bid + 1, fair + 3)
            orders.append(Order(product, our_bid, buy_capacity))

        # Passive sell only when heavily extended
        excess_inv = max(0, pos - 75)
        if excess_inv > 0 and best_ask is not None:
            if drawdown_from_peak >= 4 or slope < 0:
                our_ask = max(best_ask - 1, fair + 10)
                sell_size = min(excess_inv, sell_capacity)
                if sell_size > 0:
                    orders.append(Order(product, our_ask, -sell_size))

        return orders