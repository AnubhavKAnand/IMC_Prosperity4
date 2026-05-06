from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle


class Trader:
    """
    Round 2 trader — improved after live run feedback.

    Key fixes vs previous version:
      - OSMIUM fair is now a dynamic EMA of mid (training mean 10000 was wrong
        for the live sim where mid hovered around 10004–10008).
      - OSMIUM make quotes are clipped to book — no more quoting deep inside a
        wide spread which got adversely selected.
      - PEPPER: keeps aggressive long ramp, but now sells a slice when price
        spikes meaningfully above trend (was trend+20 — never triggered).
      - PEPPER: drift assumption lowered (0.05 vs 0.1) so the trend tracker
        doesn't over-lead price in slow-drift days; still long-biased overall.
    """

    POSITION_LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    # ---- Osmium ----
    OSM_FAIR_INIT  = 10000
    OSM_FAIR_ALPHA = 0.02      # slow EMA on mid — tracks level without chasing noise
    OSM_SKEW_THRESH = 30       # start skewing quotes beyond ±30 position
    OSM_EDGE       = 1         # only take/quote when we have at least 1 tick of edge

    # ---- Pepper ----
    PEP_DRIFT        = 0.05    # conservative — live drift can be slower than training
    PEP_TREND_ALPHA  = 0.08    # moderate EMA
    PEP_BUY_PREMIUM  = 5       # pay up to trend+5 to accumulate fast
    PEP_SELL_SPIKE   = 10      # lowered from 20 — harvest real mean-reversion
    PEP_KEEP_MIN     = 60      # don't sell below this position (preserve drift exposure)

    # ---- MAF bid ----
    def bid(self) -> int:
        return 299

    # ---- main ----
    def run(self, state: TradingState):
        osm_fair = self.OSM_FAIR_INIT
        pep_trend = None
        if state.traderData:
            try:
                d = jsonpickle.decode(state.traderData)
                osm_fair  = d.get("osm_fair",  self.OSM_FAIR_INIT)
                pep_trend = d.get("pep_trend", None)
            except Exception:
                pass

        result: Dict[str, List[Order]] = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            orders, osm_fair = self._trade_osmium(state, osm_fair)
            result["ASH_COATED_OSMIUM"] = orders

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            orders, pep_trend = self._trade_pepper(state, pep_trend)
            result["INTARIAN_PEPPER_ROOT"] = orders

        td = jsonpickle.encode({"osm_fair": osm_fair, "pep_trend": pep_trend})
        return result, 0, td

    # -------- OSMIUM (dynamic fair, book-aware MM) --------
    def _trade_osmium(self, state: TradingState, fair: float):
        product = "ASH_COATED_OSMIUM"
        depth = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMIT[product]
        orders: List[Order] = []

        if not depth.buy_orders or not depth.sell_orders:
            return orders, fair

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        # Update fair via slow EMA of mid — tracks the true level of osmium
        # which in live runs sat around 10003–10008, not exactly 10000.
        fair = (1 - self.OSM_FAIR_ALPHA) * fair + self.OSM_FAIR_ALPHA * mid

        buy_cap = limit - pos
        sell_cap = limit + pos

        # TAKE: require strict edge vs dynamic fair
        take_buy_px  = fair - self.OSM_EDGE
        take_sell_px = fair + self.OSM_EDGE

        for ask_px, ask_vol in sorted(depth.sell_orders.items()):
            if buy_cap <= 0:
                break
            if ask_px <= take_buy_px:
                q = min(-ask_vol, buy_cap)
                orders.append(Order(product, int(ask_px), q))
                buy_cap -= q
            # flatten-short free trade at fair (±0.5)
            elif abs(ask_px - fair) < 0.5 and pos < 0:
                q = min(-ask_vol, -pos, buy_cap)
                if q > 0:
                    orders.append(Order(product, int(ask_px), q))
                    buy_cap -= q

        for bid_px, bid_vol in sorted(depth.buy_orders.items(), reverse=True):
            if sell_cap <= 0:
                break
            if bid_px >= take_sell_px:
                q = min(bid_vol, sell_cap)
                orders.append(Order(product, int(bid_px), -q))
                sell_cap -= q
            elif abs(bid_px - fair) < 0.5 and pos > 0:
                q = min(bid_vol, pos, sell_cap)
                if q > 0:
                    orders.append(Order(product, int(bid_px), -q))
                    sell_cap -= q

        # MAKE: respect the book — don't quote deep inside a wide spread.
        # If market spread is tight, join one tick inside; otherwise sit near fair.
        skew = 0
        if pos >  self.OSM_SKEW_THRESH: skew = -1
        elif pos < -self.OSM_SKEW_THRESH: skew =  1

        fair_int = int(round(fair))

        if buy_cap > 0:
            # quote at best_bid + 1, but never above (fair - 1 + skew)
            mm_bid = min(best_bid + 1, fair_int - 1 + skew)
            # and never at/above best_ask (would cross)
            mm_bid = min(mm_bid, best_ask - 1)
            orders.append(Order(product, mm_bid, buy_cap))

        if sell_cap > 0:
            # quote at best_ask - 1, but never below (fair + 1 + skew)
            mm_ask = max(best_ask - 1, fair_int + 1 + skew)
            # and never at/below best_bid
            mm_ask = max(mm_ask, best_bid + 1)
            orders.append(Order(product, mm_ask, -sell_cap))

        return orders, fair

    # -------- PEPPER (long-biased, harvest spikes) --------
    def _trade_pepper(self, state: TradingState, trend):
        product = "INTARIAN_PEPPER_ROOT"
        depth = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMIT[product]
        orders: List[Order] = []

        if not depth.buy_orders or not depth.sell_orders:
            return orders, trend

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        if trend is None:
            trend = mid
        else:
            trend = (1 - self.PEP_TREND_ALPHA) * (trend + self.PEP_DRIFT) \
                    + self.PEP_TREND_ALPHA * mid

        buy_cap = limit - pos
        sell_cap = limit + pos

        # TAKE (BUY): accumulate aggressively
        buy_thresh = trend + self.PEP_BUY_PREMIUM
        for ask_px, ask_vol in sorted(depth.sell_orders.items()):
            if buy_cap <= 0:
                break
            if ask_px <= buy_thresh:
                q = min(-ask_vol, buy_cap)
                orders.append(Order(product, int(ask_px), q))
                buy_cap -= q

        # TAKE (SELL): harvest spikes, but protect drift exposure.
        # Only sell down to PEP_KEEP_MIN — keeps us long enough to earn drift.
        sellable = max(0, pos - self.PEP_KEEP_MIN)   # only willing to sell this much
        sell_thresh = trend + self.PEP_SELL_SPIKE
        for bid_px, bid_vol in sorted(depth.buy_orders.items(), reverse=True):
            if sellable <= 0 or sell_cap <= 0:
                break
            if bid_px >= sell_thresh:
                q = min(bid_vol, sellable, sell_cap)
                if q > 0:
                    orders.append(Order(product, int(bid_px), -q))
                    sellable -= q
                    sell_cap -= q

        # MAKE: aggressive long-biased bid
        if buy_cap > 0:
            # Join one tick above best_bid when book is wide enough; otherwise sit
            if best_ask - best_bid > 2:
                mm_bid = best_bid + 1
            else:
                mm_bid = best_bid
            # Cap so we don't pay more than trend + buy premium
            mm_bid = min(mm_bid, int(trend + self.PEP_BUY_PREMIUM))
            # Don't cross the ask
            mm_bid = min(mm_bid, best_ask - 1)
            if mm_bid > 0:
                orders.append(Order(product, mm_bid, buy_cap))

        # MAKE: place a defensive resting sell above trend to catch flash spikes,
        # but only with the "sellable" portion (preserves core long position).
        extra_sellable = max(0, pos - self.PEP_KEEP_MIN)
        if sell_cap > 0 and extra_sellable > 0:
            mm_ask_floor = int(trend + self.PEP_SELL_SPIKE)
            mm_ask = max(best_ask - 1, mm_ask_floor)
            mm_ask = max(mm_ask, best_bid + 1)
            qty = min(sell_cap, extra_sellable)
            orders.append(Order(product, mm_ask, -qty))

        return orders, trend