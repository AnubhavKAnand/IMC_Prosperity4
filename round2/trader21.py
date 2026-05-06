import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    """
    Round 2 Trading Algorithm — v2
    ===============================
    Fixes over v1:
      - CRITICAL BUG FIX: sell_room was computed using pos AFTER aggressive buys.
        IMC checks against the INITIAL position, causing position limit violations.
        Fix: track initial_pos separately and compute all capacities from it.
      - Dynamic EMA fair value for ASH (starts at 10000, converges to true mid).
      - Passive order sizing capped by true remaining capacity.

    ASH_COATED_OSMIUM : Mean-reverting. Market-make around a dynamic fair value.
    INTARIAN_PEPPER_ROOT : Linear uptrend ~1000/day. Always hold maximum long (80).
    """

    # ── Market Access Fee ────────────────────────────────────────────────────
    # Top 50% of bids win extra 25% order book depth.
    # Empirically: extra access adds ~25% more ASH flow → ~+2,000/day.
    # Bidding 1500 comfortably beats median while keeping net EV positive.
    def bid(self) -> int:
        return 1500

    # ── Constants ────────────────────────────────────────────────────────────
    ASH    = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"
    LIMIT  = 80

    # ASH EMA parameters (fair value tracks the mid price)
    EMA_ALPHA  = 0.02   # converges to true mid in ~100 steps
    EMA_INIT   = 10000  # prior; market average ~10004 in Round 2

    def run(self, state: TradingState):
        # ── Load persisted state ──────────────────────────────────────────
        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}

        ash_fair_ema = float(trader_state.get("ash_fair_ema", self.EMA_INIT))

        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == self.ASH:
                orders, ash_fair_ema = self._trade_ash(order_depth, pos, ash_fair_ema)
            elif product == self.PEPPER:
                orders = self._trade_pepper(order_depth, pos)
            else:
                orders = []
            result[product] = orders

        # ── Persist state ─────────────────────────────────────────────────
        trader_state["ash_fair_ema"] = ash_fair_ema
        return result, 1, json.dumps(trader_state)

    # ── ASH: Market-Making with dynamic fair value ───────────────────────
    def _trade_ash(self, ob: OrderDepth, pos: int, fair_ema: float):
        orders: List[Order] = []
        LIMIT = self.LIMIT
        initial_pos = pos  # save for capacity checks (IMC uses this for limit validation)

        # Update EMA fair value from current order book mid
        if ob.buy_orders and ob.sell_orders:
            mid = (max(ob.buy_orders) + min(ob.sell_orders)) / 2
            fair_ema = (1 - self.EMA_ALPHA) * fair_ema + self.EMA_ALPHA * mid
        elif ob.buy_orders:
            fair_ema = (1 - self.EMA_ALPHA / 2) * fair_ema + (self.EMA_ALPHA / 2) * max(ob.buy_orders)
        elif ob.sell_orders:
            fair_ema = (1 - self.EMA_ALPHA / 2) * fair_ema + (self.EMA_ALPHA / 2) * min(ob.sell_orders)
        fair = round(fair_ema)

        # Track submitted volumes for correct capacity math
        total_buys  = 0
        total_sells = 0

        # ── 1. Aggressively BUY underpriced asks ──────────────────────────
        if ob.sell_orders:
            for ask_px, ask_vol in sorted(ob.sell_orders.items()):
                if ask_px >= fair:
                    break
                # Capacity: IMC checks initial_pos + all_buys <= LIMIT
                can_buy = LIMIT - initial_pos - total_buys
                qty = min(-ask_vol, can_buy)  # ask_vol is negative in datamodel
                if qty > 0:
                    orders.append(Order(self.ASH, ask_px, qty))
                    total_buys += qty

        # ── 2. Aggressively SELL overpriced bids ─────────────────────────
        if ob.buy_orders:
            for bid_px, bid_vol in sorted(ob.buy_orders.items(), reverse=True):
                if bid_px <= fair:
                    break
                # Capacity: IMC checks initial_pos - all_sells >= -LIMIT
                can_sell = LIMIT + initial_pos - total_sells
                qty = min(bid_vol, can_sell)
                if qty > 0:
                    orders.append(Order(self.ASH, bid_px, -qty))
                    total_sells += qty

        # ── 3. Passive limit orders inside the spread ─────────────────────
        OFFSET     = 2   # quote this many ticks inside fair
        QUOTE_SIZE = 15  # max units per passive level

        best_bid = max(ob.buy_orders)  if ob.buy_orders  else fair - 10
        best_ask = min(ob.sell_orders) if ob.sell_orders else fair + 10

        passive_bid = fair - OFFSET
        passive_ask = fair + OFFSET

        # Nudge into the spread if needed (don't improve market by more than 1 tick)
        if passive_bid < best_bid:
            passive_bid = best_bid + 1   # join at best bid + 1
        if passive_ask > best_ask:
            passive_ask = best_ask - 1   # join at best ask - 1

        # ── Passive BUY ───────────────────────────────────────────────────
        # IMC capacity: initial_pos + total_buys + passive_buy_qty <= LIMIT
        buy_cap = LIMIT - initial_pos - total_buys
        if passive_bid < fair and buy_cap > 0:
            qty = min(QUOTE_SIZE, buy_cap)
            orders.append(Order(self.ASH, passive_bid, qty))
            total_buys += qty  # update for any further logic (not strictly needed)

        # ── Passive SELL ──────────────────────────────────────────────────
        # IMC capacity: initial_pos - total_sells - passive_sell_qty >= -LIMIT
        # CRITICAL: use initial_pos NOT the post-aggressive pos.
        #           Using updated pos here caused the position limit violations.
        sell_cap = LIMIT + initial_pos - total_sells
        if passive_ask > fair and sell_cap > 0:
            qty = min(QUOTE_SIZE, sell_cap)
            orders.append(Order(self.ASH, passive_ask, -qty))

        return orders, fair_ema

    # ── PEPPER: Trend-following (always hold max long) ───────────────────
    def _trade_pepper(self, ob: OrderDepth, pos: int) -> List[Order]:
        """
        Price rises ~1000 XIRECs/day linearly (+0.001/timestamp unit).
        Optimal: always hold position = +80. Buy aggressively at best available ask.
        """
        orders: List[Order] = []
        remaining = self.LIMIT - pos

        if remaining <= 0 or not ob.sell_orders:
            return orders

        for ask_px, ask_vol in sorted(ob.sell_orders.items()):
            qty = min(-ask_vol, remaining)
            if qty > 0:
                orders.append(Order(self.PEPPER, ask_px, qty))
                remaining -= qty
            if remaining == 0:
                break

        return orders