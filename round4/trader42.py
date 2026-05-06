"""
Round 4 – "The More The Merrier": Trading algorithm.

Built on top of the Round 3 algorithm (+39,802 PnL over 3 days).

══════════════════════════════════════════════════════════════════════════════
NEW IN ROUND 4: COUNTERPARTY INTELLIGENCE
══════════════════════════════════════════════════════════════════════════════

HYDROGEL_PACK
─────────────
• Mark 38 is in 100% of all HYDROGEL trades — he is the sole market-taker.
  He buys at best-ask and sells at best-bid, ~50/50 direction, no memory.
• Mark 14 is the primary (profitable) market-maker — we copy his role.
• NEW: when Mark 38 was recently buying, add extra aggressive sells at ask
  (and vice-versa). Exploits the short-term clustering in his order batches.

VELVETFRUIT_EXTRACT
───────────────────
• Mark 55 (87% of trades): symmetric market-taker (buys ask / sells bid).
  No directional edge — just keep collecting his spread.
• Mark 67 (165 buy-only trades per 3 days, 0 sells): perpetual buyer at
  best-ask, NEVER sells. Creates net upward buy pressure.
  → When active: reduce soft-buy cap so we don't accumulate long inventory;
    post extra sells at best-ask for him to lift.
• Mark 49 (net seller, 105 sells / 17 buys): adds cheap supply below fair.
  → When active: post extra buys at best-bid.

VEV OPTIONS
───────────
• TTE changes from 5 days (R3) to 4 days (R4). Updated below.
• VEV_6000 / VEV_6500: Mark 22 sells these to Mark 01 at price 0.
  Buying at 1 is essentially free positive EV — any payoff is pure profit.
  R3 skipped them entirely; R4 aggressively bids 1 for up to the full limit.
• Smart-money follow: when market_trades shows Mark 01 / Mark 14 buying any
  ARB_VOUCHER, take one extra 10-lot at the ask (ride their flow signal).
• All other R3 logic (smile arb, deep-ITM arb, IV offsets, EWMA smile fit)
  is preserved intact.
══════════════════════════════════════════════════════════════════════════════
"""

from datamodel import Order, OrderDepth, Symbol, TradingState, Listing, Observation
from typing import Dict, List, Set
import jsonpickle
import math


# ─────────────────────── Black-Scholes utilities ──────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return max(0.0, S - K)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_call_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    return S * _norm_pdf(d1) * sqrtT


def implied_vol(C: float, S: float, K: float, T: float) -> float:
    if T <= 0.0:
        return 0.0
    intrinsic = max(0.0, S - K)
    if C <= intrinsic + 1e-6:
        return 0.0
    lo, hi = 1e-4, 5.0
    if bs_call_price(S, K, T, hi) < C:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if bs_call_price(S, K, T, mid) > C:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ─────────────────────────── Constants ───────────────────────────────────────

UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDRO      = "HYDROGEL_PACK"

VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

POSITION_LIMITS = {HYDRO: 200, UNDERLYING: 200}
for _sym in VOUCHER_STRIKES:
    POSITION_LIMITS[_sym] = 300

SMILE_FIT_STRIKES   = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
ARB_VOUCHERS        = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5400", "VEV_5500"]
PASSIVE_MM_VOUCHERS = []
DEEP_ITM            = ["VEV_4000", "VEV_4500"]
DEEP_OTM            = ["VEV_6000", "VEV_6500"]  # R4 new: free-lottery buys

IV_OFFSETS = {
    "VEV_5000": -0.0007,
    "VEV_5100": -0.0008,
    "VEV_5200": +0.0029,
    "VEV_5300": +0.0051,
    "VEV_5400": -0.0117,
    "VEV_5500": +0.0052,
}

# Counterparty IDs revealed in R4
SMART_MONEY_BUYERS = {"Mark 01", "Mark 14"}
MARK38 = "Mark 38"
MARK67 = "Mark 67"
MARK49 = "Mark 49"


# ─────────────────────────── Trader ──────────────────────────────────────────

class Trader:

    def __init__(self):
        self.smile_a = 0.0293
        self.smile_b = 0.0030
        self.smile_c = 0.2394
        # KEY R4 CHANGE: TTE is now 4 trading days at round start (was 5 in R3)
        self.TTE_at_round_start_days = 4
        self.YEAR = 365.0

        self.MM_PARAMS = {
            HYDRO: {
                "soft_pos": 20, "max_pos": 50, "passive_size": 20,
                "edge": 1, "take_edge": 6,
            },
            UNDERLYING: {
                "soft_pos": 20, "max_pos": 50, "passive_size": 20,
                "edge": 1, "take_edge": 3,
            },
        }
        self.ARB_PARAMS = {
            "max_pos":       75,
            "threshold":     0.7,
            "max_take_size": 30,
        }
        self.VOUCHER_PASSIVE_PARAMS = {}

    # ──────────────────── helpers ─────────────────────────────────────────────

    @staticmethod
    def _best_bid_ask(depth: OrderDepth):
        bid = max(depth.buy_orders.keys())  if depth.buy_orders  else None
        ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        return bid, ask

    def _mid(self, depth: OrderDepth):
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    def _compute_TTE_years(self, timestamp: int) -> float:
        frac     = timestamp / 1_000_000.0
        tte_days = max(self.TTE_at_round_start_days - frac, 1e-6)
        return tte_days / self.YEAR

    # ──────────────────── R4 NEW: counterparty signal extraction ──────────────

    def _extract_signals(self, state: TradingState) -> dict:
        """
        Read state.market_trades once per tick and return lightweight signals.
        No state is mutated here — all signals are derived purely from the
        latest batch of completed trades visible in the order book feed.
        """
        sig = {
            "mark38_active": False,
            "mark38_buying": False,   # True => Mark 38 lifted asks recently
            "mark67_active": False,   # True => lean short on VE
            "mark49_active": False,   # True => lean long on VE
            "smart_buys":    set(),   # voucher strikes Mark01/14 bought
        }

        for t in state.market_trades.get(HYDRO, []):
            if t.buyer == MARK38 or t.seller == MARK38:
                sig["mark38_active"] = True
                sig["mark38_buying"] = (t.buyer == MARK38)

        for t in state.market_trades.get(UNDERLYING, []):
            if t.buyer == MARK67:
                sig["mark67_active"] = True
            if t.buyer == MARK49 or t.seller == MARK49:
                sig["mark49_active"] = True

        for sym in ARB_VOUCHERS + DEEP_ITM:
            for t in state.market_trades.get(sym, []):
                if t.buyer in SMART_MONEY_BUYERS:
                    sig["smart_buys"].add(VOUCHER_STRIKES[sym])

        return sig

    # ──────────────────── smile EWMA fit (unchanged from R3) ─────────────────

    def _fit_smile_online(self, S: float, T: float,
                          voucher_mids: Dict[str, float]) -> None:
        m_vals, iv_vals = [], []
        for sym in SMILE_FIT_STRIKES:
            if sym not in voucher_mids:
                continue
            K  = VOUCHER_STRIKES[sym]
            iv = implied_vol(voucher_mids[sym], S, K, T)
            if iv <= 0.05 or iv >= 1.0:
                continue
            m_vals.append(math.log(K / S) / math.sqrt(T))
            iv_vals.append(iv)
        if len(m_vals) < 4:
            return
        n    = len(m_vals)
        sm0  = n
        sm1  = sum(m_vals)
        sm2  = sum(m * m      for m in m_vals)
        sm3  = sum(m ** 3     for m in m_vals)
        sm4  = sum(m ** 4     for m in m_vals)
        sy   = sum(iv_vals)
        sym1 = sum(iv_vals[i] * m_vals[i]             for i in range(n))
        sym2 = sum(iv_vals[i] * m_vals[i] * m_vals[i] for i in range(n))
        M    = [[sm4, sm3, sm2], [sm3, sm2, sm1], [sm2, sm1, sm0]]
        rhs  = [sym2, sym1, sy]
        sol  = self._solve_3x3(M, rhs)
        if sol is None:
            return
        new_a, new_b, new_c = sol
        if not (0.0 < new_c < 1.0 and abs(new_b) < 1.0 and -1.0 < new_a < 1.0):
            return
        alpha        = 0.10
        self.smile_a = (1 - alpha) * self.smile_a + alpha * new_a
        self.smile_b = (1 - alpha) * self.smile_b + alpha * new_b
        self.smile_c = (1 - alpha) * self.smile_c + alpha * new_c

    @staticmethod
    def _solve_3x3(M, b):
        def det3(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        D = det3(M)
        if abs(D) < 1e-12:
            return None
        out = []
        for i in range(3):
            Mi = [row[:] for row in M]
            for r in range(3):
                Mi[r][i] = b[r]
            out.append(det3(Mi) / D)
        return out

    def _fair_price(self, sym: str, S: float, T: float) -> float:
        K  = VOUCHER_STRIKES[sym]
        m  = math.log(K / S) / math.sqrt(T)
        iv = self.smile_a * m * m + self.smile_b * m + self.smile_c
        iv += IV_OFFSETS.get(sym, 0.0)
        iv = max(0.05, min(1.0, iv))
        return bs_call_price(S, K, T, iv)

    # ──────────────────── core MM (unchanged from R3) ─────────────────────────

    def _market_make(self, sym: str, depth: OrderDepth, position: int,
                     params: dict, fair_filter: float = None) -> List[Order]:
        orders: List[Order] = []
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return orders

        spread       = ask - bid
        edge         = params["edge"]
        max_pos      = params["max_pos"]
        soft_pos     = params["soft_pos"]
        passive_size = params["passive_size"]
        take_edge    = params.get("take_edge", 0)

        # Aggressive take
        if take_edge > 0 and fair_filter is None:
            mid = (bid + ask) / 2.0
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p <= mid - take_edge and position < max_pos:
                    avail = -depth.sell_orders[ask_p]
                    qty   = min(avail, max_pos - position)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        position += qty
                else:
                    break
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p >= mid + take_edge and position > -max_pos:
                    avail = depth.buy_orders[bid_p]
                    qty   = min(avail, max_pos + position)
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        position -= qty
                else:
                    break

        # Passive quotes
        if spread <= 2 * edge:
            return orders
        my_bid = bid + edge
        my_ask = ask - edge

        if fair_filter is not None:
            if my_bid >= fair_filter:
                my_bid = None
            if my_ask is not None and my_ask <= fair_filter:
                my_ask = None

        post_bid = (position < soft_pos)  and (position < max_pos)  and (my_bid is not None)
        post_ask = (position > -soft_pos) and (position > -max_pos) and (my_ask is not None)

        if post_bid:
            qty = min(passive_size, max_pos - position)
            if qty > 0:
                orders.append(Order(sym, int(my_bid), qty))
        if post_ask:
            qty = min(passive_size, max_pos + position)
            if qty > 0:
                orders.append(Order(sym, int(my_ask), -qty))

        return orders

    # ──────────────────── R4 NEW: lean helpers ───────────────────────────────

    def _hydrogel_lean(self, depth: OrderDepth, position: int,
                       sig: dict) -> List[Order]:
        """
        Extra directional layer when Mark 38 is active.
        He tends to continue in the same direction within a tick batch, so we
        lean into him: sell more at ask when he buys, buy more at bid when he sells.
        """
        if not sig["mark38_active"]:
            return []
        orders: List[Order] = []
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return orders
        max_pos = self.MM_PARAMS[HYDRO]["max_pos"]

        if sig["mark38_buying"]:
            cap = max_pos + position          # remaining sell capacity
            qty = min(25, cap)
            if qty > 0:
                orders.append(Order(HYDRO, ask, -qty))
        else:
            cap = max_pos - position
            qty = min(25, cap)
            if qty > 0:
                orders.append(Order(HYDRO, bid, qty))
        return orders

    def _ve_lean(self, depth: OrderDepth, position: int,
                 sig: dict) -> List[Order]:
        """
        Mark 67 lean (short) and Mark 49 lean (long).
        These are ADDITIONAL orders on top of the MM layer.
        """
        orders: List[Order] = []
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return orders
        max_pos = self.MM_PARAMS[UNDERLYING]["max_pos"]

        if sig["mark67_active"]:
            # Mark 67 always lifts best-ask — sell extra volume there
            cap = max_pos + position
            qty = min(35, cap)
            if qty > 0:
                orders.append(Order(UNDERLYING, ask, -qty))

        if sig["mark49_active"]:
            # Mark 49 hits best-bid — buy extra volume there
            cap = max_pos - position
            qty = min(20, cap)
            if qty > 0:
                orders.append(Order(UNDERLYING, bid, qty))

        return orders

    # ──────────────────── R4 NEW: deep-OTM free lottery ─────────────────────

    def _trade_deep_otm(self, sym: str, depth: OrderDepth,
                        position: int) -> List[Order]:
        """
        VEV_6000 / VEV_6500: Mark 22 sells these for 0 to Mark 01 every day.
        We bid 1. Maximum loss = 1 per unit. Any nonzero payoff = pure profit.
        Take any existing ask <= 1, then rest a bid at 1 for remaining capacity.
        """
        orders: List[Order] = []
        cap = POSITION_LIMITS[sym] - position
        if cap <= 0:
            return orders

        # Take any ask priced at 0 or 1 immediately
        for ask_p in sorted(depth.sell_orders.keys()):
            if ask_p <= 1 and cap > 0:
                avail = -depth.sell_orders[ask_p]
                qty   = min(avail, cap)
                if qty > 0:
                    orders.append(Order(sym, ask_p, qty))
                    cap -= qty
            else:
                break

        # Rest a passive bid at 1 for any leftover capacity
        if cap > 0:
            orders.append(Order(sym, 1, cap))

        return orders

    # ──────────────────── smile-arb voucher ──────────────────────────────────

    def _arb_voucher(self, sym: str, depth: OrderDepth, position: int,
                     S: float, T: float, smart_buy: bool = False) -> List[Order]:
        """
        Unchanged from R3 except for the smart_buy tail:
        if Mark 01 / Mark 14 recently bought this strike, take an extra 10-lot
        at best-ask to ride their informed flow.
        """
        orders: List[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        fair      = self._fair_price(sym, S, T)
        max_pos   = self.ARB_PARAMS["max_pos"]
        threshold = self.ARB_PARAMS["threshold"]
        max_take  = self.ARB_PARAMS["max_take_size"]

        for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_p - fair > threshold and position > -max_pos:
                avail = depth.buy_orders[bid_p]
                qty   = min(avail, max_pos + position, max_take)
                if qty > 0:
                    orders.append(Order(sym, bid_p, -qty))
                    position -= qty
            else:
                break

        for ask_p in sorted(depth.sell_orders.keys()):
            if fair - ask_p > threshold and position < max_pos:
                avail = -depth.sell_orders[ask_p]
                qty   = min(avail, max_pos - position, max_take)
                if qty > 0:
                    orders.append(Order(sym, ask_p, qty))
                    position += qty
            else:
                break

        # R4: follow smart money
        if smart_buy and position < max_pos and depth.sell_orders:
            ask_p = min(depth.sell_orders.keys())
            qty   = min(10, max_pos - position, -depth.sell_orders[ask_p])
            if qty > 0:
                orders.append(Order(sym, ask_p, qty))

        return orders

    # ──────────────────── deep-ITM arb (unchanged from R3) ───────────────────

    def _trade_deep_itm(self, sym: str, depth: OrderDepth, position: int,
                        S: float) -> List[Order]:
        orders: List[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders
        K    = VOUCHER_STRIKES[sym]
        fair = S - K
        edge = 2.0
        max_pos = POSITION_LIMITS[sym]

        for ask_p in sorted(depth.sell_orders.keys()):
            if ask_p < fair - edge and position < max_pos:
                avail = -depth.sell_orders[ask_p]
                qty   = min(avail, max_pos - position, 30)
                if qty > 0:
                    orders.append(Order(sym, ask_p, qty))
                    position += qty
            else:
                break
        for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_p > fair + edge and position > -max_pos:
                avail = depth.buy_orders[bid_p]
                qty   = min(avail, max_pos + position, 30)
                if qty > 0:
                    orders.append(Order(sym, bid_p, -qty))
                    position -= qty
            else:
                break
        return orders

    # ──────────────────── main entry ─────────────────────────────────────────

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        try:
            data = jsonpickle.decode(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        if isinstance(data, dict):
            self.smile_a = data.get("smile_a", self.smile_a)
            self.smile_b = data.get("smile_b", self.smile_b)
            self.smile_c = data.get("smile_c", self.smile_c)

        order_depths = state.order_depths
        positions    = state.position

        # Extract counterparty signals (R4 new)
        sig = self._extract_signals(state)

        S = self._mid(order_depths[UNDERLYING]) if UNDERLYING in order_depths else None
        T = self._compute_TTE_years(state.timestamp)

        # ── HYDROGEL_PACK ─────────────────────────────────────────────────────
        if HYDRO in order_depths:
            depth = order_depths[HYDRO]
            pos   = positions.get(HYDRO, 0)
            orders = (self._market_make(HYDRO, depth, pos, self.MM_PARAMS[HYDRO])
                      + self._hydrogel_lean(depth, pos, sig))
            if orders:
                result[HYDRO] = orders

        # ── VELVETFRUIT_EXTRACT ───────────────────────────────────────────────
        if UNDERLYING in order_depths:
            depth = order_depths[UNDERLYING]
            pos   = positions.get(UNDERLYING, 0)
            # Reduce bid aggressiveness when Mark 67 is active (pure buyer =
            # upward pressure; we don't want long inventory we'll have to cover)
            ve_params = dict(self.MM_PARAMS[UNDERLYING])
            if sig["mark67_active"]:
                ve_params["soft_pos"] = 0   # only quote bids when flat or short
            orders = (self._market_make(UNDERLYING, depth, pos, ve_params)
                      + self._ve_lean(depth, pos, sig))
            if orders:
                result[UNDERLYING] = orders

        # ── VEV VOUCHERS ──────────────────────────────────────────────────────
        if S is not None and T > 0:
            # Update smile
            voucher_mids: Dict[str, float] = {}
            for sym in SMILE_FIT_STRIKES:
                if sym in order_depths:
                    m = self._mid(order_depths[sym])
                    if m is not None:
                        voucher_mids[sym] = m
            self._fit_smile_online(S, T, voucher_mids)

            # Smile-arb vouchers
            for sym in ARB_VOUCHERS:
                if sym not in order_depths:
                    continue
                depth     = order_depths[sym]
                pos       = positions.get(sym, 0)
                smart_buy = VOUCHER_STRIKES[sym] in sig["smart_buys"]

                arb_orders = self._arb_voucher(sym, depth, pos, S, T, smart_buy)
                if arb_orders:
                    result.setdefault(sym, []).extend(arb_orders)
                    for o in arb_orders:
                        pos += o.quantity

                if sym in PASSIVE_MM_VOUCHERS:
                    fair = self._fair_price(sym, S, T)
                    passive_orders = self._market_make(
                        sym, depth, pos, self.VOUCHER_PASSIVE_PARAMS[sym],
                        fair_filter=fair,
                    )
                    if passive_orders:
                        result.setdefault(sym, []).extend(passive_orders)

            # Deep-ITM arb (unchanged from R3)
            for sym in DEEP_ITM:
                if sym not in order_depths:
                    continue
                depth  = order_depths[sym]
                pos    = positions.get(sym, 0)
                orders = self._trade_deep_itm(sym, depth, pos, S)
                if orders:
                    result[sym] = orders

            # R4 NEW: deep-OTM free lottery
            for sym in DEEP_OTM:
                if sym not in order_depths:
                    continue
                depth  = order_depths[sym]
                pos    = positions.get(sym, 0)
                orders = self._trade_deep_otm(sym, depth, pos)
                if orders:
                    result[sym] = orders

        new_data = {"smile_a": self.smile_a, "smile_b": self.smile_b, "smile_c": self.smile_c}
        return result, 0, jsonpickle.encode(new_data)