"""
Round 4 – "The More The Merrier": Trading algorithm.
V3: Dynamic Skew + Trap Removals
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
# REMOVED DEEP_OTM entirely. Bidding 1 guaranteed a loss.

IV_OFFSETS = {
    "VEV_5000": -0.0007,
    "VEV_5100": -0.0008,
    "VEV_5200": +0.0029,
    "VEV_5300": +0.0051,
    "VEV_5400": -0.0117,
    "VEV_5500": +0.0052,
}

SMART_MONEY_BUYERS = {"Mark 01", "Mark 14"}
MARK38 = "Mark 38"

# ─────────────────────────── Trader ──────────────────────────────────────────

class Trader:

    def __init__(self):
        self.smile_a = 0.0293
        self.smile_b = 0.0030
        self.smile_c = 0.2394
        
        self.TTE_at_round_start_days = 4
        self.YEAR = 365.0

        self.MM_PARAMS = {
            HYDRO: {
                "max_pos": 60, "passive_size": 20,
                "edge": 1, "take_edge": 6,
            },
            UNDERLYING: {
                "max_pos": 60, "passive_size": 25,
                "edge": 1, "take_edge": 3,
            },
        }
        self.ARB_PARAMS = {
            "max_pos":       75,
            "threshold":     1.5,  # INCREASED: We only take high-confidence arb now
            "max_take_size": 30,
        }

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

    def _update_signals_state(self, state: TradingState, data: dict):
        if data.get("mark38_ttl", 0) > 0:
            data["mark38_ttl"] -= 1
        else:
            data["mark38_dir"] = 0 

        if data.get("smart_ttl", 0) > 0:
            data["smart_ttl"] -= 1
        else:
            data["smart_buys"] = []

        for t in state.market_trades.get(HYDRO, []):
            if t.buyer == MARK38:
                data["mark38_dir"] = 1
                data["mark38_ttl"] = 8  
            elif t.seller == MARK38:
                data["mark38_dir"] = -1
                data["mark38_ttl"] = 8

        smart_strikes_this_tick = set()
        for sym in ARB_VOUCHERS + DEEP_ITM:
            for t in state.market_trades.get(sym, []):
                if t.buyer in SMART_MONEY_BUYERS:
                    smart_strikes_this_tick.add(VOUCHER_STRIKES[sym])
        
        if smart_strikes_this_tick:
            current_smart = set(data.get("smart_buys", []))
            data["smart_buys"] = list(current_smart.union(smart_strikes_this_tick))
            data["smart_ttl"] = 5  # Reduced TTL to prevent bagholding

    # ──────────────────── smile EWMA fit (unchanged) ─────────────────

    def _fit_smile_online(self, S: float, T: float, voucher_mids: Dict[str, float]) -> None:
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
        
        def det3(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        
        D = det3(M)
        if abs(D) < 1e-12:
            return
        sol = []
        for i in range(3):
            Mi = [row[:] for row in M]
            for r in range(3):
                Mi[r][i] = rhs[r]
            sol.append(det3(Mi) / D)
            
        new_a, new_b, new_c = sol
        if not (0.0 < new_c < 1.0 and abs(new_b) < 1.0 and -1.0 < new_a < 1.0):
            return
        alpha        = 0.10
        self.smile_a = (1 - alpha) * self.smile_a + alpha * new_a
        self.smile_b = (1 - alpha) * self.smile_b + alpha * new_b
        self.smile_c = (1 - alpha) * self.smile_c + alpha * new_c


    def _fair_price(self, sym: str, S: float, T: float) -> float:
        K  = VOUCHER_STRIKES[sym]
        m  = math.log(K / S) / math.sqrt(T)
        iv = self.smile_a * m * m + self.smile_b * m + self.smile_c
        iv += IV_OFFSETS.get(sym, 0.0)
        iv = max(0.05, min(1.0, iv))
        return bs_call_price(S, K, T, iv)

    # ──────────────────── DYNAMIC MARKET MAKING ─────────────────────────

    def _market_make_dynamic(self, sym: str, depth: OrderDepth, position: int,
                             params: dict, is_velvetfruit: bool = False) -> List[Order]:
        """
        Replaces static soft_pos with dynamic inventory skewing.
        """
        orders: List[Order] = []
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return orders

        mid = (bid + ask) / 2.0
        edge = params["edge"]
        max_pos = params["max_pos"]
        passive_size = params["passive_size"]

        # Calculate Skew based on current position
        inventory_ratio = position / max_pos
        
        # We shift our perceived fair value down if we are long, up if we are short
        # This keeps us trading constantly and stops us from getting stuck
        skew = inventory_ratio * 1.5 
        
        # Velvetfruit Structural Lean: Mark 67 is a heavy buyer, so we naturally want 
        # to shift our mid-price higher to capture more of his spread
        if is_velvetfruit:
            skew -= 0.7  

        my_bid = mid - edge - skew
        my_ask = mid + edge - skew

        # Round to valid tick prices
        my_bid = math.floor(my_bid)
        my_ask = math.ceil(my_ask)

        if position < max_pos:
            qty = min(passive_size, max_pos - position)
            orders.append(Order(sym, my_bid, qty))
            
        if position > -max_pos:
            qty = min(passive_size, max_pos + position)
            orders.append(Order(sym, my_ask, -qty))

        return orders

    # ──────────────────── R4 REFINED: Stateful Lean Helpers ───────────────────────

    def _hydrogel_lean(self, depth: OrderDepth, position: int, data: dict) -> List[Order]:
        orders: List[Order] = []
        mark38_dir = data.get("mark38_dir", 0)
        if mark38_dir == 0:
            return orders
            
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return orders
            
        max_pos = self.MM_PARAMS[HYDRO]["max_pos"]

        if mark38_dir == 1:
            qty = min(25, max_pos + position) 
            if qty > 0:
                orders.append(Order(HYDRO, ask, -qty))
        elif mark38_dir == -1: 
            qty = min(25, max_pos - position)
            if qty > 0:
                orders.append(Order(HYDRO, bid, qty))
                
        return orders

    # ──────────────────── smile-arb voucher ──────────────────────────────────

    def _arb_voucher(self, sym: str, depth: OrderDepth, position: int,
                     S: float, T: float, smart_buy: bool = False) -> List[Order]:
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

        if smart_buy and position < max_pos and depth.sell_orders:
            ask_p = min(depth.sell_orders.keys())
            qty   = min(10, max_pos - position, -depth.sell_orders[ask_p])
            if qty > 0:
                orders.append(Order(sym, ask_p, qty))

        return orders

    def _trade_deep_itm(self, sym: str, depth: OrderDepth, position: int, S: float) -> List[Order]:
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
            
        if not isinstance(data, dict):
            data = {}

        self.smile_a = data.get("smile_a", self.smile_a)
        self.smile_b = data.get("smile_b", self.smile_b)
        self.smile_c = data.get("smile_c", self.smile_c)

        order_depths = state.order_depths
        positions    = state.position

        self._update_signals_state(state, data)
        smart_buys_state = set(data.get("smart_buys", []))

        S = self._mid(order_depths[UNDERLYING]) if UNDERLYING in order_depths else None
        T = self._compute_TTE_years(state.timestamp)

        # ── HYDROGEL_PACK ─────────────────────────────────────────────────────
        if HYDRO in order_depths:
            depth = order_depths[HYDRO]
            pos   = positions.get(HYDRO, 0)
            orders = (self._market_make_dynamic(HYDRO, depth, pos, self.MM_PARAMS[HYDRO], False)
                      + self._hydrogel_lean(depth, pos, data))
            if orders:
                result[HYDRO] = orders

        # ── VELVETFRUIT_EXTRACT ───────────────────────────────────────────────
        if UNDERLYING in order_depths:
            depth = order_depths[UNDERLYING]
            pos   = positions.get(UNDERLYING, 0)
            orders = self._market_make_dynamic(UNDERLYING, depth, pos, self.MM_PARAMS[UNDERLYING], True)
            if orders:
                result[UNDERLYING] = orders

        # ── VEV VOUCHERS ──────────────────────────────────────────────────────
        if S is not None and T > 0:
            voucher_mids: Dict[str, float] = {}
            for sym in SMILE_FIT_STRIKES:
                if sym in order_depths:
                    m = self._mid(order_depths[sym])
                    if m is not None:
                        voucher_mids[sym] = m
            self._fit_smile_online(S, T, voucher_mids)

            for sym in ARB_VOUCHERS:
                if sym not in order_depths:
                    continue
                depth     = order_depths[sym]
                pos       = positions.get(sym, 0)
                smart_buy = VOUCHER_STRIKES[sym] in smart_buys_state

                arb_orders = self._arb_voucher(sym, depth, pos, S, T, smart_buy)
                if arb_orders:
                    result.setdefault(sym, []).extend(arb_orders)
                    for o in arb_orders:
                        pos += o.quantity

            for sym in DEEP_ITM:
                if sym not in order_depths:
                    continue
                depth  = order_depths[sym]
                pos    = positions.get(sym, 0)
                orders = self._trade_deep_itm(sym, depth, pos, S)
                if orders:
                    result[sym] = orders

        data["smile_a"] = self.smile_a
        data["smile_b"] = self.smile_b
        data["smile_c"] = self.smile_c
        
        return result, 0, jsonpickle.encode(data)