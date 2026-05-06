"""
Round 3 - Gloves Off: Delta-Hedged & Dynamic Smile Calibration
Refinement: "Cold-Start Snap" implemented. 
On the first tick, the algorithm instantly aligns its IV models with the 
live market (alpha=1.0) to prevent false arbs caused by historical data lag.
"""

from datamodel import Order, OrderDepth, Symbol, TradingState, Listing, Observation
from typing import Dict, List
import jsonpickle
import math

# ----------------------- Black-Scholes utilities ----------------------------
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

def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Calculate the Delta of a European Call option."""
    if T <= 0.0 or sigma <= 0.0:
        return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    return _norm_cdf(d1)

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

# ------------------------------- Constants ---------------------------------
UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"

VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

POSITION_LIMITS = {HYDRO: 200, UNDERLYING: 200}
for _sym in VOUCHER_STRIKES:
    POSITION_LIMITS[_sym] = 300

SMILE_FIT_STRIKES = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
ARB_VOUCHERS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
PASSIVE_MM_VOUCHERS = ["VEV_5300"]
DEEP_ITM = ["VEV_4000", "VEV_4500"]

# ----------------------------- Trader ---------------------------------------
class Trader:
    def __init__(self):
        self.smile_a = 0.0293
        self.smile_b = 0.0030
        self.smile_c = 0.2394
        
        self.iv_offsets = {sym: 0.0 for sym in SMILE_FIT_STRIKES}
        self.last_S = None
        self.initialized_iv = False
        
        self.TTE_at_round_start_days = 5  
        self.YEAR = 365.0

        self.MM_PARAMS = {
            HYDRO: {"soft_pos": 20, "max_pos": 50, "passive_size": 20, "edge": 1, "take_edge": 6},
            UNDERLYING: {"soft_pos": 20, "max_pos": 200, "passive_size": 20, "edge": 1, "take_edge": 3},
        }
        
        self.ARB_PARAMS = {
            "max_pos": 150,
            "threshold": 1.0, 
            "max_take_size": 30, 
        }
        
        self.VOUCHER_PASSIVE_PARAMS = {
            "VEV_5300": {"soft_pos": 30, "max_pos": 50, "passive_size": 15, "edge": 1},
        }

    @staticmethod
    def _best_bid_ask(depth: OrderDepth):
        bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        return bid, ask

    def _mid(self, depth: OrderDepth):
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    def _compute_TTE_years(self, timestamp: int) -> float:
        frac = timestamp / 1_000_000.0
        tte_days = max(self.TTE_at_round_start_days - frac, 1e-6)
        return tte_days / self.YEAR

    def _fit_smile_online(self, S: float, T: float, voucher_mids: Dict[str, float], is_cold_start: bool) -> bool:
        m_vals, iv_vals = [], []
        for sym in SMILE_FIT_STRIKES:
            if sym not in voucher_mids:
                continue
            K = VOUCHER_STRIKES[sym]
            iv = implied_vol(voucher_mids[sym], S, K, T)
            if iv <= 0.05 or iv >= 1.0:
                continue
            m_vals.append(math.log(K / S) / math.sqrt(T))
            iv_vals.append(iv)
            
        if len(m_vals) < 4:
            return False
            
        n = len(m_vals)
        sm0, sm1 = n, sum(m_vals)
        sm2 = sum(m * m for m in m_vals)
        sm3 = sum(m ** 3 for m in m_vals)
        sm4 = sum(m ** 4 for m in m_vals)
        sy = sum(iv_vals)
        sym1 = sum(iv_vals[i] * m_vals[i] for i in range(n))
        sym2 = sum(iv_vals[i] * m_vals[i] * m_vals[i] for i in range(n))
        
        M = [[sm4, sm3, sm2], [sm3, sm2, sm1], [sm2, sm1, sm0]]
        rhs = [sym2, sym1, sy]
        sol = self._solve_3x3(M, rhs)
        if sol is None:
            return False
            
        new_a, new_b, new_c = sol
        if not (0.0 < new_c < 1.0 and abs(new_b) < 1.0 and -1.0 < new_a < 1.0):
            return False
            
        # Cold start snaps instantly. Otherwise, smooth EWMA.
        alpha = 1.0 if is_cold_start else 0.10
        self.smile_a = (1 - alpha) * self.smile_a + alpha * new_a
        self.smile_b = (1 - alpha) * self.smile_b + alpha * new_b
        self.smile_c = (1 - alpha) * self.smile_c + alpha * new_c
        return True

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
        K = VOUCHER_STRIKES[sym]
        m = math.log(K / S) / math.sqrt(T)
        iv = self.smile_a * m * m + self.smile_b * m + self.smile_c
        iv += self.iv_offsets.get(sym, 0.0) 
        iv = max(0.05, min(1.0, iv))
        return bs_call_price(S, K, T, iv)

    def _market_make(self, sym: str, depth: OrderDepth, position: int,
                      params: dict, fair_filter: float = None, target_pos: int = 0) -> List[Order]:
        orders: List[Order] = []
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return orders

        spread = ask - bid
        edge = params["edge"]
        max_pos = params["max_pos"]
        soft_pos = params["soft_pos"]
        passive_size = params["passive_size"]
        
        effective_pos = position - target_pos

        # ---- Aggressive take ----
        take_edge = params.get("take_edge", 0)
        if take_edge > 0 and fair_filter is None:
            mid = (bid + ask) / 2.0
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p <= mid - take_edge and position < max_pos:
                    avail = -depth.sell_orders[ask_p]
                    qty = min(avail, max_pos - position)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        position += qty
                else:
                    break
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p >= mid + take_edge and position > -max_pos:
                    avail = depth.buy_orders[bid_p]
                    qty = min(avail, max_pos + position)
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        position -= qty
                else:
                    break

        # ---- Passive quotes ----
        if spread <= 2 * edge:
            return orders

        my_bid = bid + edge
        my_ask = ask - edge

        if fair_filter is not None:
            if my_bid >= fair_filter:
                my_bid = None
            if my_ask is not None and my_ask <= fair_filter:
                my_ask = None

        effective_pos = position - target_pos

        post_bid = (effective_pos < soft_pos) and (position < max_pos) and (my_bid is not None)
        post_ask = (effective_pos > -soft_pos) and (position > -max_pos) and (my_ask is not None)

        if post_bid:
            buy_capacity = max_pos - position
            qty = min(passive_size, buy_capacity)
            if qty > 0:
                orders.append(Order(sym, int(my_bid), qty))
        if post_ask:
            sell_capacity = max_pos + position
            qty = min(passive_size, sell_capacity)
            if qty > 0:
                orders.append(Order(sym, int(my_ask), -qty))

        return orders

    def _arb_voucher(self, sym: str, depth: OrderDepth, position: int,
                      S: float, T: float) -> List[Order]:
        orders: List[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders
            
        fair = self._fair_price(sym, S, T)
        max_pos = self.ARB_PARAMS["max_pos"]
        threshold = self.ARB_PARAMS["threshold"]
        max_take = self.ARB_PARAMS["max_take_size"]

        for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_p - fair > threshold and position > -max_pos:
                avail = depth.buy_orders[bid_p]
                qty = min(avail, max_pos + position, max_take)
                if qty > 0:
                    orders.append(Order(sym, bid_p, -qty))
                    position -= qty
            else:
                break
                
        for ask_p in sorted(depth.sell_orders.keys()):
            if fair - ask_p > threshold and position < max_pos:
                avail = -depth.sell_orders[ask_p]
                qty = min(avail, max_pos - position, max_take)
                if qty > 0:
                    orders.append(Order(sym, ask_p, qty))
                    position += qty
            else:
                break
                
        return orders

    def _trade_deep_itm(self, sym: str, depth: OrderDepth, position: int,
                         S: float) -> List[Order]:
        orders: List[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders
            
        K = VOUCHER_STRIKES[sym]
        fair = S - K
        max_pos = POSITION_LIMITS[sym]
        edge = 2.0

        for ask_p in sorted(depth.sell_orders.keys()):
            if ask_p < fair - edge and position < max_pos:
                avail = -depth.sell_orders[ask_p]
                qty = min(avail, max_pos - position, 30)
                if qty > 0:
                    orders.append(Order(sym, ask_p, qty))
                    position += qty
            else:
                break
                
        for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_p > fair + edge and position > -max_pos:
                avail = depth.buy_orders[bid_p]
                qty = min(avail, max_pos + position, 30)
                if qty > 0:
                    orders.append(Order(sym, bid_p, -qty))
                    position -= qty
            else:
                break
                
        return orders

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
            self.iv_offsets = data.get("iv_offsets", self.iv_offsets)
            self.last_S = data.get("last_S", self.last_S)
            self.initialized_iv = data.get("initialized_iv", False)
        else:
            self.initialized_iv = False

        is_cold_start = not self.initialized_iv

        order_depths = state.order_depths
        positions = state.position

        S = None
        if UNDERLYING in order_depths:
            S = self._mid(order_depths[UNDERLYING])
            self.last_S = S
        else:
            S = self.last_S 

        T = self._compute_TTE_years(state.timestamp)

        # ---------------------------------------------
        #  Calculate Portfolio Delta
        # ---------------------------------------------
        total_delta = 0.0
        if S is not None and T > 0:
            for sym, pos in positions.items():
                if sym.startswith("VEV_"):
                    K = VOUCHER_STRIKES[sym]
                    m = math.log(K / S) / math.sqrt(T)
                    iv = self.smile_a * m * m + self.smile_b * m + self.smile_c + self.iv_offsets.get(sym, 0.0)
                    iv = max(0.05, min(1.0, iv))
                    delta = bs_call_delta(S, K, T, iv)
                    total_delta += pos * delta

        target_underlying_pos = -int(round(total_delta))
        target_underlying_pos = max(-150, min(150, target_underlying_pos))

        # ---- HYDROGEL_PACK ----
        if HYDRO in order_depths:
            depth = order_depths[HYDRO]
            pos = positions.get(HYDRO, 0)
            orders = self._market_make(HYDRO, depth, pos, self.MM_PARAMS[HYDRO], target_pos=0)
            if orders:
                result[HYDRO] = orders

        # ---- VELVETFRUIT_EXTRACT (Delta Hedged) ----
        if UNDERLYING in order_depths:
            depth = order_depths[UNDERLYING]
            pos = positions.get(UNDERLYING, 0)
            orders = self._market_make(UNDERLYING, depth, pos, self.MM_PARAMS[UNDERLYING], target_pos=target_underlying_pos)
            if orders:
                result[UNDERLYING] = orders

        # ---- Vouchers ----
        if S is not None and T > 0:
            voucher_mids: Dict[str, float] = {}
            for sym in SMILE_FIT_STRIKES:
                if sym in order_depths:
                    m = self._mid(order_depths[sym])
                    if m is not None:
                        voucher_mids[sym] = m
                        
            # Real-time smile calibration
            smile_updated = self._fit_smile_online(S, T, voucher_mids, is_cold_start)

            # Dynamic IV Offset Tracking
            offsets_updated = False
            for sym in SMILE_FIT_STRIKES:
                if sym in voucher_mids:
                    m = math.log(VOUCHER_STRIKES[sym] / S) / math.sqrt(T)
                    base_iv = self.smile_a * m * m + self.smile_b * m + self.smile_c
                    actual_iv = implied_vol(voucher_mids[sym], S, VOUCHER_STRIKES[sym], T)
                    
                    if 0.05 < actual_iv < 1.0:
                        residual = actual_iv - base_iv
                        # Snap instantly if cold start, else smooth
                        alpha_off = 1.0 if is_cold_start else 0.05
                        self.iv_offsets[sym] = (1 - alpha_off) * self.iv_offsets.get(sym, 0.0) + alpha_off * residual
                        offsets_updated = True

            # Mark as initialized only after a successful first-pass calibration
            if is_cold_start and (smile_updated or offsets_updated):
                self.initialized_iv = True

            for sym in ARB_VOUCHERS:
                if sym not in order_depths:
                    continue
                depth = order_depths[sym]
                pos = positions.get(sym, 0)
                
                # 1. Aggressive arb 
                arb_orders = self._arb_voucher(sym, depth, pos, S, T)
                if arb_orders:
                    result.setdefault(sym, []).extend(arb_orders)
                    for o in arb_orders:
                        pos += o.quantity
                        
                # 2. Passive MM filtered by Fair
                if sym in PASSIVE_MM_VOUCHERS:
                    fair = self._fair_price(sym, S, T)
                    params = self.VOUCHER_PASSIVE_PARAMS[sym]
                    passive_orders = self._market_make(
                        sym, depth, pos, params, fair_filter=fair
                    )
                    if passive_orders:
                        result.setdefault(sym, []).extend(passive_orders)

            for sym in DEEP_ITM:
                if sym not in order_depths:
                    continue
                depth = order_depths[sym]
                pos = positions.get(sym, 0)
                orders = self._trade_deep_itm(sym, depth, pos, S)
                if orders:
                    result[sym] = orders

        # Persist dynamic state to next tick
        new_data = {
            "smile_a": self.smile_a,
            "smile_b": self.smile_b,
            "smile_c": self.smile_c,
            "iv_offsets": self.iv_offsets,
            "last_S": self.last_S,
            "initialized_iv": self.initialized_iv,
        }
        
        conversions = 0
        return result, conversions, jsonpickle.encode(new_data)