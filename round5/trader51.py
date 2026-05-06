"""
IMC Prosperity Round 5 — "The Final Stretch"
==============================================
Strategy overview (estimated ~825k PnL on training data):

1. PEBBLES ARB (≈143k)
   The 5 PEBBLES products always sum to ~50,000.
   We compute a rolling EMA per product; fair value = EMA + (50000 - current_sum)/5.
   When price deviates > threshold from fair value, we fade the deviation.

2. ROBOT_DISHES — EMA Mean Reversion (≈139k)
   Lag-1 return autocorrelation = -0.23 (strong mean reversion).
   Aggressively buy when ask < EMA - 3, sell when bid > EMA + 3.

3. All other 44 products — Directional Hold (≈543k)
   Historical analysis over 3 days determines the optimal direction.
   We take max position from first tick and hold.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


# ─── IMC Prosperity types (provided by the platform) ──────────────────────────
Symbol = str
Product = str
Position = int

@dataclass
class Order:
    symbol: Symbol
    price: int
    quantity: int

@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)   # price → qty (positive)
    sell_orders: Dict[int, int] = field(default_factory=dict)  # price → qty (negative)

@dataclass
class TradingState:
    timestamp: int
    listings: Dict[Symbol, object]
    order_depths: Dict[Symbol, OrderDepth]
    own_trades: Dict[Symbol, List]
    market_trades: Dict[Symbol, List]
    position: Dict[Product, Position]
    observations: object


# ─── Strategy configuration ───────────────────────────────────────────────────

PEBBLES = ['PEBBLES_L', 'PEBBLES_M', 'PEBBLES_S', 'PEBBLES_XL', 'PEBBLES_XS']
PEBBLES_SUM = 50000  # hard constraint enforced by the exchange

# Direction +10 = long, -10 = short, determined by 3-day historical analysis
HOLD_DIRECTIONS: Dict[str, int] = {
    'GALAXY_SOUNDS_BLACK_HOLES':    10,
    'GALAXY_SOUNDS_DARK_MATTER':    10,
    'GALAXY_SOUNDS_PLANETARY_RINGS': -10,
    'GALAXY_SOUNDS_SOLAR_FLAMES':   10,
    'GALAXY_SOUNDS_SOLAR_WINDS':    10,
    'MICROCHIP_CIRCLE':             10,
    'MICROCHIP_OVAL':              -10,
    'MICROCHIP_RECTANGLE':         -10,
    'MICROCHIP_SQUARE':             10,
    'MICROCHIP_TRIANGLE':          -10,
    'OXYGEN_SHAKE_CHOCOLATE':       10,
    'OXYGEN_SHAKE_EVENING_BREATH': -10,
    'OXYGEN_SHAKE_GARLIC':          10,
    'OXYGEN_SHAKE_MINT':            10,
    'OXYGEN_SHAKE_MORNING_BREATH': -10,
    'PANEL_1X2':                   -10,
    'PANEL_1X4':                   -10,
    'PANEL_2X2':                   -10,
    'PANEL_2X4':                    10,
    'PANEL_4X4':                   -10,
    'ROBOT_IRONING':               -10,
    'ROBOT_LAUNDRY':               -10,
    'ROBOT_MOPPING':                10,
    'ROBOT_VACUUMING':             -10,
    'SLEEP_POD_COTTON':             10,
    'SLEEP_POD_LAMB_WOOL':          10,
    'SLEEP_POD_NYLON':              10,
    'SLEEP_POD_POLYESTER':          10,
    'SLEEP_POD_SUEDE':              10,
    'SNACKPACK_CHOCOLATE':         -10,
    'SNACKPACK_PISTACHIO':         -10,
    'SNACKPACK_RASPBERRY':          10,
    'SNACKPACK_STRAWBERRY':         10,
    'SNACKPACK_VANILLA':            10,
    'TRANSLATOR_ASTRO_BLACK':      -10,
    'TRANSLATOR_ECLIPSE_CHARCOAL': -10,
    'TRANSLATOR_GRAPHITE_MIST':    -10,
    'TRANSLATOR_SPACE_GRAY':       -10,
    'TRANSLATOR_VOID_BLUE':         10,
    'UV_VISOR_AMBER':              -10,
    'UV_VISOR_MAGENTA':             10,
    'UV_VISOR_ORANGE':             -10,
    'UV_VISOR_RED':                 10,
    'UV_VISOR_YELLOW':              10,
}

POSITION_LIMIT = 10


# ─── Trader class ─────────────────────────────────────────────────────────────

class Trader:

    def __init__(self):
        # ---- PEBBLES state ----
        self.pebble_ema: Dict[str, Optional[float]] = {p: None for p in PEBBLES}
        self.pebble_ema_alpha: float = 2.0 / (1000 + 1)  # span-1000 EMA
        self.pebble_threshold: float = 50.0
        self.pebble_qty: int = 3

        # ---- ROBOT_DISHES state ----
        self.dishes_ema: Optional[float] = None
        self.dishes_ema_alpha: float = 2.0 / (50 + 1)  # span-50 EMA
        self.dishes_edge: float = 3.0
        self.dishes_max_qty: int = 5

        # Track whether hold positions have been initialised (per product per day)
        self._hold_entered: Dict[str, bool] = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _best_ask(order_depth: OrderDepth) -> Optional[tuple]:
        """Return (price, volume) of the best ask, or None."""
        if not order_depth.sell_orders:
            return None
        price = min(order_depth.sell_orders)
        return price, abs(order_depth.sell_orders[price])

    @staticmethod
    def _best_bid(order_depth: OrderDepth) -> Optional[tuple]:
        """Return (price, volume) of the best bid, or None."""
        if not order_depth.buy_orders:
            return None
        price = max(order_depth.buy_orders)
        return price, order_depth.buy_orders[price]

    @staticmethod
    def _mid(order_depth: OrderDepth) -> Optional[float]:
        """Return mid-price, or None if the book is one-sided."""
        ask = Trader._best_ask(order_depth)
        bid = Trader._best_bid(order_depth)
        if ask and bid:
            return (ask[0] + bid[0]) / 2.0
        elif ask:
            return float(ask[0])
        elif bid:
            return float(bid[0])
        return None

    # ── PEBBLES arbitrage ────────────────────────────────────────────────────

    def _pebbles_orders(
        self,
        state: TradingState,
        orders: Dict[str, List[Order]],
    ) -> None:
        """
        Compute fair values for all 5 PEBBLES using the hard constraint
        (sum == 50000) and trade deviations above the threshold.
        """
        # 1. Read current mid-prices and update EMAs
        mids: Dict[str, float] = {}
        for prod in PEBBLES:
            od = state.order_depths.get(prod)
            if od is None:
                return  # can't act without all books
            mid = self._mid(od)
            if mid is None:
                return
            mids[prod] = mid

            # Update EMA
            if self.pebble_ema[prod] is None:
                self.pebble_ema[prod] = mid
            else:
                a = self.pebble_ema_alpha
                self.pebble_ema[prod] = a * mid + (1 - a) * self.pebble_ema[prod]

        # 2. Compute constraint deviation
        current_sum = sum(mids[p] for p in PEBBLES)
        constraint_adj = (PEBBLES_SUM - current_sum) / 5.0  # distributed equally

        # 3. Generate orders per product
        for prod in PEBBLES:
            mid = mids[prod]
            ema = self.pebble_ema[prod]
            fair = ema + constraint_adj

            dev = mid - fair
            pos = state.position.get(prod, 0)
            od = state.order_depths[prod]

            if dev > self.pebble_threshold and pos > -POSITION_LIMIT:
                # Overpriced → sell
                best_bid = self._best_bid(od)
                if best_bid:
                    qty = min(POSITION_LIMIT + pos, self.pebble_qty)
                    if qty > 0:
                        orders[prod].append(Order(prod, best_bid[0], -qty))

            elif dev < -self.pebble_threshold and pos < POSITION_LIMIT:
                # Underpriced → buy
                best_ask = self._best_ask(od)
                if best_ask:
                    qty = min(POSITION_LIMIT - pos, self.pebble_qty)
                    if qty > 0:
                        orders[prod].append(Order(prod, best_ask[0], qty))

    # ── ROBOT_DISHES mean reversion ──────────────────────────────────────────

    def _dishes_orders(
        self,
        state: TradingState,
        orders: Dict[str, List[Order]],
    ) -> None:
        """
        Fast mean-reversion for ROBOT_DISHES (lag-1 AC ≈ -0.23).
        Buy aggressively when ask < EMA - edge; sell when bid > EMA + edge.
        """
        prod = 'ROBOT_DISHES'
        od = state.order_depths.get(prod)
        if od is None:
            return

        mid = self._mid(od)
        if mid is None:
            return

        # Update EMA
        if self.dishes_ema is None:
            self.dishes_ema = mid
        else:
            a = self.dishes_ema_alpha
            self.dishes_ema = a * mid + (1 - a) * self.dishes_ema

        ema = self.dishes_ema
        pos = state.position.get(prod, 0)

        best_ask = self._best_ask(od)
        best_bid = self._best_bid(od)

        # Buy when ask is cheap relative to EMA
        if best_ask and best_ask[0] < ema - self.dishes_edge and pos < POSITION_LIMIT:
            qty = min(POSITION_LIMIT - pos, self.dishes_max_qty)
            orders[prod].append(Order(prod, best_ask[0], qty))

        # Sell when bid is expensive relative to EMA
        if best_bid and best_bid[0] > ema + self.dishes_edge and pos > -POSITION_LIMIT:
            qty = min(POSITION_LIMIT + pos, self.dishes_max_qty)
            orders[prod].append(Order(prod, best_bid[0], -qty))

        # Inventory management: reduce large positions when price is neutral
        if pos > 7 and best_bid and best_bid[0] >= ema:
            orders[prod].append(Order(prod, best_bid[0], -1))
        elif pos < -7 and best_ask and best_ask[0] <= ema:
            orders[prod].append(Order(prod, best_ask[0], 1))

    # ── Directional hold positions ────────────────────────────────────────────

    def _hold_orders(
        self,
        state: TradingState,
        orders: Dict[str, List[Order]],
    ) -> None:
        """
        For each product in HOLD_DIRECTIONS:
        - Take maximum position in the desired direction immediately.
        - Maintain it throughout the day (do nothing once filled).
        """
        for prod, target in HOLD_DIRECTIONS.items():
            od = state.order_depths.get(prod)
            if od is None:
                continue

            pos = state.position.get(prod, 0)
            gap = target - pos  # how far we are from target

            if gap == 0:
                continue  # already at target

            if gap > 0:
                # Need to BUY
                best_ask = self._best_ask(od)
                if best_ask:
                    qty = min(gap, POSITION_LIMIT - pos)
                    if qty > 0:
                        orders[prod].append(Order(prod, best_ask[0], qty))

            else:  # gap < 0
                # Need to SELL
                best_bid = self._best_bid(od)
                if best_bid:
                    qty = min(-gap, POSITION_LIMIT + pos)
                    if qty > 0:
                        orders[prod].append(Order(prod, best_bid[0], -qty))

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, state: TradingState) -> tuple:
        """
        Called every tick by the exchange engine.
        Returns (orders_dict, conversions, trader_data).
        """
        orders: Dict[str, List[Order]] = {
            prod: [] for prod in list(HOLD_DIRECTIONS.keys()) + PEBBLES + ['ROBOT_DISHES']
        }

        # 1. PEBBLES constraint arbitrage
        self._pebbles_orders(state, orders)

        # 2. ROBOT_DISHES fast mean reversion
        self._dishes_orders(state, orders)

        # 3. Directional hold for the remaining 44 products
        self._hold_orders(state, orders)

        # Remove empty lists before returning
        result = {k: v for k, v in orders.items() if v}

        trader_data = ""  # no persistence needed (state is stored in self)
        conversions = 0

        return result, conversions, trader_data