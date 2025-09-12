from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .config import AppConfig, load_config

# ================= Types =================

class Side(str, Enum):
    BACK = "BACK"
    LAY = "LAY"

@dataclass
class StakingResult:
    ok: bool
    price: float
    size: float
    liability: Optional[float]
    reason: str

# ================= Odds ladder (ticks) =================
# Barème standard Betfair
_TICKS = [
    (1.01, 2.0, 0.01),
    (2.0, 3.0, 0.02),
    (3.0, 4.0, 0.05),
    (4.0, 6.0, 0.10),
    (6.0, 10.0, 0.20),
    (10.0, 20.0, 0.50),
    (20.0, 30.0, 1.00),
    (30.0, 50.0, 2.00),
    (50.0, 100.0, 5.00),
    (100.0, 1000.0, 10.0),
]

def round_to_tick(price: float) -> float:
    p = max(1.01, min(float(price), 1000.0))
    for lo, hi, step in _TICKS:
        if lo <= p < hi:
            # arrondi "down" de sécurité (coté demande)
            steps = int((p - lo) // step)
            return max(lo, min(lo + steps * step, hi - step))
    return p

# ================= StakingEngine =================

class StakingEngine:
    """
    Sizing simple basé sur:
      - CAPITAL (global, .env)
      - LTP (prix instantané)
      - EDGE_{FAMILY}_{SLOT} (coefficient par slot, .env)

    Règles:
      BACK: stake = CAPITAL * EDGE / LTP
      LAY : liability = CAPITAL * EDGE ; stake = liability / (LTP - 1)

    Contraintes:
      - arrondi au tick
      - MIN_STAKE / MIN_LIABILITY
      - MAX_MARKET_STAKE (cap marché)
      - MAX_RUNNER_STAKE (cap par chien, global ou par slot)
      - MAX_DAILY_EXPOSURE à gérer côté appelant (agrégé)
    """
    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or load_config()

    def _edge_for(self, family: str, slot: int) -> float:
        return self.cfg.edges.get(f"EDGE_{family}_{slot}", 0.0)

    def _runner_cap_for(self, family: str, slot: int) -> float:
        # cap spécifique par slot si défini, sinon cap global
        key = f"MAX_RUNNER_STAKE_{family}_{slot}"
        return float(self.cfg.per_slot_runner_caps.get(key, self.cfg.max_runner_stake))

    def compute(
        self,
        side: Side,
        price_ltp: float,
        family: str,
        slot: int,
        market_id: str,
        selection_id: int,
        course_id: str,
        strategy_tag: str,
    ) -> StakingResult:
        edge = self._edge_for(family, slot)
        price = max(1.01, float(price_ltp))

        if edge <= 0.0:
            return StakingResult(False, round_to_tick(price), 0.0, None, "edge_zero_or_missing")

        if side == Side.BACK:
            stake_raw = (self.cfg.capital * edge) / price
            liability = None
            reason = "back_capital_edge_over_price"
        else:
            liability_raw = self.cfg.capital * edge
            stake_raw = liability_raw / max(0.01, price - 1.0)
            liability = liability_raw
            reason = "lay_capital_edge_liability"

        # Minima
        stake = max(stake_raw, self.cfg.risk.min_stake)
        if side == Side.LAY:
            liability = max(liability or 0.0, self.cfg.risk.min_liability)
            stake = max(stake, liability / max(0.01, price - 1.0))

        # Cap par marché (existant)
        stake = min(stake, self.cfg.risk.max_market_stake)

        # ---- NOUVEAU : cap par chien (global puis par slot si défini) ----
        runner_cap = self._runner_cap_for(family, slot)
        stake = min(stake, runner_cap)

        # Arrondi odds (sécurité)
        price_req = round_to_tick(price)

        # Pour LAY, recalc liability finale avec la stake capée (plus parlant dans le CSV)
        if side == Side.LAY:
            liability = round(stake * max(0.01, price_req - 1.0), 2)

        # Taille finale arrondie à 2 décimales (compat ex. GBP/EUR)
        return StakingResult(True, price_req, round(stake, 2), liability, reason)
