from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
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
    staking_formula: str = ""
    staking_alpha: float | None = None
    staking_back_alpha: float | None = None
    staking_lay_alpha: float | None = None
    stake_raw_before_caps: float | None = None
    stake_after_caps: float | None = None
    lay_liability_after_sizing: float | None = None
    lay_liability_cap: float | None = None
    lay_liability_cap_hit: bool = False

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

    def _size_from_edge(
        self,
        side: Side,
        price_ltp: float,
        edge: float,
        max_runner_cap: Optional[float],
        reason_back: str,
        reason_lay: str,
    ) -> StakingResult:
        price = max(1.01, float(price_ltp))
        back_alpha = float(self.cfg.stake_back_odds_decay_alpha)
        lay_alpha = float(self.cfg.stake_lay_odds_decay_alpha)

        if edge <= 0.0:
            return StakingResult(
                False,
                round_to_tick(price),
                0.0,
                None,
                "edge_zero_or_missing",
                staking_formula="capital_edge_over_odds_power",
                staking_back_alpha=back_alpha,
                staking_lay_alpha=lay_alpha,
            )

        if side == Side.BACK:
            alpha = back_alpha
            stake_raw = (self.cfg.capital * edge) / (price ** alpha)
            liability = None
            reason = reason_back
        else:
            alpha = lay_alpha
            stake_raw = (self.cfg.capital * edge) / (price ** alpha)
            liability = None
            reason = reason_lay

        stake = stake_raw

        # Cap par marche existant
        stake = min(stake, self.cfg.risk.max_market_stake)

        # Cap par runner, global ou fourni par la strategie
        runner_cap = self.cfg.max_runner_stake if max_runner_cap is None else max_runner_cap
        stake = min(stake, float(runner_cap))

        price_req = round_to_tick(price)
        lay_liability_cap = None
        lay_liability_cap_hit = False
        if side == Side.LAY:
            lay_liability_cap = float(self.cfg.max_lay_liability_per_order)
            liability_after_caps = stake * max(0.01, price_req - 1.0)
            if lay_liability_cap > 0.0 and liability_after_caps > lay_liability_cap:
                stake = lay_liability_cap / max(0.01, price_req - 1.0)
                lay_liability_cap_hit = True

        if not isinstance(stake, (int, float)) or not math.isfinite(stake) or stake <= 0.0:
            return StakingResult(
                False,
                price_req,
                0.0,
                None,
                "stake_zero_or_invalid",
                staking_formula="capital_edge_over_odds_power",
                staking_alpha=alpha,
                staking_back_alpha=back_alpha,
                staking_lay_alpha=lay_alpha,
                stake_raw_before_caps=stake_raw,
                stake_after_caps=stake,
                lay_liability_after_sizing=(
                    stake * max(0.01, price_req - 1.0)
                    if side == Side.LAY and isinstance(stake, (int, float)) and math.isfinite(stake)
                    else None
                ),
                lay_liability_cap=lay_liability_cap,
                lay_liability_cap_hit=lay_liability_cap_hit,
            )
        min_stake = 1.0
        if 0.0 < stake < min_stake:
            stake = min_stake

        stake_req = round(stake, 2)

        if side == Side.LAY:
            liability = round(stake_req * max(0.01, price_req - 1.0), 2)

        return StakingResult(
            True,
            price_req,
            stake_req,
            liability,
            reason,
            staking_formula="capital_edge_over_odds_power",
            staking_alpha=alpha,
            staking_back_alpha=back_alpha,
            staking_lay_alpha=lay_alpha,
            stake_raw_before_caps=stake_raw,
            stake_after_caps=stake_req,
            lay_liability_after_sizing=liability,
            lay_liability_cap=lay_liability_cap,
            lay_liability_cap_hit=lay_liability_cap_hit,
        )

    def quote(
        self,
        side: Side,
        price_ltp: float,
        edge: float,
        max_runner_cap: Optional[float] = None,
    ) -> StakingResult:
        return self._size_from_edge(
            side=side,
            price_ltp=price_ltp,
            edge=edge,
            max_runner_cap=max_runner_cap,
            reason_back="back_capital_edge_over_odds_power",
            reason_lay="lay_capital_edge_over_odds_power",
        )

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
        max_runner_cap: Optional[float] = None,
    ) -> StakingResult:
        edge = self._edge_for(family, slot)
        runner_cap = self._runner_cap_for(family, slot) if max_runner_cap is None else max_runner_cap
        return self._size_from_edge(
            side=side,
            price_ltp=price_ltp,
            edge=edge,
            max_runner_cap=runner_cap,
            reason_back="back_capital_edge_over_odds_power",
            reason_lay="lay_capital_edge_over_odds_power",
        )
