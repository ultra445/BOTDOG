# src/dogbot/risk.py
from __future__ import annotations
import datetime as dt
import time, os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from loguru import logger

# =======================
# 1) TON CODE EXISTANT — CONSERVÉ
# =======================

@dataclass
class RiskLimits:
    max_daily_exposure: float
    max_market_stake: float
    block_in_play: bool
    trading_start_hhmm: str
    trading_end_hhmm: str

class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self._reset_day = dt.date.today()
        self._daily_exposure = 0.0
        self._market_exposure = {}
        # throttle warnings
        self._last_warn = {}
        self._warn_throttle = float(os.getenv("RISK_WARN_THROTTLE_SECS", "5.0"))

    def _warn_once(self, key: str, msg: str):
        now = time.time()
        last = self._last_warn.get(key, 0.0)
        if now - last >= self._warn_throttle:
            logger.warning(msg)
            self._last_warn[key] = now

    def _maybe_rollover(self):
        today = dt.date.today()
        if today != self._reset_day:
            logger.info("Rollover: resetting daily exposure")
            self._reset_day = today
            self._daily_exposure = 0.0
            self._market_exposure.clear()

    def within_trading_window(self) -> bool:
        now = dt.datetime.now().time()
        start_h, start_m = map(int, self.limits.trading_start_hhmm.split(":"))
        end_h, end_m = map(int, self.limits.trading_end_hhmm.split(":"))
        start = dt.time(start_h, start_m)
        end = dt.time(end_h, end_m)
        return start <= now <= end

    def can_place(self, market_id: str, intended_stake: float, in_play: bool) -> bool:
        self._maybe_rollover()

        if not self.within_trading_window():
            self._warn_once("window", "Blocked: outside trading window")
            return False

        if self.limits.block_in_play and in_play:
            self._warn_once(f"inplay:{market_id}", "Blocked: market is in-play and policy forbids in-play bets")
            return False

        if self._daily_exposure + intended_stake > self.limits.max_daily_exposure:
            self._warn_once("daily", "Blocked: daily exposure limit reached")
            return False

        mexp = self._market_exposure.get(market_id, 0.0)
        if mexp + intended_stake > self.limits.max_market_stake:
            self._warn_once(f"mexp:{market_id}", "Blocked: per-market exposure limit reached")
            return False

        return True

    def register(self, market_id: str, stake: float):
        self._daily_exposure += stake
        self._market_exposure[market_id] = self._market_exposure.get(market_id, 0.0) + stake

# =======================
# 2) AJOUT : EXPOSURE MANAGER “LIVE-READY”
# =======================

# On réutilise la config de la brique mises (capital, caps, etc.)
from .config import AppConfig, load_config
from .staking import Side

@dataclass
class RiskSnapshot:
    exposure_day: float                         # somme stakes BACK + liabilities LAY (du jour)
    per_market: Dict[str, float]                # market_id -> stake cumulé (BACK) ou somme des stakes posées (LAY aussi)
    per_runner: Dict[Tuple[str,int], float]     # (market_id, selection_id) -> stake cumulé

class ExposureManager:
    """
    Gestion d'exposition runtime pour le passage en réel :
      - MAX_DAILY_EXPOSURE (jour)
      - MAX_MARKET_STAKE (par marché)
      - (cap par runner déjà appliqué dans StakingEngine sur la taille unitaire)
    + compat UTC day rollover.
    """
    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or load_config()
        self._day = dt.datetime.utcnow().strftime("%Y%m%d")
        self._exposure_day = 0.0
        self._per_market: Dict[str, float] = {}
        self._per_runner: Dict[Tuple[str,int], float] = {}

    def _rollover_if_needed(self) -> None:
        today = dt.datetime.utcnow().strftime("%Y%m%d")
        if today != self._day:
            self._day = today
            self._exposure_day = 0.0
            self._per_market.clear()
            self._per_runner.clear()
            logger.info("Exposure rollover (UTC day)")

    def can_place(
        self,
        side: Side,
        *,
        market_id: str,
        selection_id: int,
        planned_stake: float,
        planned_liability: float | None,
    ) -> tuple[bool, str]:
        """
        Vérifie les garde-fous AVANT le place_orders.
        - Pour BACK : on additionne la stake.
        - Pour LAY  : on raisonne en 'exposition' = liability.
        """
        self._rollover_if_needed()

        # Expo 'day' : BACK=stake, LAY=liability
        size_like = float(planned_stake)
        if side == Side.LAY and planned_liability is not None:
            size_like = float(planned_liability)

        if self._exposure_day + size_like > self.cfg.risk.max_daily_exposure:
            return False, "cap_daily_exposure"

        # Cap par marché : on cumule les 'stakes' envoyées sur ce marché
        cur_mkt = self._per_market.get(market_id, 0.0)
        if cur_mkt + float(planned_stake) > self.cfg.risk.max_market_stake:
            return False, "cap_market_stake"

        # (Info) Par runner cumulé : on ne bloque pas ici, StakingEngine applique déjà un cap unitaire
        # Si tu veux bloquer aussi en cumulé runner, tu peux décommenter :
        # key = (market_id, int(selection_id))
        # cur_run = self._per_runner.get(key, 0.0)
        # if cur_run + float(planned_stake) > self.cfg.max_runner_stake:
        #     return False, "cap_runner_stake"

        return True, "ok"

    def on_placed(
        self,
        side: Side,
        *,
        market_id: str,
        selection_id: int,
        stake: float,
        liability: float | None,
    ) -> None:
        """
        À appeler après un place_orders OK (on réserve l'exposition).
        """
        self._rollover_if_needed()

        size_like = float(stake)
        if side == Side.LAY and liability is not None:
            size_like = float(liability)

        self._exposure_day += max(0.0, size_like)
        self._per_market[market_id] = self._per_market.get(market_id, 0.0) + float(stake)
        key = (market_id, int(selection_id))
        self._per_runner[key] = self._per_runner.get(key, 0.0) + float(stake)

    def snapshot(self) -> RiskSnapshot:
        self._rollover_if_needed()
        return RiskSnapshot(
            exposure_day=self._exposure_day,
            per_market=dict(self._per_market),
            per_runner=dict(self._per_runner),
        )
