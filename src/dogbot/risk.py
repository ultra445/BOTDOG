from __future__ import annotations
import datetime as dt
import time, os
from dataclasses import dataclass
from loguru import logger

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
