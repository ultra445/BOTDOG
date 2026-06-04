from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dogbot.gruss.gruss_mapper import GrussRunner, GrussSnapshot


@dataclass(frozen=True)
class GrussMomentumAnchor:
    timestamp: datetime
    countdown_seconds: int
    win_base_by_trap: dict[int, float]
    win_ltp_by_trap: dict[int, float]
    win_best_back_by_trap: dict[int, float]
    place_ltp_by_trap: dict[int, float]
    place_best_back_by_trap: dict[int, float]


@dataclass(frozen=True)
class GrussMomentumValue:
    trap: int
    has_mom45: bool
    mom45: float | None
    source_timestamp: datetime | None
    source_countdown_seconds: int | None
    current_value: float | None
    anchor_value: float | None
    reason: str | None
    win_best_back_anchor: float | None
    win_ltp_anchor: float | None
    place_best_back_anchor: float | None
    place_ltp_anchor: float | None
    first_seen_countdown: int | None
    t45_anchor_found: bool


@dataclass(frozen=True)
class GrussMomentumCourseStatus:
    has_mom45: bool
    mom45_reason: str | None
    first_seen_countdown: int | None
    t45_anchor_found: bool


class GrussMomentumBuffer:
    """Keeps recent Gruss snapshots and exposes the closest T-45 anchor."""

    def __init__(
        self,
        retention_seconds: int = 90,
        anchor_min_seconds: int = 42,
        anchor_max_seconds: int = 48,
    ) -> None:
        self.retention_seconds = retention_seconds
        self.anchor_min_seconds = anchor_min_seconds
        self.anchor_max_seconds = anchor_max_seconds
        self._history: dict[str, deque[GrussMomentumAnchor]] = defaultdict(deque)
        self._captured_anchor_keys: set[str] = set()
        self._first_seen_countdown: dict[str, int] = {}

    def add_snapshot_pair(
        self,
        win_snapshot: GrussSnapshot,
        place_snapshot: GrussSnapshot,
        *,
        timestamp: datetime | None = None,
    ) -> tuple[bool, GrussMomentumAnchor | None]:
        timestamp = timestamp or datetime.now(timezone.utc)
        key = gruss_momentum_key(win_snapshot, place_snapshot)
        seconds = _countdown_seconds(win_snapshot, place_snapshot)
        if seconds is None:
            return False, None
        self._first_seen_countdown.setdefault(key, seconds)

        anchor = _build_anchor(timestamp, seconds, win_snapshot, place_snapshot)
        history = self._history[key]
        history.append(anchor)
        self._prune(history, timestamp)

        captured = False
        if self.anchor_min_seconds <= seconds <= self.anchor_max_seconds:
            if key not in self._captured_anchor_keys:
                self._captured_anchor_keys.add(key)
                captured = True
        return captured, anchor if captured else None

    def closest_t45_anchor(
        self,
        win_snapshot: GrussSnapshot,
        place_snapshot: GrussSnapshot,
    ) -> GrussMomentumAnchor | None:
        key = gruss_momentum_key(win_snapshot, place_snapshot)
        candidates = [
            anchor
            for anchor in self._history.get(key, ())
            if self.anchor_min_seconds <= anchor.countdown_seconds <= self.anchor_max_seconds
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda anchor: abs(anchor.countdown_seconds - 45))

    def momentum_by_trap(
        self,
        win_snapshot: GrussSnapshot,
        place_snapshot: GrussSnapshot,
    ) -> dict[int, GrussMomentumValue]:
        anchor = self.closest_t45_anchor(win_snapshot, place_snapshot)
        traps = {
            runner.trap
            for runner in win_snapshot.runners + place_snapshot.runners
            if runner.trap is not None
        }
        first_seen_countdown = self._first_seen_countdown.get(gruss_momentum_key(win_snapshot, place_snapshot))
        if anchor is None:
            reason = self._missing_anchor_reason(win_snapshot, place_snapshot)
            return {
                int(trap): GrussMomentumValue(
                    trap=int(trap),
                    has_mom45=False,
                    mom45=None,
                    source_timestamp=None,
                    source_countdown_seconds=None,
                    current_value=None,
                    anchor_value=None,
                    reason=reason,
                    win_best_back_anchor=None,
                    win_ltp_anchor=None,
                    place_best_back_anchor=None,
                    place_ltp_anchor=None,
                    first_seen_countdown=first_seen_countdown,
                    t45_anchor_found=False,
                )
                for trap in traps
            }

        result: dict[int, GrussMomentumValue] = {}
        for trap in traps:
            trap = int(trap)
            current_runner = _runner_by_trap(win_snapshot, trap)
            current_base = gruss_win_base_price(current_runner) if current_runner is not None else None
            anchor_base = anchor.win_base_by_trap.get(trap)
            reason = None
            mom45 = None
            if current_runner is None:
                reason = "runner_missing_current"
            elif anchor_base is None:
                reason = "runner_missing_at_t45"
            elif current_base is None:
                reason = "current_base_missing"
            elif anchor_base == 0:
                reason = "anchor_zero"
            else:
                mom45 = (current_base / anchor_base) - 1.0
            result[trap] = GrussMomentumValue(
                trap=trap,
                has_mom45=mom45 is not None,
                mom45=mom45,
                source_timestamp=anchor.timestamp,
                source_countdown_seconds=anchor.countdown_seconds,
                current_value=current_base,
                anchor_value=anchor_base,
                reason=reason,
                win_best_back_anchor=anchor.win_best_back_by_trap.get(trap),
                win_ltp_anchor=anchor.win_ltp_by_trap.get(trap),
                place_best_back_anchor=anchor.place_best_back_by_trap.get(trap),
                place_ltp_anchor=anchor.place_ltp_by_trap.get(trap),
                first_seen_countdown=first_seen_countdown,
                t45_anchor_found=True,
            )
        return result

    def course_status(
        self,
        win_snapshot: GrussSnapshot,
        place_snapshot: GrussSnapshot,
    ) -> GrussMomentumCourseStatus:
        key = gruss_momentum_key(win_snapshot, place_snapshot)
        anchor = self.closest_t45_anchor(win_snapshot, place_snapshot)
        first_seen_countdown = self._first_seen_countdown.get(key)
        if anchor is None:
            return GrussMomentumCourseStatus(
                has_mom45=False,
                mom45_reason=self._missing_anchor_reason(win_snapshot, place_snapshot),
                first_seen_countdown=first_seen_countdown,
                t45_anchor_found=False,
            )
        values = self.momentum_by_trap(win_snapshot, place_snapshot)
        has_mom45 = any(value.has_mom45 for value in values.values())
        reasons = {value.reason for value in values.values() if value.reason}
        reason = "runner_missing_at_t45" if "runner_missing_at_t45" in reasons else (sorted(reasons)[0] if reasons else None)
        return GrussMomentumCourseStatus(
            has_mom45=has_mom45,
            mom45_reason=reason,
            first_seen_countdown=first_seen_countdown,
            t45_anchor_found=True,
        )

    def _missing_anchor_reason(
        self,
        win_snapshot: GrussSnapshot,
        place_snapshot: GrussSnapshot,
    ) -> str:
        key = gruss_momentum_key(win_snapshot, place_snapshot)
        first_seen = self._first_seen_countdown.get(key)
        if first_seen is not None and first_seen < self.anchor_min_seconds:
            return "watcher_started_after_t45"
        return "no_t45_anchor"

    def _prune(self, history: deque[GrussMomentumAnchor], now: datetime) -> None:
        while history and (now - history[0].timestamp).total_seconds() > self.retention_seconds:
            history.popleft()


def gruss_momentum_key(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> str:
    parent_id = win_snapshot.metadata.parent_id or place_snapshot.metadata.parent_id
    if parent_id:
        return f"parent:{parent_id}"
    return f"markets:{win_snapshot.metadata.market_id}:{place_snapshot.metadata.market_id}"


def gruss_win_base_price(runner: GrussRunner | None, moyltp_tolerance_pct: float = 30.0) -> float | None:
    if runner is None:
        return None
    ltp = _positive_or_none(runner.ltp)
    best_back = _positive_or_none(runner.best_back)
    best_lay = _positive_or_none(runner.best_lay)
    mid = ((best_back + best_lay) / 2.0) if best_back is not None and best_lay is not None else None
    moyltp = _trusted_moyltp(ltp, mid, moyltp_tolerance_pct)
    return moyltp or ltp or best_back


def _build_anchor(
    timestamp: datetime,
    countdown_seconds: int,
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
) -> GrussMomentumAnchor:
    return GrussMomentumAnchor(
        timestamp=timestamp,
        countdown_seconds=countdown_seconds,
        win_base_by_trap={
            int(runner.trap): value
            for runner in win_snapshot.runners
            if runner.trap is not None and (value := gruss_win_base_price(runner)) is not None
        },
        win_ltp_by_trap=_value_by_trap(win_snapshot.runners, "ltp"),
        win_best_back_by_trap=_value_by_trap(win_snapshot.runners, "best_back"),
        place_ltp_by_trap=_value_by_trap(place_snapshot.runners, "ltp"),
        place_best_back_by_trap=_value_by_trap(place_snapshot.runners, "best_back"),
    )


def _value_by_trap(runners: list[GrussRunner], field_name: str) -> dict[int, float]:
    values = {}
    for runner in runners:
        if runner.trap is None:
            continue
        value = _positive_or_none(getattr(runner, field_name, None))
        if value is not None:
            values[int(runner.trap)] = value
    return values


def _countdown_seconds(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> int | None:
    seconds = win_snapshot.metadata.countdown_seconds
    if seconds is None:
        seconds = place_snapshot.metadata.countdown_seconds
    return seconds


def _runner_by_trap(snapshot: GrussSnapshot, trap: int) -> GrussRunner | None:
    return next((runner for runner in snapshot.runners if runner.trap == trap), None)


def _trusted_moyltp(ltp: float | None, mid: float | None, tolerance_pct: float) -> float | None:
    if ltp is None or mid is None or ltp <= 0:
        return None
    gap = abs(mid - ltp) / ltp * 100.0
    return ((ltp + mid) / 2.0) if gap <= tolerance_pct else None


def _positive_or_none(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None
