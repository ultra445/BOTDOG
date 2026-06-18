# src/dogbot/executor.py
# Snapshots WIN + PLACE, jalons 300/150/80/45/2
# - BSP_WIN / BSP_PLACE (priorité: SP_EST si dispo, sinon NEAR) + SP_AVAILABLE_* (1/0)
# - WINPROB=(BSP_WIN+LTP_WIN)/2 ; PLACEPROB=(BSP_PLACE+LTP_PLACE)/2
# - MID/MOYLTP avec seuil de confiance MOYLTP_TOLERANCE_PCT (%.env)
# - DIFF/MOM sur BASE_WIN = (WINPROB -> MOYLTP_WIN -> LTP_WIN -> BEST_BACK)
# - PLACETHEORIQUE = cote (1/q) via Plackett–Luce (Top-K) à partir des prix WIN
# - Duplication : PLACETHEORIQUE_PLACE (même valeur que sur WIN) affichée sur les lignes PLACE

from __future__ import annotations
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Dict, List, Tuple, Set
from collections import defaultdict, deque
import csv, math, os, re

from .types import Instruction, RunnerMeta
from .indexer import MarketIndex
from .feeds import create_price_feed_from_env

# --- import Plackett–Luce (supporte plackett OU placket pour éviter la confusion) ---
try:
    from .plackett import odds_to_win_probs, place_probabilities, fair_place_odds
except Exception:
    from .placket import odds_to_win_probs, place_probabilities, fair_place_odds  # type: ignore

# --- AJOUTS (Staking/Stratégies) ---
from .config import ORDER_PROVIDER_GRUSS_EXCEL_REAL, load_config
from .staking import StakingEngine, Side
from . import strategies as _strategies
from .strategies import build_registry, try_fire_slot, RunnerCtx, ExecMode  # <-- ExecMode ajouté
from .pre_ladder import (
    BETFAIR_PRICE_BANDS,
    PRE_LADDER_SYSTEM_IDS,
    build_pre_ladder_from_same_side_offer,
    decide_pre_ladder_step,
    plan_gruss_pre_ladder_trigger,
    round_final_lim_to_ladder_tick,
    round_to_betfair_tick,
)

# --- AJOUTS LIVE (Step 3) ---
from .execution.orders import OrderExecutor
from .risk import ExposureManager

PRE_LADDER_DEFAULT_STEPS_SECONDS = (45, 32, 20, 14)
PRE_SEND_SECONDS_BEFORE_OFF = 45
POST_SEND_SECONDS_BEFORE_OFF = 1
POST_SEND_SECONDS_BEFORE_OFF_DEFAULT = 1


@dataclass
class _StrategyOrderCandidate:
    slot: Any
    market_id: str
    market_type: str
    selection_id: int
    course_id: str
    side: str
    price: float
    size: float
    liability: float
    reason: str
    exec_mode: ExecMode
    sp_limit: float | None
    execution_phase: str
    triggered_systems: list[str]
    triggered_prices: list[float]
    bet_per_market_key: tuple[str, int, str]
    phase_send_seconds_before_off: int | None = None
    best_unmatched_back_offer: float | None = None
    best_unmatched_lay_offer: float | None = None
    market_reference_price: float | None = None
    strategy_group: Any = None
    strategy_region: Any = None
    strategy_signal: Any = None
    strategy_bucket: Any = None
    strategy_edge: float | None = None
    strategy_score: float | None = None
    staking_formula: str = ""
    staking_alpha: float | None = None
    staking_back_alpha: float | None = None
    staking_lay_alpha: float | None = None
    stake_raw_before_caps: float | None = None
    stake_after_caps: float | None = None
    lay_liability_after_sizing: float | None = None
    lay_liability_cap: float | None = None
    lay_liability_cap_hit: bool = False
    runner_name: str = ""
    conflict_detected: bool = False
    conflict_type: str = ""
    conflict_group_key: str = ""
    conflict_candidates_count: int = 0
    selected_side: str = ""
    rejected_side: str = ""
    back_systems: str = ""
    lay_systems: str = ""
    conflict_resolution_reason: str = ""
    pre_back_lay_conflict: bool = False
    pre_conflict_resolution: str = ""
    pre_conflict_chosen_side: str = ""
    pre_conflict_rejected_side: str = ""
    pre_conflict_reason: str = ""
    pre_conflict_group_key: str = ""
    pre_conflict_course_id: str = ""
    pre_conflict_market_id: str = ""
    pre_conflict_market_type: str = ""
    pre_conflict_selection_id: str = ""
    pre_conflict_runner_name: str = ""
    pre_back_target_price: float | None = None
    pre_lay_target_price: float | None = None
    pre_current_best_lay: float | None = None
    pre_current_best_back: float | None = None
    pre_back_distance_ticks: float | None = None
    pre_lay_distance_ticks: float | None = None

    @property
    def merge_key(self) -> tuple[str, int, str, str, str]:
        return (
            self.market_id,
            self.selection_id,
            self.market_type,
            self.side,
            self.execution_phase,
        )

    @property
    def final_system(self) -> str:
        return self.triggered_systems[0] if self.triggered_systems else str(getattr(self.slot, "tag", ""))


@dataclass(frozen=True)
class _FrozenPreLadderPlan:
    ladder_id: str
    side: str
    start_price: float | None
    start_price_raw: float | None
    start_price_tick: float | None
    final_lim_price_raw: float
    final_lim_price_tick: float
    ladder_prices: list[float]
    reason: str
    created_step: int
    created_at_countdown: int
    best_same_side_offer_at_creation: float | None
    ladder_direction: str
    disabled_lim_not_in_ladder_direction: bool = False
    invalid_reason: str = ""


def _execution_phase_for_milestone(milestone: int | None) -> str | None:
    if milestone in _pre_ladder_steps_from_env():
        return _strategies.EXECUTION_PHASE_PRE
    if milestone == _post_send_seconds_before_off_from_env():
        return _strategies.EXECUTION_PHASE_POST
    return None


def _pre_ladder_steps_from_env() -> tuple[int, ...]:
    raw = os.getenv("DOGBOT_PRE_LADDER_STEPS")
    if raw in (None, ""):
        return PRE_LADDER_DEFAULT_STEPS_SECONDS
    steps: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if value > 0:
            steps.append(value)
    return tuple(steps or PRE_LADDER_DEFAULT_STEPS_SECONDS)


def _post_send_seconds_before_off_from_env() -> int:
    raw = os.getenv("DOGBOT_POST_SEND_SECONDS_BEFORE_OFF")
    if raw in (None, ""):
        return POST_SEND_SECONDS_BEFORE_OFF_DEFAULT
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return POST_SEND_SECONDS_BEFORE_OFF_DEFAULT
    return min(60, max(0, value))


def _pre_ladder_prices_are_valid_for_side(
    side: str,
    prices: list[float],
    final_lim_price_tick: float,
    *,
    direct_plan: bool,
) -> bool:
    if not prices:
        return False
    upper_side = str(side or "").upper()
    if upper_side == "BACK":
        if any(price < final_lim_price_tick for price in prices):
            return False
        if direct_plan:
            return all(abs(price - final_lim_price_tick) <= 1e-9 for price in prices)
        if len(prices) > 1 and len(set(prices)) == 1:
            return False
        return all(prices[index] >= prices[index + 1] for index in range(len(prices) - 1))
    if upper_side == "LAY":
        if any(price > final_lim_price_tick for price in prices):
            return False
        if direct_plan:
            return all(abs(price - final_lim_price_tick) <= 1e-9 for price in prices)
        if len(prices) > 1 and len(set(prices)) == 1:
            return False
        return all(prices[index] <= prices[index + 1] for index in range(len(prices) - 1))
    return False


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_false(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"0", "false", "no", "n", "off"}


def _gruss_real_post_ready_for_trade_log() -> bool:
    return (
        _env_false("DRY_RUN")
        and str(os.getenv("DOGBOT_ORDER_PROVIDER", "")).strip().lower() == ORDER_PROVIDER_GRUSS_EXCEL_REAL
        and _env_flag("DOGBOT_GRUSS_ENABLE_REAL_ORDERS", False)
        and not _env_flag("DOGBOT_GRUSS_REAL_PREVIEW", False)
        and not _env_flag("DOGBOT_GRUSS_WRITE_NO_TRIGGER", False)
    )


def _merge_order_candidates(
    candidates: list[_StrategyOrderCandidate],
) -> list[_StrategyOrderCandidate]:
    grouped: dict[tuple[str, int, str, str, str], list[_StrategyOrderCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.merge_key].append(candidate)

    merged: list[_StrategyOrderCandidate] = []
    for group in grouped.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        first = group[0]
        price = min(candidate.price for candidate in group) if first.side == "BACK" else max(
            candidate.price for candidate in group
        )
        size = sum(candidate.size for candidate in group)
        liability = sum(candidate.liability for candidate in group)
        systems = [system for candidate in group for system in candidate.triggered_systems]
        prices = [price for candidate in group for price in candidate.triggered_prices]
        merged.append(
            replace(
                first,
                price=price,
                size=round(size, 2),
                liability=round(liability, 2),
                reason=f"merged_{first.execution_phase.lower()} systems={','.join(systems)}",
                triggered_systems=systems,
                triggered_prices=prices,
            )
        )
    return merged


def _resolve_back_lay_same_phase_candidates(
    candidates: list[_StrategyOrderCandidate],
) -> tuple[list[_StrategyOrderCandidate], list[_StrategyOrderCandidate]]:
    grouped: dict[tuple[str, str, int, str, str], list[_StrategyOrderCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[
            (
                candidate.course_id,
                candidate.market_id,
                candidate.selection_id,
                candidate.market_type,
                candidate.execution_phase,
            )
        ].append(candidate)

    rejected_ids: set[int] = set()
    selected_replacements: dict[int, _StrategyOrderCandidate] = {}
    rejected: list[_StrategyOrderCandidate] = []
    for key, group in grouped.items():
        backs = [candidate for candidate in group if candidate.side == "BACK"]
        lays = [candidate for candidate in group if candidate.side == "LAY"]
        if not backs or not lays:
            continue
        back_systems = [system for candidate in backs for system in candidate.triggered_systems]
        lay_systems = [system for candidate in lays for system in candidate.triggered_systems]
        if str(key[4] or "").upper() == _strategies.EXECUTION_PHASE_PRE:
            selected_side, pre_reason, back_distance, lay_distance = _pre_conflict_choice(backs, lays)
            back_price = _first_positive_price(backs)
            lay_price = _first_positive_price(lays)
            best_lay = _first_positive_reference(backs, reference_side="LAY")
            best_back = _first_positive_reference(lays, reference_side="BACK")
            pre_base = {
                "conflict_detected": True,
                "conflict_type": "back_lay_same_runner_market_phase",
                "conflict_group_key": "|".join(str(part) for part in key),
                "conflict_candidates_count": len(backs) + len(lays),
                "back_systems": "|".join(back_systems),
                "lay_systems": "|".join(lay_systems),
                "conflict_resolution_reason": "per_runner_nearest_price",
                "pre_back_lay_conflict": True,
                "pre_conflict_resolution": "per_runner_nearest_price",
                "pre_conflict_chosen_side": selected_side,
                "pre_conflict_rejected_side": "LAY" if selected_side == "BACK" else ("BACK" if selected_side == "LAY" else "BOTH"),
                "pre_conflict_reason": pre_reason,
                "back_distance": back_distance,
                "lay_distance": lay_distance,
                "selected_side": selected_side if selected_side in {"BACK", "LAY"} else "NONE",
                "rejected_side": "LAY" if selected_side == "BACK" else ("BACK" if selected_side == "LAY" else "BOTH"),
                "pre_conflict_group_key": "|".join(str(part) for part in key),
                "pre_conflict_course_id": str(key[0]),
                "pre_conflict_market_id": str(key[1]),
                "pre_conflict_selection_id": str(key[2]),
                "pre_conflict_market_type": str(key[3]),
                "pre_conflict_runner_name": backs[0].runner_name or lays[0].runner_name,
                "pre_back_target_price": back_price,
                "pre_lay_target_price": lay_price,
                "pre_current_best_lay": best_lay,
                "pre_current_best_back": best_back,
                "pre_back_distance_ticks": back_distance,
                "pre_lay_distance_ticks": lay_distance,
            }
            if selected_side in {"BACK", "LAY"}:
                for candidate in group:
                    if candidate.side == selected_side:
                        selected_replacements[id(candidate)] = replace(candidate, **pre_base)
                        continue
                    if candidate.side not in {"BACK", "LAY"}:
                        continue
                    rejected_ids.add(id(candidate))
                    rejected.append(replace(candidate, **pre_base, reason="conflicting_back_lay_lost_priority"))
                for candidate in group:
                    if candidate.side == selected_side:
                        rejected_ids.discard(id(candidate))
                continue
            fields = {**pre_base, "reason": pre_reason}
            for candidate in group:
                if candidate.side not in {"BACK", "LAY"}:
                    continue
                rejected_ids.add(id(candidate))
                rejected.append(replace(candidate, **fields))
            continue

        fields = {
            "conflict_detected": True,
            "conflict_type": "back_lay_same_runner_market_phase",
            "conflict_group_key": "|".join(str(part) for part in key),
            "conflict_candidates_count": len(backs) + len(lays),
            "selected_side": "NONE",
            "rejected_side": "BOTH",
            "back_systems": "|".join(back_systems),
            "lay_systems": "|".join(lay_systems),
            "conflict_resolution_reason": "conflicting_back_lay_no_bet",
            "reason": "conflicting_back_lay_no_bet",
        }
        for candidate in group:
            if candidate.side not in {"BACK", "LAY"}:
                continue
            rejected_ids.add(id(candidate))
            rejected.append(replace(candidate, **fields))

    selected = [
        selected_replacements.get(id(candidate), candidate)
        for candidate in candidates
        if id(candidate) not in rejected_ids
    ]
    return selected, rejected


def _pre_conflict_choice(
    backs: list[_StrategyOrderCandidate],
    lays: list[_StrategyOrderCandidate],
) -> tuple[str, str, float | None, float | None]:
    back_distance = _nearest_pre_conflict_distance(backs, reference_side="LAY")
    lay_distance = _nearest_pre_conflict_distance(lays, reference_side="BACK")
    if back_distance is None or lay_distance is None:
        return "NONE", "pre_conflict_missing_reference_no_bet", back_distance, lay_distance
    if math.isclose(back_distance, lay_distance, rel_tol=1e-9, abs_tol=1e-9):
        return "NONE", "pre_conflict_equal_distance_no_bet", back_distance, lay_distance
    if back_distance < lay_distance:
        return "BACK", "pre_conflict_back_nearer", back_distance, lay_distance
    return "LAY", "pre_conflict_lay_nearer", back_distance, lay_distance


def _first_positive_price(candidates: list[_StrategyOrderCandidate]) -> float | None:
    for candidate in candidates:
        value = _positive_float(candidate.price)
        if value is not None:
            return value
    return None


def _first_positive_reference(candidates: list[_StrategyOrderCandidate], *, reference_side: str) -> float | None:
    for candidate in candidates:
        reference = (
            candidate.best_unmatched_lay_offer
            if reference_side == "LAY"
            else candidate.best_unmatched_back_offer
        )
        value = _positive_float(reference)
        if value is not None:
            return value
    return None


def _nearest_pre_conflict_distance(
    candidates: list[_StrategyOrderCandidate],
    *,
    reference_side: str,
) -> float | None:
    distances: list[float] = []
    for candidate in candidates:
        reference = (
            candidate.best_unmatched_lay_offer
            if reference_side == "LAY"
            else candidate.best_unmatched_back_offer
        )
        distance = _betfair_tick_distance(candidate.price, reference)
        if distance is not None:
            distances.append(distance)
    return min(distances) if distances else None


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _betfair_tick_distance(price: Any, reference: Any) -> float | None:
    price_value = _positive_float(price)
    reference_value = _positive_float(reference)
    if price_value is None or reference_value is None:
        return None
    ticks = _betfair_ticks_for_distance()
    try:
        price_tick = round_to_betfair_tick(price_value)
        reference_tick = round_to_betfair_tick(reference_value)
        return float(abs(ticks.index(price_tick) - ticks.index(reference_tick)))
    except (ValueError, TypeError):
        return None


def _betfair_ticks_for_distance() -> tuple[float, ...]:
    ticks: list[float] = []
    for start, end, step in BETFAIR_PRICE_BANDS:
        value = start
        while value < end - 1e-9:
            rounded = round(value, 2)
            if not ticks or not math.isclose(ticks[-1], rounded, rel_tol=1e-9, abs_tol=1e-9):
                ticks.append(rounded)
            value += step
    ticks.append(1000.0)
    return tuple(ticks)


def _pre_pipeline_candidate_id(ctx: RunnerCtx, side: str, strategy_id: str) -> str:
    return "|".join(
        str(part)
        for part in (
            ctx.course_id,
            ctx.market_id,
            ctx.market_type,
            ctx.selection_id,
            ctx.execution_phase,
            side,
            strategy_id,
        )
    )


def _runner_debug_summary(runner_name: Optional[str], signals: Iterable[dict[str, Any]]) -> str:
    values = [
        f"{signal.get('side')}:{signal.get('strategy_id')}"
        for signal in signals
    ]
    return _value_by_runner(runner_name, "|".join(str(value) for value in values if value))


def _candidate_debug_summary(
    runner_name: Optional[str],
    candidates: Iterable[_StrategyOrderCandidate],
) -> str:
    values = [
        f"{candidate.side}:{'&'.join(candidate.triggered_systems)}"
        for candidate in candidates
    ]
    return _value_by_runner(runner_name, "|".join(str(value) for value in values if value))


def _value_by_runner(runner_name: Optional[str], value: str) -> str:
    name = str(runner_name or "").strip() or "<unknown>"
    return f"{name}={value}" if value else f"{name}="


def _relative_distance(price: float | None, reference: float | None) -> float | None:
    try:
        price_value = float(price)
        reference_value = float(reference)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price_value) or not math.isfinite(reference_value) or reference_value <= 0:
        return None
    return abs(price_value - reference_value) / reference_value


# ---------- utilitaires généraux ----------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _tz_utc(dt: Optional[datetime]):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _best_at_side(ex_obj, side: str) -> tuple[Optional[float], Optional[float]]:
    if ex_obj is None:
        return (None, None)
    ladder = getattr(ex_obj, "available_to_back", None) if side == "BACK" else getattr(ex_obj, "available_to_lay", None)
    if not ladder:
        return (None, None)
    top = ladder[0]
    try:
        return float(top.price), float(top.size)
    except Exception:
        p = top.get("price") if isinstance(top, dict) else None
        s = top.get("size") if isinstance(top, dict) else None
        return (float(p) if p is not None else None, float(s) if s is not None else None)

def _compress_ladder(ladder, n=3) -> Optional[str]:
    if not ladder:
        return None
    items = []
    for x in ladder[:n]:
        try:
            items.append(f"{x.price}:{x.size}")
        except Exception:
            p = x.get("price") if isinstance(x, dict) else None
            s = x.get("size") if isinstance(x, dict) else None
            if p is None or s is None:
                continue
            items.append(f"{p}:{s}")
    return "|".join(items) if items else None

def _runner_lpt_or_back(r) -> Optional[float]:
    lpt = getattr(r, "last_price_traded", None)
    try:
        if lpt is not None:
            return float(lpt)
    except Exception:
        pass
    ex = getattr(r, "ex", None)
    pb, _ = _best_at_side(ex, "BACK")
    return pb


def _finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _market_status(md) -> Optional[str]:
    try:
        return getattr(md, "status", None)
    except Exception:
        return None

def _num_winners(md) -> Optional[int]:
    try:
        v = getattr(md, "number_of_winners", None) or getattr(md, "numberOfWinners", None)
        return int(v) if v is not None else None
    except Exception:
        return None

def _fallback_k_place_used(n_active: int) -> int:
    return 3 if n_active >= 8 else 2

def _active_count(runners) -> int:
    n = 0
    for r in (runners or []):
        st = getattr(r, "status", None)
        if (st or "ACTIVE").upper() == "ACTIVE":
            n += 1
    return n

def _overround_back(runners) -> Optional[float]:
    s = 0.0; k = 0
    for r in (runners or []):
        pb = _best_at_side(getattr(r, "ex", None), "BACK")[0]
        if pb and pb > 1e-9:
            s += 1.0 / pb; k += 1
    return (s * 100.0) if k else None

def _overround_lay(runners) -> Optional[float]:
    s = 0.0; k = 0
    for r in (runners or []):
        pl = _best_at_side(getattr(r, "ex", None), "LAY")[0]
        if pl and pl > 1e-9:
            s += 1.0 / pl; k += 1
    return (s * 100.0) if k else None

def _spread_pct(runners) -> Optional[float]:
    vals = []
    for r in (runners or []):
        pb = _best_at_side(getattr(r, "ex", None), "BACK")[0]
        pl = _best_at_side(getattr(r, "ex", None), "LAY")[0]
        if pb and pl and pb > 0:
            mid = (pb + pl) / 2.0
            if mid > 0:
                vals.append((pl - pb) / mid)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals)//2] * 100.0

def _first_positive(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 1.0:
            return number
    return None

def _market_reference_price(
    last_traded_price: Optional[float],
    best_back: Optional[float],
    best_lay: Optional[float],
) -> Optional[float]:
    midpoint = None
    if best_back is not None and best_lay is not None:
        try:
            midpoint = (float(best_back) + float(best_lay)) / 2.0
        except (TypeError, ValueError):
            midpoint = None
    return _first_positive(last_traded_price, midpoint, best_back, best_lay)

def _rankings(runners) -> tuple[Dict[int,int], Dict[int,int]]:
    arr_ltp, arr_bb = [], []
    for r in (runners or []):
        sid = getattr(r, "selection_id", None)
        if sid is None: continue
        lpt = _runner_lpt_or_back(r)
        ex = getattr(r, "ex", None)
        pb, _ = _best_at_side(ex, "BACK")
        if lpt is not None: arr_ltp.append((sid, lpt))
        if pb is not None:  arr_bb.append((sid, pb))
    arr_ltp.sort(key=lambda t: t[1])
    arr_bb.sort(key=lambda t: t[1])
    return ({sid:i+1 for i,(sid,_) in enumerate(arr_ltp)},
            {sid:i+1 for i,(sid,_) in enumerate(arr_bb)})


def _gor_bin(gor: float | None) -> tuple[float | None, float | None, float | None]:
    """Return (gapmin, gapmax, gor) per bin rules. Bins: [1.00,1.10,1.25,1.50,2.00,3.00,inf)
    If gor is None -> (None,None,None).
    """
    if gor is None:
        return (None, None, None)
    edges = [1.0, 1.10, 1.25, 1.50, 2.00, 3.00]
    if gor < edges[0]:
        return (edges[0], edges[0], gor)  # shouldn't happen, clamp
    for i in range(len(edges)-1):
        left, right = edges[i], edges[i+1]
        # left-closed, right-open
        if gor >= left and gor < right:
            return (left, right, gor)
    # last bin: >= 3.00
    return (edges[-1], float('inf'), gor)

# -------- TRAP parsing --------
_TRAP_PATTS = [
    re.compile(r"^\s*(?:TRAP|T)\s*([1-9]\d?)\b", re.I),
    re.compile(r"^\s*([1-9]\d?)\s*[.\-)\s]"),
    re.compile(r"\(\s*(?:TRAP|T)\s*([1-9]\d?)\s*\)", re.I),
]
def _parse_trap(meta: Optional[RunnerMeta], runner_name: Optional[str]) -> Optional[int]:
    for v in [getattr(meta, "trap", None), getattr(meta, "draw", None)]:
        if v is None: continue
        s = str(v).strip()
        m = re.match(r"^[^\d]*([1-9]\d?)", s)
        if m:
            try: return int(m.group(1))
            except Exception: pass
    if runner_name:
        for patt in _TRAP_PATTS:
            m = patt.search(runner_name)
            if m:
                try: return int(m.group(1))
                except Exception: continue
    return None

class _MetaStub:
    def __init__(self, runner_name=None, draw=None, sort_priority=None, trap=None):
        self.runner_name = runner_name
        self.draw = draw
        self.sort_priority = sort_priority
        self.trap = trap


class Executor:
    MILESTONES = sorted(
        {300, 150, 80, 45, 2, *PRE_LADDER_DEFAULT_STEPS_SECONDS, _post_send_seconds_before_off_from_env()},
        reverse=True,
    )
    TOLERANCE_S = 1.5

    MARKET_HEADER = [
        "SNAP_TS_UTC","MARKET_ID","COURSE_ID","MARKET_TYPE",
        "EVENT_ID","EVENT_NAME","VENUE","COUNTRY_CODE",
        "MARKET_START_TIME_UTC","SECONDS_TO_OFF",
        "INPLAY","MARKET_STATUS","NUMBER_OF_WINNERS","N_RUNNERS_ACTIVE","MARKET_TOTAL_MATCHED",
        "BACK_OVERROUND","LAY_OVERROUND","SPREAD_PCT",
        "WIN_MARKET_ID","PLACE_MARKET_ID","COURSE_LINK_OK","N_PLACES",
        "RULE_OK_TIME_WINDOW","RULE_OK_PRICE_BOUNDS","RULE_OK_LIQUIDITY","RULE_OK_OVERROUND","RULES_ALL_OK",
        "STRATEGY_NAME","SIGNAL","SIDE","TARGET_PRICE","STAKE","PERSISTENCE","HEDGE_TICKS","STOP_TICKS",
        "EST_LIABILITY","REASON_CODE","DRYRUN_WOULD_PLACE",
        "MILESTONE_S",
        "FETCH_LATENCY_MS","CACHE_AGE_S","RETRY_COUNT","THROTTLE_WEIGHT","CODE_VERSION",
    ]

    RUNNER_HEADER = [
    "SNAP_TS_UTC","COURSE_ID","VENUE","MARKET_START_TIME_UTC","SECONDS_TO_OFF","MILESTONE_S",
    "WIN_MARKET_ID","PLACE_MARKET_ID","MARKET_ID","MARKET_TYPE",
    "SELECTION_ID","RUNNER_NAME","RUNNER_STATUS","DRAW","TRAP","VIRTUAL_TRAP","SORT_PRIORITY",
    "N_RUNNERS_ACTIVE","N_PLACES",
    # WIN
    "LTP_WIN","BEST_BACK_PRICE_1_WIN","BEST_BACK_SIZE_1_WIN","BEST_LAY_PRICE_1_WIN","BEST_LAY_SIZE_1_WIN",
    "MID_WIN","MOYLTP_WIN",
    "BACK_LADDER_WIN","LAY_LADDER_WIN","RUNNER_TOTAL_MATCHED_WIN",
    "FAV_RANK_LTP_WIN","FAV_RANK_BACK_WIN","WIN_IMPLIED_PROB_WIN",
    "BSP_WIN","SP_AVAILABLE_WIN","WINPROB",
    "LTP_300_WIN","LTP_150_WIN","LTP_80_WIN","LTP_45_WIN","LTP_2_WIN",
    "DIFF150_300_WIN","DIFF80_150_WIN","DIFF45_80_WIN",
    "MOM45_WIN","MOM80_WIN","MOM150_WIN","MOM300_WIN",
    "PRICE_DELTA_5S_WIN","PRICE_DELTA_30S_WIN","VOLATILITY_60S_WIN","LIQUIDITY_SCORE_WIN",
    "IS_FAVOURITE_WIN","PLACETHEORIQUE",
    # PLACE
    "LTP_PLACE","BEST_BACK_PRICE_1_PLACE","BEST_BACK_SIZE_1_PLACE","BEST_LAY_PRICE_1_PLACE","BEST_LAY_SIZE_1_PLACE",
    "MID_PLACE","MOYLTP_PLACE",
    "BACK_LADDER_PLACE","LAY_LADDER_PLACE","RUNNER_TOTAL_MATCHED_PLACE",
    "BSP_PLACE","SP_AVAILABLE_PLACE","PLACEPROB",
    "PLACETHEORIQUE_PLACE","EV_PLACE",
    "LTP_300_PLACE","LTP_150_PLACE","LTP_80_PLACE","LTP_45_PLACE","LTP_2_PLACE",
    # NEW: gap @ T−2s (WIN; dupliqué sur PLACE depuis le WIN lié)
    "GAPMIN","GAPMAX","GOR",
]

    TRADE_HEADER = [
        "ts","run_id","evaluation_id","parent_market_id","milestone","complete_after_post",
        "post_checked","post_signal_count","post_evaluated","post_missing_reason",
        "market_id","market_type","selection_id","course_id",
        "side","price_req","size_req","liability","strategy",
        "market_family","strategy_group","strategy_region","strategy_signal","strategy_bucket",
        "strategy_edge","strategy_score",
        "staking_formula","staking_alpha","staking_back_alpha","staking_lay_alpha",
        "stake_raw_before_caps","stake_after_caps","lay_liability_after_sizing",
        "lay_liability_cap","lay_liability_cap_hit",
        "execution_phase","triggered_systems","triggered_prices","final_system","final_price","final_stake","merged",
        "stake_pre","stake_post","total_stake_same_runner_side","pre_post_cumulative",
        "pre_existing_order_detected","pre_existing_order_side","pre_existing_order_market_type",
        "pre_existing_order_stake","post_stake",
        "processed_key","exec_mode",
        "ladder_enabled","ladder_preview","ladder_id","ladder_tracking_key","ladder_step",
        "ladder_seconds_before_off","final_lim_price","final_lim_price_raw","final_lim_price_tick",
        "start_price","start_price_raw","start_price_tick","tick_rounding_mode","ladder_prices",
        "ladder_plan_frozen","ladder_plan_created_step","ladder_prices_frozen",
        "current_ladder_price_from_frozen_plan","best_same_side_offer_at_creation",
        "best_back_displayed","best_lay_displayed","start_price_source",
        "ladder_direction","ladder_disabled_lim_not_in_ladder_direction",
        "direct_lim_order_planned","direct_lim_order_written","no_replace_steps_for_direct_lim",
        "current_ladder_price","best_unmatched_back_offer","best_unmatched_lay_offer",
        "best_same_side_back_offer","best_same_side_lay_offer","source_fields_used",
        "market_reference_price_at_signal",
        "no_better_ladder_range_reason",
        "previous_order_status","matched_stake","remaining_stake","cancelled_previous",
        "cancel_failed","stop_reason","current_step_stake",
        "gruss_planned_trigger","gruss_trigger_allowed","gruss_bet_ref_required",
        "gruss_bet_ref_present","gruss_bet_ref","gruss_replace_confirmed",
        "gruss_no_stack",
        "conflict_detected","conflict_type","conflict_group_key","conflict_candidates_count",
        "selected_side","rejected_side","back_systems","lay_systems","conflict_resolution_reason",
        "pre_back_lay_conflict","pre_conflict_resolution","pre_conflict_chosen_side",
        "pre_conflict_rejected_side","pre_conflict_reason",
        "pre_conflict_group_key","pre_conflict_course_id","pre_conflict_market_id",
        "pre_conflict_market_type","pre_conflict_selection_id","pre_conflict_runner_name",
        "pre_back_target_price","pre_lay_target_price","pre_current_best_lay",
        "pre_current_best_back","pre_back_distance_ticks","pre_lay_distance_ticks",
        "status","reason",
    ]

    STRATEGY_DEBUG_HEADER = [
        "ts","market_id","market_type","selection_id","runner_name",
        "milestone","secs_to_off","tag","market_family","strategy_group","strategy_signal",
        "execution_phase",
        "condition_result","fail_reason",
        "trap","region","winbet","place_price","ev_place","has_mom45","mom45","mom45_source",
        "mom45_injected_before_strategy_eval","place_theo","bb","bl",
        "place_winners","k_place_used","fallback_k_place_used","requires_mom45",
    ]

    STRATEGY_EVAL_SUMMARY_HEADER = [
        "ts","market_id","market_type","selection_id","runner_name",
        "trap","region","winbet","place_price","place_theo","ev_place","has_mom45","mom45",
        "mom45_available_count","mom45_missing_count","mom45_source",
        "mom45_injected_before_strategy_eval","mom45_strategy_slots_evaluated_count",
        "mom45_strategy_slots_missing_count",
        "place_winners","k_place_used","fallback_k_place_used",
        "execution_phase","slots_tested","conditions_true_count","true_tags","error_count",
    ]

    PRE_PIPELINE_DEBUG_HEADER = [
        "ts","row_type",
        "course_id","market_id","market_type",
        "pre_pipeline_course_id","pre_pipeline_market_id","pre_pipeline_market_type",
        "pre_pipeline_region","pre_pipeline_runner_count",
        "selection_id","runner_name","trap",
        "candidate_id","side","strategy_id","execution_phase",
        "created_from_tag","candidate_created","removed",
        "removed_stage","removed_reason","removed_detail",
        "pre_strategy_signals_count","pre_strategy_signals_by_runner",
        "pre_strategy_back_signals_by_runner","pre_strategy_lay_signals_by_runner",
        "pre_raw_candidates_count","pre_raw_candidates_by_runner",
        "pre_raw_back_candidates_by_runner","pre_raw_lay_candidates_by_runner",
        "pre_after_filters_count","pre_after_filters_by_runner",
        "pre_removed_candidates_count","pre_removed_candidates_detail",
        "pre_conflict_groups_count","pre_conflict_groups_by_runner","pre_conflict_group_key",
        "pre_after_conflict_resolution_count","pre_after_conflict_resolution_by_runner",
        "pre_after_ladder_planning_count","pre_after_ladder_planning_by_runner",
        "pre_trades_created_count","pre_trades_created_by_runner",
        "pre_gruss_orders_attempted_count","pre_gruss_orders_attempted_by_runner",
        "strategy_back_signal","strategy_lay_signal",
        "raw_back_candidate_created","raw_lay_candidate_created",
        "conflict_detected","chosen_side","rejected_side",
        "conflict_group_key","conflict_course_id","conflict_market_id",
        "conflict_market_type","conflict_selection_id","conflict_runner_name",
        "conflict_execution_phase","conflict_back_candidates_count",
        "conflict_lay_candidates_count","conflict_resolution",
        "conflict_chosen_side","conflict_rejected_side","conflict_reason",
        "final_pre_order_created","removed_reason_if_no_order",
        "lay_signal_seen","lay_candidate_created","lay_reached_conflict_resolver",
        "lay_removed_stage","lay_removed_reason","lay_selected",
        "lay_reached_trade","lay_reached_gruss_writer",
        "pre_lay_signal_seen","pre_lay_candidate_created","pre_lay_candidate_removed",
        "pre_lay_removed_stage","pre_lay_removed_reason",
        "pre_lay_reached_conflict_resolver","pre_lay_reached_ladder_planner",
        "pre_lay_reached_gruss_writer",
        "pre_existing_order_detected","pre_existing_order_side",
        "pre_existing_order_market_type","pre_existing_order_stake",
        "post_stake","total_stake_same_runner_side","pre_post_cumulative",
    ]


    ENTRY_MIN_T_S = 120
    ENTRY_MAX_T_S = 7200
    PRICE_MIN = 1.30
    PRICE_MAX = 12.0
    MIN_LIQUIDITY_MARKET = 0.0

    def __init__(self, client: Any, strategy, market_index: MarketIndex, dry_run: bool = True, data_dir: str = "./data"):
        self.client = client
        self.strategy = strategy
        self.market_index = market_index
        self.dry_run = dry_run
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.market_csv = self.data_dir / f"{today}_snapshots.csv"
        self.runner_csv = self.data_dir / f"{today}_snapshots_runners.csv"
        self._ensure_header(self.market_csv, self.MARKET_HEADER)
        self._ensure_header(self.runner_csv, self.RUNNER_HEADER)

        self._pre_ladder_steps = _pre_ladder_steps_from_env()
        self._configured_milestones = sorted(
            {300, 150, 80, 45, 2, *self._pre_ladder_steps, _post_send_seconds_before_off_from_env()},
            reverse=True,
        )
        self._next_ms: Dict[str, List[int]] = defaultdict(lambda: list(self._configured_milestones))
        self._last_tto: Dict[str, float] = {}

        # Milestones:
        self._base_win_ms: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        self._ltp_place_ms: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))

        # Historique LTP (pour d5/d30/vol)
        self._hist: Dict[str, Dict[int, deque[Tuple[float,float]]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=600)))

        # Cache PLACETHEORIQUE -> duplication PLACE
        self._last_place_theo_by_market: Dict[str, Dict[int, float]] = defaultdict(dict)
        self._last_base_win_by_market: Dict[str, Dict[int, float]] = defaultdict(dict)
        self._last_ev_place_by_market: Dict[str, Dict[int, float]] = defaultdict(dict)
        self._last_rank_ltp_by_market: Dict[str, Dict[int, int]] = defaultdict(dict)
        self._last_rank_back_by_market: Dict[str, Dict[int, int]] = defaultdict(dict)
        self._last_mom45_by_market: Dict[str, Dict[int, float]] = defaultdict(dict)
        self._place_winners_by_market: Dict[str, int] = {}
        self._k_place_used_by_market: Dict[str, int] = {}
        self._fallback_k_place_used_by_market: Dict[str, bool] = {}
        self._k_place_fallback_logged: Set[str] = set()

        # Cache TRAP & GAP by race (shared between WIN/PLACE)
        self._trap_by_race: dict[tuple[str,int], int] = {}
        self._gap_by_race: dict[tuple[str,int], tuple[float|None,float|None,float|None]] = {}


        # BSP / SP feed (REST aujourd'hui, STREAM demain sans toucher l'executor)
        self.price_feed = create_price_feed_from_env()

        # Diag
        self._diag_fetch_latency_ms: Optional[float] = None
        self._diag_retry_count: Optional[float] = None
        self._diag_throttle_weight: Optional[float] = None
        self._code_version: Optional[str] = os.environ.get("CODE_VERSION")

        # Tolérance MOYLTP (% sur LTP)
        try:
            self._moyltp_tol = float(os.getenv("MOYLTP_TOLERANCE_PCT", "30"))
        except Exception:
            self._moyltp_tol = 30.0

        # --- AJOUTS (init Staking/Stratégies + dossier trades) ---
        self.cfg = load_config()
        self.staking_engine = StakingEngine(self.cfg)
        self.strategy_registry = build_registry()
        self.trades_dir = self.data_dir  # on réutilise ./data pour trades_YYYYMMDD.csv

        # --- AJOUT: mémoire "un pari par marché" par slot ---
        self._slot_market_fired: Set[tuple[str,int,str]] = set()  # {(family, slot, market_id)}
        self._phase_stakes_by_runner_side: Dict[tuple[str, int, str, str], Dict[str, float]] = defaultdict(
            lambda: {_strategies.EXECUTION_PHASE_PRE: 0.0, _strategies.EXECUTION_PHASE_POST: 0.0}
        )
        self._pre_ladder_price_plans: dict[str, _FrozenPreLadderPlan] = {}

        # --- AJOUTS LIVE (OrderExecutor + ExposureManager) ---
        self.order_executor = OrderExecutor(
            client=self.client,
            data_dir=str(self.data_dir),
            throttle_max_per_minute=int(os.getenv("THROTTLE_MAX_PER_MIN", "25")),
            default_persistence=os.getenv("PERSISTENCE", "LAPSE"),
            fok_ms=int(os.getenv("FOK_MS", "0")) or None,
        )
        self.exposure = ExposureManager(self.cfg)

    def set_diagnostics(self, fetch_latency_ms: Optional[float] = None, retry_count: Optional[int] = None,
                        throttle_weight: Optional[float] = None, code_version: Optional[str] = None) -> None:
        if fetch_latency_ms is not None:
            self._diag_fetch_latency_ms = float(fetch_latency_ms)
        if retry_count is not None:
            self._diag_retry_count = int(retry_count)
        if throttle_weight is not None:
            self._diag_throttle_weight = float(throttle_weight)
        if code_version is not None:
            self._code_version = str(code_version)

    def _ensure_header(self, path: Path, header: Iterable[str]) -> None:
        header_list = list(header)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    first = f.readline().strip()
                current = first.split(",")
                if current != header_list:
                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    backup = path.with_name(path.stem + f"_old_{ts}" + path.suffix)
                    path.rename(backup)
                    with path.open("w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(header_list)
                    return
                else:
                    return
            except Exception:
                pass
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header_list)

    def _append(self, path: Path, row: Iterable[Any]) -> None:
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(list(row))

    def _strategy_debug_enabled(self, ctx: Optional[RunnerCtx] = None) -> bool:
        raw = str(os.getenv("STRATEGY_DEBUG", "")).strip().lower()
        enabled = raw in ("1", "true", "yes", "y", "on")
        if not enabled:
            return False
        return ctx is None or ctx.milestone == 2

    def _debug_place_price(self, ctx: RunnerCtx) -> Optional[float]:
        if (ctx.market_type or "").upper() != "PLACE":
            return None
        if ctx.bsp_place and ctx.bsp_place > 1.0:
            return ctx.bsp_place
        if ctx.ltp and ctx.ltp > 1.0:
            return ctx.ltp
        return None

    def _debug_evaluate_slot(self, slot, ctx: RunnerCtx) -> tuple[bool, str]:
        market_type = (ctx.market_type or "").upper()
        family = str(getattr(slot, "family", "")).upper()
        if "WIN" in family and market_type != "WIN":
            return False, "market_type_mismatch"
        if "PLACE" in family and market_type != "PLACE":
            return False, "market_type_mismatch"
        if getattr(slot, "requires_mom45", False) and ctx.mom45 is None:
            return False, "missing_mom45"
        try:
            passed = bool(slot.condition(ctx))
        except Exception as e:
            return False, f"condition_error={e!r}"
        if passed:
            return True, ""
        return False, self._debug_fail_reason(slot, ctx)

    def _debug_fail_reason(self, slot, ctx: RunnerCtx) -> str:
        region = getattr(slot, "strategy_region", None)
        if region and ctx.region != region:
            return "region_mismatch"
        signal = str(getattr(slot, "strategy_signal", "") or "").upper()
        if signal == "TRAP1" and ctx.trap != 1:
            return "trap_mismatch"
        if signal == "TRAP8" and ctx.trap != 8:
            return "trap_mismatch"
        market_family = str(getattr(slot, "market_family", "") or "").upper()
        if market_family == "WIN" and ctx.winbet is None:
            return "missing_winbet"
        if market_family == "PLACE" and self._debug_place_price(ctx) is None:
            return "missing_place_price"
        if signal in ("TRAP1", "TRAP8") and ctx.ev_place is None:
            return "missing_ev_place"
        if getattr(slot, "requires_mom45", False) and ctx.mom45 is None:
            return "missing_mom45"
        return "condition_false"

    def _debug_try_fire_slot_none_detail(self, slot, ctx: RunnerCtx) -> str:
        parts: list[str] = ["condition_true_but_try_fire_slot_returned_none"]
        try:
            side = getattr(getattr(slot, "side", None), "value", getattr(slot, "side", ""))
            mode = getattr(slot, "exec_mode", None)
            limit_style = getattr(slot, "limit_style", None)
            decision: dict[str, Any] = {}
            effective_mode = mode
            effective_limit_style = limit_style
            sp_limit = getattr(slot, "sp_limit", None)
            if getattr(slot, "sp_limit_fn", None) is not None:
                try:
                    computed_sp_limit = slot.sp_limit_fn(ctx)
                    if computed_sp_limit is not None and float(computed_sp_limit) > 1.0:
                        sp_limit = float(computed_sp_limit)
                except Exception as exc:
                    parts.append(f"sp_limit_fn_error={exc!r}")
            if mode == _strategies.ExecMode.HYB:
                try:
                    decision = _strategies._hyb_decide(ctx, slot)
                    effective_mode = decision.get("mode", effective_mode)
                    effective_limit_style = decision.get("limit_style", effective_limit_style)
                    sp_limit = decision.get("sp_limit", sp_limit)
                except Exception as exc:
                    parts.append(f"hyb_decide_error={exc!r}")
            try:
                bounds_price = _strategies._pick_bounds_price(ctx, getattr(slot, "price_for_bounds", "BASE"))
            except Exception as exc:
                bounds_price = None
                parts.append(f"bounds_price_error={exc!r}")
            order_price = None
            limit_choice = decision.get("limit_price") if decision else ""
            if bounds_price is None:
                reason = "no_bounds_price"
            else:
                reason = "unknown_no_fire"
                if effective_mode == _strategies.ExecMode.LIMIT_LTP:
                    if getattr(slot, "sp_limit_fn", None) is not None:
                        if sp_limit is None or float(sp_limit) <= 1.0:
                            reason = "no_theo_limit_price"
                        else:
                            order_price = float(sp_limit)
                            limit_choice = "THEO_LIMIT"
                    else:
                        choice = str(limit_choice or "").upper()
                        if choice == "CROSS":
                            order_price = ctx.bl if str(side).upper() == "BACK" else ctx.bb
                        elif choice == "OWN":
                            order_price = ctx.bb if str(side).upper() == "BACK" else ctx.bl
                        elif choice == "MID":
                            if ctx.bb is not None and ctx.bl is not None:
                                order_price = (ctx.bb + ctx.bl) / 2.0
                        if order_price is None:
                            order_price = _strategies._choose_limit_price(slot.side, effective_limit_style, ctx)
                    if order_price is None:
                        fallback_enabled = str(os.getenv("HYB_FALLBACK_TO_SP_MOC", "true")).strip().lower() not in {
                            "0",
                            "false",
                            "no",
                            "off",
                        }
                        if fallback_enabled:
                            order_price = _strategies._choose_limit_price(slot.side, effective_limit_style, ctx) or bounds_price
                            reason = "fallback_sp_moc_price_ref" if order_price is not None else "no_order_price"
                        else:
                            reason = "no_order_price"
                else:
                    order_price = _strategies._choose_limit_price(slot.side, effective_limit_style, ctx) or bounds_price

                if order_price is not None:
                    edge_env = getattr(slot, "edge_env", None) or f"EDGE_{slot.family}_{slot.slot}"
                    edge = _strategies._env_float(edge_env, 0.02)
                    cap_env = getattr(slot, "max_runner_stake_env", None) or f"MAX_RUNNER_STAKE_{slot.family}_{slot.slot}"
                    cap_raw = os.getenv(cap_env)
                    max_runner_cap = float(cap_raw) if cap_raw not in (None, "") else None
                    sr = _strategies._compute_stake_safe(
                        self.staking_engine,
                        slot.side,
                        float(order_price),
                        edge,
                        max_runner_cap,
                    )
                    reason = getattr(sr, "reason", "stake_not_ok") if not getattr(sr, "ok", False) else "unexpected_ok"
                    parts.extend(
                        [
                            f"edge_env={edge_env}",
                            f"edge={edge}",
                            f"max_runner_cap_env={cap_env}",
                            f"max_runner_cap={max_runner_cap}",
                            f"stake_ok={getattr(sr, 'ok', False)}",
                            f"stake={getattr(sr, 'size', None)}",
                            f"liability={getattr(sr, 'liability', None)}",
                        ]
                    )
            parts.extend(
                [
                    f"reason={reason}",
                    f"side={side}",
                    f"exec_mode={effective_mode}",
                    f"limit_style={effective_limit_style}",
                    f"limit_choice={limit_choice}",
                    f"bounds_price={bounds_price}",
                    f"order_price={order_price}",
                    f"bb={ctx.bb}",
                    f"bl={ctx.bl}",
                    f"ltp={ctx.ltp}",
                    f"bsp_place={ctx.bsp_place}",
                    f"ev_place={ctx.ev_place}",
                    f"place_theo={ctx.place_theo}",
                ]
            )
        except Exception as exc:
            parts.append(f"diagnostic_error={exc!r}")
        return ";".join(str(part) for part in parts)

    def _log_strategy_debug_row(
        self,
        slot,
        ctx: RunnerCtx,
        runner_name: Optional[str],
        condition_result: bool,
        fail_reason: str,
    ) -> None:
        if not self._strategy_debug_enabled(ctx):
            return
        fname = self.data_dir / f"strategy_debug_{datetime.now(timezone.utc):%Y%m%d}.csv"
        self._ensure_header(fname, self.STRATEGY_DEBUG_HEADER)
        row = {
            "ts": _now_utc_iso(),
            "market_id": ctx.market_id,
            "market_type": ctx.market_type,
            "selection_id": ctx.selection_id,
            "runner_name": runner_name,
            "milestone": ctx.milestone,
            "secs_to_off": ctx.secs_to_off,
            "tag": getattr(slot, "tag", None),
            "market_family": getattr(slot, "market_family", None),
            "strategy_group": getattr(slot, "strategy_group", None),
            "strategy_signal": getattr(slot, "strategy_signal", None),
            "execution_phase": getattr(slot, "execution_phase", None),
            "condition_result": condition_result,
            "fail_reason": fail_reason,
            "trap": ctx.trap,
            "region": ctx.region,
            "winbet": ctx.winbet,
            "place_price": self._debug_place_price(ctx),
            "ev_place": ctx.ev_place,
            "has_mom45": ctx.mom45 is not None,
            "mom45": ctx.mom45,
            "mom45_source": getattr(self, "_gruss_mom45_diagnostics", {}).get("mom45_source", ""),
            "mom45_injected_before_strategy_eval": getattr(self, "_gruss_mom45_diagnostics", {}).get(
                "mom45_injected_before_strategy_eval",
                "",
            ),
            "place_theo": ctx.place_theo,
            "bb": ctx.bb,
            "bl": ctx.bl,
            "place_winners": getattr(ctx, "place_winners", None),
            "k_place_used": getattr(ctx, "k_place_used", None),
            "fallback_k_place_used": getattr(ctx, "fallback_k_place_used", None),
            "requires_mom45": getattr(slot, "requires_mom45", False),
        }
        with fname.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.STRATEGY_DEBUG_HEADER).writerow(row)

    def _pre_pipeline_debug_enabled(self, ctx: RunnerCtx) -> bool:
        raw = str(os.getenv("DOGBOT_PRE_PIPELINE_DEBUG", "")).strip().lower()
        explicit = raw in ("1", "true", "yes", "y", "on")
        return explicit or self._strategy_debug_enabled(ctx)

    def _log_pre_pipeline_debug_row(
        self,
        ctx: RunnerCtx,
        runner_name: Optional[str],
        *,
        row_type: str,
        candidate_id: str = "",
        side: str = "",
        strategy_id: str = "",
        removed_stage: str = "",
        removed_reason: str = "",
        removed_detail: str = "",
        summary: dict[str, Any] | None = None,
    ) -> None:
        if not self._pre_pipeline_debug_enabled(ctx):
            return
        summary = summary or {}
        fname = self.data_dir / f"pre_pipeline_debug_{datetime.now(timezone.utc):%Y%m%d}.csv"
        self._ensure_header(fname, self.PRE_PIPELINE_DEBUG_HEADER)
        row = {field: "" for field in self.PRE_PIPELINE_DEBUG_HEADER}
        row.update(
            {
                "ts": _now_utc_iso(),
                "row_type": row_type,
                "course_id": ctx.course_id,
                "market_id": ctx.market_id,
                "market_type": ctx.market_type,
                "pre_pipeline_course_id": ctx.course_id,
                "pre_pipeline_market_id": ctx.market_id,
                "pre_pipeline_market_type": ctx.market_type,
                "pre_pipeline_region": ctx.region,
                "selection_id": ctx.selection_id,
                "runner_name": runner_name,
                "trap": ctx.trap,
                "candidate_id": candidate_id,
                "side": side,
                "strategy_id": strategy_id,
                "execution_phase": ctx.execution_phase,
                "created_from_tag": strategy_id,
                "candidate_created": str(row_type in {"raw_candidate", "trade_created"}),
                "removed": str(row_type == "candidate_removed"),
                "removed_stage": removed_stage,
                "removed_reason": removed_reason,
                "removed_detail": removed_detail,
            }
        )
        for key, value in summary.items():
            if key in row:
                row[key] = value
        with fname.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.PRE_PIPELINE_DEBUG_HEADER).writerow(row)

    def _log_pre_pipeline_summary_row(
        self,
        ctx: RunnerCtx,
        runner_name: Optional[str],
        *,
        pre_strategy_signals: list[dict[str, Any]],
        pre_raw_candidates: list[_StrategyOrderCandidate],
        pre_after_conflict: list[_StrategyOrderCandidate],
        pre_after_merge: list[_StrategyOrderCandidate],
        pre_removed_candidates: list[dict[str, Any]],
    ) -> None:
        if not pre_strategy_signals and not pre_raw_candidates and not pre_removed_candidates:
            return
        back_signals = [signal for signal in pre_strategy_signals if str(signal.get("side", "")).upper() == "BACK"]
        lay_signals = [signal for signal in pre_strategy_signals if str(signal.get("side", "")).upper() == "LAY"]
        raw_backs = [candidate for candidate in pre_raw_candidates if candidate.side == "BACK"]
        raw_lays = [candidate for candidate in pre_raw_candidates if candidate.side == "LAY"]
        final_backs = [candidate for candidate in pre_after_merge if candidate.side == "BACK"]
        final_lays = [candidate for candidate in pre_after_merge if candidate.side == "LAY"]
        conflict_groups = {
            candidate.pre_conflict_group_key or candidate.conflict_group_key
            for candidate in [*pre_after_conflict]
            if candidate.pre_back_lay_conflict
        }
        conflict_groups.update(
            str(removed.get("removed_detail") or "")
            for removed in pre_removed_candidates
            if str(removed.get("removed_stage") or "") == "conflict_resolution"
        )
        conflict_groups.discard("")
        lay_removed = next(
            (
                removed
                for removed in pre_removed_candidates
                if str(removed.get("side", "")).upper() == "LAY"
            ),
            {},
        )
        conflict_candidates = [
            candidate for candidate in pre_after_conflict if candidate.conflict_detected or candidate.pre_back_lay_conflict
        ]
        conflict_candidate = conflict_candidates[0] if conflict_candidates else None
        conflict_rejections = [
            removed for removed in pre_removed_candidates if str(removed.get("removed_stage") or "") == "conflict_resolution"
        ]
        conflict_rejection = conflict_rejections[0] if conflict_rejections else {}
        conflict_back_count = len(raw_backs) if conflict_groups else 0
        conflict_lay_count = len(raw_lays) if conflict_groups else 0
        removed_detail = "|".join(
            f"{item.get('strategy_id')}:{item.get('side')}:{item.get('removed_stage')}:{item.get('removed_reason')}"
            for item in pre_removed_candidates
        )
        summary = {
            "pre_strategy_signals_count": len(pre_strategy_signals),
            "pre_strategy_signals_by_runner": _runner_debug_summary(runner_name, pre_strategy_signals),
            "pre_strategy_back_signals_by_runner": _runner_debug_summary(runner_name, back_signals),
            "pre_strategy_lay_signals_by_runner": _runner_debug_summary(runner_name, lay_signals),
            "pre_raw_candidates_count": len(pre_raw_candidates),
            "pre_raw_candidates_by_runner": _candidate_debug_summary(runner_name, pre_raw_candidates),
            "pre_raw_back_candidates_by_runner": _candidate_debug_summary(runner_name, raw_backs),
            "pre_raw_lay_candidates_by_runner": _candidate_debug_summary(runner_name, raw_lays),
            "pre_after_filters_count": len(pre_raw_candidates),
            "pre_after_filters_by_runner": _candidate_debug_summary(runner_name, pre_raw_candidates),
            "pre_removed_candidates_count": len(pre_removed_candidates),
            "pre_removed_candidates_detail": removed_detail,
            "pre_conflict_groups_count": len(conflict_groups),
            "pre_conflict_groups_by_runner": _value_by_runner(runner_name, "|".join(sorted(conflict_groups))),
            "pre_conflict_group_key": "|".join(sorted(conflict_groups)),
            "pre_after_conflict_resolution_count": len(pre_after_conflict),
            "pre_after_conflict_resolution_by_runner": _candidate_debug_summary(runner_name, pre_after_conflict),
            "pre_after_ladder_planning_count": len(pre_after_merge),
            "pre_after_ladder_planning_by_runner": _candidate_debug_summary(runner_name, pre_after_merge),
            "pre_trades_created_count": len(pre_after_merge),
            "pre_trades_created_by_runner": _candidate_debug_summary(runner_name, pre_after_merge),
            "pre_gruss_orders_attempted_count": "",
            "pre_gruss_orders_attempted_by_runner": "",
            "strategy_back_signal": str(bool(back_signals)),
            "strategy_lay_signal": str(bool(lay_signals)),
            "raw_back_candidate_created": str(bool(raw_backs)),
            "raw_lay_candidate_created": str(bool(raw_lays)),
            "conflict_detected": str(bool(conflict_groups)),
            "chosen_side": (
                pre_after_conflict[0].pre_conflict_chosen_side
                if conflict_groups and pre_after_conflict
                else conflict_rejection.get("selected_side", "")
            ),
            "rejected_side": (
                pre_after_conflict[0].pre_conflict_rejected_side
                if conflict_groups and pre_after_conflict
                else conflict_rejection.get("rejected_side", "")
            ),
            "conflict_group_key": (
                getattr(conflict_candidate, "conflict_group_key", "")
                or getattr(conflict_candidate, "pre_conflict_group_key", "")
                or "|".join(sorted(conflict_groups))
            ),
            "conflict_course_id": getattr(conflict_candidate, "pre_conflict_course_id", "") or ctx.course_id,
            "conflict_market_id": getattr(conflict_candidate, "pre_conflict_market_id", "") or ctx.market_id,
            "conflict_market_type": getattr(conflict_candidate, "pre_conflict_market_type", "") or ctx.market_type,
            "conflict_selection_id": getattr(conflict_candidate, "pre_conflict_selection_id", "") or ctx.selection_id,
            "conflict_runner_name": getattr(conflict_candidate, "pre_conflict_runner_name", "") or runner_name,
            "conflict_execution_phase": ctx.execution_phase,
            "conflict_back_candidates_count": conflict_back_count,
            "conflict_lay_candidates_count": conflict_lay_count,
            "conflict_resolution": (
                getattr(conflict_candidate, "conflict_resolution_reason", "")
                or conflict_rejection.get("conflict_resolution_reason", "")
            ),
            "conflict_chosen_side": (
                getattr(conflict_candidate, "selected_side", "")
                or getattr(conflict_candidate, "pre_conflict_chosen_side", "")
                or conflict_rejection.get("selected_side", "")
            ),
            "conflict_rejected_side": (
                getattr(conflict_candidate, "rejected_side", "")
                or getattr(conflict_candidate, "pre_conflict_rejected_side", "")
                or conflict_rejection.get("rejected_side", "")
            ),
            "conflict_reason": (
                getattr(conflict_candidate, "conflict_resolution_reason", "")
                or conflict_rejection.get("conflict_resolution_reason", "")
                or conflict_rejection.get("removed_reason", "")
            ),
            "final_pre_order_created": str(bool(pre_after_merge)),
            "removed_reason_if_no_order": "" if pre_after_merge else removed_detail,
            "lay_signal_seen": str(bool(lay_signals)),
            "lay_candidate_created": str(bool(raw_lays)),
            "lay_reached_conflict_resolver": str(bool(raw_lays)),
            "lay_removed_stage": lay_removed.get("removed_stage", ""),
            "lay_removed_reason": lay_removed.get("removed_reason", ""),
            "lay_selected": str(bool(final_lays)),
            "lay_reached_trade": str(bool(final_lays)),
            "lay_reached_gruss_writer": str(bool(final_lays)),
            "pre_lay_signal_seen": str(bool(lay_signals)),
            "pre_lay_candidate_created": str(bool(raw_lays)),
            "pre_lay_candidate_removed": str(bool(lay_removed)),
            "pre_lay_removed_stage": lay_removed.get("removed_stage", ""),
            "pre_lay_removed_reason": lay_removed.get("removed_reason", ""),
            "pre_lay_reached_conflict_resolver": str(bool(raw_lays)),
            "pre_lay_reached_ladder_planner": str(bool(final_lays)),
            "pre_lay_reached_gruss_writer": str(bool(final_lays)),
        }
        self._log_pre_pipeline_debug_row(
            ctx,
            runner_name,
            row_type="runner_summary",
            summary=summary,
        )

    def _log_strategy_eval_summary_row(
        self,
        ctx: RunnerCtx,
        runner_name: Optional[str],
        slots_tested: int,
        true_tags: List[str],
        error_count: int,
        mom45_slots_tested: int = 0,
        mom45_slots_missing: int = 0,
    ) -> None:
        if ctx.milestone != 2:
            return
        mom45_diagnostics = getattr(self, "_gruss_mom45_diagnostics", {}) or {}
        fname = self.data_dir / f"strategy_eval_summary_{datetime.now(timezone.utc):%Y%m%d}.csv"
        self._ensure_header(fname, self.STRATEGY_EVAL_SUMMARY_HEADER)
        row = {
            "ts": _now_utc_iso(),
            "market_id": ctx.market_id,
            "market_type": ctx.market_type,
            "selection_id": ctx.selection_id,
            "runner_name": runner_name,
            "trap": ctx.trap,
            "region": ctx.region,
            "winbet": ctx.winbet,
            "place_price": self._debug_place_price(ctx),
            "place_theo": ctx.place_theo,
            "ev_place": ctx.ev_place,
            "has_mom45": ctx.mom45 is not None,
            "mom45": ctx.mom45,
            "mom45_available_count": mom45_diagnostics.get("mom45_available_count", ""),
            "mom45_missing_count": mom45_diagnostics.get("mom45_missing_count", ""),
            "mom45_source": mom45_diagnostics.get("mom45_source", ""),
            "mom45_injected_before_strategy_eval": mom45_diagnostics.get(
                "mom45_injected_before_strategy_eval",
                "",
            ),
            "mom45_strategy_slots_evaluated_count": mom45_slots_tested,
            "mom45_strategy_slots_missing_count": mom45_slots_missing,
            "place_winners": getattr(ctx, "place_winners", None),
            "k_place_used": getattr(ctx, "k_place_used", None),
            "fallback_k_place_used": getattr(ctx, "fallback_k_place_used", None),
            "execution_phase": getattr(ctx, "execution_phase", None),
            "slots_tested": slots_tested,
            "conditions_true_count": len(true_tags),
            "true_tags": "|".join(true_tags),
            "error_count": error_count,
        }
        with fname.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.STRATEGY_EVAL_SUMMARY_HEADER).writerow(row)

    def _mindex_vals(self) -> List[Any]:
        for try_fn in (lambda i: list(i.values()),
                       lambda i: [v for _, v in i.items()],
                       lambda i: list(i),):
            try: return try_fn(self.market_index)  # type: ignore
            except Exception: pass
        for attr in ("by_id","_by_id","map","_map","d","_d"):
            d = getattr(self.market_index, attr, None)
            if isinstance(d, dict): return list(d.values())
        return []

    def _resolve_link_ids(self, current_market_id: str, mie: Any, md: Any, market_type_hint: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        win_id = getattr(mie, "win_market_id", None) if mie is not None else None
        place_id = getattr(mie, "place_market_id", None) if mie is not None else None
        if win_id and place_id:
            return win_id, place_id

        cur_mt = (getattr(md, "market_type", None) or market_type_hint or "").upper()
        if cur_mt not in ("WIN","PLACE"):
            if getattr(md, "number_of_winners", None) or getattr(md, "numberOfWinners", None):
                nw = int(getattr(md, "number_of_winners", None) or getattr(md, "numberOfWinners", 1))
                cur_mt = "PLACE" if nw >= 2 else "WIN"

        ev_id = getattr(getattr(mie, "event", None), "id", None) if mie is not None else None
        start = _tz_utc(getattr(mie, "market_start_time", None)) if mie is not None else None
        if start is None:
            start = _tz_utc(getattr(md, "market_time", None))

        if ev_id and start:
            for other in self._mindex_vals():
                omid = getattr(other, "market_id", None)
                if not omid or str(omid) == str(current_market_id):
                    continue
                oe = getattr(other, "event", None)
                if getattr(oe, "id", None) != ev_id:
                    continue
                ostart = _tz_utc(getattr(other, "market_start_time", None))
                if ostart and abs((ostart - start).total_seconds()) > 180:
                    continue
                omt = (getattr(other, "market_type", None) or "").upper()
                if omt not in ("WIN","PLACE"):
                    name2 = getattr(other, "market_name", None) or getattr(other, "marketName", None)
                    omt = "PLACE" if (name2 and "place" in name2.lower()) else "WIN"
                if omt == "WIN" and not win_id: win_id = omid
                if omt == "PLACE" and not place_id: place_id = omid

        if cur_mt == "WIN" and not win_id:
            win_id = current_market_id
        if cur_mt == "PLACE" and not place_id:
            place_id = current_market_id

        return (str(win_id) if win_id else None, str(place_id) if place_id else None)

    def _remember_place_winners(
        self,
        current_market_id: str,
        place_id: Optional[str],
        mie: Any,
        md: Any,
        market_type: Optional[str],
    ) -> None:
        winners = self._official_place_winners(current_market_id, place_id, mie, md, market_type)
        if winners is not None and place_id:
            self._place_winners_by_market[str(place_id)] = int(winners)

    def _official_place_winners(
        self,
        current_market_id: str,
        place_id: Optional[str],
        mie: Any,
        md: Any,
        market_type: Optional[str],
    ) -> Optional[int]:
        current_type = (market_type or "").upper()
        current_winners = _num_winners(md)
        if current_type == "PLACE" and current_winners is not None:
            return int(current_winners)

        if place_id:
            cached = self._place_winners_by_market.get(str(place_id))
            if cached is not None:
                return int(cached)
            place_entry = None
            try:
                place_entry = self.market_index.get(str(place_id))  # type: ignore
            except Exception:
                place_entry = None
            indexed = getattr(place_entry, "n_places", None)
            if indexed is not None:
                return int(indexed)

        indexed_current = getattr(mie, "n_places", None) if mie is not None else None
        if current_type == "PLACE" and indexed_current is not None:
            return int(indexed_current)
        if place_id and str(current_market_id) == str(place_id) and current_winners is not None:
            return int(current_winners)
        return None

    def _resolve_k_place_used(
        self,
        current_market_id: str,
        place_id: Optional[str],
        mie: Any,
        md: Any,
        market_type: Optional[str],
        n_active: int,
    ) -> tuple[int, Optional[int], bool]:
        place_winners = self._official_place_winners(current_market_id, place_id, mie, md, market_type)
        if place_winners is not None:
            k_place_used = int(place_winners)
            fallback = False
        else:
            k_place_used = _fallback_k_place_used(n_active)
            fallback = True

        cache_key = str(place_id or current_market_id)
        self._k_place_used_by_market[cache_key] = int(k_place_used)
        self._fallback_k_place_used_by_market[cache_key] = bool(fallback)
        if place_winners is not None:
            self._place_winners_by_market[cache_key] = int(place_winners)
        elif fallback and cache_key not in self._k_place_fallback_logged:
            self._k_place_fallback_logged.add(cache_key)
            print(
                "[K_PLACE_FALLBACK] "
                f"market_id={current_market_id} place_market_id={place_id or ''} "
                f"n_active={n_active} k_place_used={k_place_used}"
            )
        return int(k_place_used), place_winners, fallback

    # ---------- cœur ----------
    def process_book(self, book: Any) -> None:
        market_id = str(getattr(book, "market_id", ""))
        md = getattr(book, "market_definition", None)
        runners = getattr(book, "runners", None) or []

        try:
            mie = self.market_index.get(market_id)  # type: ignore
        except Exception:
            mie = None

        # Infos catalogue
        info = self._extract_catalogue_info(mie, md)
        market_type = info["market_type"]
        start_utc = info["start_utc"]
        now = datetime.now(timezone.utc)
        t_to_off_s = (start_utc - now).total_seconds() if start_utc else None

        if getattr(book, "inplay", False):
            self._next_ms.pop(market_id, None)
            self._last_tto[market_id] = t_to_off_s if t_to_off_s is not None else 0.0
            return

        # Historique LTP pour d5/d30/vol (WIN uniquement, indicatif)
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: continue
            lpt = _runner_lpt_or_back(r)
            if lpt is not None:
                self._hist[market_id][int(sid)].append((datetime.now(timezone.utc).timestamp(), float(lpt)))

        rank_ltp, rank_back = _rankings(runners)
        milestone = self._milestone_due(market_id, t_to_off_s)

        self._write_market_row(book, info, market_type, t_to_off_s, milestone, runners)
        if milestone is not None:
            self._write_runner_rows(book, info, market_type, t_to_off_s, milestone, runners, rank_ltp, rank_back)

        # stratégie (legacy)
        try:
            instructions = self.strategy.decide_all(book, mie, datetime.now(timezone.utc)) or []
        except Exception as e:
            print(f"[STRATEGY_ERR] {market_id}: {e}")
            instructions = []

        if instructions:
            if self.dry_run:
                for ins in instructions:
                    print("[DRY] would place", ins.asdict(), "on", market_id)
            else:
                for ins in instructions:
                    print("[LIVE] placing", ins.asdict(), "on", market_id)

        if t_to_off_s is not None:
            self._last_tto[market_id] = t_to_off_s

    def _write_market_row(self, book, info: Dict[str,Any], market_type: Optional[str],
                          tto: Optional[float], milestone: Optional[int], runners) -> None:
        md = getattr(book, "market_definition", None)
        back_over = _overround_back(runners)
        lay_over  = _overround_lay(runners)
        spread    = _spread_pct(runners)

        mie_current = getattr(self.market_index, "get", lambda _:_)(getattr(book,"market_id",None))
        win_id, place_id = self._resolve_link_ids(str(getattr(book,"market_id","")), mie_current, md, market_type)
        n_active = _active_count(runners)
        k_place_used, place_winners, fallback_k_place_used = self._resolve_k_place_used(
            str(getattr(book, "market_id", "")),
            place_id,
            mie_current,
            md,
            market_type,
            n_active,
        )
        self._remember_place_winners(str(getattr(book, "market_id", "")), place_id, mie_current, md, market_type)

        row = [
            _now_utc_iso(),
            str(getattr(book, "market_id", "")),
            None,
            market_type,
            info["event_id"],
            info["event_name"],
            info["venue"],
            info["country_code"],
            (info["start_utc"].isoformat().replace("+00:00","Z") if info["start_utc"] else None),
            float(tto) if tto is not None else None,
            bool(getattr(book, "inplay", False)),
            _market_status(md),
            _num_winners(md),
            n_active,
            float(getattr(book, "total_matched", None) or 0.0),
            back_over, lay_over, spread,
            win_id, place_id,
            (1 if (win_id and place_id) else 0),
            k_place_used,
            (tto is not None and self.ENTRY_MIN_T_S <= tto <= self.ENTRY_MAX_T_S),
            self._fav_price_ok(runners),
            (float(getattr(book, "total_matched", 0.0)) >= self.MIN_LIQUIDITY_MARKET),
            (back_over is not None and lay_over is not None),
            None,
            (getattr(self.strategy, "name", None) or None),
            None, None, None, None, None, None, None, None,
            False,
            milestone,
            self._diag_fetch_latency_ms,
            None,
            self._diag_retry_count,
            self._diag_throttle_weight,
            self._code_version,
        ]
        self._append(self.market_csv, row)

    def _fav_price_ok(self, runners) -> Optional[bool]:
        ex_prices = []
        for r in (runners or []):
            pb = _best_at_side(getattr(r, "ex", None), "BACK")[0]
            if pb:
                ex_prices.append(pb)
        if not ex_prices:
            return None
        fav_price = min(ex_prices)
        return (self.PRICE_MIN <= fav_price <= self.PRICE_MAX)

    def _trusted_moyltp(self, ltp: Optional[float], mid: Optional[float]) -> Optional[float]:
        if ltp is None or mid is None or ltp <= 0:
            return None
        gap = abs(mid - ltp) / ltp * 100.0
        return ((ltp + mid) / 2.0) if (gap <= self._moyltp_tol) else None

    def _write_runner_rows(self, book, info: Dict[str,Any], market_type: Optional[str], tto: Optional[float], milestone: int,
                           runners, rank_ltp: Dict[int,int], rank_back: Dict[int,int]) -> None:
        market_id = str(getattr(book, "market_id", ""))
        n_active = _active_count(runners)
        md = getattr(book, "market_definition", None)

        meta_by_sid: Dict[int, RunnerMeta] = info.get("meta_by_sid", {}) or {}
        vmap = self._compute_virtual_traps(runners, meta_by_sid)

        mie_current = getattr(self.market_index, "get", lambda _:_)(market_id)
        win_id, place_id = self._resolve_link_ids(market_id, mie_current, md, market_type)
        n_places, place_winners, fallback_k_place_used = self._resolve_k_place_used(
            market_id,
            place_id,
            mie_current,
            md,
            market_type,
            n_active,
        )
        self._remember_place_winners(market_id, place_id, mie_current, md, market_type)

        is_win  = (market_type == "WIN")
        is_place = (market_type == "PLACE")
        race_key = (win_id or place_id or market_id)
        country_code = info.get("country_code") or getattr(md, "country_code", None) or getattr(md, "countryCode", None)
        country_code = str(country_code).upper() if country_code else ""
        region = "UK" if country_code in ("GB", "IE") else "ROW"


        # GOR map at T-2s on WIN: ratio of next favourite's price / current price
        gor_map: dict[int, float | None] = {}
        if is_win and milestone == 2:
            arr = []
            for rr in (runners or []):
                if (getattr(rr, "status", None) or "ACTIVE").upper() != "ACTIVE":
                    continue
                sid2 = getattr(rr, "selection_id", None)
                if sid2 is None: continue
                lpt2 = _runner_lpt_or_back(rr)
                if lpt2 is None or lpt2 <= 1.0:  # skip invalid
                    continue
                arr.append((int(sid2), float(lpt2)))
            arr.sort(key=lambda t: t[1])  # favourite -> outsider
            for i, (sid2, p2) in enumerate(arr):
                if i+1 < len(arr):
                    p_next = arr[i+1][1]
                    gor_map[sid2] = (p_next / p2) if p2 > 0 else None
                else:
                    gor_map[sid2] = None  # last = NO_NEXT
        # Prépare prix + caches
        sid_prices_for_pl: List[Tuple[int,float]] = []  # (sid, BASE_WIN)
        cache: Dict[int, Dict[str, Optional[float]]] = {}

        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: continue
            sid = int(sid)

            lpt = _runner_lpt_or_back(r)
            ex = getattr(r, "ex", None)
            bb, _ = _best_at_side(ex, "BACK")
            bl, _ = _best_at_side(ex, "LAY")

            mid = ((bb + bl) / 2.0) if (bb is not None and bl is not None) else None
            moyltp = self._trusted_moyltp(lpt, mid)

            # SP / BSP via feed (REST: SP_AVAILABLE; STREAM: SP_EST)
            tmp = {}
            self.price_feed.enrich_runner_row(tmp, "WIN" if is_win else "PLACE", r)
            if is_win:
                sp_est = tmp.get("SP_EST_WIN")
                near   = tmp.get("NEAR_SP_WIN")
                sp_av  = 1 if tmp.get("SP_AVAILABLE_WIN") else 0
                bsp    = sp_est if (sp_est is not None) else near
                bsp_win = float(bsp) if bsp is not None else None
                sp_av_win = sp_av
                bsp_place = None
                sp_av_place = 0
            else:
                sp_est = tmp.get("SP_EST_PLACE")
                near   = tmp.get("NEAR_SP_PLACE")
                sp_av  = 1 if tmp.get("SP_AVAILABLE_PLACE") else 0
                bsp    = sp_est if (sp_est is not None) else near
                bsp_place = float(bsp) if bsp is not None else None
                sp_av_place = sp_av
                bsp_win = None
                sp_av_win = 0

            # Cache par type
            cache[sid] = {
                "LTP": lpt, "BB": bb, "BL": bl,
                "MID": mid,
                "MOYLTP": moyltp,
                "BSP_WIN": bsp_win,
                "SP_AV_WIN": float(sp_av_win),
                "BSP_PLACE": bsp_place,
                "SP_AV_PLACE": float(sp_av_place),
            }

            # BASE_WIN pour Plackett-Luce (uniquement marché WIN)
            if is_win:
                winprob = ((bsp_win + lpt)/2.0) if (bsp_win is not None and lpt is not None) else None
                base_win = winprob or moyltp or lpt or bb  # hiérarchie
                cache[sid]["WINPROB"] = winprob
                cache[sid]["BASE_WIN"] = base_win
                if base_win and base_win > 1.0:
                    sid_prices_for_pl.append((sid, float(base_win)))

        # PLACETHEORIQUE via Plackett–Luce (cote = 1/q) pour WIN
        place_theo_by_sid: Dict[int, Optional[float]] = {}
        if is_win and len(sid_prices_for_pl) >= max(2, min(5, n_active)):
            sid_prices_for_pl.sort(key=lambda t: t[0])
            sids = [sid for sid,_ in sid_prices_for_pl]
            odds = [price for _,price in sid_prices_for_pl]
            try:
                p = odds_to_win_probs(odds, beta=1.0)
                q = place_probabilities(p, K=n_places)
                fair = fair_place_odds(q)
                fair_list = list(fair) if not hasattr(fair, "tolist") else fair.tolist()
                for sid, odd_place in zip(sids, fair_list):
                    place_theo_by_sid[sid] = float(odd_place)
            except Exception as e:
                print(f"[PL_ERR] {market_id}: {e}")
                place_theo_by_sid = {sid: None for sid,_ in sid_prices_for_pl}

            # cache duplication pour PLACE
            cache_key = (win_id or market_id)
            if cache_key:
                self._last_place_theo_by_market[str(cache_key)] = {
                    sid: v for sid, v in place_theo_by_sid.items() if v is not None
                }

        # Écriture des lignes runners
        rank_ltp, rank_back = _rankings(runners)
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: continue
            sid = int(sid)
            status = getattr(r, "status", None)
            rm: RunnerMeta | None = meta_by_sid.get(sid)
            runner_name = (rm.runner_name if rm else None)
            trap = _parse_trap(rm, runner_name)
            vtrap = vmap.get(sid, trap)
            # trap cache & optional vtrap fallback
            key_r = (race_key, sid)
            if trap is None:
                trap = self._trap_by_race.get(key_r)
            if trap is None and str(os.getenv("ALLOW_VTRAP_AS_TRAP", "false")).lower() in ("1","true","yes","y") and vtrap is not None:
                trap = vtrap
            if trap is not None:
                try:
                    self._trap_by_race[key_r] = int(trap)
                except Exception:
                    pass

            lpt = cache[sid].get("LTP")
            ex = getattr(r, "ex", None)
            bb, bs = _best_at_side(ex, "BACK")
            bl, ls = _best_at_side(ex, "LAY")
            mid = cache[sid].get("MID")
            moyltp = cache[sid].get("MOYLTP")

            ladder_b = _compress_ladder(getattr(ex, "available_to_back", None), 3)
            ladder_l = _compress_ladder(getattr(ex, "available_to_lay", None), 3)
            total_matched_runner = getattr(r, "total_matched", None)

            rk_ltp = rank_ltp.get(sid)
            rk_bb  = rank_back.get(sid)
            linked_rank_ltp = self._last_rank_ltp_by_market.get(str(win_id), {}).get(sid) if (is_place and win_id) else None
            linked_rank_back = self._last_rank_back_by_market.get(str(win_id), {}).get(sid) if (is_place and win_id) else None
            implied = (1.0 / (bb or lpt)) if (bb or lpt) else None

            bsp_win = cache[sid].get("BSP_WIN") if is_win else None
            sp_av_win = cache[sid].get("SP_AV_WIN") if is_win else None
            bsp_place = cache[sid].get("BSP_PLACE") if is_place else None
            sp_av_place = cache[sid].get("SP_AV_PLACE") if is_place else None

            winprob = cache[sid].get("WINPROB") if is_win else None
            base_win = cache[sid].get("BASE_WIN") if is_win else None
            linked_winbet = None
            if is_place and win_id:
                linked_winbet = self._last_base_win_by_market.get(str(win_id), {}).get(sid)

            # Milestones:
            if is_win and base_win is not None:
                self._base_win_ms[market_id][sid][milestone] = float(base_win)
                self._last_base_win_by_market[str(win_id or market_id)][sid] = float(base_win)
            if is_win and rk_ltp is not None:
                self._last_rank_ltp_by_market[str(win_id or market_id)][sid] = int(rk_ltp)
            if is_win and rk_bb is not None:
                self._last_rank_back_by_market[str(win_id or market_id)][sid] = int(rk_bb)
            if is_place and lpt is not None:
                self._ltp_place_ms[market_id][sid][milestone] = float(lpt)

            def _get(msdict, ms):
                return msdict.get(market_id, {}).get(sid, {}).get(ms)
            def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
                if a is None or b is None or b == 0: return None
                return (a / b) - 1.0

            base_300 = _get(self._base_win_ms, 300) if is_win else None
            base_150 = _get(self._base_win_ms, 150) if is_win else None
            base_80  = _get(self._base_win_ms, 80)  if is_win else None
            base_45  = _get(self._base_win_ms, 45)  if is_win else None
            base_2   = _get(self._base_win_ms, 2)   if is_win else None

            diff150_300 = ratio(base_150, base_300) if is_win else None
            diff80_150  = ratio(base_80,  base_150) if is_win else None
            diff45_80   = ratio(base_45,  base_80)  if is_win else None
            mom45  = ratio(base_2,  base_45) if is_win else None
            mom80  = ratio(base_2,  base_80)  if is_win else None
            mom150 = ratio(base_2,  base_150) if is_win else None
            mom300 = ratio(base_2,  base_300) if is_win else None
            external_mom45 = (
                self._last_mom45_by_market.get(str(win_id or market_id), {}).get(sid)
                if is_win
                else None
            )
            if is_win and mom45 is None and external_mom45 is not None:
                mom45 = external_mom45
            if is_win and mom45 is not None:
                self._last_mom45_by_market[str(win_id or market_id)][sid] = float(mom45)
            linked_mom45 = self._last_mom45_by_market.get(str(win_id), {}).get(sid) if (is_place and win_id) else None

            # deltas/vol (info) sur LTP côté WIN
            d5 = d30 = vol = None
            if is_win:
                dq = self._hist.get(market_id, {}).get(sid)
                if dq:
                    now_ts = datetime.now(timezone.utc).timestamp()
                    for window, varname in ((5.0, "d5"), (30.0, "d30")):
                        target = now_ts - window
                        older = None
                        for ts, p in reversed(dq):
                            older = (ts, p)
                            if ts <= target:
                                break
                        if older is not None:
                            p_now = dq[-1][1]; p_old = older[1]
                            if varname == "d5": d5 = p_now - p_old
                            else: d30 = p_now - p_old
                    vals = [p for ts, p in dq if ts >= now_ts - 60.0]
                    if len(vals) >= 3:
                        m = sum(vals)/len(vals)
                        var = sum((x-m)**2 for x in vals)/(len(vals)-1)
                        vol = math.sqrt(var)

            # PLACETHEORIQUE (cote) :
            place_theo_win = place_theo_by_sid.get(sid) if is_win else None
            place_theo_place = None
            if is_place and win_id:
                cached = self._last_place_theo_by_market.get(str(win_id))
                if cached is not None:
                    place_theo_place = cached.get(sid)
            place_theo_ctx = place_theo_win if is_win else place_theo_place

            # Milestones PLACE LTP
            def _get_place(ms):
                return self._ltp_place_ms.get(market_id, {}).get(sid, {}).get(ms)
            ltp_300_p = _get_place(300) if is_place else None
            ltp_150_p = _get_place(150) if is_place else None
            ltp_80_p  = _get_place(80)  if is_place else None
            ltp_45_p  = _get_place(45)  if is_place else None
            ltp_2_p   = _get_place(2)   if is_place else None


            # PLACE momentum 45s->2s
            mom45p = None
            if is_place and (ltp_2_p is not None) and (ltp_45_p is not None) and (ltp_45_p != 0):
                try:
                    mom45p = (ltp_2_p / ltp_45_p) - 1.0
                except Exception:
                    mom45p = None

            # EV_PLACE uses PLACE_BSP_THEN_LTP semantics: BSP_PLACE if available, else LTP_PLACE.
            place_price_for_ev = None
            if is_place and bsp_place is not None and bsp_place > 1.0:
                place_price_for_ev = bsp_place
            elif is_place and lpt is not None and lpt > 1.0:
                place_price_for_ev = lpt

            if place_theo_place is not None and place_theo_place > 1.0 and place_price_for_ev is not None:
                ev_place = (float(place_price_for_ev) / float(place_theo_place)) - 1.0
            else:
                ev_place = None
            if is_place and ev_place is not None:
                self._last_ev_place_by_market[str(place_id or market_id)][sid] = float(ev_place)
            linked_ev_place = self._last_ev_place_by_market.get(str(place_id), {}).get(sid) if (is_win and place_id) else None
            ev_place_ctx = linked_ev_place if is_win else ev_place

            # GOR & gap bins (@ WIN T-2s) and duplicate to PLACE from cache
            gapmin = gapmax = None
            gor_val = None
            key_r = (race_key, sid)
            if is_win and milestone == 2:
                gor_val = gor_map.get(sid)
                gm, gM, g = _gor_bin(gor_val)
                gapmin, gapmax = gm, gM
                # store for PLACE duplication
                try:
                    self._gap_by_race[key_r] = (gapmin, gapmax, gor_val)
                except Exception:
                    pass
            else:
                # try from cache (PLACE rows or WIN non-2s)
                tpl = self._gap_by_race.get(key_r)
                if tpl:
                    gapmin, gapmax, gor_val = tpl

            row = [
                # base
                _now_utc_iso(),
                None,  # COURSE_ID (optionnel)
                info["venue"],
                (info["start_utc"].isoformat().replace("+00:00","Z") if info["start_utc"] else None),
                float(tto) if tto is not None else None,
                milestone,
                win_id, place_id,
                market_id, (market_type or ""),
                sid, runner_name, status,
                (getattr(rm, "draw", None) if rm else None),
                trap, vtrap,
                (getattr(rm, "sort_priority", None) if rm else None),
                n_active, n_places,
                # WIN block
                (lpt if is_win else None),
                (bb if is_win else None),
                (bs if is_win else None),
                (bl if is_win else None),
                (ls if is_win else None),
                (mid if is_win else None),
                (moyltp if is_win else None),
                (ladder_b if is_win else None),
                (ladder_l if is_win else None),
                (total_matched_runner if is_win else None),
                (rk_ltp if is_win else None),
                (rk_bb  if is_win else None),
                (implied if is_win else None),
                (bsp_win if is_win else None),
                (sp_av_win if is_win else None),
                (winprob if is_win else None),
                (base_300 if is_win else None),
                (base_150 if is_win else None),
                (base_80  if is_win else None),
                (base_45  if is_win else None),
                (base_2   if is_win else None),
                (diff150_300 if is_win else None),
                (diff80_150  if is_win else None),
                (diff45_80   if is_win else None),
                (mom45  if is_win else None),
                (mom80  if is_win else None),
                (mom150 if is_win else None),
                (mom300 if is_win else None),
                (d5 if is_win else None),
                (d30 if is_win else None),
                (vol if is_win else None),
                ((bs or 0.0) + (ls or 0.0) if is_win else None),
                (((rk_bb or 0) == 1) if is_win else None),
                (place_theo_win if is_win else None),
                # PLACE block
                (lpt if is_place else None),
                (bb if is_place else None),
                (bs if is_place else None),
                (bl if is_place else None),
                (ls if is_place else None),
                (mid if is_place else None),
                (moyltp if is_place else None),
                (ladder_b if is_place else None),
                (ladder_l if is_place else None),
                (total_matched_runner if is_place else None),
                (bsp_place if is_place else None),
                (sp_av_place if is_place else None),
                (((bsp_place + lpt)/2.0) if (is_place and bsp_place is not None and lpt is not None) else None),
                (place_theo_place if is_place else None),
                ev_place,
                (ltp_300_p if is_place else None),
                (ltp_150_p if is_place else None),
                (ltp_80_p  if is_place else None),
                (ltp_45_p  if is_place else None),
                (ltp_2_p   if is_place else None),
                # NEW: gap outputs
                gapmin, gapmax, gor_val,
            ]
            self._append(self.runner_csv, row)

            # --- AJOUT : déclenchement staking + LIVE ---
            try:
                # course_id simple: VENUE-YYYYMMDDHHMM
                start_utc = info.get("start_utc")
                venue = (info.get("venue") or "NA")
                if start_utc is not None:
                    course_id = f"{venue}-{start_utc:%Y%m%d%H%M}"
                else:
                    course_id = str(info.get("event_id") or market_id)

                ltp_val = cache.get(sid, {}).get("LTP")
                if ltp_val is not None and ltp_val >= 1.01:
                    ctx = RunnerCtx(
                        market_id=market_id,
                        market_type=(market_type or ""),
                        selection_id=int(sid),
                        course_id=str(course_id),
                        ltp=float(ltp_val),
                        milestone=milestone,                           # jalon courant
                        secs_to_off=float(tto) if tto is not None else None,
                        # RunnerCtx fields below mirror CSV features used by strategies.
                        trap=(int(trap) if trap is not None else None),
                        fav_rank_ltp=(rk_ltp if is_win else linked_rank_ltp),
                        fav_rank_back=(rk_bb if is_win else linked_rank_back),
                        gor=gor_val,
                        mom45=(mom45 if is_win else linked_mom45),
                        mom45_place=(mom45p if is_place else None),
                        d5=(d5 if is_win else None),
                        d30=(d30 if is_win else None),
                        vol60=(vol if is_win else None),
                        base_win=(base_win if is_win else None),
                        bb=bb,
                        bl=bl,
                        region=region,
                        winbet=(base_win if is_win else linked_winbet),
                        place_theo=place_theo_ctx,
                        ev_place=ev_place_ctx,
                        bsp_place=(bsp_place if is_place else None),
                        execution_phase=_strategies.EXECUTION_PHASE_POST,
                        phase_send_seconds_before_off=None,
                    )
                    ctx.place_winners = place_winners
                    ctx.k_place_used = n_places
                    ctx.fallback_k_place_used = fallback_k_place_used
                    execution_phase = _execution_phase_for_milestone(milestone)
                    if execution_phase is None:
                        continue
                    ctx.execution_phase = execution_phase
                    ctx.phase_send_seconds_before_off = milestone
                    ctx.milestone = 2

                    slots_tested = 0
                    true_tags: List[str] = []
                    error_count = 0
                    mom45_slots_tested = 0
                    mom45_slots_missing = 0
                    pre_orders: List[_StrategyOrderCandidate] = []
                    post_orders: List[_StrategyOrderCandidate] = []
                    pre_strategy_signals: list[dict[str, Any]] = []
                    pre_raw_candidates: list[_StrategyOrderCandidate] = []
                    pre_removed_candidates: list[dict[str, Any]] = []

                    for slot in self.strategy_registry:
                        slot_phase = str(
                            getattr(slot, "execution_phase", _strategies.EXECUTION_PHASE_POST)
                        ).upper()
                        if slot_phase != execution_phase:
                            continue
                        # Bet per market (par slot)
                        key = (slot.family, slot.slot, market_id)
                        if getattr(slot, "bet_per_market", False) and key in self._slot_market_fired:
                            self._log_strategy_debug_row(slot, ctx, runner_name, False, "bet_per_market_already_fired")
                            continue

                        slots_tested += 1
                        if getattr(slot, "requires_mom45", False):
                            mom45_slots_tested += 1
                            if ctx.mom45 is None:
                                mom45_slots_missing += 1
                        debug_enabled = self._strategy_debug_enabled(ctx)
                        debug_condition_result: Optional[bool] = None
                        debug_fail_reason = ""
                        if getattr(slot, "requires_mom45", False) and ctx.mom45 is None:
                            debug_condition_result = False
                            debug_fail_reason = "missing_mom45"
                        elif debug_enabled:
                            debug_condition_result, debug_fail_reason = self._debug_evaluate_slot(slot, ctx)
                            if debug_fail_reason.startswith("condition_error="):
                                error_count += 1
                        else:
                            try:
                                debug_condition_result = bool(slot.condition(ctx))
                            except Exception:
                                debug_condition_result = False
                                error_count += 1
                        if debug_condition_result is True:
                            true_tags.append(str(slot.tag))
                            signal_side = str(getattr(getattr(slot, "side", None), "value", getattr(slot, "side", "")))
                            signal = {
                                "candidate_id": _pre_pipeline_candidate_id(
                                    ctx,
                                    signal_side,
                                    str(getattr(slot, "tag", "")),
                                ),
                                "side": signal_side,
                                "strategy_id": str(getattr(slot, "tag", "")),
                            }
                            pre_strategy_signals.append(signal)
                            self._log_pre_pipeline_debug_row(
                                ctx,
                                runner_name,
                                row_type="strategy_signal",
                                candidate_id=signal["candidate_id"],
                                side=signal_side,
                                strategy_id=signal["strategy_id"],
                            )

                        res = try_fire_slot(self.staking_engine, slot, ctx)
                        if debug_enabled:
                            if debug_condition_result is True and not res:
                                debug_fail_reason = "no_fire_result_after_condition"
                            self._log_strategy_debug_row(
                                slot, ctx, runner_name, bool(debug_condition_result), debug_fail_reason
                            )
                        if not res:
                            if debug_condition_result is True:
                                signal_side = str(getattr(getattr(slot, "side", None), "value", getattr(slot, "side", "")))
                                candidate_id = _pre_pipeline_candidate_id(
                                    ctx,
                                    signal_side,
                                    str(getattr(slot, "tag", "")),
                                )
                                detail = self._debug_try_fire_slot_none_detail(slot, ctx)
                                pre_removed_candidates.append(
                                    {
                                        "candidate_id": candidate_id,
                                        "side": signal_side,
                                        "strategy_id": str(getattr(slot, "tag", "")),
                                        "removed_stage": "strategy_to_candidate",
                                        "removed_reason": "no_fire_result_after_condition",
                                        "removed_detail": detail,
                                    }
                                )
                                self._log_pre_pipeline_debug_row(
                                    ctx,
                                    runner_name,
                                    row_type="candidate_removed",
                                    candidate_id=candidate_id,
                                    side=signal_side,
                                    strategy_id=str(getattr(slot, "tag", "")),
                                    removed_stage="strategy_to_candidate",
                                    removed_reason="no_fire_result_after_condition",
                                    removed_detail=detail,
                                )
                            continue

                        candidate = _StrategyOrderCandidate(
                            slot=slot,
                            market_id=market_id,
                            market_type=(market_type or ""),
                            selection_id=int(sid),
                            course_id=str(course_id),
                            side=slot.side.value,
                            price=float(res.price),
                            size=float(res.size),
                            liability=round(res.liability or 0.0, 2),
                            reason=res.reason,
                            exec_mode=res.exec_mode,
                            sp_limit=res.sp_limit,
                            execution_phase=execution_phase,
                            triggered_systems=[str(slot.tag)],
                            triggered_prices=[float(res.price)],
                            bet_per_market_key=key,
                            phase_send_seconds_before_off=milestone,
                            best_unmatched_back_offer=bb,
                            best_unmatched_lay_offer=bl,
                            market_reference_price=_market_reference_price(lpt, bb, bl),
                            strategy_group=getattr(slot, "strategy_group", None),
                            strategy_region=getattr(slot, "strategy_region", None),
                            strategy_signal=getattr(slot, "strategy_signal", None),
                            strategy_bucket=getattr(slot, "strategy_bucket", None),
                            strategy_edge=_finite_float_or_none(getattr(ctx, "ev_place", None)),
                            strategy_score=None,
                            staking_formula=getattr(res, "staking_formula", "") or "",
                            staking_alpha=getattr(res, "staking_alpha", None),
                            staking_back_alpha=getattr(res, "staking_back_alpha", None),
                            staking_lay_alpha=getattr(res, "staking_lay_alpha", None),
                            stake_raw_before_caps=getattr(res, "stake_raw_before_caps", None),
                            stake_after_caps=getattr(res, "stake_after_caps", None),
                            lay_liability_after_sizing=getattr(res, "lay_liability_after_sizing", None),
                            lay_liability_cap=getattr(res, "lay_liability_cap", None),
                            lay_liability_cap_hit=bool(getattr(res, "lay_liability_cap_hit", False)),
                            runner_name=runner_name or "",
                        )
                        if execution_phase == _strategies.EXECUTION_PHASE_PRE:
                            pre_orders.append(candidate)
                        else:
                            post_orders.append(candidate)
                        pre_raw_candidates.append(candidate)
                        self._log_pre_pipeline_debug_row(
                            ctx,
                            runner_name,
                            row_type="raw_candidate",
                            candidate_id=_pre_pipeline_candidate_id(ctx, candidate.side, str(slot.tag)),
                            side=candidate.side,
                            strategy_id=str(slot.tag),
                        )
                        continue

                    pre_orders, pre_conflict_rejections = _resolve_back_lay_same_phase_candidates(pre_orders)
                    post_orders, post_conflict_rejections = _resolve_back_lay_same_phase_candidates(post_orders)
                    current_conflict_rejections = (
                        pre_conflict_rejections
                        if execution_phase == _strategies.EXECUTION_PHASE_PRE
                        else post_conflict_rejections
                    )
                    for rejected_order in current_conflict_rejections:
                        removed = {
                            "candidate_id": _pre_pipeline_candidate_id(
                                ctx,
                                rejected_order.side,
                                "|".join(rejected_order.triggered_systems),
                            ),
                            "side": rejected_order.side,
                            "strategy_id": "|".join(rejected_order.triggered_systems),
                            "removed_stage": "conflict_resolution",
                            "removed_reason": rejected_order.reason,
                            "removed_detail": rejected_order.pre_conflict_group_key or rejected_order.conflict_group_key,
                            "conflict_resolution_reason": rejected_order.conflict_resolution_reason,
                            "selected_side": rejected_order.selected_side,
                            "rejected_side": rejected_order.rejected_side,
                        }
                        pre_removed_candidates.append(removed)
                        self._log_pre_pipeline_debug_row(
                            ctx,
                            runner_name,
                            row_type="candidate_removed",
                            candidate_id=removed["candidate_id"],
                            side=removed["side"],
                            strategy_id=removed["strategy_id"],
                            removed_stage=removed["removed_stage"],
                            removed_reason=removed["removed_reason"],
                            removed_detail=removed["removed_detail"],
                        )
                    for rejected_order in pre_conflict_rejections + post_conflict_rejections:
                        self._log_trade_row(
                            {
                                **self._trade_base_for_final_order(rejected_order, record_phase_stake=False),
                                "status": "REJECTED_REAL",
                                "reason": rejected_order.reason,
                            }
                        )
                    merged_pre_orders = _merge_order_candidates(pre_orders)
                    merged_post_orders = _merge_order_candidates(post_orders)
                    current_orders = pre_orders if execution_phase == _strategies.EXECUTION_PHASE_PRE else post_orders
                    current_merged_orders = (
                        merged_pre_orders
                        if execution_phase == _strategies.EXECUTION_PHASE_PRE
                        else merged_post_orders
                    )
                    if len(current_orders) > len(current_merged_orders):
                        for merged_order in current_merged_orders:
                            if len(merged_order.triggered_systems) <= 1:
                                continue
                            self._log_pre_pipeline_debug_row(
                                ctx,
                                runner_name,
                                row_type="duplicate_merge",
                                candidate_id=_pre_pipeline_candidate_id(
                                    ctx,
                                    merged_order.side,
                                    "|".join(merged_order.triggered_systems),
                                ),
                                side=merged_order.side,
                                strategy_id="|".join(merged_order.triggered_systems),
                                removed_stage="duplicate_merge",
                                removed_reason="same_phase_same_runner_side_merged",
                                removed_detail=merged_order.merge_key,
                            )
                    for final_order in current_merged_orders:
                        self._log_pre_pipeline_debug_row(
                            ctx,
                            runner_name,
                            row_type="trade_created",
                            candidate_id=_pre_pipeline_candidate_id(
                                ctx,
                                final_order.side,
                                "|".join(final_order.triggered_systems),
                            ),
                            side=final_order.side,
                            strategy_id="|".join(final_order.triggered_systems),
                        )
                    self._log_pre_pipeline_summary_row(
                        ctx,
                        runner_name,
                        pre_strategy_signals=pre_strategy_signals,
                        pre_raw_candidates=pre_raw_candidates,
                        pre_after_conflict=current_orders,
                        pre_after_merge=current_merged_orders,
                        pre_removed_candidates=pre_removed_candidates,
                    )
                    for final_order in merged_pre_orders + merged_post_orders:
                        self._handle_final_strategy_order(final_order)
                    self._log_strategy_eval_summary_row(
                        ctx,
                        runner_name,
                        slots_tested,
                        true_tags,
                        error_count,
                        mom45_slots_tested=mom45_slots_tested,
                        mom45_slots_missing=mom45_slots_missing,
                    )

            except Exception as e:
                # Ne jamais casser le snapshotting pour une erreur staking/ordre
                print(f"[STRATEGY_LOOP_ERR] {market_id} sid={sid} err={e!r}")

    # ----- helpers -----
    def _trade_base_for_final_order(self, order: _StrategyOrderCandidate, *, record_phase_stake: bool = True) -> dict:
        final_system = (
            f"MERGED_{order.execution_phase}"
            if len(order.triggered_systems) > 1
            else order.final_system
        )
        stake_fields = (
            self._record_phase_stake(order)
            if record_phase_stake
            else self._phase_stake_preview_fields(order)
        )
        return {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_id": self._trade_run_id(),
            "evaluation_id": self._trade_evaluation_id(order),
            "parent_market_id": order.course_id,
            "milestone": "" if order.phase_send_seconds_before_off is None else order.phase_send_seconds_before_off,
            "complete_after_post": str(order.execution_phase == _strategies.EXECUTION_PHASE_POST),
            "post_checked": str(order.execution_phase == _strategies.EXECUTION_PHASE_POST),
            "post_signal_count": 1 if order.execution_phase == _strategies.EXECUTION_PHASE_POST else 0,
            "post_evaluated": str(order.execution_phase == _strategies.EXECUTION_PHASE_POST),
            "post_missing_reason": "post_logged" if order.execution_phase == _strategies.EXECUTION_PHASE_POST else "post_not_checked_yet",
            "market_id": order.market_id,
            "market_type": order.market_type,
            "selection_id": order.selection_id,
            "course_id": order.course_id,
            "side": order.side,
            "price_req": order.price,
            "size_req": order.size,
            "liability": round(order.liability or 0.0, 2),
            "strategy": final_system,
            "market_family": getattr(order.slot, "market_family", None),
            "strategy_group": order.strategy_group,
            "strategy_region": order.strategy_region,
            "strategy_signal": order.strategy_signal,
            "strategy_bucket": order.strategy_bucket,
            "strategy_edge": "" if order.strategy_edge is None else order.strategy_edge,
            "strategy_score": "" if order.strategy_score is None else order.strategy_score,
            "staking_formula": order.staking_formula,
            "staking_alpha": "" if order.staking_alpha is None else order.staking_alpha,
            "staking_back_alpha": "" if order.staking_back_alpha is None else order.staking_back_alpha,
            "staking_lay_alpha": "" if order.staking_lay_alpha is None else order.staking_lay_alpha,
            "stake_raw_before_caps": "" if order.stake_raw_before_caps is None else order.stake_raw_before_caps,
            "stake_after_caps": "" if order.stake_after_caps is None else order.stake_after_caps,
            "lay_liability_after_sizing": "" if order.lay_liability_after_sizing is None else order.lay_liability_after_sizing,
            "lay_liability_cap": "" if order.lay_liability_cap is None else order.lay_liability_cap,
            "lay_liability_cap_hit": str(bool(order.lay_liability_cap_hit)),
            "execution_phase": order.execution_phase,
            "triggered_systems": "|".join(order.triggered_systems),
            "triggered_prices": "|".join(str(price) for price in order.triggered_prices),
            "final_system": final_system,
            "final_price": order.price,
            "final_stake": order.size,
            "merged": str(len(order.triggered_systems) > 1),
            "processed_key": self._strategy_processed_key(order),
            "exec_mode": order.exec_mode.value,
            "ladder_enabled": "",
            "ladder_preview": "",
            "ladder_id": "",
            "ladder_tracking_key": "",
            "ladder_step": "",
            "ladder_seconds_before_off": "",
            "final_lim_price": "",
            "final_lim_price_raw": "",
            "final_lim_price_tick": "",
            "start_price": "",
            "start_price_raw": "",
            "start_price_tick": "",
            "tick_rounding_mode": "",
            "ladder_prices": "",
            "ladder_plan_frozen": "",
            "ladder_plan_created_step": "",
            "ladder_prices_frozen": "",
            "current_ladder_price_from_frozen_plan": "",
            "best_same_side_offer_at_creation": "",
            "best_back_displayed": "",
            "best_lay_displayed": "",
            "start_price_source": "",
            "ladder_direction": "",
            "ladder_disabled_lim_not_in_ladder_direction": "",
            "direct_lim_order_planned": "",
            "direct_lim_order_written": "",
            "no_replace_steps_for_direct_lim": "",
            "current_ladder_price": "",
            "best_unmatched_back_offer": "",
            "best_unmatched_lay_offer": "",
            "best_same_side_back_offer": "",
            "best_same_side_lay_offer": "",
            "source_fields_used": "",
            "market_reference_price_at_signal": "" if order.market_reference_price is None else order.market_reference_price,
            "no_better_ladder_range_reason": "",
            "previous_order_status": "",
            "matched_stake": "",
            "remaining_stake": "",
            "cancelled_previous": "",
            "cancel_failed": "",
            "stop_reason": "",
            "current_step_stake": "",
            "gruss_planned_trigger": "",
            "gruss_trigger_allowed": "",
            "gruss_bet_ref_required": "",
            "gruss_bet_ref_present": "",
            "gruss_bet_ref": "",
            "gruss_replace_confirmed": "",
            "gruss_no_stack": "",
            "conflict_detected": str(bool(order.conflict_detected)),
            "conflict_type": order.conflict_type,
            "conflict_group_key": order.conflict_group_key,
            "conflict_candidates_count": order.conflict_candidates_count or "",
            "selected_side": order.selected_side,
            "rejected_side": order.rejected_side,
            "back_systems": order.back_systems,
            "lay_systems": order.lay_systems,
            "conflict_resolution_reason": order.conflict_resolution_reason,
            "pre_back_lay_conflict": str(bool(order.pre_back_lay_conflict)),
            "pre_conflict_resolution": order.pre_conflict_resolution,
            "pre_conflict_chosen_side": order.pre_conflict_chosen_side,
            "pre_conflict_rejected_side": order.pre_conflict_rejected_side,
            "pre_conflict_reason": order.pre_conflict_reason,
            "pre_conflict_group_key": order.pre_conflict_group_key,
            "pre_conflict_course_id": order.pre_conflict_course_id,
            "pre_conflict_market_id": order.pre_conflict_market_id,
            "pre_conflict_market_type": order.pre_conflict_market_type,
            "pre_conflict_selection_id": order.pre_conflict_selection_id,
            "pre_conflict_runner_name": order.pre_conflict_runner_name,
            "pre_back_target_price": order.pre_back_target_price,
            "pre_lay_target_price": order.pre_lay_target_price,
            "pre_current_best_lay": order.pre_current_best_lay,
            "pre_current_best_back": order.pre_current_best_back,
            "pre_back_distance_ticks": order.pre_back_distance_ticks,
            "pre_lay_distance_ticks": order.pre_lay_distance_ticks,
            **stake_fields,
        }

    def _strategy_processed_key(self, order: _StrategyOrderCandidate) -> str:
        return "|".join(
            str(part)
            for part in (
                order.course_id,
                order.market_id,
                order.selection_id,
                order.side,
                order.market_type,
                order.execution_phase,
            )
        )

    def _trade_run_id(self) -> str:
        run_id = getattr(self, "_run_id", None)
        if run_id:
            return str(run_id)
        run_id = os.getenv("DOGBOT_RUN_ID")
        if run_id:
            self._run_id = run_id
            return run_id
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self._run_id = run_id
        return run_id

    def _trade_evaluation_id(self, order: _StrategyOrderCandidate) -> str:
        return "|".join(str(part) for part in (self._trade_run_id(), order.course_id))

    def _phase_stake_preview_fields(self, order: _StrategyOrderCandidate) -> dict:
        key = (order.market_id, order.selection_id, order.market_type, order.side)
        phase_stakes = self._phase_stakes_by_runner_side[key]
        stake_pre = round(phase_stakes.get(_strategies.EXECUTION_PHASE_PRE, 0.0), 2)
        stake_post = round(phase_stakes.get(_strategies.EXECUTION_PHASE_POST, 0.0), 2)
        if order.execution_phase == _strategies.EXECUTION_PHASE_PRE:
            stake_pre = round(max(stake_pre, float(order.size)), 2)
        elif order.execution_phase == _strategies.EXECUTION_PHASE_POST:
            stake_post = round(max(stake_post, float(order.size)), 2)
        return {
            "stake_pre": stake_pre,
            "stake_post": stake_post,
            "total_stake_same_runner_side": round(stake_pre + stake_post, 2),
            "pre_post_cumulative": str(stake_pre > 0 and stake_post > 0),
            "pre_existing_order_detected": str(stake_pre > 0),
            "pre_existing_order_side": order.side if stake_pre > 0 else "",
            "pre_existing_order_market_type": order.market_type if stake_pre > 0 else "",
            "pre_existing_order_stake": stake_pre if stake_pre > 0 else "",
            "post_stake": stake_post if order.execution_phase == _strategies.EXECUTION_PHASE_POST else "",
        }

    def _record_phase_stake(self, order: _StrategyOrderCandidate) -> dict:
        key = (order.market_id, order.selection_id, order.market_type, order.side)
        phase_stakes = self._phase_stakes_by_runner_side[key]
        phase_stakes[order.execution_phase] = round(
            phase_stakes.get(order.execution_phase, 0.0) + float(order.size),
            2,
        )
        stake_pre = round(phase_stakes.get(_strategies.EXECUTION_PHASE_PRE, 0.0), 2)
        stake_post = round(phase_stakes.get(_strategies.EXECUTION_PHASE_POST, 0.0), 2)
        return {
            "stake_pre": stake_pre,
            "stake_post": stake_post,
            "total_stake_same_runner_side": round(stake_pre + stake_post, 2),
            "pre_post_cumulative": str(stake_pre > 0 and stake_post > 0),
            "pre_existing_order_detected": str(stake_pre > 0),
            "pre_existing_order_side": order.side if stake_pre > 0 else "",
            "pre_existing_order_market_type": order.market_type if stake_pre > 0 else "",
            "pre_existing_order_stake": stake_pre if stake_pre > 0 else "",
            "post_stake": stake_post if order.execution_phase == _strategies.EXECUTION_PHASE_POST else "",
        }

    def _remember_pre_phase_stake(self, order: _StrategyOrderCandidate) -> None:
        if order.execution_phase != _strategies.EXECUTION_PHASE_PRE:
            return
        key = (order.market_id, order.selection_id, order.market_type, order.side)
        phase_stakes = self._phase_stakes_by_runner_side[key]
        phase_stakes[_strategies.EXECUTION_PHASE_PRE] = round(
            max(phase_stakes.get(_strategies.EXECUTION_PHASE_PRE, 0.0), float(order.size)),
            2,
        )

    def _is_pre_ladder_order(self, order: _StrategyOrderCandidate) -> bool:
        if order.execution_phase != _strategies.EXECUTION_PHASE_PRE:
            return False
        if order.exec_mode != ExecMode.LIMIT_LTP:
            return False
        return any(system in PRE_LADDER_SYSTEM_IDS for system in order.triggered_systems)

    def _pre_ladder_id(self, order: _StrategyOrderCandidate) -> str:
        final_system = (
            f"MERGED_{order.execution_phase}"
            if len(order.triggered_systems) > 1
            else order.final_system
        )
        return f"{final_system}:{order.market_id}:{order.selection_id}:{order.market_type}:{order.side}:PRE"

    def _pre_ladder_tracking_key(self, order: _StrategyOrderCandidate) -> str:
        return "|".join(
            str(part)
            for part in (
                order.market_id,
                order.selection_id,
                order.market_type,
                order.side,
                order.execution_phase,
                self._pre_ladder_id(order),
            )
        )

    def _ensure_pre_ladder_price_plans(self) -> dict[str, _FrozenPreLadderPlan]:
        plans = getattr(self, "_pre_ladder_price_plans", None)
        if plans is None:
            plans = {}
            self._pre_ladder_price_plans = plans
        return plans

    def _create_frozen_pre_ladder_plan(
        self,
        order: _StrategyOrderCandidate,
        *,
        ladder_id: str,
        step_index: int,
        seconds: int,
    ) -> _FrozenPreLadderPlan:
        steps = tuple(self._pre_ladder_steps)
        upper_side = str(order.side or "").upper()
        final_lim_price_raw = float(order.price)
        final_tick_price = round_final_lim_to_ladder_tick(order.side, final_lim_price_raw)
        best_back_displayed = order.best_unmatched_back_offer
        best_lay_displayed = order.best_unmatched_lay_offer
        start_reference = best_lay_displayed if upper_side == "BACK" else best_back_displayed
        direction = "BACK_DESCENDING" if upper_side == "BACK" else "LAY_ASCENDING"
        start_price_raw = None
        start_price_tick = None
        start_price = None
        reason = ""
        disabled_direct = False
        if start_reference is None:
            ladder_prices = [final_tick_price]
            reason = "no_start_price_source"
            disabled_direct = True
        else:
            start_price_raw = float(start_reference)
            price_plan = build_pre_ladder_from_same_side_offer(
                order.side,
                start_price_raw,
                final_lim_price_raw,
                steps=len(steps),
            )
            reason = price_plan.reason
            start_price = price_plan.start_price
            start_price_tick = price_plan.start_price
            if reason:
                ladder_prices = [final_tick_price]
                disabled_direct = reason in {"no_better_back_ladder_range", "no_better_lay_ladder_range"}
            else:
                ladder_prices = list(price_plan.prices)

        invalid_reason = ""
        if not _pre_ladder_prices_are_valid_for_side(
            upper_side,
            ladder_prices,
            final_tick_price,
            direct_plan=bool(reason),
        ):
            invalid_reason = "invalid_non_monotonic_ladder_plan"

        return _FrozenPreLadderPlan(
            ladder_id=ladder_id,
            side=upper_side,
            start_price=start_price,
            start_price_raw=start_price_raw,
            start_price_tick=start_price_tick,
            final_lim_price_raw=final_lim_price_raw,
            final_lim_price_tick=final_tick_price,
            ladder_prices=ladder_prices,
            reason=reason,
            created_step=step_index + 1,
            created_at_countdown=seconds,
            best_same_side_offer_at_creation=None if start_reference is None else float(start_reference),
            ladder_direction=direction,
            disabled_lim_not_in_ladder_direction=disabled_direct,
            invalid_reason=invalid_reason,
        )

    def _frozen_pre_ladder_plan(
        self,
        order: _StrategyOrderCandidate,
        *,
        ladder_id: str,
        step_index: int,
        seconds: int,
    ) -> _FrozenPreLadderPlan:
        plans = self._ensure_pre_ladder_price_plans()
        if step_index == 0 or ladder_id not in plans:
            plans[ladder_id] = self._create_frozen_pre_ladder_plan(
                order,
                ladder_id=ladder_id,
                step_index=step_index,
                seconds=seconds,
            )
        return plans[ladder_id]

    def _pre_ladder_step_payload(self, order: _StrategyOrderCandidate) -> dict | None:
        steps = tuple(self._pre_ladder_steps)
        seconds = int(order.phase_send_seconds_before_off or 0)
        if seconds not in steps:
            return None
        step_index = steps.index(seconds)
        step_label = f"{step_index + 1}/{len(steps)}"
        upper_side = str(order.side or "").upper()
        best_back = order.best_unmatched_back_offer
        best_lay = order.best_unmatched_lay_offer
        ladder_id = self._pre_ladder_id(order)
        price_plan = self._frozen_pre_ladder_plan(
            order,
            ladder_id=ladder_id,
            step_index=step_index,
            seconds=seconds,
        )
        final_lim_price = price_plan.final_lim_price_raw
        fallback_reason = price_plan.invalid_reason or price_plan.reason
        final_tick_price = price_plan.final_lim_price_tick
        tick_rounding_mode = "BACK_CEIL" if upper_side == "BACK" else "LAY_FLOOR"
        if upper_side == "BACK":
            source_fields_used = (
                "best_lay_displayed=runner.ex.available_to_lay[0].price;"
                "best_back_displayed=runner.ex.available_to_back[0].price;"
                "start_price_source=best_lay_displayed"
            )
        else:
            source_fields_used = (
                "best_back_displayed=runner.ex.available_to_back[0].price;"
                "best_lay_displayed=runner.ex.available_to_lay[0].price;"
                "start_price_source=best_back_displayed"
            )

        ladder_prices = list(price_plan.ladder_prices)
        if not ladder_prices:
            return None
        if step_index >= len(ladder_prices):
            return None
        current_price = ladder_prices[min(step_index, len(ladder_prices) - 1)]

        decision = decide_pre_ladder_step(None, full_stake=float(order.size))
        trigger_plan = plan_gruss_pre_ladder_trigger(
            side=order.side,
            step_index=step_index,
            bet_ref=None,
            replace_confirmed=False,
        )
        return {
            "ladder_id": ladder_id,
            "ladder_tracking_key": self._pre_ladder_tracking_key(order),
            "ladder_step": step_label,
            "ladder_seconds_before_off": seconds,
            "final_lim_price": final_lim_price,
            "final_lim_price_raw": final_lim_price,
            "final_lim_price_tick": final_tick_price,
            "start_price": "" if price_plan.start_price is None else price_plan.start_price,
            "start_price_raw": "" if price_plan.start_price_raw is None else price_plan.start_price_raw,
            "start_price_tick": "" if price_plan.start_price_tick is None else price_plan.start_price_tick,
            "tick_rounding_mode": tick_rounding_mode,
            "ladder_prices": "|".join(str(price) for price in ladder_prices),
            "ladder_plan_frozen": str(True),
            "ladder_plan_created_step": price_plan.created_step,
            "ladder_prices_frozen": "|".join(str(price) for price in ladder_prices),
            "current_ladder_price_from_frozen_plan": str(True),
            "best_same_side_offer_at_creation": (
                "" if price_plan.best_same_side_offer_at_creation is None else price_plan.best_same_side_offer_at_creation
            ),
            "best_back_displayed": "" if best_back is None else best_back,
            "best_lay_displayed": "" if best_lay is None else best_lay,
            "start_price_source": (
                "best_lay_displayed" if upper_side == "BACK" else "best_back_displayed"
            ),
            "ladder_direction": price_plan.ladder_direction,
            "ladder_disabled_lim_not_in_ladder_direction": str(
                bool(price_plan.disabled_lim_not_in_ladder_direction)
            ),
            "direct_lim_order_planned": str(bool(price_plan.disabled_lim_not_in_ladder_direction)),
            "direct_lim_order_written": str(False),
            "no_replace_steps_for_direct_lim": str(bool(price_plan.disabled_lim_not_in_ladder_direction)),
            "current_ladder_price": current_price,
            "best_unmatched_back_offer": "" if best_back is None else best_back,
            "best_unmatched_lay_offer": "" if best_lay is None else best_lay,
            "best_same_side_back_offer": "" if best_back is None else best_back,
            "best_same_side_lay_offer": "" if best_lay is None else best_lay,
            "source_fields_used": source_fields_used,
            "market_reference_price_at_signal": _first_positive(
                order.market_reference_price,
                (best_back + best_lay) / 2.0 if best_back is not None and best_lay is not None else None,
                price_plan.best_same_side_offer_at_creation,
            ),
            "no_better_ladder_range_reason": (
                fallback_reason
                if fallback_reason in {"no_better_back_ladder_range", "no_better_lay_ladder_range"}
                else ""
            ),
            "previous_order_status": decision.previous_order_status,
            "matched_stake": decision.matched_stake,
            "remaining_stake": decision.remaining_stake,
            "cancelled_previous": str(decision.cancelled_previous),
            "cancel_failed": str(decision.cancel_failed),
            "stop_reason": decision.stop_reason,
            "current_step_stake": decision.current_step_stake,
            "gruss_planned_trigger": trigger_plan.trigger,
            "gruss_trigger_allowed": str(trigger_plan.allowed),
            "gruss_bet_ref_required": str(trigger_plan.bet_ref_required),
            "gruss_bet_ref_present": str(trigger_plan.bet_ref_present),
            "gruss_bet_ref": "",
            "gruss_replace_confirmed": str(False),
            "gruss_no_stack": str(trigger_plan.no_stack),
            "trigger_plan_reason": trigger_plan.reason,
            "fallback_reason": fallback_reason,
        }

    def _handle_pre_ladder_order(self, order: _StrategyOrderCandidate) -> bool:
        if not self._is_pre_ladder_order(order):
            return False
        ladder_enabled = _env_flag("DOGBOT_PRE_LADDER_ENABLED", False)
        ladder_preview = _env_flag("DOGBOT_PRE_LADDER_PREVIEW", True)
        payload = self._pre_ladder_step_payload(order)
        if payload is None:
            return True

        trade_base = self._trade_base_for_final_order(order, record_phase_stake=False)
        row = {
            **trade_base,
            "price_req": payload["current_ladder_price"],
            "size_req": payload["current_step_stake"] or order.size,
            "final_price": payload["current_ladder_price"],
            "final_stake": payload["current_step_stake"] or order.size,
            "ladder_enabled": str(ladder_enabled),
            "ladder_preview": str(ladder_preview),
            "ladder_id": payload["ladder_id"],
            "ladder_tracking_key": payload["ladder_tracking_key"],
            "ladder_step": payload["ladder_step"],
            "ladder_seconds_before_off": payload["ladder_seconds_before_off"],
            "final_lim_price": payload["final_lim_price"],
            "final_lim_price_raw": payload["final_lim_price_raw"],
            "final_lim_price_tick": payload["final_lim_price_tick"],
            "start_price": payload["start_price"],
            "start_price_raw": payload["start_price_raw"],
            "start_price_tick": payload["start_price_tick"],
            "tick_rounding_mode": payload["tick_rounding_mode"],
            "ladder_prices": payload["ladder_prices"],
            "ladder_plan_frozen": payload["ladder_plan_frozen"],
            "ladder_plan_created_step": payload["ladder_plan_created_step"],
            "ladder_prices_frozen": payload["ladder_prices_frozen"],
            "current_ladder_price_from_frozen_plan": payload["current_ladder_price_from_frozen_plan"],
            "best_same_side_offer_at_creation": payload["best_same_side_offer_at_creation"],
            "best_back_displayed": payload["best_back_displayed"],
            "best_lay_displayed": payload["best_lay_displayed"],
            "start_price_source": payload["start_price_source"],
            "ladder_direction": payload["ladder_direction"],
            "ladder_disabled_lim_not_in_ladder_direction": payload[
                "ladder_disabled_lim_not_in_ladder_direction"
            ],
            "direct_lim_order_planned": payload["direct_lim_order_planned"],
            "direct_lim_order_written": payload["direct_lim_order_written"],
            "no_replace_steps_for_direct_lim": payload["no_replace_steps_for_direct_lim"],
            "current_ladder_price": payload["current_ladder_price"],
            "best_unmatched_back_offer": payload["best_unmatched_back_offer"],
            "best_unmatched_lay_offer": payload["best_unmatched_lay_offer"],
            "best_same_side_back_offer": payload["best_same_side_back_offer"],
            "best_same_side_lay_offer": payload["best_same_side_lay_offer"],
            "source_fields_used": payload["source_fields_used"],
            "market_reference_price_at_signal": payload["market_reference_price_at_signal"],
            "no_better_ladder_range_reason": payload["no_better_ladder_range_reason"],
            "previous_order_status": payload["previous_order_status"],
            "matched_stake": payload["matched_stake"],
            "remaining_stake": payload["remaining_stake"],
            "cancelled_previous": payload["cancelled_previous"],
            "cancel_failed": payload["cancel_failed"],
            "stop_reason": payload["stop_reason"],
            "current_step_stake": payload["current_step_stake"],
            "gruss_planned_trigger": payload["gruss_planned_trigger"],
            "gruss_trigger_allowed": payload["gruss_trigger_allowed"],
            "gruss_bet_ref_required": payload["gruss_bet_ref_required"],
            "gruss_bet_ref_present": payload["gruss_bet_ref_present"],
            "gruss_bet_ref": payload["gruss_bet_ref"],
            "gruss_replace_confirmed": payload["gruss_replace_confirmed"],
            "gruss_no_stack": payload["gruss_no_stack"],
        }
        if payload["fallback_reason"] == "invalid_non_monotonic_ladder_plan":
            self._log_trade_row({**row, "status": "PRE_LADDER_INVALID", "reason": payload["fallback_reason"]})
            return True
        self._remember_pre_phase_stake(order)
        if ladder_preview:
            reason = payload["fallback_reason"] or payload["trigger_plan_reason"] or "pre_ladder_preview"
            self._log_trade_row({**row, "status": "PRE_LADDER_PREVIEW", "reason": reason})
            return True
        if ladder_enabled:
            reason = payload["fallback_reason"] or payload["trigger_plan_reason"] or "pre_ladder_real_ready"
            self._log_trade_row({**row, "status": "PRE_LADDER_REAL_READY", "reason": reason})
            return True

        self._log_trade_row(
            {
                **row,
                "status": "PRE_LADDER_DISABLED",
                "reason": "real_pre_ladder_requires_cancel_replace",
                "stop_reason": "real_ladder_not_implemented",
            }
        )
        return True

    def _handle_final_strategy_order(self, order: _StrategyOrderCandidate) -> None:
        if self._handle_pre_ladder_order(order):
            return
        trade_base = self._trade_base_for_final_order(order)
        final_system = str(trade_base["final_system"])
        if self.dry_run:
            real_ready = (
                order.execution_phase == _strategies.EXECUTION_PHASE_POST
                and _gruss_real_post_ready_for_trade_log()
            )
            self._log_trade_row(
                {
                    **trade_base,
                    "status": "REAL_READY" if real_ready else "DRYRUN",
                    "reason": "post_real_provider_armed" if real_ready else order.reason,
                }
            )
            if getattr(order.slot, "bet_per_market", False):
                self._slot_market_fired.add(order.bet_per_market_key)
            return

        self._log_trade_row({**trade_base, "status": "LIVE_INTENT", "reason": order.reason})
        print(
            f"[LIVE_INTENT] {order.market_id} sid={order.selection_id} strategy={final_system} "
            f"phase={order.execution_phase} side={order.side} price={order.price} size={order.size}"
        )

        if getattr(order.slot, "bet_per_market", False):
            self._slot_market_fired.add(order.bet_per_market_key)

        can, reason = self.exposure.can_place(
            Side(order.side),
            market_id=order.market_id,
            selection_id=order.selection_id,
            planned_stake=float(order.size),
            planned_liability=float(order.liability or 0.0),
        )
        if not can:
            self._log_trade_row({**trade_base, "status": "LIVE_BLOCKED", "reason": reason})
            print(f"[LIVE_BLOCKED] {order.market_id} sid={order.selection_id} strategy={final_system} reason={reason}")
            return

        orr = None
        try:
            if order.exec_mode == ExecMode.LIMIT_LTP:
                idem_key = f"{order.market_id}:{order.selection_id}:{final_system}:{order.execution_phase}:{order.price}"
                orr = self.order_executor.place_limit(
                    market_id=order.market_id,
                    selection_id=order.selection_id,
                    side=order.side,
                    price=float(order.price),
                    size=float(order.size),
                    strategy=final_system,
                    persistence=os.getenv("PERSISTENCE", "LAPSE"),
                    idem_key=idem_key,
                    retries=int(os.getenv("ORDER_RETRIES", "2")),
                    backoff_ms=int(os.getenv("ORDER_BACKOFF_MS", "250")),
                )
            elif order.exec_mode == ExecMode.SP_MOC:
                qty = float(order.size) if order.side == "BACK" else float(order.liability or 0.0)
                idem_key = f"{order.market_id}:{order.selection_id}:{final_system}:{order.execution_phase}:SP_MOC"
                orr = self.order_executor.place_sp_market_on_close(
                    market_id=order.market_id,
                    selection_id=order.selection_id,
                    side=order.side,
                    size_or_liability=qty,
                    strategy=final_system,
                    idem_key=idem_key,
                    retries=int(os.getenv("ORDER_RETRIES", "2")),
                    backoff_ms=int(os.getenv("ORDER_BACKOFF_MS", "250")),
                )
            elif order.exec_mode == ExecMode.SP_LOC:
                sp_lim = order.sp_limit if order.sp_limit is not None else getattr(order.slot, "sp_limit", None)
                if sp_lim is None:
                    sp_lim = order.price
                sp_lim = float(sp_lim)
                qty = float(order.size) if order.side == "BACK" else float(order.liability or 0.0)
                idem_key = f"{order.market_id}:{order.selection_id}:{final_system}:{order.execution_phase}:SP_LOC:{sp_lim}"
                orr = self.order_executor.place_sp_limit_on_close(
                    market_id=order.market_id,
                    selection_id=order.selection_id,
                    side=order.side,
                    size_or_liability=qty,
                    sp_limit_price=sp_lim,
                    strategy=final_system,
                    idem_key=idem_key,
                    retries=int(os.getenv("ORDER_RETRIES", "2")),
                    backoff_ms=int(os.getenv("ORDER_BACKOFF_MS", "250")),
                )
            else:
                raise ValueError(f"unsupported_exec_mode={order.exec_mode!r}")
        except Exception as e:
            reject_reason = f"order_exception={e!r}"
            self._log_trade_row({**trade_base, "status": "LIVE_REJECTED", "reason": reject_reason})
            self._log_strategy_error(order.market_id, order.selection_id, reject_reason)
            print(f"[LIVE_REJECTED] {order.market_id} sid={order.selection_id} strategy={final_system} reason={reject_reason}")
            return

        order_reason = self._order_result_reason(orr)
        order_status = getattr(orr, "status", None) if orr is not None else None
        if order_status == "IDEMPOTENT_SKIPPED":
            self._log_trade_row({**trade_base, "status": "LIVE_SKIPPED", "reason": "IDEMPOTENT_SKIPPED"})
            print(f"[LIVE_SKIPPED] {order.market_id} sid={order.selection_id} strategy={final_system} reason=IDEMPOTENT_SKIPPED")
        elif orr and getattr(orr, "ok", False):
            self._log_trade_row({**trade_base, "status": "LIVE_PLACED", "reason": order_reason})
            print(f"[LIVE_PLACED] {order.market_id} sid={order.selection_id} strategy={final_system} reason={order_reason}")
            self.exposure.on_placed(
                Side(order.side),
                market_id=order.market_id,
                selection_id=order.selection_id,
                stake=float(order.size),
                liability=float(order.liability or 0.0),
            )
        else:
            self._log_trade_row({**trade_base, "status": "LIVE_REJECTED", "reason": order_reason})
            self._log_strategy_error(order.market_id, order.selection_id, order_reason)
            print(f"[LIVE_REJECTED] {order.market_id} sid={order.selection_id} strategy={final_system} reason={order_reason}")

    def _ensure_trade_header(self, path: Path) -> None:
        header = list(self.TRADE_HEADER)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    first = f.readline().strip()
                if first.split(",") == header:
                    return
            except Exception:
                first = ""
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path.rename(path.with_name(path.stem + f"_old_{ts}" + path.suffix))
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def _log_trade_row(self, row: dict) -> None:
        fname = self.trades_dir / f"trades_{datetime.now(timezone.utc):%Y%m%d}.csv"
        self._ensure_trade_header(fname)
        with fname.open("a", newline="", encoding="utf-8") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=self.TRADE_HEADER)
            w.writerow(row)

    def _order_result_reason(self, orr: Any) -> str:
        if orr is None:
            return "order_result_missing"
        parts = []
        for name in ("bet_id", "order_id", "status", "error_code", "message"):
            try:
                value = getattr(orr, name, None)
            except Exception:
                value = None
            if value not in (None, ""):
                parts.append(f"{name}={value}")
        if not parts:
            parts.append(repr(orr))
        return " ".join(parts)

    def _log_strategy_error(self, market_id: str, sid: Any, err: Any) -> None:
        fname = self.data_dir / f"strategy_errors_{datetime.now(timezone.utc):%Y%m%d}.log"
        with fname.open("a", encoding="utf-8") as f:
            f.write(f"{_now_utc_iso()} market_id={market_id} sid={sid} err={err!r}\n")

    def _extract_catalogue_info(self, mie: Any, md: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "event_id": None, "event_name": None, "venue": None, "country_code": None,
            "start_utc": None, "market_type": None, "market_name": None, "meta_by_sid": {},
        }
        info["market_name"] = getattr(mie, "market_name", None) or getattr(mie, "marketName", None)
        start = getattr(mie, "event_open_utc", None) or getattr(mie, "market_start_time", None)
        if start is None and md is not None:
            start = getattr(md, "market_time", None)
        info["start_utc"] = _tz_utc(start)
        ev = getattr(mie, "event", None)
        if ev is not None:
            info["event_id"] = getattr(ev, "id", None)
            info["event_name"] = getattr(ev, "name", None)
            info["venue"] = getattr(ev, "venue", None)
            info["country_code"] = getattr(ev, "country_code", None) or getattr(ev, "countryCode", None)
        m = {}
        try:
            for rc in getattr(mie, "runners", []) or []:
                sid = getattr(rc, "selection_id", None) or getattr(rc, "selectionId", None)
                rn  = getattr(rc, "runner_name", None) or getattr(rc, "runnerName", None)
                mdict = getattr(rc, "metadata", None) or {}
                draw = mdict.get("CLOTH_NUMBER") or mdict.get("clothNumber") or mdict.get("TRAP")
                m[int(sid)] = _MetaStub(runner_name=rn, draw=draw, sort_priority=getattr(rc, "sort_priority", None))
        except Exception:
            pass
        info["meta_by_sid"] = m
        # market_type robuste
        mt_hint = getattr(md, "market_type", None)
        if isinstance(mt_hint, str) and mt_hint.upper() in ("WIN","PLACE"):
            info["market_type"] = mt_hint.upper()
        else:
            nw = _num_winners(md)
            if isinstance(nw, int):
                info["market_type"] = "PLACE" if nw >= 2 else "WIN"
            else:
                nm = info["market_name"] or ""
                info["market_type"] = "PLACE" if ("place" in nm.lower()) else "WIN"
        return info

    @staticmethod
    def _compute_virtual_traps(runners, meta_by_sid):
        active = []
        absent_traps = set()
        for r in (runners or []):
            st = (getattr(r, "status", None) or "ACTIVE").upper()
            sid = getattr(r, "selection_id", None)
            rm = meta_by_sid.get(int(sid)) if sid is not None else None
            runner_name = getattr(rm, "runner_name", None) if rm else None
            trap = _parse_trap(rm, runner_name)
            if trap is None or sid is None:
                continue
            if st == "ACTIVE":
                active.append((int(sid), int(trap)))
            else:
                absent_traps.add(int(trap))
        active.sort(key=lambda t: t[1])
        N = len(active)
        vmap = {}
        if N == 0:
            return vmap
        for i, (sid, _trap) in enumerate(active):
            vmap[sid] = i + 1
        if N >= 7 and 8 in absent_traps:
            rightmost_sid = active[-1][0]
            vmap[rightmost_sid] = 8
        return vmap

    def _milestone_due(self, mid: str, tto: Optional[float]) -> Optional[int]:
        if tto is None:
            return None
        ms_list = self._next_ms[mid]
        forced = getattr(self, "_forced_strategy_milestones", None)
        if isinstance(forced, dict) and mid in forced:
            ms = forced.pop(mid)
            if ms in ms_list:
                ms_list.remove(ms)
            return ms
        for ms in list(ms_list):
            if abs(tto - ms) <= self.TOLERANCE_S:
                ms_list.remove(ms)
                return ms
        last = self._last_tto.get(mid)
        if last is None:
            return None
        for ms in list(ms_list):
            if (last - ms) > 0 and (ms - tto) >= 0:
                ms_list.remove(ms)
                return ms
        return None
