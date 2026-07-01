from __future__ import annotations

import os
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dogbot.config import (
    DATA_PROVIDER_GRUSS_EXCEL,
    ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN,
    ORDER_PROVIDER_GRUSS_EXCEL_REAL,
    ProviderConfig,
    load_provider_config,
)
from dogbot.executor import Executor
from dogbot.executor import (
    _execution_phase_for_milestone,
    _post_send_seconds_before_off_from_env,
    _pre_ladder_steps_from_env,
)
from dogbot.feed_selector import create_data_feed_from_config
from dogbot.gruss.gruss_engine_adapter import GrussEngineBundle, build_engine_bundle
from dogbot.gruss.gruss_feed import GrussFeed
from dogbot.gruss.gruss_mapper import GrussMapper, GrussSnapshot
from dogbot.gruss.gruss_momentum import (
    GrussMomentumBuffer,
    GrussMomentumCourseStatus,
    GrussMomentumValue,
)
from dogbot.gruss.gruss_orders import GrussOrderProvider, OrderIntent, make_order_intent
from dogbot.gruss.gruss_region import (
    is_gruss_result_screen,
    normalize_gruss_meeting_name,
    normalize_gruss_region,
)
from dogbot.strategies import build_registry, try_fire_slot


GRUSS_TRADE_DIAGNOSTIC_FIELDS = [
    "data_provider",
    "order_provider",
    "win_market_id",
    "place_market_id",
    "parent_id",
    "countdown_seconds",
    "countdown_display",
    "tradable",
    "has_mom45",
    "mom45_reason",
    "first_seen_countdown",
    "t45_anchor_found",
    "mom45",
    "mom45_win_best_back",
    "mom45_win_ltp",
    "mom45_place_best_back",
    "mom45_place_ltp",
    "win_best_back",
    "win_best_lay",
    "place_best_back",
    "place_best_lay",
    "place_winners",
    "k_place_used",
    "fallback_k_place_used",
    "place_theorique",
    "ev_place",
    "gruss_event_path",
    "gruss_win_market_title",
    "gruss_place_market_title",
]


@dataclass(frozen=True)
class GrussDryRunRead:
    win_snapshot: GrussSnapshot
    place_snapshot: GrussSnapshot
    validation_warnings: list[str]
    tradable: bool
    skip_reason: str | None


@dataclass
class GrussDryRunRunner:
    data_dir: Path
    processed_store: "ProcessedRaceStore"
    executor: Executor | None = None

    def __init__(self, data_dir: str | Path = "./data") -> None:
        self.data_dir = Path(data_dir)
        self.processed_store = ProcessedRaceStore(self.data_dir / "gruss_dryrun_processed.csv")
        self.executor = None

    def ensure_executor(self, bundle: GrussEngineBundle) -> Executor:
        if self.executor is None:
            self.executor = Executor(
                client=_DryRunClient(),
                strategy=_NoopLegacyStrategy(),
                market_index=bundle.market_index,
                dry_run=True,
                data_dir=str(self.data_dir),
            )
        else:
            self.executor.market_index = bundle.market_index
        return self.executor

    def evaluate(
        self,
        win_snapshot: GrussSnapshot,
        place_snapshot: GrussSnapshot,
        *,
        debug_strategies: bool = False,
        momentum_buffer: GrussMomentumBuffer | None = None,
        force_milestone: int | None = None,
    ) -> None:
        bundle = build_engine_bundle(win_snapshot, place_snapshot)
        executor = self.ensure_executor(bundle)
        if force_milestone is not None:
            executor._forced_strategy_milestones = {
                bundle.win_book.market_id: force_milestone,
                bundle.place_book.market_id: force_milestone,
            }
        momentum_values = seed_gruss_momentum_into_executor(
            executor,
            bundle,
            win_snapshot,
            place_snapshot,
            momentum_buffer,
        )
        momentum_course_status = (
            momentum_buffer.course_status(win_snapshot, place_snapshot)
            if momentum_buffer is not None
            else None
        )
        install_gruss_trade_diagnostics(
            executor,
            win_snapshot,
            place_snapshot,
            momentum_values,
            momentum_course_status,
        )
        if debug_strategies:
            install_gruss_strategy_debug_logger(executor)
        _seed_strategy_milestone_if_due(executor, bundle, win_snapshot, place_snapshot)
        executor.process_book(bundle.win_book)
        executor.process_book(bundle.place_book)

    def trade_row_count(self) -> int:
        path = self.trade_path()
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return max(0, sum(1 for _ in handle) - 1)

    def trade_path(self) -> Path:
        return self.data_dir / f"trades_{datetime.now(timezone.utc):%Y%m%d}.csv"

    def trade_rows_since(self, row_count_before: int) -> list[dict[str, str]]:
        path = self.trade_path()
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return rows[row_count_before:]

    def log_gruss_order_intents(
        self,
        trade_rows: list[dict[str, str]],
        win_snapshot: GrussSnapshot,
        place_snapshot: GrussSnapshot,
    ) -> list[Any]:
        provider = GrussOrderProvider(self.data_dir)
        results = []
        for intent in build_order_intents_from_trade_rows(trade_rows, win_snapshot, place_snapshot):
            results.append(provider.place_order(intent))
        return results


class ProcessedRaceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen = self._load()

    def has_seen(self, key: str) -> bool:
        return key in self._seen

    def mark_seen(self, key: str, win_market_id: str | None, place_market_id: str | None) -> None:
        if key in self._seen:
            return
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(["ts", "race_key", "win_market_id", "place_market_id"])
            writer.writerow([datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), key, win_market_id, place_market_id])
        self._seen.add(key)

    def _load(self) -> set[str]:
        if not self.path.exists():
            return set()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return {row["race_key"] for row in reader if row.get("race_key")}
        except Exception:
            return set()


def read_gruss_dryrun_state(feed: GrussFeed) -> GrussDryRunRead:
    feed.connect_open_workbook()
    if not feed.bridge.has_sheet("WIN"):
        raise RuntimeError("onglet WIN manquant")
    if not feed.bridge.has_sheet("PLACE"):
        raise RuntimeError("onglet PLACE manquant")
    win_snapshot = feed.read_snapshot("WIN")
    place_snapshot = feed.read_snapshot("PLACE")
    validation_warnings = GrussMapper.get_win_place_validation_warnings(win_snapshot, place_snapshot)
    tradable = _is_tradable(win_snapshot, place_snapshot, validation_ok=not validation_warnings)
    skip_reason = get_skip_reason(win_snapshot, place_snapshot, validation_warnings, tradable)
    return GrussDryRunRead(
        win_snapshot=win_snapshot,
        place_snapshot=place_snapshot,
        validation_warnings=validation_warnings,
        tradable=tradable,
        skip_reason=skip_reason,
    )


def get_skip_reason(
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
    validation_warnings: list[str],
    tradable: bool,
) -> str | None:
    if is_result_screen_for_snapshots(win_snapshot, place_snapshot):
        return "result_screen"
    if validation_warnings:
        return "win_place_mismatch: " + "; ".join(validation_warnings)
    if not tradable:
        return "not_tradable"
    countdown_seconds = win_snapshot.metadata.countdown_seconds
    if countdown_seconds is None:
        countdown_seconds = place_snapshot.metadata.countdown_seconds
    if countdown_seconds is None:
        return "countdown_seconds_unavailable"
    normalized_region = gruss_region_for_snapshots(win_snapshot, place_snapshot)
    if normalized_region == "UNKNOWN":
        return "unknown_gruss_region"
    if _missing_runner_prices(win_snapshot, place_snapshot):
        return "missing_runner_prices"
    return None


def race_key(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> str:
    parent_id = win_snapshot.metadata.parent_id or place_snapshot.metadata.parent_id
    if parent_id:
        return f"parent:{parent_id}"
    return f"markets:{win_snapshot.metadata.market_id}:{place_snapshot.metadata.market_id}"


def validate_gruss_dryrun_provider_config(config: ProviderConfig | None = None) -> ProviderConfig:
    config = config or load_provider_config()
    if config.data_provider != DATA_PROVIDER_GRUSS_EXCEL:
        raise RuntimeError(
            "Gruss dry-run requires DOGBOT_DATA_PROVIDER=gruss_excel "
            f"(current: {config.data_provider})"
        )
    if config.order_provider != ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN:
        raise RuntimeError(
            "Gruss dry-run requires DOGBOT_ORDER_PROVIDER=gruss_excel_dryrun "
            f"(current: {config.order_provider})"
        )
    os.environ["DRY_RUN"] = "1"
    return config


def create_configured_gruss_feed(config: ProviderConfig | None = None) -> GrussFeed:
    config = validate_gruss_dryrun_provider_config(config)
    feed = create_data_feed_from_config(config)
    if not isinstance(feed, GrussFeed):
        raise RuntimeError("configured data feed is not GrussFeed")
    return feed


def describe_state(state: GrussDryRunRead) -> str:
    win_meta = state.win_snapshot.metadata
    place_meta = state.place_snapshot.metadata
    countdown = win_meta.countdown_display or place_meta.countdown_display or ""
    seconds = win_meta.countdown_seconds
    if seconds is None:
        seconds = place_meta.countdown_seconds
    validation = "OK" if not state.validation_warnings else "KO"
    tradable = "OK" if state.tradable else "KO"
    meeting_name = gruss_meeting_name_for_snapshots(state.win_snapshot, state.place_snapshot)
    normalized_region = gruss_region_for_snapshots(state.win_snapshot, state.place_snapshot)
    return (
        f"course={win_meta.event_path or place_meta.event_path} "
        f"meeting={meeting_name} region={normalized_region} "
        f"countdown={countdown} seconds={seconds} "
        f"win_market_id={win_meta.market_id} place_market_id={place_meta.market_id} "
        f"validation={validation} tradable={tradable} "
        f"runners={len(state.win_snapshot.runners)}/{len(state.place_snapshot.runners)}"
    )


def gruss_meeting_name_for_snapshots(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> str:
    event_path = win_snapshot.metadata.event_path or place_snapshot.metadata.event_path
    title = win_snapshot.metadata.market_title or place_snapshot.metadata.market_title
    return normalize_gruss_meeting_name(event_path or title)


def gruss_region_for_snapshots(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> str:
    event_path = win_snapshot.metadata.event_path or place_snapshot.metadata.event_path
    market_title = win_snapshot.metadata.market_title or place_snapshot.metadata.market_title
    meeting_name = gruss_meeting_name_for_snapshots(win_snapshot, place_snapshot)
    return normalize_gruss_region(
        event_path=event_path,
        market_title=market_title,
        meeting_name=meeting_name,
    )


def is_result_screen_for_snapshots(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> bool:
    return is_gruss_result_screen(
        event_path=win_snapshot.metadata.event_path or place_snapshot.metadata.event_path,
        market_title=win_snapshot.metadata.market_title or place_snapshot.metadata.market_title,
        meeting_name=gruss_meeting_name_for_snapshots(win_snapshot, place_snapshot),
    )


def strategy_registry_diagnostics() -> dict[str, Any]:
    slots = build_registry()
    by_side = Counter(_slot_side(slot) for slot in slots)
    by_market_type = Counter(_slot_market_type(slot) for slot in slots)
    by_execution_phase = Counter(str(getattr(slot, "execution_phase", "POST")).upper() for slot in slots)
    place_slots = [slot for slot in slots if "PLACE" in _slot_id(slot).upper()]
    lay_place_slots = [slot for slot in slots if "LAY_PLACE" in _slot_id(slot).upper()]
    pre_slots = [slot for slot in slots if str(getattr(slot, "execution_phase", "POST")).upper() == "PRE"]
    post_slots = [slot for slot in slots if str(getattr(slot, "execution_phase", "POST")).upper() == "POST"]
    return {
        "total": len(slots),
        "by_side": dict(by_side),
        "by_market_type": dict(by_market_type),
        "by_execution_phase": dict(by_execution_phase),
        "pre_ids": [_slot_id(slot) for slot in pre_slots],
        "post_ids": [_slot_id(slot) for slot in post_slots],
        "place_ids": [_slot_id(slot) for slot in place_slots],
        "lay_place_ids": [_slot_id(slot) for slot in lay_place_slots],
        "details": [
            {
                "strategy_id": _slot_id(slot),
                "side": _slot_side(slot),
                "market_type": _slot_market_type(slot),
                "mode": _slot_mode(slot),
                "execution_phase": getattr(slot, "execution_phase", "POST"),
                "requires_mom45": bool(getattr(slot, "requires_mom45", False)),
            }
            for slot in slots
        ],
    }


def print_strategy_registry_diagnostics() -> None:
    diagnostics = strategy_registry_diagnostics()
    print("Strategies registry diagnostics")
    print(f"  total: {diagnostics['total']}")
    print(f"  by side: {_format_counts(diagnostics['by_side'])}")
    print(f"  by market_type: {_format_counts(diagnostics['by_market_type'])}")
    print(f"  by execution_phase: {_format_counts(diagnostics['by_execution_phase'])}")
    print(f"  PRE total: {len(diagnostics['pre_ids'])}")
    print(f"  POST total: {len(diagnostics['post_ids'])}")
    print(f"  PRE strategy_ids: {', '.join(diagnostics['pre_ids']) or '-'}")
    print(f"  POST strategy_ids: {', '.join(diagnostics['post_ids']) or '-'}")
    print(f"  PLACE strategy_ids: {', '.join(diagnostics['place_ids']) or '-'}")
    print(f"  LAY_PLACE strategy_ids: {', '.join(diagnostics['lay_place_ids']) or '-'}")
    print("  registry details:")
    for detail in diagnostics["details"]:
        if "PLACE" not in detail["strategy_id"].upper():
            continue
        print(
            "    "
            f"{detail['strategy_id']} side={detail['side']} "
            f"market_type={detail['market_type']} mode={detail['mode']} "
            f"execution_phase={detail['execution_phase']} "
            f"requires_mom45={detail['requires_mom45']}"
        )


def active_strategy_milestones() -> tuple[int, ...]:
    milestones = {_post_send_seconds_before_off_from_env()}
    has_pre_systems = any(
        str(getattr(slot, "execution_phase", "POST")).upper() == "PRE"
        for slot in build_registry()
    )
    if _env_true(os.getenv("DOGBOT_PRE_LADDER_PREVIEW"), default=True) or has_pre_systems:
        milestones.update(_pre_ladder_steps_from_env())
    return tuple(sorted(milestones, reverse=True))


def current_strategy_milestone(
    win_countdown_seconds: int | None,
    place_countdown_seconds: int | None,
) -> int | None:
    seconds = win_countdown_seconds
    if seconds is None:
        seconds = place_countdown_seconds
    if seconds is None:
        return None
    try:
        value = int(seconds)
    except (TypeError, ValueError):
        return None
    return value if value in active_strategy_milestones() else None


def countdown_wait_reason(
    win_countdown_seconds: int | None,
    place_countdown_seconds: int | None,
) -> str | None:
    seconds = win_countdown_seconds
    if seconds is None:
        seconds = place_countdown_seconds
    if seconds is None:
        return "skip: countdown_seconds_unavailable"
    try:
        value = int(seconds)
    except (TypeError, ValueError):
        return "skip: countdown_seconds_unavailable"
    milestones = active_strategy_milestones()
    if value in milestones:
        return None
    future = [milestone for milestone in milestones if milestone < value]
    if future:
        next_milestone = max(future)
        phase = _execution_phase_for_milestone(next_milestone) or "UNKNOWN"
        return (
            f"wait: countdown_seconds={value} next_milestone={next_milestone} "
            f"execution_phase={phase}"
        )
    return "wait: no_active_milestone_remaining"


def describe_current_strategy_milestone(
    win_countdown_seconds: int | None,
    place_countdown_seconds: int | None,
) -> str:
    milestone = current_strategy_milestone(win_countdown_seconds, place_countdown_seconds)
    if milestone is None:
        return "milestone=none execution_phase=UNKNOWN"
    phase = _execution_phase_for_milestone(milestone) or "UNKNOWN"
    parts = [f"milestone={milestone}", f"execution_phase={phase}"]
    if phase == "PRE":
        steps = _pre_ladder_steps_from_env()
        if milestone in steps:
            parts.append(f"pre_ladder_step={steps.index(milestone) + 1}/{len(steps)}")
        parts.append("evaluating PRE systems")
    elif phase == "POST":
        parts.append("evaluating POST systems")
    return " ".join(parts)


def strategy_milestone_key(base_key: str, milestone: int | None) -> str:
    phase = _execution_phase_for_milestone(milestone) if milestone is not None else None
    if phase is None:
        return base_key
    return f"{base_key}|milestone={milestone}|phase={phase}"


def install_gruss_trade_diagnostics(
    executor: Executor,
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
    momentum_values: dict[int, GrussMomentumValue] | None = None,
    momentum_course_status: GrussMomentumCourseStatus | None = None,
) -> None:
    """Attach Gruss-only diagnostics to dry-run trade rows."""
    base_header = list(getattr(executor, "_gruss_trade_base_header", executor.TRADE_HEADER))
    executor._gruss_trade_base_header = base_header
    executor.TRADE_HEADER = list(dict.fromkeys(base_header + GRUSS_TRADE_DIAGNOSTIC_FIELDS))
    executor._gruss_trade_diag_context = {
        "win_snapshot": win_snapshot,
        "place_snapshot": place_snapshot,
        "tradable": _is_tradable(win_snapshot, place_snapshot, validation_ok=True),
        "momentum_values": momentum_values or {},
        "momentum_course_status": momentum_course_status,
    }

    if getattr(executor, "_gruss_trade_diag_installed", False):
        return

    original_log_trade_row = executor._log_trade_row

    def _log_trade_row_with_gruss_diagnostics(row: dict) -> None:
        enriched = dict(row)
        enriched.update(build_gruss_trade_diagnostics(executor, row))
        original_log_trade_row(enriched)

    executor._log_trade_row = _log_trade_row_with_gruss_diagnostics
    executor._gruss_trade_diag_installed = True


def build_gruss_trade_diagnostics(executor: Executor, row: dict) -> dict[str, Any]:
    context = getattr(executor, "_gruss_trade_diag_context", {}) or {}
    win_snapshot = context.get("win_snapshot")
    place_snapshot = context.get("place_snapshot")
    if not isinstance(win_snapshot, GrussSnapshot) or not isinstance(place_snapshot, GrussSnapshot):
        return {field: None for field in GRUSS_TRADE_DIAGNOSTIC_FIELDS}

    win_meta = win_snapshot.metadata
    place_meta = place_snapshot.metadata
    selection_id = _as_int_or_none(row.get("selection_id"))
    win_runner = _runner_by_trap(win_snapshot, selection_id)
    place_runner = _runner_by_trap(place_snapshot, selection_id)
    win_market_id = win_meta.market_id
    place_market_id = place_meta.market_id
    momentum_value = _momentum_value_for_selection(context, selection_id)
    momentum_course_status = context.get("momentum_course_status")
    place_winners = place_meta.winners
    k_place_used = None
    fallback_k_place_used = None
    if place_market_id:
        k_place_used = getattr(executor, "_k_place_used_by_market", {}).get(str(place_market_id))
        fallback_k_place_used = getattr(executor, "_fallback_k_place_used_by_market", {}).get(str(place_market_id))

    place_theorique = None
    if selection_id is not None and win_market_id:
        place_theorique = getattr(executor, "_last_place_theo_by_market", {}).get(str(win_market_id), {}).get(selection_id)

    ev_place = None
    if selection_id is not None and place_market_id:
        ev_place = getattr(executor, "_last_ev_place_by_market", {}).get(str(place_market_id), {}).get(selection_id)

    mom45 = None
    if selection_id is not None and win_market_id:
        mom45 = getattr(executor, "_last_mom45_by_market", {}).get(str(win_market_id), {}).get(selection_id)
    if mom45 is None and momentum_value is not None:
        mom45 = momentum_value.mom45

    return {
        "data_provider": DATA_PROVIDER_GRUSS_EXCEL,
        "order_provider": _diagnostic_order_provider_for_trade_row(row),
        "win_market_id": win_market_id,
        "place_market_id": place_market_id,
        "parent_id": win_meta.parent_id or place_meta.parent_id,
        "countdown_seconds": win_meta.countdown_seconds
        if win_meta.countdown_seconds is not None
        else place_meta.countdown_seconds,
        "countdown_display": win_meta.countdown_display or place_meta.countdown_display,
        "tradable": "1" if context.get("tradable") else "0",
        "has_mom45": getattr(momentum_value, "has_mom45", getattr(momentum_course_status, "has_mom45", None)),
        "mom45_reason": getattr(momentum_value, "reason", None)
        or getattr(momentum_course_status, "mom45_reason", None),
        "first_seen_countdown": getattr(momentum_value, "first_seen_countdown", None)
        if momentum_value is not None
        else getattr(momentum_course_status, "first_seen_countdown", None),
        "t45_anchor_found": getattr(momentum_value, "t45_anchor_found", None)
        if momentum_value is not None
        else getattr(momentum_course_status, "t45_anchor_found", None),
        "mom45": mom45,
        "mom45_win_best_back": getattr(momentum_value, "win_best_back_anchor", None),
        "mom45_win_ltp": getattr(momentum_value, "win_ltp_anchor", None),
        "mom45_place_best_back": getattr(momentum_value, "place_best_back_anchor", None),
        "mom45_place_ltp": getattr(momentum_value, "place_ltp_anchor", None),
        "win_best_back": getattr(win_runner, "best_back", None),
        "win_best_lay": getattr(win_runner, "best_lay", None),
        "place_best_back": getattr(place_runner, "best_back", None),
        "place_best_lay": getattr(place_runner, "best_lay", None),
        "place_winners": place_winners,
        "k_place_used": k_place_used,
        "fallback_k_place_used": fallback_k_place_used,
        "place_theorique": place_theorique,
        "ev_place": ev_place,
        "gruss_event_path": win_meta.event_path or place_meta.event_path,
        "gruss_win_market_title": win_meta.market_title,
        "gruss_place_market_title": place_meta.market_title,
    }


def _diagnostic_order_provider_for_trade_row(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "").strip().upper()
    configured = str(os.getenv("DOGBOT_ORDER_PROVIDER") or "").strip().lower()
    if (
        status in {"DRYRUN", "REAL_READY", "PRE_LADDER_REAL_READY"}
        and configured == ORDER_PROVIDER_GRUSS_EXCEL_REAL
        and _env_true(os.getenv("DOGBOT_GRUSS_ENABLE_REAL_ORDERS"), default=False)
        and not _env_true(os.getenv("DOGBOT_GRUSS_REAL_PREVIEW"), default=False)
        and not _env_true(os.getenv("DOGBOT_GRUSS_WRITE_NO_TRIGGER"), default=False)
    ):
        return ORDER_PROVIDER_GRUSS_EXCEL_REAL
    return ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN


def install_gruss_strategy_debug_logger(executor: Executor) -> None:
    """Print detailed PLACE strategy evaluation without changing strategy logic."""
    os.environ["STRATEGY_DEBUG"] = "1"
    if getattr(executor, "_gruss_strategy_debug_installed", False):
        return

    original_log_strategy_debug_row = executor._log_strategy_debug_row

    def _log_strategy_debug_row(slot, ctx, runner_name, condition_result, fail_reason) -> None:
        original_log_strategy_debug_row(slot, ctx, runner_name, condition_result, fail_reason)
        if _slot_market_type(slot) != "PLACE":
            return
        context = getattr(executor, "_gruss_trade_diag_context", {}) or {}
        win_snapshot = context.get("win_snapshot")
        place_snapshot = context.get("place_snapshot")
        gruss_event_path = ""
        meeting_name = ""
        normalized_region = ""
        if isinstance(win_snapshot, GrussSnapshot) and isinstance(place_snapshot, GrussSnapshot):
            gruss_event_path = win_snapshot.metadata.event_path or place_snapshot.metadata.event_path or ""
            meeting_name = gruss_meeting_name_for_snapshots(win_snapshot, place_snapshot)
            normalized_region = gruss_region_for_snapshots(win_snapshot, place_snapshot)
        momentum_value = _momentum_value_for_selection(context, getattr(ctx, "selection_id", None))
        momentum_course_status = context.get("momentum_course_status")
        has_mom45 = (
            bool(momentum_value.has_mom45)
            if momentum_value is not None
            else bool(getattr(momentum_course_status, "has_mom45", False))
        )
        mom45_source_timestamp = ""
        if momentum_value is not None and momentum_value.source_timestamp is not None:
            mom45_source_timestamp = momentum_value.source_timestamp.isoformat().replace("+00:00", "Z")
        eligible = bool(condition_result)
        reason = fail_reason or ("eligible" if eligible else "condition_false")
        price_req = None
        stake = None
        if eligible:
            try:
                result = try_fire_slot(executor.staking_engine, slot, ctx)
                if result is not None:
                    price_req = result.price
                    stake = result.size
                else:
                    reason = "condition_true_but_no_fire_result"
            except Exception as exc:
                reason = f"debug_fire_error={exc!r}"
        print(
            "[GRUSS_STRATEGY_DEBUG] "
            f"strategy_id={_slot_id(slot)} "
            f"gruss_event_path={gruss_event_path!r} "
            f"meeting_name={meeting_name!r} normalized_region={normalized_region} "
            f"runner={runner_name or ''} trap={getattr(ctx, 'trap', None)} "
            f"side={_slot_side(slot)} market_type={getattr(ctx, 'market_type', None)} "
            f"requires_mom45={getattr(slot, 'requires_mom45', False)} "
            f"has_mom45={has_mom45} "
            f"mom45_source_timestamp={mom45_source_timestamp} "
            f"mom45_source_countdown={getattr(momentum_value, 'source_countdown_seconds', '') if momentum_value is not None else ''} "
            f"mom45_current_value={getattr(momentum_value, 'current_value', '') if momentum_value is not None else ''} "
            f"mom45_anchor_value={getattr(momentum_value, 'anchor_value', '') if momentum_value is not None else ''} "
            f"mom45={getattr(ctx, 'mom45', None)} "
            f"mom45_reason={getattr(momentum_value, 'reason', None) or getattr(momentum_course_status, 'mom45_reason', '')} "
            f"first_seen_countdown={getattr(momentum_course_status, 'first_seen_countdown', '')} "
            f"t45_anchor_found={getattr(momentum_course_status, 't45_anchor_found', '')} "
            f"place_winners={getattr(ctx, 'place_winners', '')} "
            f"k_place_used={getattr(ctx, 'k_place_used', '')} "
            f"fallback_k_place_used={getattr(ctx, 'fallback_k_place_used', '')} "
            f"place_theorique={getattr(ctx, 'place_theo', '')} "
            f"ev_place={getattr(ctx, 'ev_place', '')} "
            f"eligible={eligible} reason={reason} "
            f"price_req={price_req if price_req is not None else ''} "
            f"stake={stake if stake is not None else ''}"
        )

    executor._log_strategy_debug_row = _log_strategy_debug_row
    executor._gruss_strategy_debug_installed = True


def build_order_intents_from_trade_rows(
    trade_rows: list[dict[str, str]],
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
) -> list[OrderIntent]:
    intents: list[OrderIntent] = []
    for row in trade_rows:
        status = str(row.get("status", "")).upper()
        if status not in {"DRYRUN", "REAL_READY", "PRE_LADDER_REAL_READY"}:
            continue
        trap = _as_int_or_none(row.get("selection_id"))
        market_type = str(row.get("market_type") or "").upper()
        snapshot = win_snapshot if market_type == "WIN" else place_snapshot
        runner = _runner_by_trap(snapshot, trap)
        pre_ladder = status == "PRE_LADDER_REAL_READY"
        intents.append(
            make_order_intent(
                provider=DATA_PROVIDER_GRUSS_EXCEL,
                market_type=market_type,
                market_id=row.get("market_id") or "",
                parent_id=row.get("parent_id") or win_snapshot.metadata.parent_id or place_snapshot.metadata.parent_id,
                runner_name=getattr(runner, "runner_name", "") or "",
                trap=trap,
                side=row.get("side") or "",
                order_type=_order_type_for_trade_row(row),
                price=_as_float_or_none(
                    _first_non_blank(
                        row.get("current_ladder_price"),
                        row.get("final_lim_price"),
                        row.get("final_lim_price_tick"),
                        row.get("price_req"),
                    )
                    if pre_ladder
                    else row.get("price_req")
                ),
                stake=_as_float_or_none(
                    _first_non_blank(
                        row.get("current_step_stake"),
                        row.get("final_stake"),
                        row.get("size_req"),
                    )
                    if pre_ladder
                    else row.get("size_req")
                ),
                strategy_id=row.get("strategy") or "",
                course_id=row.get("course_id") or None,
                timestamp=row.get("ts") or None,
                dry_run=True,
                selection_id=row.get("selection_id") or trap,
                execution_phase=row.get("execution_phase") or "POST",
                triggered_systems=row.get("triggered_systems") or row.get("strategy") or "",
                triggered_prices=row.get("triggered_prices") or row.get("price_req") or "",
                pre_ladder=pre_ladder,
                ladder_id=row.get("ladder_id") or None,
                ladder_step=row.get("ladder_step") or None,
                ladder_tracking_key=row.get("ladder_tracking_key") or None,
                gruss_planned_trigger=row.get("gruss_planned_trigger") or None,
                matched_stake=_as_float_or_none(row.get("matched_stake")),
                signal_countdown_seconds=_as_int_or_none(row.get("milestone")),
                market_reference_price_at_signal=_as_float_or_none(
                    row.get("market_reference_price_at_signal")
                ),
                best_same_side_back_offer=_as_float_or_none(row.get("best_same_side_back_offer")),
                best_same_side_lay_offer=_as_float_or_none(row.get("best_same_side_lay_offer")),
                ladder_plan_frozen=_env_true(row.get("ladder_plan_frozen"), default=False),
                ladder_plan_created_step=row.get("ladder_plan_created_step") or None,
                ladder_prices_frozen=row.get("ladder_prices_frozen") or None,
                current_ladder_price_from_frozen_plan=_env_true(
                    row.get("current_ladder_price_from_frozen_plan"),
                    default=False,
                ),
                computed_limit_price_raw=_as_float_or_none(row.get("computed_limit_price_raw")),
                computed_limit_price_effective=_as_float_or_none(row.get("computed_limit_price_effective")),
                min_price_floor_applied=_env_true(row.get("min_price_floor_applied"), default=False),
                pre_value_target_price=_as_float_or_none(row.get("pre_value_target_price")),
                ladder_planned_price=_as_float_or_none(row.get("ladder_planned_price")),
                sent_price_before_value_clamp=_as_float_or_none(row.get("sent_price_before_value_clamp")),
                sent_price_after_value_clamp=_as_float_or_none(row.get("sent_price_after_value_clamp")),
                value_clamp_applied=_env_true(row.get("value_clamp_applied"), default=False),
                value_limit_breached=_env_true(row.get("value_limit_breached"), default=False),
                value_limit_skip_reason=row.get("value_limit_skip_reason") or None,
                tick_rounding_direction=row.get("tick_rounding_direction") or None,
                best_same_side_offer_at_creation=_as_float_or_none(
                    row.get("best_same_side_offer_at_creation")
                ),
                best_back_displayed=_as_float_or_none(row.get("best_back_displayed")),
                best_lay_displayed=_as_float_or_none(row.get("best_lay_displayed")),
                start_price_source=row.get("start_price_source") or None,
                ladder_direction=row.get("ladder_direction") or None,
                ladder_disabled_lim_not_in_ladder_direction=_env_true(
                    row.get("ladder_disabled_lim_not_in_ladder_direction"),
                    default=False,
                ),
                direct_lim_order_planned=_env_true(
                    row.get("direct_lim_order_planned")
                    or row.get("direct_lim_order_written"),
                    default=False,
                ),
                direct_lim_order_written=False,
                no_replace_steps_for_direct_lim=_env_true(
                    row.get("no_replace_steps_for_direct_lim"),
                    default=False,
                ),
                strategy_edge=_as_float_or_none(
                    row.get("strategy_edge")
                    or row.get("signal_edge")
                    or row.get("edge")
                    or row.get("ev_place")
                    or row.get("EV_PLACE")
                ),
                strategy_score=_as_float_or_none(
                    row.get("strategy_score")
                    or row.get("signal_score")
                    or row.get("score")
                ),
                conflict_detected=_env_true(row.get("conflict_detected"), default=False),
                conflict_type=row.get("conflict_type") or None,
                back_price=_as_float_or_none(row.get("back_price")),
                lay_price=_as_float_or_none(row.get("lay_price")),
                market_reference_price=_as_float_or_none(row.get("market_reference_price")),
                back_distance=_as_float_or_none(row.get("back_distance")),
                lay_distance=_as_float_or_none(row.get("lay_distance")),
                selected_side=row.get("selected_side") or None,
                rejected_side=row.get("rejected_side") or None,
                conflict_resolution_reason=row.get("conflict_resolution_reason") or None,
                conflict_group_key=row.get("conflict_group_key") or None,
                conflict_candidates_count=_as_int_or_none(row.get("conflict_candidates_count")),
                back_systems=row.get("back_systems") or None,
                lay_systems=row.get("lay_systems") or None,
                pre_back_lay_conflict=_env_true(row.get("pre_back_lay_conflict"), default=False),
                pre_conflict_resolution=row.get("pre_conflict_resolution") or None,
                pre_conflict_chosen_side=row.get("pre_conflict_chosen_side") or None,
                pre_conflict_rejected_side=row.get("pre_conflict_rejected_side") or None,
                pre_conflict_reason=row.get("pre_conflict_reason") or None,
                pre_conflict_group_key=row.get("pre_conflict_group_key") or None,
                pre_conflict_course_id=row.get("pre_conflict_course_id") or None,
                pre_conflict_market_id=row.get("pre_conflict_market_id") or None,
                pre_conflict_market_type=row.get("pre_conflict_market_type") or None,
                pre_conflict_selection_id=row.get("pre_conflict_selection_id") or None,
                pre_conflict_runner_name=row.get("pre_conflict_runner_name") or None,
                pre_back_target_price=_as_float_or_none(row.get("pre_back_target_price")),
                pre_lay_target_price=_as_float_or_none(row.get("pre_lay_target_price")),
                pre_current_best_lay=_as_float_or_none(row.get("pre_current_best_lay")),
                pre_current_best_back=_as_float_or_none(row.get("pre_current_best_back")),
                pre_back_distance_ticks=_as_float_or_none(row.get("pre_back_distance_ticks")),
                pre_lay_distance_ticks=_as_float_or_none(row.get("pre_lay_distance_ticks")),
                staking_formula=row.get("staking_formula") or None,
                staking_alpha=_as_float_or_none(row.get("staking_alpha")),
                staking_back_alpha=_as_float_or_none(row.get("staking_back_alpha")),
                staking_lay_alpha=_as_float_or_none(row.get("staking_lay_alpha")),
                stake_raw_before_caps=_as_float_or_none(row.get("stake_raw_before_caps")),
                stake_after_caps=_as_float_or_none(row.get("stake_after_caps")),
                lay_liability_after_sizing=_as_float_or_none(row.get("lay_liability_after_sizing")),
                lay_liability_cap=_as_float_or_none(row.get("lay_liability_cap")),
                lay_liability_cap_hit=_env_true(row.get("lay_liability_cap_hit"), default=False),
            )
        )
    return intents


def _first_non_blank(*values: object) -> object:
    for value in values:
        if str(value or "").strip():
            return value
    return ""


def _order_type_for_trade_row(row: dict[str, str]) -> str:
    explicit_mode = str(row.get("exec_mode") or "").upper()
    if explicit_mode == "SP_MOC":
        return "SP_MOC"
    if explicit_mode in {"LIMIT", "LIMIT_LTP", "LIM"}:
        return "LIMIT"

    strategy_id = str(row.get("strategy") or "")
    for slot in build_registry():
        if _slot_id(slot) != strategy_id:
            continue
        return "SP_MOC" if _slot_mode(slot) == "SP_MOC" else "LIMIT"
    return "LIMIT"


def seed_gruss_momentum_into_executor(
    executor: Executor,
    bundle: GrussEngineBundle,
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
    momentum_buffer: GrussMomentumBuffer | None,
) -> dict[int, GrussMomentumValue]:
    if momentum_buffer is None:
        return {}
    momentum_values = momentum_buffer.momentum_by_trap(win_snapshot, place_snapshot)
    if not hasattr(executor, "_last_mom45_by_market"):
        executor._last_mom45_by_market = defaultdict(dict)
    available_count = 0
    missing_count = 0
    for trap, value in momentum_values.items():
        if value.anchor_value is not None:
            executor._base_win_ms[bundle.win_book.market_id][int(trap)][45] = float(value.anchor_value)
        if value.place_ltp_anchor is not None:
            executor._ltp_place_ms[bundle.place_book.market_id][int(trap)][45] = float(value.place_ltp_anchor)
        if value.has_mom45 and value.mom45 is not None:
            executor._last_mom45_by_market[str(bundle.win_book.market_id)][int(trap)] = float(value.mom45)
            available_count += 1
        else:
            missing_count += 1
    executor._gruss_mom45_diagnostics = {
        "mom45_available_count": available_count,
        "mom45_missing_count": missing_count,
        "mom45_source": "gruss_momentum_buffer",
        "mom45_injected_before_strategy_eval": True,
    }
    return momentum_values


def _seed_strategy_milestone_if_due(
    executor: Executor,
    bundle: GrussEngineBundle,
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
) -> None:
    seconds = win_snapshot.metadata.countdown_seconds
    if seconds is None:
        seconds = place_snapshot.metadata.countdown_seconds
    milestone = current_strategy_milestone(seconds, seconds)
    if seconds is None or milestone is None:
        return
    for market_id in (bundle.win_book.market_id, bundle.place_book.market_id):
        if milestone in executor._next_ms[market_id] and market_id not in executor._last_tto:
            executor._last_tto[market_id] = float(milestone + 1)


def _env_true(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().casefold() in {"1", "true", "yes", "on", "y"}


def _is_tradable(
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
    validation_ok: bool,
) -> bool:
    if not validation_ok:
        return False
    if _is_suspended(win_snapshot.metadata.suspend_status) or _is_suspended(place_snapshot.metadata.suspend_status):
        return False
    for win_runner in win_snapshot.runners:
        if win_runner.trap is None:
            continue
        place_runner = next((runner for runner in place_snapshot.runners if runner.trap == win_runner.trap), None)
        if place_runner is None:
            continue
        if _has_available_odds(win_runner) and _has_available_odds(place_runner):
            return True
    return False


def _missing_runner_prices(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> bool:
    for snapshot in (win_snapshot, place_snapshot):
        for runner in snapshot.runners:
            if runner.trap is None:
                continue
            if runner.ltp is None and not _has_available_odds(runner):
                return True
    return False


def _is_suspended(value: Any) -> bool:
    return str(value or "").strip().casefold() == "suspended"


def _has_available_odds(runner: Any) -> bool:
    return _positive(getattr(runner, "best_back", None)) or _positive(getattr(runner, "best_lay", None))


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _runner_by_trap(snapshot: GrussSnapshot, trap: int | None) -> Any:
    if trap is None:
        return None
    return next((runner for runner in snapshot.runners if runner.trap == trap), None)


def _momentum_value_for_selection(context: dict[str, Any], selection_id: Any) -> GrussMomentumValue | None:
    selection_id = _as_int_or_none(selection_id)
    if selection_id is None:
        return None
    values = context.get("momentum_values") or {}
    value = values.get(selection_id)
    return value if isinstance(value, GrussMomentumValue) else None


def _as_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slot_id(slot: Any) -> str:
    return str(getattr(slot, "tag", None) or f"{getattr(slot, 'family', '')}_{getattr(slot, 'slot', '')}")


def _slot_side(slot: Any) -> str:
    side = getattr(slot, "side", None)
    return str(getattr(side, "value", side) or "").upper()


def _slot_market_type(slot: Any) -> str:
    market_family = getattr(slot, "market_family", None)
    if market_family:
        return str(market_family).upper()
    family = str(getattr(slot, "family", "")).upper()
    if "PLACE" in family:
        return "PLACE"
    if "WIN" in family:
        return "WIN"
    return "UNKNOWN"


def _slot_mode(slot: Any) -> str:
    mode = getattr(slot, "exec_mode", None)
    return str(getattr(mode, "value", mode) or "")


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "-"


class _NoopLegacyStrategy:
    name = "gruss_dryrun_noop_legacy"

    def decide_all(self, market_book: Any, market_index_entry: Any, now_utc: datetime) -> list[Any]:
        return []


class _DryRunClient:
    """Placeholder client; dry_run=True prevents order methods from being called."""

    betting = None
