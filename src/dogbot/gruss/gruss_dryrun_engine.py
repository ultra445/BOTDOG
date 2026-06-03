from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dogbot.config import (
    DATA_PROVIDER_GRUSS_EXCEL,
    ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN,
    ProviderConfig,
    load_provider_config,
)
from dogbot.executor import Executor
from dogbot.feed_selector import create_data_feed_from_config
from dogbot.gruss.gruss_engine_adapter import GrussEngineBundle, build_engine_bundle
from dogbot.gruss.gruss_feed import GrussFeed
from dogbot.gruss.gruss_mapper import GrussMapper, GrussSnapshot


GRUSS_TRADE_DIAGNOSTIC_FIELDS = [
    "data_provider",
    "order_provider",
    "win_market_id",
    "place_market_id",
    "parent_id",
    "countdown_seconds",
    "countdown_display",
    "tradable",
    "win_best_back",
    "win_best_lay",
    "place_best_back",
    "place_best_lay",
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

    def evaluate(self, win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> None:
        bundle = build_engine_bundle(win_snapshot, place_snapshot)
        executor = self.ensure_executor(bundle)
        install_gruss_trade_diagnostics(executor, win_snapshot, place_snapshot)
        _seed_t2_milestone_if_due(executor, bundle, win_snapshot, place_snapshot)
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
    if validation_warnings:
        return "win_place_mismatch: " + "; ".join(validation_warnings)
    if not tradable:
        return "not_tradable"
    countdown_seconds = win_snapshot.metadata.countdown_seconds
    if countdown_seconds is None:
        countdown_seconds = place_snapshot.metadata.countdown_seconds
    if countdown_seconds is None:
        return "countdown_seconds_unavailable"
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
    return (
        f"course={win_meta.event_path or place_meta.event_path} "
        f"countdown={countdown} seconds={seconds} "
        f"win_market_id={win_meta.market_id} place_market_id={place_meta.market_id} "
        f"validation={validation} tradable={tradable} "
        f"runners={len(state.win_snapshot.runners)}/{len(state.place_snapshot.runners)}"
    )


def install_gruss_trade_diagnostics(
    executor: Executor,
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
) -> None:
    """Attach Gruss-only diagnostics to dry-run trade rows."""
    base_header = list(getattr(executor, "_gruss_trade_base_header", executor.TRADE_HEADER))
    executor._gruss_trade_base_header = base_header
    executor.TRADE_HEADER = list(dict.fromkeys(base_header + GRUSS_TRADE_DIAGNOSTIC_FIELDS))
    executor._gruss_trade_diag_context = {
        "win_snapshot": win_snapshot,
        "place_snapshot": place_snapshot,
        "tradable": _is_tradable(win_snapshot, place_snapshot, validation_ok=True),
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

    place_theorique = None
    if selection_id is not None and win_market_id:
        place_theorique = getattr(executor, "_last_place_theo_by_market", {}).get(str(win_market_id), {}).get(selection_id)

    ev_place = None
    if selection_id is not None and place_market_id:
        ev_place = getattr(executor, "_last_ev_place_by_market", {}).get(str(place_market_id), {}).get(selection_id)

    return {
        "data_provider": DATA_PROVIDER_GRUSS_EXCEL,
        "order_provider": ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN,
        "win_market_id": win_market_id,
        "place_market_id": place_market_id,
        "parent_id": win_meta.parent_id or place_meta.parent_id,
        "countdown_seconds": win_meta.countdown_seconds
        if win_meta.countdown_seconds is not None
        else place_meta.countdown_seconds,
        "countdown_display": win_meta.countdown_display or place_meta.countdown_display,
        "tradable": "1" if context.get("tradable") else "0",
        "win_best_back": getattr(win_runner, "best_back", None),
        "win_best_lay": getattr(win_runner, "best_lay", None),
        "place_best_back": getattr(place_runner, "best_back", None),
        "place_best_lay": getattr(place_runner, "best_lay", None),
        "place_theorique": place_theorique,
        "ev_place": ev_place,
        "gruss_event_path": win_meta.event_path or place_meta.event_path,
        "gruss_win_market_title": win_meta.market_title,
        "gruss_place_market_title": place_meta.market_title,
    }


def _seed_t2_milestone_if_due(
    executor: Executor,
    bundle: GrussEngineBundle,
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
) -> None:
    seconds = win_snapshot.metadata.countdown_seconds
    if seconds is None:
        seconds = place_snapshot.metadata.countdown_seconds
    if seconds is None or seconds > 2:
        return
    for market_id in (bundle.win_book.market_id, bundle.place_book.market_id):
        if 2 in executor._next_ms[market_id] and market_id not in executor._last_tto:
            executor._last_tto[market_id] = 3.0


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


def _as_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _NoopLegacyStrategy:
    name = "gruss_dryrun_noop_legacy"

    def decide_all(self, market_book: Any, market_index_entry: Any, now_utc: datetime) -> list[Any]:
        return []


class _DryRunClient:
    """Placeholder client; dry_run=True prevents order methods from being called."""

    betting = None
