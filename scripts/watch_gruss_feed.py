from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH
from dogbot.gruss.gruss_feed import GrussFeed
from dogbot.gruss.gruss_mapper import GrussMapper, GrussRunner, GrussSnapshot


ROWS = 80
COLUMNS = 50
DEFAULT_OUTPUT_PATH = ROOT / "data" / "gruss_live_snapshots.csv"
CSV_FIELDS = [
    "timestamp",
    "parent_id",
    "win_market_id",
    "place_market_id",
    "event_path",
    "win_market_title",
    "place_market_title",
    "countdown",
    "market_status",
    "suspend_status",
    "trap",
    "runner_name",
    "win_best_back",
    "win_best_lay",
    "win_ltp",
    "win_total_matched",
    "place_best_back",
    "place_best_lay",
    "place_ltp",
    "place_total_matched",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.interval <= 0:
        print("ERREUR: --interval doit etre superieur a 0")
        return 1
    if args.max_ticks is not None and args.max_ticks <= 0:
        print("ERREUR: --max-ticks doit etre superieur a 0")
        return 1
    if not DEFAULT_WORKBOOK_PATH.exists():
        print(f"ERREUR: workbook absent: {DEFAULT_WORKBOOK_PATH}")
        return 1

    feed = GrussFeed(DEFAULT_WORKBOOK_PATH)
    print("Gruss feed watcher dry-run")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print(f"CSV diagnostic: {args.output}")
    print("Aucun ordre ne sera envoye.")

    tick = 0
    while args.max_ticks is None or tick < args.max_ticks:
        tick += 1
        timestamp = datetime.now().isoformat(timespec="seconds")
        status_code = _run_tick(feed, Path(args.output), timestamp)
        if status_code != 0 and args.max_ticks == 1:
            return status_code
        if args.max_ticks is not None and tick >= args.max_ticks:
            break
        time.sleep(args.interval)

    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch the live Gruss Excel feed in dry-run mode.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between Excel reads.")
    parser.add_argument("--max-ticks", type=int, default=None, help="Stop after N reads.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="CSV snapshot output path.")
    return parser.parse_args(argv)


def _run_tick(feed: GrussFeed, output_path: Path, timestamp: str) -> int:
    try:
        feed.connect_open_workbook()
        _require_sheet(feed, "WIN")
        _require_sheet(feed, "PLACE")
        win_snapshot = feed.read_snapshot("WIN", rows=ROWS, columns=COLUMNS)
        place_snapshot = feed.read_snapshot("PLACE", rows=ROWS, columns=COLUMNS)
    except Exception as exc:
        print(f"{timestamp} |  |  |  |  | ERROR | runners=0/0 | validation KO | tradable KO | {exc}")
        return 1

    warnings = GrussMapper.get_win_place_validation_warnings(win_snapshot, place_snapshot)
    validation_label = "OK" if not warnings else "KO"
    tradable_label = "OK" if is_tradable(win_snapshot, place_snapshot, validation_ok=not warnings) else "KO"
    _print_compact_line(timestamp, win_snapshot, place_snapshot, validation_label, tradable_label)

    if warnings:
        for warning in warnings:
            print(f"  validation reason: {warning}")
        return 1

    try:
        rows = build_csv_rows(timestamp, win_snapshot, place_snapshot)
        append_snapshot_rows(output_path, rows)
    except Exception as exc:
        print(f"  csv write error: {exc}")
        return 1

    return 0


def build_csv_rows(
    timestamp: str,
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
) -> list[dict[str, Any]]:
    win_by_trap = _runners_by_trap(win_snapshot)
    place_by_trap = _runners_by_trap(place_snapshot)
    traps = sorted(set(win_by_trap) | set(place_by_trap))
    win_meta = win_snapshot.metadata
    place_meta = place_snapshot.metadata

    rows = []
    for trap in traps:
        win_runner = win_by_trap.get(trap)
        place_runner = place_by_trap.get(trap)
        rows.append(
            {
                "timestamp": timestamp,
                "parent_id": win_meta.parent_id or place_meta.parent_id,
                "win_market_id": win_meta.market_id,
                "place_market_id": place_meta.market_id,
                "event_path": win_meta.event_path or place_meta.event_path,
                "win_market_title": win_meta.market_title,
                "place_market_title": place_meta.market_title,
                "countdown": win_meta.countdown_display or place_meta.countdown_display,
                "market_status": win_meta.market_status or place_meta.market_status,
                "suspend_status": win_meta.suspend_status or place_meta.suspend_status,
                "trap": trap,
                "runner_name": _runner_name(win_runner, place_runner),
                "win_best_back": _runner_value(win_runner, "best_back"),
                "win_best_lay": _runner_value(win_runner, "best_lay"),
                "win_ltp": _runner_value(win_runner, "ltp"),
                "win_total_matched": _runner_value(win_runner, "total_amount_matched"),
                "place_best_back": _runner_value(place_runner, "best_back"),
                "place_best_lay": _runner_value(place_runner, "best_lay"),
                "place_ltp": _runner_value(place_runner, "ltp"),
                "place_total_matched": _runner_value(place_runner, "total_amount_matched"),
            }
        )
    return rows


def append_snapshot_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def is_tradable(
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
    validation_ok: bool,
) -> bool:
    if not validation_ok:
        return False
    if _is_suspended(win_snapshot.metadata.suspend_status) or _is_suspended(place_snapshot.metadata.suspend_status):
        return False

    win_by_trap = _runners_by_trap(win_snapshot)
    place_by_trap = _runners_by_trap(place_snapshot)
    for trap in set(win_by_trap) & set(place_by_trap):
        win_runner = win_by_trap[trap]
        place_runner = place_by_trap[trap]
        if _has_available_odds(win_runner) and _has_available_odds(place_runner):
            return True
    return False


def _print_compact_line(
    timestamp: str,
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
    validation_label: str,
    tradable_label: str,
) -> None:
    win_meta = win_snapshot.metadata
    place_meta = place_snapshot.metadata
    countdown_display = win_meta.countdown_display or place_meta.countdown_display or ""
    countdown_seconds = win_meta.countdown_seconds
    if countdown_seconds is None:
        countdown_seconds = place_meta.countdown_seconds
    countdown_seconds_text = "" if countdown_seconds is None else str(countdown_seconds)
    market_status = win_meta.market_status or place_meta.market_status or ""
    suspend_status = win_meta.suspend_status or place_meta.suspend_status or ""
    status = f"{market_status}/{suspend_status}".strip("/")
    runners = f"{len(win_snapshot.runners)}/{len(place_snapshot.runners)}"
    print(
        f"{timestamp} | {countdown_display} | {countdown_seconds_text} | {win_meta.market_id or ''} | "
        f"{place_meta.market_id or ''} | {status} | runners={runners} | "
        f"validation {validation_label} | tradable {tradable_label}"
    )


def _require_sheet(feed: GrussFeed, sheet_name: str) -> None:
    if not feed.bridge.has_sheet(sheet_name):
        raise RuntimeError(f"onglet {sheet_name} manquant")


def _runners_by_trap(snapshot: GrussSnapshot) -> dict[int, GrussRunner]:
    return {runner.trap: runner for runner in snapshot.runners if runner.trap is not None}


def _runner_name(win_runner: GrussRunner | None, place_runner: GrussRunner | None) -> str:
    runner = win_runner or place_runner
    return runner.runner_name if runner else ""


def _runner_value(runner: GrussRunner | None, field_name: str) -> Any:
    if runner is None:
        return None
    return getattr(runner, field_name)


def _is_suspended(value: Any) -> bool:
    return str(value or "").strip().casefold() == "suspended"


def _has_available_odds(runner: GrussRunner) -> bool:
    return _positive(runner.best_back) or _positive(runner.best_lay)


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
