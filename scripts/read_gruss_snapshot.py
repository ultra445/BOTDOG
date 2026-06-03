from __future__ import annotations

import sys
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


def main() -> int:
    print("Gruss live snapshot diagnostic")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")

    if not DEFAULT_WORKBOOK_PATH.exists():
        print(f"ERREUR: workbook absent: {DEFAULT_WORKBOOK_PATH}")
        return 1

    feed = GrussFeed(DEFAULT_WORKBOOK_PATH)
    try:
        feed.bridge.connect_open_workbook()
    except Exception as exc:
        print(f"ERREUR: impossible de se connecter au workbook Excel ouvert: {exc}")
        return 1

    print("Workbook ouvert trouve: OUI")

    for sheet_name in ("WIN", "PLACE"):
        try:
            exists = feed.bridge.has_sheet(sheet_name)
        except Exception as exc:
            print(f"ERREUR: impossible de verifier l'onglet {sheet_name}: {exc}")
            return 1
        if not exists:
            print(f"ERREUR: onglet {sheet_name} manquant")
            return 1
        print(f"Onglet {sheet_name} trouve: OUI")

    try:
        win_snapshot = feed.read_snapshot("WIN", rows=ROWS, columns=COLUMNS)
        place_snapshot = feed.read_snapshot("PLACE", rows=ROWS, columns=COLUMNS)
    except Exception as exc:
        print(f"ERREUR: lecture/parsing Gruss impossible: {exc}")
        return 1

    _print_market_summary("WIN", win_snapshot)
    _print_market_summary("PLACE", place_snapshot)
    _print_validation(win_snapshot, place_snapshot)
    _print_runner_table(win_snapshot, place_snapshot)
    return 0


def _print_market_summary(label: str, snapshot: GrussSnapshot) -> None:
    meta = snapshot.metadata
    print()
    print(f"=== Marche {label} ===")
    print(f"market_title: {_fmt(meta.market_title)}")
    print(f"market_id: {_fmt(meta.market_id)}")
    print(f"parent_id: {_fmt(meta.parent_id)}")
    print(f"event_path: {_fmt(meta.event_path)}")
    print(f"countdown: {_fmt(meta.countdown)}")
    print(f"market_status: {_fmt(meta.market_status)}")
    print(f"suspend_status: {_fmt(meta.suspend_status)}")
    print(f"total_matched: {_fmt(meta.total_matched)}")
    print(f"nombre de runners: {len(snapshot.runners)}")


def _print_validation(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> None:
    warnings = GrussMapper.get_win_place_validation_warnings(win_snapshot, place_snapshot)
    print()
    print("=== Validation ===")
    if not warnings:
        print("WIN/PLACE: OK")
        return

    print("WIN/PLACE: KO")
    for warning in warnings:
        print(f"- {warning}")


def _print_runner_table(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> None:
    win_by_trap = _runners_by_trap(win_snapshot)
    place_by_trap = _runners_by_trap(place_snapshot)
    traps = sorted(set(win_by_trap) | set(place_by_trap))

    print()
    print("=== Runners appaires par trap ===")
    headers = (
        "trap",
        "runner_name",
        "WIN best_back",
        "WIN best_lay",
        "WIN ltp",
        "PLACE best_back",
        "PLACE best_lay",
        "PLACE ltp",
    )
    rows = []
    for trap in traps:
        win_runner = win_by_trap.get(trap)
        place_runner = place_by_trap.get(trap)
        rows.append(
            (
                trap,
                _runner_name(win_runner, place_runner),
                _runner_value(win_runner, "best_back"),
                _runner_value(win_runner, "best_lay"),
                _runner_value(win_runner, "ltp"),
                _runner_value(place_runner, "best_back"),
                _runner_value(place_runner, "best_lay"),
                _runner_value(place_runner, "ltp"),
            )
        )
    _print_table(headers, rows)


def _runners_by_trap(snapshot: GrussSnapshot) -> dict[int, GrussRunner]:
    return {runner.trap: runner for runner in snapshot.runners if runner.trap is not None}


def _runner_name(win_runner: GrussRunner | None, place_runner: GrussRunner | None) -> str:
    runner = win_runner or place_runner
    return runner.runner_name if runner else ""


def _runner_value(runner: GrussRunner | None, field_name: str) -> Any:
    if runner is None:
        return None
    return getattr(runner, field_name)


def _print_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    rendered_rows = [tuple(_fmt(value) for value in row) for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        widths = [max(widths[index], len(row[index])) for index in range(len(headers))]
    print(" | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rendered_rows:
        print(" | ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
