from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay PRE pipeline diagnostics from CSV files.")
    parser.add_argument("--strategy-debug", type=Path, required=True)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=False)
    parser.add_argument("--course-id", default="")
    parser.add_argument("--market-type", default="PLACE")
    args = parser.parse_args()

    strategy_rows = _read_csv(args.strategy_debug)
    trade_rows = _read_csv(args.trades)
    attempt_rows = _read_csv(args.attempts) if args.attempts else []

    market_type = args.market_type.upper()
    signals: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"BACK": [], "LAY": []})
    for row in strategy_rows:
        if str(row.get("execution_phase", "")).upper() != "PRE":
            continue
        if str(row.get("market_type", "")).upper() != market_type:
            continue
        if str(row.get("condition_result", "")).lower() not in {"true", "1"}:
            continue
        side = _side_from_strategy(row.get("tag", ""))
        if side not in {"BACK", "LAY"}:
            continue
        key = _runner_key(row)
        signals[key][side].append(str(row.get("tag", "")))

    trades_by_runner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trade_rows:
        if str(row.get("execution_phase", "")).upper() != "PRE":
            continue
        if str(row.get("market_type", "")).upper() != market_type:
            continue
        if args.course_id and str(row.get("course_id") or row.get("parent_market_id") or "") != args.course_id:
            continue
        trades_by_runner[_runner_key(row)].append(row)

    attempts_by_runner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempt_rows:
        if str(row.get("execution_phase", "")).upper() != "PRE":
            continue
        if str(row.get("market_type", "")).upper() != market_type:
            continue
        attempts_by_runner[_runner_key(row)].append(row)

    print("runner,strategy_back_signal,strategy_lay_signal,back_systems,lay_systems,trade_statuses,attempt_statuses,diagnosis")
    for key in sorted(set(signals) | set(trades_by_runner) | set(attempts_by_runner)):
        back = signals[key]["BACK"]
        lay = signals[key]["LAY"]
        trades = trades_by_runner.get(key, [])
        attempts = attempts_by_runner.get(key, [])
        diagnosis = _diagnosis(back, lay, trades, attempts)
        print(
            ",".join(
                _csv_cell(value)
                for value in (
                    key,
                    bool(back),
                    bool(lay),
                    "|".join(back),
                    "|".join(lay),
                    "|".join(_status_reason(row) for row in trades),
                    "|".join(_status_reason(row) for row in attempts),
                    diagnosis,
                )
            )
        )
    return 0


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _runner_key(row: dict[str, Any]) -> str:
    selection = row.get("selection_id") or row.get("selection") or row.get("trap") or ""
    runner = row.get("runner_name") or row.get("runner") or ""
    return f"{selection}:{runner}"


def _side_from_strategy(strategy_id: Any) -> str:
    text = str(strategy_id or "").upper()
    if text.startswith("BACK_"):
        return "BACK"
    if text.startswith("LAY_"):
        return "LAY"
    return ""


def _status_reason(row: dict[str, Any]) -> str:
    return f"{row.get('status', '')}:{row.get('reason', '')}"


def _diagnosis(
    back: list[str],
    lay: list[str],
    trades: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> str:
    if (back or lay) and not trades:
        return "signal_seen_but_no_trade_row"
    if trades and not attempts:
        return "trade_created_but_no_gruss_attempt"
    if attempts:
        return "gruss_attempt_logged"
    return "no_pre_signal"


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(char in text for char in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


if __name__ == "__main__":
    raise SystemExit(main())
