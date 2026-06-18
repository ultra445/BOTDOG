from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_gruss_bet_ref_place as bet_ref_diag
from dogbot.gruss.gruss_mapper import parse_countdown_seconds
from dogbot.gruss.gruss_real_orders import (
    is_terminal_bet_status,
    is_valid_bet_ref,
    normalise_gruss_bet_ref,
    strip_gruss_ref_suffix,
)


DEFAULT_WORKBOOK_PATH = bet_ref_diag.DEFAULT_WORKBOOK_PATH
GrussExcelBridge = bet_ref_diag.GrussExcelBridge

SHEET = "PLACE"
TRIGGER_COLUMN = "Q"
ODDS_COLUMN = "R"
STAKE_COLUMN = "S"
BET_REF_COLUMN = "T"
MATCHED_STAKE_COLUMN = "W"
ARM_ENV = "DOGBOT_GRUSS_PRE_LADDER_TIMED_DIAGNOSTIC_PLACE"


@dataclass(frozen=True)
class TimedPreLadderConfig:
    row: int = 5
    side: str = "BACK"
    stake: float = 2.0
    steps: tuple[int, ...] = (45, 32, 20, 14)
    prices: tuple[float, ...] = (50.0, 48.0, 46.0, 44.0)
    cancel_before_post_seconds: int = 1
    replace_min_countdown_seconds: int = 10
    interval_seconds: float = 1.0
    output_dir: Path = ROOT / "data"
    max_wait_seconds: int = 90


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        validate_environment()
        bridge = GrussExcelBridge(DEFAULT_WORKBOOK_PATH)
        bridge.connect_open_workbook()
        bet_ref_diag.ensure_open_visible_place_sheet(bridge)
        session_dir = run_diagnostic(bridge, config)
    except Exception as exc:
        print(f"ERREUR: {exc}")
        return 1
    print(f"diagnostic_output={session_dir}")
    print("Aucun POST reel n'est envoye par ce diagnostic.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic timed PRE ladder Gruss PLACE BACKR/LAYR puis CANCEL.")
    parser.add_argument("--runner-row", type=int, default=5)
    parser.add_argument("--side", default="BACK")
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument("--steps", default=os.getenv("DOGBOT_PRE_LADDER_STEPS", "45,32,20,14"))
    parser.add_argument("--prices", default=os.getenv("DOGBOT_GRUSS_TIMED_PRE_LADDER_PRICES", "50,48,46,44"))
    parser.add_argument("--cancel-before-post-seconds", type=int, default=1)
    parser.add_argument("--replace-min-countdown-seconds", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-wait-seconds", type=int, default=90)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    return parser


def config_from_args(args: argparse.Namespace) -> TimedPreLadderConfig:
    side = str(args.side or "").strip().upper()
    if side not in {"BACK", "LAY"}:
        raise ValueError("side doit etre BACK ou LAY")
    steps = _parse_int_list(args.steps)
    prices = _parse_float_list(args.prices)
    if len(steps) != 4 or len(prices) != 4:
        raise ValueError("steps et prices doivent contenir exactement 4 valeurs")
    if float(args.stake) != 2.0:
        raise ValueError("stake doit etre exactement 2 EUR")
    if args.runner_row < 5:
        raise ValueError("runner-row doit etre >= 5")
    return TimedPreLadderConfig(
        row=int(args.runner_row),
        side=side,
        stake=float(args.stake),
        steps=tuple(steps),
        prices=tuple(prices),
        cancel_before_post_seconds=int(args.cancel_before_post_seconds),
        replace_min_countdown_seconds=int(args.replace_min_countdown_seconds),
        interval_seconds=float(args.interval),
        output_dir=Path(args.output_dir),
        max_wait_seconds=int(args.max_wait_seconds),
    )


def validate_environment(env: dict[str, str] | None = None) -> None:
    values = env if env is not None else os.environ
    required_true = (
        ARM_ENV,
        "DOGBOT_GRUSS_ENABLE_REAL_ORDERS",
        "DOGBOT_GRUSS_REAL_TEST_MODE",
        "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED",
    )
    for name in required_true:
        if not _env_true(values.get(name)):
            raise RuntimeError(f"{name}=true est obligatoire")
    required_values = {
        "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
        "DOGBOT_GRUSS_REAL_MAX_STAKE": "2",
        "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE": "2",
    }
    for name, expected in required_values.items():
        if str(values.get(name, "")).strip() != expected:
            raise RuntimeError(f"{name} doit etre exactement {expected!r}")
    for name in ("DOGBOT_GRUSS_REAL_PREVIEW", "DOGBOT_GRUSS_WRITE_NO_TRIGGER"):
        if _env_true(values.get(name)):
            raise RuntimeError(f"{name}=true est incompatible avec ce diagnostic")


def run_diagnostic(
    bridge: Any,
    config: TimedPreLadderConfig,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = datetime.now,
) -> Path:
    session_dir = _create_session_dir(config.output_dir, now_fn=now_fn)
    summary_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    ticks_path = session_dir / "ticks.csv"
    with ticks_path.open("w", newline="", encoding="utf-8-sig") as tick_handle:
        tick_writer = csv.DictWriter(tick_handle, fieldnames=_tick_fieldnames())
        tick_writer.writeheader()
        for index, (seconds, price) in enumerate(zip(config.steps, config.prices), start=1):
            _wait_until_countdown(bridge, config, seconds, tick_writer, tick_rows, sleep_fn=sleep_fn, now_fn=now_fn)
            if index == 1:
                row = write_initial_step(bridge, config, price)
            else:
                row = maybe_write_replace_step(bridge, config, index, price)
            summary_rows.append(row)
            tick = dump_tick(bridge, config, now_fn=now_fn, event=row["phase"])
            tick_rows.append(tick)
            tick_writer.writerow(tick)
        _wait_until_countdown(
            bridge,
            config,
            config.cancel_before_post_seconds,
            tick_writer,
            tick_rows,
            sleep_fn=sleep_fn,
            now_fn=now_fn,
        )
        cancel_row = maybe_write_cancel_before_post(bridge, config)
        summary_rows.append(cancel_row)
        tick = dump_tick(bridge, config, now_fn=now_fn, event=cancel_row["phase"])
        tick_rows.append(tick)
        tick_writer.writerow(tick)

    _write_summary(session_dir, summary_rows)
    final = {
        "initial_written": any(row["phase"] == "step_1" and row["result"] == "written" for row in summary_rows),
        "replace_written_count": sum(1 for row in summary_rows if row["phase"].startswith("step_") and row["trigger"] in {"BACKR", "LAYR"} and row["result"] == "written"),
        "cancel_written": any(row["phase"] == "cancel_before_post" and row["result"] == "written" for row in summary_rows),
        "last_bet_ref": _read_bet_ref(bridge, config.row),
        "last_matched_stake": _read_cell(bridge, f"{MATCHED_STAKE_COLUMN}{config.row}"),
    }
    (session_dir / "summary_final.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return session_dir


def write_initial_step(bridge: Any, config: TimedPreLadderConfig, price: float) -> dict[str, Any]:
    cells = (
        (f"{ODDS_COLUMN}{config.row}", price),
        (f"{STAKE_COLUMN}{config.row}", config.stake),
        (f"{TRIGGER_COLUMN}{config.row}", config.side),
    )
    written = bridge.write_cells(SHEET, cells, allow_write=True)
    return _summary_row("step_1", config.side, price, "pre_ladder_step_written", "written", cells_written=written)


def maybe_write_replace_step(
    bridge: Any,
    config: TimedPreLadderConfig,
    step_index: int,
    price: float,
) -> dict[str, Any]:
    trigger = "BACKR" if config.side == "BACK" else "LAYR"
    status_reason = _replace_block_reason(bridge, config)
    bet_ref = _read_bet_ref(bridge, config.row)
    matched_stake = _read_cell(bridge, f"{MATCHED_STAKE_COLUMN}{config.row}")
    if status_reason:
        return _summary_row(
            f"step_{step_index}",
            trigger,
            price,
            status_reason,
            "skipped",
            bet_ref=bet_ref,
            matched_stake=matched_stake,
        )
    cells: list[tuple[str, Any]] = []
    clean_ref = strip_gruss_ref_suffix(bet_ref)
    if clean_ref != bet_ref:
        cells.append((f"{BET_REF_COLUMN}{config.row}", clean_ref))
    cells.extend(
        [
            (f"{ODDS_COLUMN}{config.row}", price),
            (f"{STAKE_COLUMN}{config.row}", config.stake),
            (f"{TRIGGER_COLUMN}{config.row}", trigger),
        ]
    )
    written = bridge.write_cells(SHEET, tuple(cells), allow_write=True)
    return _summary_row(
        f"step_{step_index}",
        trigger,
        price,
        "pre_ladder_replace_written",
        "written",
        bet_ref=bet_ref,
        matched_stake=matched_stake,
        cells_written=written,
    )


def maybe_write_cancel_before_post(bridge: Any, config: TimedPreLadderConfig) -> dict[str, Any]:
    reason = _cancel_block_reason(bridge, config)
    bet_ref = _read_bet_ref(bridge, config.row)
    matched_stake = _read_cell(bridge, f"{MATCHED_STAKE_COLUMN}{config.row}")
    if reason:
        return _summary_row("cancel_before_post", "CANCEL", "", reason, "skipped", bet_ref=bet_ref, matched_stake=matched_stake)
    cells: list[tuple[str, Any]] = []
    clean_ref = strip_gruss_ref_suffix(bet_ref)
    if clean_ref != bet_ref:
        cells.append((f"{BET_REF_COLUMN}{config.row}", clean_ref))
    cells.append((f"{TRIGGER_COLUMN}{config.row}", "CANCEL"))
    written = bridge.write_cells(SHEET, tuple(cells), allow_write=True)
    return _summary_row(
        "cancel_before_post",
        "CANCEL",
        "",
        "pre_cancel_before_post_written",
        "written",
        bet_ref=bet_ref,
        matched_stake=matched_stake,
        cells_written=written,
    )


def _replace_block_reason(bridge: Any, config: TimedPreLadderConfig) -> str:
    market_status = _read_cell(bridge, "F2")
    if _is_suspended_or_closed(market_status):
        return "market_suspended_no_replace"
    countdown = _countdown(bridge)
    if countdown is None:
        return "countdown_unavailable_no_replace"
    if countdown <= config.replace_min_countdown_seconds:
        return "countdown_too_low_no_replace"
    bet_ref = _read_bet_ref(bridge, config.row)
    if not bet_ref or bet_ref == "PENDING":
        return "pre_ladder_replace_skipped_bet_ref_not_ready"
    if is_terminal_bet_status(bet_ref):
        return "row_status_not_replaceable"
    if not is_valid_bet_ref(bet_ref):
        return "invalid_bet_ref_for_replace"
    matched_stake = _numeric_or_none(_read_cell(bridge, f"{MATCHED_STAKE_COLUMN}{config.row}"))
    if matched_stake is None:
        return "matched_stake_unavailable_no_replace"
    if matched_stake > 0:
        return "pre_ladder_replace_skipped_matched_stake_gt_zero"
    return ""


def _cancel_block_reason(bridge: Any, config: TimedPreLadderConfig) -> str:
    if _is_suspended_or_closed(_read_cell(bridge, "F2")):
        return "pre_cancel_skipped_market_suspended"
    bet_ref = _read_bet_ref(bridge, config.row)
    if not is_valid_bet_ref(bet_ref):
        return "pre_cancel_before_post_skipped_invalid_bet_ref"
    matched_stake = _numeric_or_none(_read_cell(bridge, f"{MATCHED_STAKE_COLUMN}{config.row}"))
    if matched_stake is None:
        return "pre_cancel_before_post_skipped_matched_stake_unavailable"
    if matched_stake > 0:
        return "pre_cancel_skipped_matched_stake_gt_zero"
    return ""


def dump_tick(
    bridge: Any,
    config: TimedPreLadderConfig,
    *,
    now_fn: Callable[[], datetime],
    event: str,
) -> dict[str, Any]:
    q_af = {}
    columns = ["Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF"]
    for column in columns:
        address = f"{column}{config.row}"
        q_af[address] = _read_cell(bridge, address)
    return {
        "timestamp": now_fn().isoformat(),
        "event": event,
        "countdown_seconds": _countdown(bridge),
        "market_status": _read_cell(bridge, "F2"),
        "bet_ref": _read_bet_ref(bridge, config.row),
        "matched_stake": _read_cell(bridge, f"{MATCHED_STAKE_COLUMN}{config.row}"),
        "place_q_af": json.dumps(q_af, default=str, sort_keys=True),
    }


def _wait_until_countdown(
    bridge: Any,
    config: TimedPreLadderConfig,
    target: int,
    writer: csv.DictWriter,
    tick_rows: list[dict[str, Any]],
    *,
    sleep_fn: Callable[[float], None],
    now_fn: Callable[[], datetime],
) -> None:
    started = time.monotonic()
    while True:
        countdown = _countdown(bridge)
        tick = dump_tick(bridge, config, now_fn=now_fn, event=f"wait_for_{target}")
        tick_rows.append(tick)
        writer.writerow(tick)
        if countdown is not None and countdown <= target:
            return
        if time.monotonic() - started > config.max_wait_seconds:
            raise TimeoutError(f"countdown {target} non atteint")
        sleep_fn(config.interval_seconds)


def _summary_row(
    phase: str,
    trigger: str,
    price: Any,
    reason: str,
    result: str,
    *,
    bet_ref: Any = "",
    matched_stake: Any = "",
    cells_written: Any = "",
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "trigger": trigger,
        "price": price,
        "stake": 2.0,
        "bet_ref_before": bet_ref,
        "matched_stake": matched_stake,
        "reason": reason,
        "result": result,
        "cells_written": ";".join(cells_written) if isinstance(cells_written, list) else cells_written,
    }


def _write_summary(session_dir: Path, rows: list[dict[str, Any]]) -> None:
    path = session_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_summary_row("", "", "", "", "").keys()))
        writer.writeheader()
        writer.writerows(rows)


def _tick_fieldnames() -> list[str]:
    return ["timestamp", "event", "countdown_seconds", "market_status", "bet_ref", "matched_stake", "place_q_af"]


def _read_cell(bridge: Any, address: str) -> Any:
    try:
        return bridge.read_cell(SHEET, address)
    except Exception as exc:
        return f"read_failed:{exc}"


def _read_bet_ref(bridge: Any, row: int) -> str:
    return normalise_gruss_bet_ref(_read_cell(bridge, f"{BET_REF_COLUMN}{row}"))


def _countdown(bridge: Any) -> int | None:
    try:
        return parse_countdown_seconds(bridge.read_cell(SHEET, "D2"))
    except Exception:
        return None


def _is_suspended_or_closed(value: Any) -> bool:
    text = str(value or "").strip().casefold()
    return "suspended" in text or "closed" in text


def _numeric_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _env_true(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _create_session_dir(output_dir: Path, *, now_fn: Callable[[], datetime]) -> Path:
    session_dir = Path(output_dir) / f"gruss_pre_ladder_timed_place_{now_fn().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


if __name__ == "__main__":
    raise SystemExit(main())

