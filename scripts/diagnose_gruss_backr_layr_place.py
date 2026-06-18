from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_gruss_bet_ref_place as bet_ref_diag
from dogbot.gruss.gruss_real_orders import (
    is_terminal_bet_status,
    is_valid_bet_ref,
    normalise_gruss_bet_ref,
    strip_gruss_ref_suffix,
)
from dogbot.gruss.gruss_mapper import parse_countdown_seconds


DEFAULT_WORKBOOK_PATH = bet_ref_diag.DEFAULT_WORKBOOK_PATH
GrussExcelBridge = bet_ref_diag.GrussExcelBridge

SHEET = "PLACE"
TRIGGER_COLUMN = "Q"
ODDS_COLUMN = "R"
STAKE_COLUMN = "S"
BET_REF_COLUMN = "T"
MATCHED_STAKE_COLUMN = "W"
ALLOWED_SIDES = frozenset({"BACK", "LAY"})
DIAGNOSTIC_ARM_ENV = "DOGBOT_GRUSS_BACKR_LAYR_DIAGNOSTIC_PLACE"

REQUIRED_ENV_TRUE = (
    "DOGBOT_GRUSS_ENABLE_REAL_ORDERS",
    "DOGBOT_GRUSS_REAL_TEST_MODE",
    "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED",
)
REQUIRED_ENV_VALUES = {
    "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
    "DOGBOT_GRUSS_REAL_MAX_STAKE": "2",
    "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE": "2",
}
FORBIDDEN_ENV_TRUE = (
    "DOGBOT_GRUSS_REAL_PREVIEW",
    "DOGBOT_GRUSS_WRITE_NO_TRIGGER",
)


@dataclass(frozen=True)
class BackrLayrDiagnosticConfig:
    row: int
    side: str
    initial_price: float
    replace_price: float
    stake: float
    seconds: int
    interval_seconds: float
    after_replace_ticks: int
    selection_rows: int
    selection_columns: int
    output_dir: Path
    second_replace_price: float | None = None
    manual_note: str = ""
    no_manual_prompt: bool = False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    print("Verification visuelle demandee: un seul ordre ouvert, cote modifiee, pas d'empilement.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic isole Gruss BACKR/LAYR PLACE. Ecrit un seul BACK/LAY reel a 2 EUR, "
            "attend une vraie Bet Ref, verifie W=0, puis tente BACKR/LAYR sur la meme ligne."
        )
    )
    parser.add_argument("--runner-row", type=int, default=int(os.getenv("DOGBOT_GRUSS_BACKR_LAYR_DIAGNOSTIC_ROW", "5")))
    parser.add_argument("--side", default=os.getenv("DOGBOT_GRUSS_BACKR_LAYR_DIAGNOSTIC_SIDE", "BACK"))
    parser.add_argument(
        "--initial-price",
        type=float,
        default=_optional_float_env("DOGBOT_GRUSS_BACKR_LAYR_INITIAL_PRICE"),
    )
    parser.add_argument(
        "--replace-price",
        type=float,
        default=_optional_float_env("DOGBOT_GRUSS_BACKR_LAYR_REPLACE_PRICE"),
    )
    parser.add_argument(
        "--second-replace-price",
        type=float,
        default=_optional_float_env("DOGBOT_GRUSS_BACKR_LAYR_SECOND_REPLACE_PRICE"),
    )
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument("--seconds", type=int, default=int(os.getenv("DOGBOT_GRUSS_BACKR_LAYR_SECONDS", "20")))
    parser.add_argument("--interval", type=float, default=float(os.getenv("DOGBOT_GRUSS_BACKR_LAYR_INTERVAL_SECONDS", "1")))
    parser.add_argument("--after-replace-ticks", type=int, default=int(os.getenv("DOGBOT_GRUSS_BACKR_LAYR_AFTER_REPLACE_TICKS", "3")))
    parser.add_argument("--selection-rows", type=int, default=int(os.getenv("DOGBOT_GRUSS_BACKR_LAYR_SELECTION_ROWS", "200")))
    parser.add_argument("--selection-columns", type=int, default=int(os.getenv("DOGBOT_GRUSS_BACKR_LAYR_SELECTION_COLUMNS", "80")))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--manual-note", default="")
    parser.add_argument("--no-manual-prompt", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> BackrLayrDiagnosticConfig:
    side = str(args.side or "").strip().upper()
    if side not in ALLOWED_SIDES:
        raise ValueError("side doit etre BACK ou LAY")
    if args.initial_price is None:
        raise ValueError("--initial-price ou DOGBOT_GRUSS_BACKR_LAYR_INITIAL_PRICE est obligatoire")
    if args.replace_price is None:
        raise ValueError("--replace-price ou DOGBOT_GRUSS_BACKR_LAYR_REPLACE_PRICE est obligatoire")
    if float(args.stake) != 2.0:
        raise ValueError("stake doit etre exactement 2 EUR pour ce diagnostic")
    if args.runner_row < 5:
        raise ValueError("runner-row doit pointer une ligne runner PLACE, typiquement >= 5")
    if args.seconds <= 0:
        raise ValueError("seconds doit etre positif")
    if args.interval <= 0:
        raise ValueError("interval doit etre positif")
    if args.after_replace_ticks < 0:
        raise ValueError("after-replace-ticks doit etre positif ou nul")
    if args.selection_rows < 200:
        raise ValueError("selection-rows doit etre au moins 200")
    return BackrLayrDiagnosticConfig(
        row=int(args.runner_row),
        side=side,
        initial_price=float(args.initial_price),
        replace_price=float(args.replace_price),
        stake=float(args.stake),
        seconds=int(args.seconds),
        interval_seconds=float(args.interval),
        after_replace_ticks=int(args.after_replace_ticks),
        selection_rows=int(args.selection_rows),
        selection_columns=int(args.selection_columns),
        output_dir=Path(args.output_dir),
        second_replace_price=None if args.second_replace_price is None else float(args.second_replace_price),
        manual_note=str(args.manual_note or ""),
        no_manual_prompt=bool(args.no_manual_prompt),
    )


def validate_environment(env: dict[str, str] | None = None) -> None:
    values = env if env is not None else os.environ
    if not _env_true(values.get(DIAGNOSTIC_ARM_ENV)):
        raise RuntimeError(f"{DIAGNOSTIC_ARM_ENV}=true est obligatoire pour armer ce diagnostic reel")
    for name in REQUIRED_ENV_TRUE:
        if not _env_true(values.get(name)):
            raise RuntimeError(f"{name}=true est obligatoire")
    for name, expected in REQUIRED_ENV_VALUES.items():
        if str(values.get(name, "")).strip() != expected:
            raise RuntimeError(f"{name} doit etre exactement {expected!r}")
    for name in FORBIDDEN_ENV_TRUE:
        if _env_true(values.get(name)):
            raise RuntimeError(f"{name}=true est incompatible avec ce diagnostic")


def run_diagnostic(
    bridge: Any,
    config: BackrLayrDiagnosticConfig,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    input_fn: Callable[[str], str] = input,
    now_fn: Callable[[], datetime] = datetime.now,
) -> Path:
    validate_target_cells(bridge, config)
    validate_market_safe_for_initial_write(bridge, config)
    session_dir = create_session_dir(config.output_dir, now_fn=now_fn)
    summary_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []

    print(f"diagnostic_session={session_dir}")
    print("Step initial: ecriture R/S/Q avec BACK ou LAY. Q/R/S ne sont pas nettoyes.")
    write_initial_order(bridge, config)
    summary_rows.append(_summary_row("initial_write", config.side, "", "", "", "written"))

    ticks_path = session_dir / "ticks.csv"
    with ticks_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=bet_ref_diag.tick_fieldnames())
        writer.writeheader()
        bet_ref = ""
        total_ticks = int(config.seconds / config.interval_seconds)
        for tick in range(total_ticks + 1):
            row = dump_tick(bridge, config, session_dir, tick=tick, now_fn=now_fn)
            tick_rows.append(row)
            writer.writerow(row)
            bet_ref_diag.print_tick_summary(row)
            bet_ref = normalise_gruss_bet_ref(row.get("bet_ref_value"))
            if is_valid_bet_ref(bet_ref):
                break
            if tick < total_ticks:
                sleep_fn(config.interval_seconds)

        replace_decision = maybe_write_replace(bridge, config, bet_ref, config.replace_price)
        summary_rows.append(replace_decision)
        for extra_tick in range(config.after_replace_ticks):
            if extra_tick > 0:
                sleep_fn(config.interval_seconds)
            row = dump_tick(
                bridge,
                config,
                session_dir,
                tick=len(tick_rows),
                now_fn=now_fn,
            )
            tick_rows.append(row)
            writer.writerow(row)
            bet_ref_diag.print_tick_summary(row)

        if config.second_replace_price is not None:
            latest_ref = _latest_row_t(tick_rows, config.row)
            second_decision = maybe_write_replace(
                bridge,
                config,
                latest_ref,
                config.second_replace_price,
                phase="second_replace",
            )
            summary_rows.append(second_decision)
            row = dump_tick(bridge, config, session_dir, tick=len(tick_rows), now_fn=now_fn)
            tick_rows.append(row)
            writer.writerow(row)

    manual_note = bet_ref_diag.collect_manual_note(
        _manual_config(config),
        input_fn=input_fn,
    )
    (session_dir / "manual_visual_check.txt").write_text(
        (
            "Questions visuelles:\n"
            "- un seul ordre reste-t-il ouvert ?\n"
            "- la cote a-t-elle change ?\n"
            "- un deuxieme ordre a-t-il ete cree ?\n"
            "- la Bet Ref recoit-elle un suffixe N ?\n\n"
            f"Note: {manual_note}"
        ),
        encoding="utf-8",
    )
    write_summary_csv(session_dir / "summary.csv", summary_rows)
    final_summary = build_final_summary(tick_rows, summary_rows)
    (session_dir / "summary_final.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    print(f"summary_final={final_summary!r}")
    return session_dir


def validate_target_cells(bridge: Any, config: BackrLayrDiagnosticConfig) -> None:
    trigger_value = bridge.read_cell(SHEET, f"{TRIGGER_COLUMN}{config.row}")
    if trigger_value not in (None, ""):
        raise RuntimeError(f"{SHEET}!Q{config.row} non vide; diagnostic refuse")
    runner = bridge.read_cell(SHEET, f"A{config.row}")
    print(f"target_runner_row={config.row} runner={runner!r}")


def validate_market_safe_for_initial_write(bridge: Any, config: BackrLayrDiagnosticConfig) -> None:
    market_status = _market_status_value(bridge)
    seconds_until_start = _seconds_until_start(bridge)

    if _is_suspended_status(market_status):
        raise RuntimeError(
            "diagnostic refuse: marche suspendu avant ordre initial "
            f"market_status={market_status!r} seconds_until_start={seconds_until_start!r}"
        )

    if seconds_until_start is None:
        raise RuntimeError(
            "diagnostic refuse: countdown illisible avant ordre initial "
            f"market_status={market_status!r}"
        )

    if seconds_until_start <= 10:
        raise RuntimeError(
            "diagnostic refuse: countdown trop bas avant ordre initial "
            f"seconds_until_start={seconds_until_start!r} market_status={market_status!r}"
        )


def write_initial_order(bridge: Any, config: BackrLayrDiagnosticConfig) -> list[str]:
    cells = (
        (f"{ODDS_COLUMN}{config.row}", config.initial_price),
        (f"{STAKE_COLUMN}{config.row}", config.stake),
        (f"{TRIGGER_COLUMN}{config.row}", config.side),
    )
    validate_write_plan(cells, row=config.row, trigger=config.side, price=config.initial_price, stake=config.stake)
    return bridge.write_cells(SHEET, cells, allow_write=True)


def maybe_write_replace(
    bridge: Any,
    config: BackrLayrDiagnosticConfig,
    bet_ref_value: Any,
    price: float,
    *,
    phase: str = "replace",
) -> dict[str, Any]:
    trigger = replace_trigger_for_side(config.side)
    market_status = _market_status_value(bridge)
    seconds_until_start = _seconds_until_start(bridge)
    raw_ref = normalise_gruss_bet_ref(bridge.read_cell(SHEET, f"{BET_REF_COLUMN}{config.row}"))
    matched_stake = bridge.read_cell(SHEET, f"{MATCHED_STAKE_COLUMN}{config.row}")
    matched_stake_value = _numeric_or_none(matched_stake)
    suffix_handled = False

    if raw_ref in {"", "PENDING"}:
        return _summary_row(phase, trigger, raw_ref, matched_stake, "bet_ref_not_ready", "skipped", market_status=market_status, seconds_until_start=seconds_until_start)
    if is_terminal_bet_status(raw_ref):
        return _summary_row(phase, trigger, raw_ref, matched_stake, "row_status_not_replaceable", "skipped", market_status=market_status, seconds_until_start=seconds_until_start)
    if not is_valid_bet_ref(raw_ref):
        return _summary_row(phase, trigger, raw_ref, matched_stake, "invalid_bet_ref_for_replace", "skipped", market_status=market_status, seconds_until_start=seconds_until_start)
    if _is_suspended_status(market_status):
        return _summary_row(
            phase,
            trigger,
            raw_ref,
            matched_stake,
            "market_suspended_no_replace",
            "skipped",
            market_status=market_status,
            seconds_until_start=seconds_until_start,
        )
    if seconds_until_start is not None and seconds_until_start <= 10:
        return _summary_row(
            phase,
            trigger,
            raw_ref,
            matched_stake,
            "seconds_until_start_lte_10_no_replace",
            "skipped",
            market_status=market_status,
            seconds_until_start=seconds_until_start,
        )
    if matched_stake_value is None:
        return _summary_row(phase, trigger, raw_ref, matched_stake, "matched_stake_unavailable_no_replace", "skipped", market_status=market_status, seconds_until_start=seconds_until_start)
    if matched_stake_value > 0:
        return _summary_row(phase, trigger, raw_ref, matched_stake, "matched_stake_positive_no_replace", "skipped", market_status=market_status, seconds_until_start=seconds_until_start)

    clean_ref = strip_gruss_ref_suffix(raw_ref)
    cells: list[tuple[str, Any]] = []
    if clean_ref != raw_ref:
        cells.append((f"{BET_REF_COLUMN}{config.row}", clean_ref))
        suffix_handled = True
    cells.extend(
        [
            (f"{ODDS_COLUMN}{config.row}", price),
            (f"{STAKE_COLUMN}{config.row}", config.stake),
            (f"{TRIGGER_COLUMN}{config.row}", trigger),
        ]
    )
    validate_write_plan(cells, row=config.row, trigger=trigger, price=price, stake=config.stake)
    written = bridge.write_cells(SHEET, tuple(cells), allow_write=True)
    row = _summary_row(
        phase,
        trigger,
        raw_ref,
        matched_stake,
        "replace_written",
        "written",
        market_status=market_status,
        seconds_until_start=seconds_until_start,
    )
    row["price"] = price
    row["bet_ref_suffix_n_handled"] = suffix_handled
    row["cells_written"] = ";".join(written)
    return row


def validate_write_plan(cells: Iterable[tuple[str, Any]], *, row: int, trigger: str, price: float, stake: float) -> None:
    plan = tuple((str(address).strip().upper(), value) for address, value in cells)
    allowed_triggers = {"BACK", "LAY", "BACKR", "LAYR"}
    if trigger not in allowed_triggers:
        raise PermissionError("ce diagnostic interdit UPDATE/CANCEL/CLEAR")
    if float(stake) != 2.0:
        raise PermissionError("stake reel force a 2 EUR")
    expected_tail = (
        (f"{ODDS_COLUMN}{row}", price),
        (f"{STAKE_COLUMN}{row}", stake),
        (f"{TRIGGER_COLUMN}{row}", trigger),
    )
    if len(plan) == 3 and plan != expected_tail:
        raise PermissionError("R et S doivent etre ecrits avant Q sur la meme ligne")
    if len(plan) == 4:
        expected = ((f"{BET_REF_COLUMN}{row}", strip_gruss_ref_suffix(plan[0][1])), *expected_tail)
        if plan != expected:
            raise PermissionError("la gestion suffixe N doit ecrire T puis R/S/Q")
    elif len(plan) != 3:
        raise PermissionError("plan d'ecriture inattendu")


def replace_trigger_for_side(side: str) -> str:
    upper = str(side or "").strip().upper()
    if upper == "BACK":
        return "BACKR"
    if upper == "LAY":
        return "LAYR"
    raise ValueError(f"side invalide={side!r}")


def dump_tick(
    bridge: Any,
    config: BackrLayrDiagnosticConfig,
    session_dir: Path,
    *,
    tick: int,
    now_fn: Callable[[], datetime],
) -> dict[str, Any]:
    base_config = _bet_ref_config(config)
    return bet_ref_diag.dump_tick(bridge, base_config, session_dir, tick=tick, now_fn=now_fn)


def build_final_summary(tick_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = bet_ref_diag.build_final_summary(tick_rows)
    base["replace_attempted"] = any(_truthy(row.get("replace_attempted")) for row in summary_rows)
    base["replace_written"] = any(row.get("result") == "written" and row.get("phase") in {"replace", "second_replace"} for row in summary_rows)
    base["replace_decisions"] = summary_rows
    return base


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "trigger",
        "bet_ref_before",
        "matched_stake",
        "reason",
        "result",
        "price",
        "bet_ref_suffix_n_handled",
        "replace_attempted",
        "market_status",
        "seconds_until_start",
        "cells_written",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def create_session_dir(output_dir: Path, *, now_fn: Callable[[], datetime]) -> Path:
    stamp = now_fn().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(output_dir) / f"gruss_backr_layr_place_{stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _summary_row(
    phase: str,
    trigger: str,
    bet_ref_before: Any,
    matched_stake: Any,
    reason: str,
    result: str,
    *,
    market_status: Any = "",
    seconds_until_start: Any = "",
) -> dict[str, Any]:
    return {
        "phase": phase,
        "trigger": trigger,
        "bet_ref_before": bet_ref_before,
        "matched_stake": matched_stake,
        "reason": reason,
        "result": result,
        "price": "",
        "bet_ref_suffix_n_handled": False,
        "replace_attempted": result == "written" and phase in {"replace", "second_replace"},
        "market_status": market_status,
        "seconds_until_start": "" if seconds_until_start is None else seconds_until_start,
        "cells_written": "",
    }


def _latest_row_t(tick_rows: list[dict[str, Any]], row: int) -> str:
    for tick_row in reversed(tick_rows):
        q_af = bet_ref_diag._json_loads_or_raw(tick_row.get("place_q_af", "{}"))
        if isinstance(q_af, dict):
            value = q_af.get(f"{BET_REF_COLUMN}{row}")
            if value not in (None, ""):
                return str(value)
    return ""


def _numeric_or_none(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _market_status_value(bridge: Any) -> str:
    values = [
        bridge.read_cell(SHEET, "E2"),
        bridge.read_cell(SHEET, "F2"),
    ]
    return " | ".join(str(value).strip() for value in values if str(value or "").strip())


def _is_suspended_status(value: Any) -> bool:
    return "suspended" in str(value or "").strip().casefold()


def _seconds_until_start(bridge: Any) -> int | None:
    return parse_countdown_seconds(bridge.read_cell(SHEET, "D2"))


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _bet_ref_config(config: BackrLayrDiagnosticConfig) -> bet_ref_diag.DiagnosticConfig:
    return bet_ref_diag.DiagnosticConfig(
        row=config.row,
        side=config.side,
        price=config.initial_price,
        stake=config.stake,
        seconds=config.seconds,
        interval_seconds=config.interval_seconds,
        selection_rows=config.selection_rows,
        selection_columns=config.selection_columns,
        output_dir=config.output_dir,
        manual_note=config.manual_note,
        no_manual_prompt=config.no_manual_prompt,
    )


def _manual_config(config: BackrLayrDiagnosticConfig) -> bet_ref_diag.DiagnosticConfig:
    return _bet_ref_config(config)


def _env_true(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _optional_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    return float(raw)


if __name__ == "__main__":
    raise SystemExit(main())
