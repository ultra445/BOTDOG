from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BRIDGE_PATH = SRC / "dogbot" / "gruss" / "gruss_excel_bridge.py"
BRIDGE_SPEC = importlib.util.spec_from_file_location("dogbot_gruss_excel_bridge_direct", BRIDGE_PATH)
if BRIDGE_SPEC is None or BRIDGE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load Gruss Excel bridge: {BRIDGE_PATH}")
BRIDGE_MODULE = importlib.util.module_from_spec(BRIDGE_SPEC)
sys.modules[BRIDGE_SPEC.name] = BRIDGE_MODULE
BRIDGE_SPEC.loader.exec_module(BRIDGE_MODULE)
DEFAULT_WORKBOOK_PATH = BRIDGE_MODULE.DEFAULT_WORKBOOK_PATH
GrussExcelBridge = BRIDGE_MODULE.GrussExcelBridge


SHEET = "PLACE"
SELECTION_SHEETS = ("PLACE_Selections", "WIN_Selections")
TRIGGER_COLUMN = "Q"
ODDS_COLUMN = "R"
STAKE_COLUMN = "S"
BET_REF_COLUMN = "T"
DUMP_START_COLUMN = "Q"
DUMP_END_COLUMN = "AF"
ALLOWED_TRIGGERS = frozenset({"BACK", "LAY"})
DEFAULT_SECONDS = 20
DEFAULT_INTERVAL_SECONDS = 1.0
DEFAULT_SELECTION_ROWS = 200
DEFAULT_SELECTION_COLUMNS = 80
DIAGNOSTIC_ARM_ENV = "DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_PLACE"
COM_REJECTED_HRESULT = -2147418111
COM_REJECTED_MARKER = "COM_REJECTED"
BET_REF_PATTERN = re.compile(r"\b[0-9]{8,}N?\b", re.IGNORECASE)

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
class DiagnosticConfig:
    row: int
    side: str
    price: float
    stake: float
    seconds: int
    interval_seconds: float
    selection_rows: int
    selection_columns: int
    output_dir: Path
    manual_note: str
    no_manual_prompt: bool


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        validate_environment()
        bridge = GrussExcelBridge(DEFAULT_WORKBOOK_PATH)
        bridge.connect_open_workbook()
        ensure_open_visible_place_sheet(bridge)
        session_dir = run_diagnostic(bridge, config)
    except Exception as exc:
        print(f"ERREUR: {exc}")
        return 1

    print(f"diagnostic_output={session_dir}")
    print("Aucun UPDATE envoye. Q/R/S sont gardes visibles volontairement.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic isole Bet Ref Gruss PLACE. Ecrit un seul ordre reel PLACE a 2 EUR, "
            "ne nettoie pas Q/R/S, puis dump Excel pendant 20 secondes."
        )
    )
    parser.add_argument("--runner-row", type=int, default=int(os.getenv("DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_ROW", "5")))
    parser.add_argument("--side", default=os.getenv("DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_SIDE", "BACK"))
    parser.add_argument(
        "--price",
        type=float,
        default=_optional_float_env("DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_PRICE"),
        help="Prix LIMIT a ecrire en R. Obligatoire si l'env DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_PRICE est absent.",
    )
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument(
        "--seconds",
        type=int,
        default=int(os.getenv("DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_SECONDS", str(DEFAULT_SECONDS))),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_INTERVAL_SECONDS", str(DEFAULT_INTERVAL_SECONDS))),
    )
    parser.add_argument(
        "--selection-rows",
        type=int,
        default=int(os.getenv("DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_SELECTION_ROWS", str(DEFAULT_SELECTION_ROWS))),
    )
    parser.add_argument(
        "--selection-columns",
        type=int,
        default=int(os.getenv("DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_SELECTION_COLUMNS", str(DEFAULT_SELECTION_COLUMNS))),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--manual-note", default="")
    parser.add_argument("--no-manual-prompt", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> DiagnosticConfig:
    side = str(args.side).strip().upper()
    if side not in ALLOWED_TRIGGERS:
        raise ValueError("side doit etre BACK ou LAY; UPDATE/CANCEL/CLEAR interdits dans ce diagnostic")
    if args.price is None:
        raise ValueError("--price ou DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_PRICE est obligatoire")
    price = float(args.price)
    stake = float(args.stake)
    if price < 1.01:
        raise ValueError("price doit etre >= 1.01")
    if stake != 2.0:
        raise ValueError("stake doit etre exactement 2 EUR pour ce diagnostic")
    if args.runner_row < 5:
        raise ValueError("runner-row doit pointer une ligne runner PLACE, typiquement >= 5")
    if args.seconds <= 0:
        raise ValueError("seconds doit etre positif")
    if args.interval <= 0:
        raise ValueError("interval doit etre positif")
    if args.selection_rows < DEFAULT_SELECTION_ROWS:
        raise ValueError("selection-rows doit etre au moins 200")
    if args.selection_columns <= 0:
        raise ValueError("selection-columns doit etre positif")
    return DiagnosticConfig(
        row=int(args.runner_row),
        side=side,
        price=price,
        stake=stake,
        seconds=int(args.seconds),
        interval_seconds=float(args.interval),
        selection_rows=int(args.selection_rows),
        selection_columns=int(args.selection_columns),
        output_dir=Path(args.output_dir),
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
        actual = values.get(name)
        if str(actual).strip() != expected:
            raise RuntimeError(f"{name} doit etre exactement {expected!r}")
    for name in FORBIDDEN_ENV_TRUE:
        if _env_true(values.get(name)):
            raise RuntimeError(f"{name}=true est incompatible avec ce diagnostic")


def ensure_open_visible_place_sheet(bridge: Any) -> None:
    if not bridge.is_workbook_visible():
        raise RuntimeError("workbook Excel Gruss non visible")
    if not bridge.has_sheet(SHEET):
        raise RuntimeError("onglet PLACE manquant")


def run_diagnostic(
    bridge: Any,
    config: DiagnosticConfig,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    input_fn: Callable[[str], str] = input,
    now_fn: Callable[[], datetime] = datetime.now,
) -> Path:
    validate_target_cells(bridge, config)
    session_dir = create_session_dir(config.output_dir, now_fn=now_fn)
    print(f"diagnostic_session={session_dir}")
    print("Ecriture reelle isolee: PLACE seulement, un runner, trigger BACK/LAY uniquement.")
    print("Q/R/S ne seront pas nettoyes par ce script.")

    written = write_initial_place_order(bridge, config)
    print(f"written_cells={written}")

    ticks_path = session_dir / "ticks.csv"
    with ticks_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=tick_fieldnames())
        writer.writeheader()
        total_ticks = int(config.seconds / config.interval_seconds)
        tick_rows: list[dict[str, Any]] = []
        for tick in range(total_ticks + 1):
            row = dump_tick(bridge, config, session_dir, tick=tick, now_fn=now_fn)
            tick_rows.append(row)
            writer.writerow(row)
            print_tick_summary(row)
            if tick < total_ticks:
                sleep_fn(config.interval_seconds)

    manual_note = collect_manual_note(config, input_fn=input_fn)
    (session_dir / "manual_open_bets_note.txt").write_text(manual_note, encoding="utf-8")
    summary = build_final_summary(tick_rows)
    (session_dir / "summary_final.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    print(f"summary_final={summary!r}")
    print(f"manual_open_bets_note={manual_note!r}")
    return session_dir


def validate_target_cells(bridge: Any, config: DiagnosticConfig) -> None:
    q_cell = f"{TRIGGER_COLUMN}{config.row}"
    r_cell = f"{ODDS_COLUMN}{config.row}"
    s_cell = f"{STAKE_COLUMN}{config.row}"
    t_cell = f"{BET_REF_COLUMN}{config.row}"
    trigger_value = bridge.read_cell(SHEET, q_cell)
    if trigger_value not in (None, ""):
        raise RuntimeError(f"{SHEET}!{q_cell} non vide; diagnostic refuse pour eviter une commande existante")
    runner = bridge.read_cell(SHEET, f"A{config.row}")
    if runner in (None, ""):
        print(f"ATTENTION: {SHEET}!A{config.row} est vide; verifie que runner-row pointe le bon chien.")
    print(f"target_runner_row={config.row} runner={runner!r}")
    pre_errors: list[str] = []
    r_value, _ = safe_read_cell(bridge, SHEET, r_cell, read_errors=pre_errors)
    s_value, _ = safe_read_cell(bridge, SHEET, s_cell, read_errors=pre_errors)
    t_value, _ = safe_read_cell(bridge, SHEET, t_cell, read_errors=pre_errors)
    print(f"pre_write {r_cell}={r_value!r} {s_cell}={s_value!r} {t_cell}={t_value!r}")
    if pre_errors:
        print(f"pre_write_read_errors={pre_errors!r}")


def write_initial_place_order(bridge: Any, config: DiagnosticConfig) -> list[str]:
    cells = (
        (f"{ODDS_COLUMN}{config.row}", config.price),
        (f"{STAKE_COLUMN}{config.row}", config.stake),
        (f"{TRIGGER_COLUMN}{config.row}", config.side),
    )
    validate_write_plan(cells, row=config.row, side=config.side, price=config.price, stake=config.stake)
    return bridge.write_cells(SHEET, cells, allow_write=True)


def validate_write_plan(cells: Iterable[tuple[str, Any]], *, row: int, side: str, price: float, stake: float) -> None:
    expected = (
        (f"{ODDS_COLUMN}{row}", price),
        (f"{STAKE_COLUMN}{row}", stake),
        (f"{TRIGGER_COLUMN}{row}", side),
    )
    plan = tuple((str(address).strip().upper(), value) for address, value in cells)
    if plan != expected:
        raise PermissionError("ce diagnostic est limite exactement a R/S/Q sur une seule ligne PLACE")
    if side not in ALLOWED_TRIGGERS:
        raise PermissionError("seuls BACK/LAY sont autorises; UPDATE/CANCEL/CLEAR interdits")
    if float(stake) != 2.0:
        raise PermissionError("stake reel force a 2 EUR")


def dump_tick(
    bridge: Any,
    config: DiagnosticConfig,
    session_dir: Path,
    *,
    tick: int,
    now_fn: Callable[[], datetime] = datetime.now,
) -> dict[str, Any]:
    timestamp = now_fn().isoformat(timespec="seconds")
    read_errors: list[str] = []
    row_dump, row_com_errors = read_row_dump_safe(
        bridge,
        SHEET,
        config.row,
        DUMP_START_COLUMN,
        DUMP_END_COLUMN,
        read_errors=read_errors,
    )
    sheet_names, sheet_name_com_errors = get_sheet_names_safe(bridge, read_errors=read_errors)
    all_sheets_dir = session_dir / f"all_sheets_tick{tick:02d}"
    all_sheets_dir.mkdir(parents=True, exist_ok=True)
    dumped_sheet_names: list[str] = []
    selection_paths: dict[str, str] = {}
    selection_summaries: dict[str, str] = {}
    selection_bet_refs: dict[str, list[str]] = {}
    sheet_com_errors = 0
    for sheet_name in sheet_names:
        has_sheet, has_sheet_error = has_sheet_safe(bridge, sheet_name, read_errors=read_errors)
        sheet_com_errors += has_sheet_error
        if not has_sheet:
            continue
        csv_path = all_sheets_dir / f"{safe_filename(sheet_name)}.csv"
        values, read_sheet_com_errors = read_sheet_safe(
            bridge,
            sheet_name,
            rows=config.selection_rows,
            columns=config.selection_columns,
            read_errors=read_errors,
        )
        sheet_com_errors += read_sheet_com_errors
        write_matrix_csv(csv_path, values)
        dumped_sheet_names.append(sheet_name)
        if sheet_name in SELECTION_SHEETS:
            selection_paths[sheet_name] = str(csv_path)
            selection_summaries[sheet_name] = json.dumps(last_non_empty_rows(values), ensure_ascii=True, default=str)
            selection_bet_refs[sheet_name] = find_bet_refs_in_matrix(values)

    for sheet_name in SELECTION_SHEETS:
        if sheet_name not in selection_paths:
            selection_paths[sheet_name] = "missing_sheet"
            selection_summaries[sheet_name] = "missing_sheet"

    bet_ref_found, bet_ref_source, bet_ref_value = detect_tick_bet_ref(row_dump, selection_bet_refs)
    runner_name, runner_com_errors = safe_read_cell(bridge, SHEET, f"A{config.row}", read_errors=read_errors)
    com_errors_count = row_com_errors + sheet_name_com_errors + sheet_com_errors + runner_com_errors
    return {
        "tick": tick,
        "timestamp": timestamp,
        "sheet": SHEET,
        "runner_row": config.row,
        "runner_name": runner_name,
        "side": config.side,
        "price": config.price,
        "stake": config.stake,
        "place_q_af": json.dumps(row_dump, ensure_ascii=True, default=str),
        "place_q": row_dump.get(f"{TRIGGER_COLUMN}{config.row}"),
        "place_r": row_dump.get(f"{ODDS_COLUMN}{config.row}"),
        "place_s": row_dump.get(f"{STAKE_COLUMN}{config.row}"),
        "place_t": row_dump.get(f"{BET_REF_COLUMN}{config.row}"),
        "workbook_sheet_names": json.dumps(sheet_names, ensure_ascii=True),
        "all_sheet_dumps_dir": str(all_sheets_dir),
        "all_sheet_dumped_names": json.dumps(dumped_sheet_names, ensure_ascii=True),
        "place_selections_csv": selection_paths.get("PLACE_Selections", ""),
        "win_selections_csv": selection_paths.get("WIN_Selections", ""),
        "place_selections_last_non_empty": selection_summaries.get("PLACE_Selections", ""),
        "win_selections_last_non_empty": selection_summaries.get("WIN_Selections", ""),
        "place_selections_bet_refs": json.dumps(selection_bet_refs.get("PLACE_Selections", []), ensure_ascii=True),
        "win_selections_bet_refs": json.dumps(selection_bet_refs.get("WIN_Selections", []), ensure_ascii=True),
        "bet_ref_found": bet_ref_found,
        "bet_ref_source": bet_ref_source,
        "bet_ref_value": bet_ref_value,
        "com_errors_count": com_errors_count,
        "read_errors": " | ".join(read_errors),
    }


def tick_fieldnames() -> list[str]:
    return [
        "tick",
        "timestamp",
        "sheet",
        "runner_row",
        "runner_name",
        "side",
        "price",
        "stake",
        "place_q_af",
        "place_q",
        "place_r",
        "place_s",
        "place_t",
        "workbook_sheet_names",
        "all_sheet_dumps_dir",
        "all_sheet_dumped_names",
        "place_selections_csv",
        "win_selections_csv",
        "place_selections_last_non_empty",
        "win_selections_last_non_empty",
        "place_selections_bet_refs",
        "win_selections_bet_refs",
        "bet_ref_found",
        "bet_ref_source",
        "bet_ref_value",
        "com_errors_count",
        "read_errors",
    ]


def print_tick_summary(row: dict[str, Any]) -> None:
    print(
        "tick={tick} runner_row={runner_row} runner={runner_name!r} "
        "Q={place_q!r} R={place_r!r} S={place_s!r} T={place_t!r} "
        "bet_ref_found={bet_ref_found} source={bet_ref_source!r} com_errors={com_errors_count} "
        "sheets={workbook_sheet_names}".format(**row)
    )
    print(f"  PLACE_Selections={row['place_selections_csv']}")
    print(f"  WIN_Selections={row['win_selections_csv']}")


def read_row_dump(bridge: Any, sheet_name: str, row: int, start_column: str, end_column: str) -> dict[str, Any]:
    start = column_index(start_column)
    end = column_index(end_column)
    values: dict[str, Any] = {}
    for column in range(start, end + 1):
        address = f"{column_letter(column)}{row}"
        values[address] = bridge.read_cell(sheet_name, address)
    return values


def read_row_dump_safe(
    bridge: Any,
    sheet_name: str,
    row: int,
    start_column: str,
    end_column: str,
    *,
    read_errors: list[str],
) -> tuple[dict[str, Any], int]:
    start = column_index(start_column)
    end = column_index(end_column)
    values: dict[str, Any] = {}
    com_errors = 0
    for column in range(start, end + 1):
        address = f"{column_letter(column)}{row}"
        value, error_count = safe_read_cell(bridge, sheet_name, address, read_errors=read_errors)
        com_errors += error_count
        values[address] = value
    return values, com_errors


def safe_read_cell(bridge: Any, sheet_name: str, address: str, *, read_errors: list[str]) -> tuple[Any, int]:
    try:
        return bridge.read_cell(sheet_name, address), 0
    except Exception as exc:
        marker = exception_marker(exc)
        read_errors.append(f"{sheet_name}!{address}:{marker}")
        return marker, 1 if marker == COM_REJECTED_MARKER else 0


def get_sheet_names_safe(bridge: Any, *, read_errors: list[str]) -> tuple[list[str], int]:
    try:
        names = get_sheet_names(bridge)
        return names or [SHEET, *SELECTION_SHEETS], 0
    except Exception as exc:
        marker = exception_marker(exc)
        read_errors.append(f"workbook.sheet_names:{marker}")
        return [SHEET, *SELECTION_SHEETS], 1 if marker == COM_REJECTED_MARKER else 0


def has_sheet_safe(bridge: Any, sheet_name: str, *, read_errors: list[str]) -> tuple[bool, int]:
    try:
        return bool(bridge.has_sheet(sheet_name)), 0
    except Exception as exc:
        marker = exception_marker(exc)
        read_errors.append(f"{sheet_name}.has_sheet:{marker}")
        return False, 1 if marker == COM_REJECTED_MARKER else 0


def read_sheet_safe(
    bridge: Any,
    sheet_name: str,
    *,
    rows: int,
    columns: int,
    read_errors: list[str],
) -> tuple[list[list[Any]], int]:
    try:
        return bridge.read_sheet(sheet_name, rows=rows, columns=columns), 0
    except Exception as exc:
        marker = exception_marker(exc)
        read_errors.append(f"{sheet_name}.read_sheet:{marker}")
        return [[marker]], 1 if marker == COM_REJECTED_MARKER else 0


def write_matrix_csv(path: Path, values: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerows(values)


def build_final_summary(tick_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_found = next((row for row in tick_rows if _truthy(row.get("bet_ref_found"))), None)
    last_row = tick_rows[-1] if tick_rows else {}
    return {
        "bet_ref_found": first_found is not None,
        "first_tick_with_bet_ref": first_found.get("tick") if first_found else None,
        "bet_ref_source": first_found.get("bet_ref_source") if first_found else "",
        "bet_ref_value": first_found.get("bet_ref_value") if first_found else "",
        "last_place_q_af": _json_loads_or_raw(last_row.get("place_q_af", "{}")),
        "last_place_selections_non_empty": _json_loads_or_raw(
            last_row.get("place_selections_last_non_empty", "[]")
        ),
        "last_win_selections_non_empty": _json_loads_or_raw(
            last_row.get("win_selections_last_non_empty", "[]")
        ),
        "com_errors_count": sum(_safe_int(row.get("com_errors_count")) for row in tick_rows),
        "ticks_written": len(tick_rows),
    }


def detect_tick_bet_ref(
    row_dump: dict[str, Any],
    selection_bet_refs: dict[str, list[str]],
) -> tuple[bool, str, str]:
    row_t = row_dump.get("T" + str(_row_from_dump(row_dump)))
    row_t_ref = first_bet_ref(row_t)
    if row_t_ref:
        return True, "ROW_T", row_t_ref

    ignored_addresses = {
        "Q" + str(_row_from_dump(row_dump)),
        "R" + str(_row_from_dump(row_dump)),
        "S" + str(_row_from_dump(row_dump)),
    }
    for address, value in row_dump.items():
        if address in ignored_addresses:
            continue
        ref = first_bet_ref(value)
        if ref:
            return True, "ROW_Q_AF", ref

    for sheet_name in SELECTION_SHEETS:
        refs = selection_bet_refs.get(sheet_name, [])
        if refs:
            return True, sheet_name, refs[0]
    return False, "", ""


def first_bet_ref(value: Any) -> str:
    if value in (None, "", COM_REJECTED_MARKER):
        return ""
    text = str(value)
    upper_text = text.strip().upper()
    if upper_text in {"PENDING", "CANCELLED", "CANCELED", "LAPSED", "VARIOUS"} or upper_text.startswith("RESULT_"):
        return ""
    match = BET_REF_PATTERN.search(text)
    return match.group(0).upper() if match else ""


def find_bet_refs_in_matrix(values: list[list[Any]]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for row in values:
        for value in row:
            ref = first_bet_ref(value)
            if ref and ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def exception_marker(exc: Exception) -> str:
    if is_com_rejected(exc):
        return COM_REJECTED_MARKER
    return f"ERROR:{type(exc).__name__}:{exc}"


def is_com_rejected(exc: Exception) -> bool:
    hresult = getattr(exc, "hresult", None)
    if hresult == COM_REJECTED_HRESULT:
        return True
    return any(arg == COM_REJECTED_HRESULT for arg in getattr(exc, "args", ()))


def _row_from_dump(row_dump: dict[str, Any]) -> int:
    for address in row_dump:
        digits = "".join(char for char in address if char.isdigit())
        if digits:
            return int(digits)
    return 0


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _json_loads_or_raw(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe or "sheet"


def last_non_empty_rows(values: list[list[Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(values, start=1):
        if any(cell not in (None, "") for cell in row):
            rows.append({"row": index, "values": row})
    return rows[-limit:]


def get_sheet_names(bridge: Any) -> list[str]:
    if hasattr(bridge, "sheet_names"):
        return list(bridge.sheet_names())
    workbook = getattr(bridge, "workbook", None)
    sheets = getattr(workbook, "sheets", [])
    return [str(getattr(sheet, "name", "")) for sheet in sheets]


def create_session_dir(output_dir: Path, *, now_fn: Callable[[], datetime] = datetime.now) -> Path:
    stamp = now_fn().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / f"gruss_bet_ref_place_{stamp}"
    counter = 1
    while session_dir.exists():
        session_dir = output_dir / f"gruss_bet_ref_place_{stamp}_{counter}"
        counter += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def collect_manual_note(config: DiagnosticConfig, *, input_fn: Callable[[str], str] = input) -> str:
    if config.manual_note:
        return config.manual_note
    if config.no_manual_prompt:
        return ""
    prompt = (
        "Observation manuelle Gruss/Open Bets pendant le diagnostic "
        "(ex: visible/open bet absent/bet ref vue), puis Entree: "
    )
    try:
        return input_fn(prompt)
    except EOFError:
        return ""


def column_index(column: str) -> int:
    value = 0
    for char in str(column).strip().upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"colonne invalide: {column}")
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value


def column_letter(index: int) -> str:
    if index <= 0:
        raise ValueError("index colonne invalide")
    chars: list[str] = []
    while index:
        index, remainder = divmod(index - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _env_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    return float(raw)


if __name__ == "__main__":
    raise SystemExit(main())
