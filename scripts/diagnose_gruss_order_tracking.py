from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
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


TARGET_SHEETS = ("WIN", "PLACE", "WIN_Selections", "PLACE_Selections")
SELECTION_SHEETS = ("WIN_Selections", "PLACE_Selections")
HEADER_SCAN_ROWS = 30
SCAN_ROWS = 250
SCAN_COLUMNS = 120
LAST_NON_EMPTY_ROWS = 12

CANDIDATE_PATTERNS: dict[str, tuple[str, ...]] = {
    "market_type": ("market type", "market_type", "market", "win/place"),
    "runner": ("runner", "selection name", "selection", "dog", "greyhound"),
    "trap": ("trap", "draw"),
    "selection_id": ("selection id", "selection_id", "selectionid"),
    "side": ("side", "back/lay", "back lay", "bet type", "bet_type"),
    "odds": ("odds", "price", "requested odds", "limit price"),
    "stake": ("stake", "size", "amount"),
    "bet_ref": ("bet ref", "bet_ref", "betref", "bet id", "bet_id", "betid", "reference"),
    "matched_stake": ("matched stake", "matched_stake", "matched size", "size matched", "matched"),
    "unmatched_stake": (
        "unmatched stake",
        "unmatched_stake",
        "remaining stake",
        "remaining_stake",
        "outstanding",
        "unmatched",
    ),
    "avg_matched": (
        "avg matched",
        "avg_matched",
        "average matched",
        "average price matched",
        "average_price_matched",
        "avg price",
    ),
    "status": ("status", "state", "order status", "bet status", "result"),
    "lapsed": ("lapsed", "lapse"),
    "cancelled": ("cancelled", "canceled", "cancel status"),
    "profit_loss": ("profit loss", "profit/loss", "p&l", "pnl", "profit_loss"),
    "timestamp": ("timestamp", "time", "placed time", "placed", "date"),
    "cancel_command": ("cancel", "cancel bet", "cancel order", "cancelled", "canceled"),
    "trigger_command": ("trigger", "command", "back", "lay", "backsp", "laysp"),
}

TRACKING_REQUIRED = (
    "side",
    "odds",
    "stake",
    "bet_ref",
    "status",
    "matched_stake",
    "unmatched_stake",
)
CANCEL_REQUIRED = ("bet_ref", "cancel_command")
LINKING_COLUMNS = ("market_type", "runner", "trap", "selection_id", "side", "odds", "stake", "bet_ref", "timestamp")


@dataclass(frozen=True)
class HeaderCell:
    sheet: str
    row: int
    column: int
    address: str
    value: str


@dataclass(frozen=True)
class CandidateColumn:
    field: str
    sheet: str
    row: int
    column: int
    address: str
    header: str


@dataclass(frozen=True)
class KeywordHit:
    sheet: str
    address: str
    value: Any
    formula: Any
    comment: str
    validation: str


def main() -> int:
    print("Gruss order tracking diagnostic")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print("READ_ONLY=true")
    print("Aucun ordre reel. Aucun cancel reel. Aucune ecriture Excel.")
    print("DOGBOT_PRE_LADDER_ENABLED doit rester false tant que tracking/cancel ne sont pas confirmes.")

    bridge = GrussExcelBridge(DEFAULT_WORKBOOK_PATH)
    try:
        bridge.connect_open_workbook()
    except Exception as exc:
        print(f"ERREUR: workbook Gruss non ouvert: {exc}")
        return 1

    workbook = bridge.workbook
    print_workbook_overview(workbook)

    all_candidates: list[CandidateColumn] = []
    all_hits: list[KeywordHit] = []
    for sheet_name in TARGET_SHEETS:
        if not bridge.has_sheet(sheet_name):
            print_section(f"{sheet_name}")
            print("missing_sheet=True")
            continue
        sheet = bridge.get_sheet(sheet_name)
        candidates, hits = diagnose_sheet(sheet_name, sheet)
        all_candidates.extend(candidates)
        all_hits.extend(hits)

    print_order_tracking_assessment(all_candidates, all_hits)
    return 0


def print_workbook_overview(workbook: Any) -> None:
    print_section("Workbook")
    print(f"fullname={safe_get(lambda: workbook.fullname)!r}")
    print(f"name={safe_get(lambda: workbook.name)!r}")
    sheet_names = [str(safe_get(lambda sheet=sheet: sheet.name, "")) for sheet in workbook_sheets(workbook)]
    print(f"sheets={sheet_names!r}")
    for sheet in workbook_sheets(workbook):
        rows, columns, address = used_range_dimensions(sheet)
        print(f"{sheet.name}: used_range={address} rows={rows} columns={columns}")


def diagnose_sheet(sheet_name: str, sheet: Any) -> tuple[list[CandidateColumn], list[KeywordHit]]:
    rows, columns, address = used_range_dimensions(sheet)
    max_rows = min(rows, SCAN_ROWS)
    max_columns = min(columns, SCAN_COLUMNS)
    print_section(sheet_name)
    print(f"used_range={address} rows={rows} columns={columns}")

    headers = detect_headers(sheet_name, sheet, rows=min(rows, HEADER_SCAN_ROWS), columns=max_columns)
    print_detected_headers(headers)

    candidates = candidate_columns_from_headers(headers)
    print_candidate_columns(sheet_name, candidates)

    hits = find_order_keyword_hits(sheet_name, sheet, rows=max_rows, columns=max_columns)
    print_keyword_hits(sheet_name, hits)

    if sheet_name.upper() in {name.upper() for name in SELECTION_SHEETS}:
        print_last_non_empty_rows(sheet_name, sheet, rows=rows, columns=max_columns)

    return candidates, hits


def detect_headers(sheet_name: str, sheet: Any, *, rows: int, columns: int) -> list[HeaderCell]:
    headers: list[HeaderCell] = []
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            value = read_cell_value(sheet, row, column)
            text = clean_text(value)
            if not text:
                continue
            if looks_like_header(text):
                headers.append(
                    HeaderCell(
                        sheet=sheet_name,
                        row=row,
                        column=column,
                        address=f"{column_letter(column)}{row}",
                        value=text,
                    )
                )
    return headers


def candidate_columns_from_headers(headers: Iterable[HeaderCell]) -> list[CandidateColumn]:
    candidates: list[CandidateColumn] = []
    seen: set[tuple[str, str, int, int]] = set()
    for header in headers:
        normalized = normalize_text(header.value)
        for field, patterns in CANDIDATE_PATTERNS.items():
            if any(normalize_text(pattern) in normalized for pattern in patterns):
                key = (field, header.sheet.upper(), header.row, header.column)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    CandidateColumn(
                        field=field,
                        sheet=header.sheet,
                        row=header.row,
                        column=header.column,
                        address=header.address,
                        header=header.value,
                    )
                )
    return candidates


def find_order_keyword_hits(sheet_name: str, sheet: Any, *, rows: int, columns: int) -> list[KeywordHit]:
    keywords = {
        "bet",
        "bet ref",
        "bet id",
        "matched",
        "unmatched",
        "average",
        "status",
        "cancel",
        "lapsed",
        "profit",
        "loss",
        "trigger",
        "command",
        "back",
        "lay",
    }
    hits: list[KeywordHit] = []
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            address = f"{column_letter(column)}{row}"
            value = read_cell_value(sheet, row, column)
            formula = normalise_empty(safe_get(lambda: sheet.range(address).formula))
            comment = normalise_empty(read_comment(sheet.range(address)))
            validation = normalise_empty(read_validation(sheet.range(address)))
            haystack = normalize_text(" ".join(str(item) for item in (value, formula, comment, validation) if item not in (None, "")))
            if not haystack:
                continue
            if any(keyword in haystack for keyword in keywords):
                hits.append(
                    KeywordHit(
                        sheet=sheet_name,
                        address=address,
                        value=value,
                        formula=formula,
                        comment=comment,
                        validation=validation,
                    )
                )
    return hits


def print_detected_headers(headers: list[HeaderCell]) -> None:
    print("headers_detected:")
    if not headers:
        print("  - none")
        return
    for header in headers[:160]:
        print(f"  - {header.sheet}!{header.address} row={header.row} col={column_letter(header.column)} value={header.value!r}")
    if len(headers) > 160:
        print(f"  ... {len(headers) - 160} headers omitted")


def print_candidate_columns(sheet_name: str, candidates: list[CandidateColumn]) -> None:
    print("candidate_columns:")
    if not candidates:
        print("  - none")
        return
    by_field: dict[str, list[CandidateColumn]] = {}
    for candidate in candidates:
        by_field.setdefault(candidate.field, []).append(candidate)
    for field in sorted(by_field):
        values = ", ".join(
            f"{item.sheet}!{column_letter(item.column)} row={item.row} header={item.header!r}"
            for item in by_field[field]
        )
        print(f"  {field}: {values}")


def print_keyword_hits(sheet_name: str, hits: list[KeywordHit]) -> None:
    print("order_keyword_hits:")
    if not hits:
        print("  - none")
        return
    for hit in hits[:120]:
        print(
            f"  - {hit.sheet}!{hit.address} "
            f"value={format_value(hit.value)} formula={format_value(hit.formula)} "
            f"comment={format_value(hit.comment)} validation={format_value(hit.validation)}"
        )
    if len(hits) > 120:
        print(f"  ... {len(hits) - 120} hits omitted")


def print_last_non_empty_rows(sheet_name: str, sheet: Any, *, rows: int, columns: int) -> None:
    print("last_non_empty_rows:")
    non_empty_rows: list[int] = []
    for row in range(1, rows + 1):
        if row_has_value(sheet, row, columns):
            non_empty_rows.append(row)
    if not non_empty_rows:
        print("  - none")
        return
    for row in non_empty_rows[-LAST_NON_EMPTY_ROWS:]:
        values = []
        for column in range(1, columns + 1):
            value = read_cell_value(sheet, row, column)
            if value not in (None, ""):
                values.append(f"{column_letter(column)}={format_value(value)}")
        print(f"  {sheet_name}!{row}: " + " | ".join(values))


def print_order_tracking_assessment(candidates: list[CandidateColumn], hits: list[KeywordHit]) -> None:
    print_section("Assessment")
    selection_candidates = [
        candidate
        for candidate in candidates
        if candidate.sheet.upper() in {name.upper() for name in SELECTION_SHEETS}
    ]
    fields = {candidate.field for candidate in selection_candidates}
    hit_text = normalize_text(
        " ".join(
            str(item)
            for hit in hits
            for item in (hit.value, hit.formula, hit.comment, hit.validation)
            if item not in (None, "")
        )
    )
    if "cancel" in hit_text:
        fields.add("cancel_command")

    missing_tracking = [field for field in TRACKING_REQUIRED if field not in fields]
    linking_missing = [field for field in LINKING_COLUMNS if field not in fields]
    missing_cancel = [field for field in CANCEL_REQUIRED if field not in fields]
    tracking_possible = not missing_tracking and any(field in fields for field in ("runner", "trap", "selection_id"))
    cancel_possible = not missing_cancel

    print(f"tracking_possible={tracking_possible}")
    print(f"cancel_possible={cancel_possible}")
    print(f"detected_columns={sorted(fields)!r}")
    print(f"required_columns={list(TRACKING_REQUIRED)!r}")
    print(f"missing_columns={missing_tracking!r}")
    print(f"linking_columns={list(LINKING_COLUMNS)!r}")
    print(f"missing_linking_columns={linking_missing!r}")
    print(f"cancel_required_columns={list(CANCEL_REQUIRED)!r}")
    print(f"missing_cancel_columns={missing_cancel!r}")

    if tracking_possible:
        print(
            "Conclusion tracking: colonnes minimales detectees sur WIN_Selections/PLACE_Selections; "
            "verification manuelle des valeurs live encore requise avant PRE ladder reel."
        )
    else:
        print(
            "Conclusion tracking: non confirme. Ne pas activer DOGBOT_PRE_LADDER_ENABLED tant que "
            "les colonnes d'etat ordre ne sont pas identifiees et testees."
        )

    if cancel_possible:
        print(
            "Conclusion cancel: une piste cancel existe dans le workbook; confirmer la commande exacte "
            "sur template/documentation avant toute annulation reelle."
        )
    else:
        print("Conclusion cancel: non confirme. Aucun cancel reel ne doit etre tente.")


def looks_like_header(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if any(any(pattern in normalized for pattern in patterns) for patterns in CANDIDATE_PATTERNS.values()):
        return True
    if re.search(r"[A-Za-z]", text) and len(text) <= 80:
        return True
    return False


def row_has_value(sheet: Any, row: int, columns: int) -> bool:
    for column in range(1, columns + 1):
        if read_cell_value(sheet, row, column) not in (None, ""):
            return True
    return False


def used_range_dimensions(sheet: Any) -> tuple[int, int, str]:
    used = sheet.used_range
    rows = int(safe_get(lambda: used.rows.count, 0) or 0)
    columns = int(safe_get(lambda: used.columns.count, 0) or 0)
    address = str(safe_get(lambda: used.address, ""))
    return rows, columns, address


def workbook_sheets(workbook: Any) -> list[Any]:
    return list(safe_iter(lambda: workbook.sheets))


def read_cell_value(sheet: Any, row: int, column: int) -> Any:
    return safe_get(lambda: sheet.range((row, column)).value)


def read_comment(cell: Any) -> str:
    comment = safe_get(lambda: cell.api.Comment)
    if comment is None:
        return ""
    return str(safe_get(lambda: comment.Text(), ""))


def read_validation(cell: Any) -> str:
    validation = safe_get(lambda: cell.api.Validation)
    if validation is None:
        return ""
    parts = []
    for attr in ("Type", "Formula1", "Formula2", "InCellDropdown", "IgnoreBlank", "ErrorTitle", "ErrorMessage"):
        value = safe_get(lambda attr=attr: getattr(validation, attr))
        if value not in (None, ""):
            parts.append(f"{attr}={value}")
    return "; ".join(parts)


def column_letter(column: int) -> str:
    letters = ""
    while column:
        column, remainder = divmod(column - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    text = clean_text(value).casefold()
    text = text.replace("_", " ")
    text = re.sub(r"[^0-9a-z&/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def format_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", "\\r").replace("\n", "\\n").replace("\t", " ")
    if len(text) > 120:
        return text[:117] + "..."
    return text


def normalise_empty(value: Any) -> Any:
    if value in (None, ""):
        return ""
    return value


def safe_get(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


def safe_iter(fn: Callable[[], Iterable[Any]]) -> Iterable[Any]:
    try:
        return fn()
    except Exception:
        return ()


def print_section(title: str) -> None:
    print("")
    print(f"=== {title} ===")


if __name__ == "__main__":
    raise SystemExit(main())
