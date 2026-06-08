from __future__ import annotations

import os
import csv
import importlib.util
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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


SHEETS = ("WIN", "PLACE")
RUNNER_ROWS = range(5, 13)
LAYOUT_ROWS = range(1, 21)
LAYOUT_COLUMNS = range(1, 27)  # A:Z
AROUND_QRS_COLUMNS = range(14, 22)  # N:U
QRS_COLUMNS = ("Q", "R", "S")
KEYWORDS = (
    "BACK",
    "LAY",
    "BACKSP",
    "LAYSP",
    "CANCEL",
    "UPDATE",
    "Bet",
    "Stake",
    "Odds",
    "Trigger",
    "Command",
    "Place",
    "Cancel",
    "SP",
    "BSP",
    "Keep",
    "Take SP",
    "Liability",
    "Backers",
    "Layers",
)
FORBIDDEN_TRIGGER_COMMANDS = frozenset({"BACK", "LAY", "BACKSP", "LAYSP"})
NEUTRAL_TEST_VALUE = "TEST_VISIBLE_WRITE"
DEFAULT_TEST_SECONDS = 10
SEARCH_ROW_CAP = 250
SEARCH_COLUMN_CAP = 80
EXPORT_PATH = ROOT / "data" / "gruss_layout_diagnostic.csv"


@dataclass(frozen=True)
class CellDetails:
    address: str
    value: Any
    formula: Any
    comment: str
    validation: str


@dataclass(frozen=True)
class TriggerCandidate:
    sheet_name: str
    column: str
    row: int
    source: str
    evidence: str

    @property
    def address(self) -> str:
        return f"{self.column}{self.row}"


def main() -> int:
    layout_test_write = env_bool("DOGBOT_GRUSS_LAYOUT_TEST_WRITE")
    hold_visual_test = env_bool("DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST")

    print("Gruss Excel layout diagnostic")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print(f"READ_ONLY_DEFAULT={not layout_test_write}")
    print(f"DOGBOT_GRUSS_LAYOUT_TEST_WRITE={layout_test_write}")
    print(f"DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST={hold_visual_test}")
    print("Aucune commande BACK/LAY/BACKSP/LAYSP ne sera ecrite.")
    if hold_visual_test:
        print("Mode hold visuel actif: uniquement valeur neutre, jamais de trigger reel.")

    bridge = GrussExcelBridge(DEFAULT_WORKBOOK_PATH)
    try:
        bridge.connect_open_workbook()
        if not bridge.is_workbook_visible():
            raise RuntimeError("workbook Excel Gruss non visible")
        for sheet_name in SHEETS:
            if not bridge.has_sheet(sheet_name):
                raise RuntimeError(f"onglet {sheet_name} manquant")
    except Exception as exc:
        print(f"ERREUR: {exc}")
        return 1

    workbook = bridge.workbook
    trigger_candidates: list[TriggerCandidate] = []
    try:
        print_workbook_identity(workbook)
        print_workbook_sheet_overview(workbook)
        print_workbook_links(workbook)
        print_named_ranges(workbook)
        export_path = export_workbook_diagnostic(workbook, EXPORT_PATH)
        print(f"Export CSV diagnostic={export_path}")
        for sheet_name in SHEETS:
            sheet = bridge.get_sheet(sheet_name)
            trigger_candidates.extend(diagnose_sheet(sheet_name, sheet))
        trigger_candidates.extend(search_all_workbook_sheets(workbook, expected_sheets=SHEETS))
        print_trigger_candidate_summary(trigger_candidates)
        print_workbook_role_assessment(workbook, trigger_candidates)
        if layout_test_write:
            run_neutral_visible_write_tests(
                bridge,
                trigger_candidates=trigger_candidates,
                hold_visual_test=hold_visual_test,
            )
        else:
            print("Mode ecriture neutre desactive: aucune cellule modifiee.")
    except Exception as exc:
        print(f"ERREUR: diagnostic interrompu: {exc}")
        return 1

    return 0


def print_workbook_identity(workbook: Any) -> None:
    print_section("Workbook")
    print(f"fullname={safe_get(lambda: workbook.fullname)!r}")
    print(f"name={safe_get(lambda: workbook.name)!r}")
    print(f"saved={safe_get(lambda: workbook.api.Saved)!r}")
    print(f"read_only={safe_get(lambda: workbook.api.ReadOnly)!r}")
    print(f"app_visible={safe_get(lambda: workbook.app.visible)!r}")
    print(f"app_caption={safe_get(lambda: workbook.app.api.Caption)!r}")


def print_workbook_sheet_overview(workbook: Any) -> None:
    print_section("Workbook sheets")
    sheet_names = workbook_sheet_names(workbook)
    print(f"sheet_count={len(sheet_names)} sheets={sheet_names!r}")
    for sheet in workbook_sheets(workbook):
        rows, columns, address = used_range_dimensions(sheet)
        visible = safe_get(lambda sheet=sheet: sheet.api.Visible)
        print(f"{sheet.name}: used_range={address} rows={rows} columns={columns} visible={visible!r}")
    extras = [name for name in sheet_names if name.upper() not in {sheet.upper() for sheet in SHEETS}]
    if extras:
        print(f"Feuilles hors WIN/PLACE detectees: {extras!r}")
    else:
        print("Aucune feuille dediee hors WIN/PLACE detectee dans ce workbook.")


def print_workbook_links(workbook: Any) -> None:
    print_section("Workbook links/connections")
    connection_count = safe_get(lambda: workbook.api.Connections.Count, 0)
    print(f"connections_count={connection_count!r}")
    for index in range(1, int(connection_count or 0) + 1):
        connection = safe_get(lambda index=index: workbook.api.Connections.Item(index))
        if connection is not None:
            print(
                f"connection[{index}] name={safe_get(lambda connection=connection: connection.Name)!r} "
                f"description={safe_get(lambda connection=connection: connection.Description)!r}"
            )
    for link_type in (1, 2):
        links = safe_get(lambda link_type=link_type: workbook.api.LinkSources(link_type), None)
        print(f"link_sources_type_{link_type}={links!r}")


def print_named_ranges(workbook: Any) -> None:
    print_section("Named ranges")
    names = collect_named_ranges(workbook)
    if not names:
        print("Aucun named range accessible.")
        return
    for name, refers_to in names:
        print(f"{name}: {refers_to}")


def diagnose_sheet(sheet_name: str, sheet: Any) -> list[TriggerCandidate]:
    print_section(f"{sheet_name} used_range")
    rows, columns, address = used_range_dimensions(sheet)
    print(f"used_range={address} rows={rows} columns={columns}")

    print_section(f"{sheet_name} layout A:Z rows 1-20")
    print_matrix(sheet_name, sheet, LAYOUT_ROWS, LAYOUT_COLUMNS)

    print_section(f"{sheet_name} columns N:U rows 1-12")
    print_matrix(sheet_name, sheet, range(1, 13), AROUND_QRS_COLUMNS)

    print_section(f"{sheet_name} runner rows 5-12 A:Z")
    print_matrix(sheet_name, sheet, RUNNER_ROWS, LAYOUT_COLUMNS)

    print_section(f"{sheet_name} non-empty runner cells A:Z rows 5-12")
    print_non_empty_cells(sheet_name, sheet, RUNNER_ROWS, LAYOUT_COLUMNS)

    print_section(f"{sheet_name} formulas/comments/validations Q:R:S runners")
    print_cell_details(sheet_name, sheet, QRS_COLUMNS, RUNNER_ROWS)

    print_section(f"{sheet_name} comments/validations N:U rows 1-12")
    print_cell_details(
        sheet_name,
        sheet,
        tuple(column_letter(column) for column in AROUND_QRS_COLUMNS),
        range(1, 13),
        only_non_empty_details=True,
    )

    print_section(f"{sheet_name} keyword cells")
    details = find_keyword_cells(sheet, rows=min(rows, SEARCH_ROW_CAP), columns=min(columns, SEARCH_COLUMN_CAP))
    if not details:
        print("Aucune cellule contenant les mots-cles recherches dans la zone inspectee.")
    for item in details:
        print(
            f"{sheet_name}!{item.address} "
            f"value={format_value(item.value)} formula={format_value(item.formula)} "
            f"comment={format_value(item.comment)} validation={format_value(item.validation)}"
        )

    return find_trigger_candidates(sheet_name, details)


def search_all_workbook_sheets(workbook: Any, *, expected_sheets: Iterable[str]) -> list[TriggerCandidate]:
    print_section("Workbook-wide keyword search")
    expected = {sheet.upper() for sheet in expected_sheets}
    candidates: list[TriggerCandidate] = []
    for sheet in workbook_sheets(workbook):
        rows, columns, _ = used_range_dimensions(sheet)
        details = find_keyword_cells(
            sheet,
            rows=min(rows, SEARCH_ROW_CAP),
            columns=min(columns, SEARCH_COLUMN_CAP),
        )
        label = "expected_market_sheet" if sheet.name.upper() in expected else "extra_sheet"
        print(f"{sheet.name}: keyword_matches={len(details)} scope={label}")
        for item in details:
            print(
                f"{sheet.name}!{item.address} "
                f"value={format_value(item.value)} formula={format_value(item.formula)} "
                f"comment={format_value(item.comment)} validation={format_value(item.validation)}"
            )
        candidates.extend(find_trigger_candidates(sheet.name, details))
    return candidates


def export_workbook_diagnostic(workbook: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sheet", "cell", "value", "formula"])
        for sheet in workbook_sheets(workbook):
            rows, columns, _ = used_range_dimensions(sheet)
            for row in range(1, min(rows, SEARCH_ROW_CAP) + 1):
                for column in range(1, min(columns, SEARCH_COLUMN_CAP) + 1):
                    address = f"{column_letter(column)}{row}"
                    item = inspect_cell(sheet, address)
                    if any(value not in (None, "") for value in (item.value, item.formula)):
                        writer.writerow([sheet.name, item.address, item.value, item.formula])
    return output_path


def print_matrix(sheet_name: str, sheet: Any, rows: Iterable[int], columns: Iterable[int]) -> None:
    column_numbers = tuple(columns)
    print("row\t" + "\t".join(column_letter(column) for column in column_numbers))
    for row in rows:
        values = []
        for column in column_numbers:
            values.append(format_value(read_cell_value(sheet, row, column)))
        print(f"{sheet_name}!{row}\t" + "\t".join(values))


def print_non_empty_cells(sheet_name: str, sheet: Any, rows: Iterable[int], columns: Iterable[int]) -> None:
    count = 0
    for row in rows:
        for column in columns:
            address = f"{column_letter(column)}{row}"
            item = inspect_cell(sheet, address)
            if any(value not in (None, "") for value in (item.value, item.formula)):
                print(
                    f"{sheet_name}!{item.address} "
                    f"value={format_value(item.value)} formula={format_value(item.formula)}"
                )
                count += 1
    if count == 0:
        print("Aucune cellule non vide dans cette zone.")


def print_cell_details(
    sheet_name: str,
    sheet: Any,
    columns: Iterable[str],
    rows: Iterable[int],
    *,
    only_non_empty_details: bool = False,
) -> None:
    printed = 0
    for row in rows:
        for column in columns:
            address = f"{column}{row}".upper()
            item = inspect_cell(sheet, address)
            has_details = any(value not in (None, "") for value in (item.formula, item.comment, item.validation))
            if only_non_empty_details and not has_details:
                continue
            print(
                f"{sheet_name}!{address} "
                f"value={format_value(item.value)} formula={format_value(item.formula)} "
                f"comment={format_value(item.comment)} validation={format_value(item.validation)}"
            )
            printed += 1
    if printed == 0:
        print("Aucun commentaire/validation/formule accessible dans cette zone.")


def find_keyword_cells(sheet: Any, *, rows: int, columns: int) -> list[CellDetails]:
    matches: list[CellDetails] = []
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            address = f"{column_letter(column)}{row}"
            item = inspect_cell(sheet, address)
            haystack = " ".join(
                str(value)
                for value in (item.value, item.formula, item.comment, item.validation)
                if value not in (None, "")
            )
            if keyword_match(haystack):
                matches.append(item)
    return matches


def find_trigger_candidates(sheet_name: str, details: Iterable[CellDetails]) -> list[TriggerCandidate]:
    candidates: list[TriggerCandidate] = []
    for item in details:
        column, row = split_address(item.address)
        evidence_parts = [
            text
            for text in (
                str(item.value or ""),
                str(item.formula or ""),
                item.comment,
                item.validation,
            )
            if text
        ]
        evidence = " | ".join(evidence_parts)
        lowered = evidence.casefold()
        if row > 20:
            continue
        if "odds" in lowered or "stake" in lowered:
            continue
        if (
            "trigger" in lowered
            or "bet" in lowered
            or "cancel" in lowered
            or "update" in lowered
            or re.search(r"\b(?:BACK|LAY|BACKSP|LAYSP)\b", evidence, flags=re.IGNORECASE)
        ):
            candidates.append(
                TriggerCandidate(
                    sheet_name=sheet_name,
                    column=column,
                    row=5,
                    source=item.address,
                    evidence=evidence[:180],
                )
            )
    return dedupe_candidates(candidates)


def print_trigger_candidate_summary(candidates: Iterable[TriggerCandidate]) -> None:
    print_section("Trigger candidates")
    unique = dedupe_candidates(candidates)
    if not unique:
        print("Aucune colonne trigger candidate autre que l'hypothese existante n'a ete detectee.")
        return
    for candidate in unique:
        print(
            f"{candidate.sheet_name}!{candidate.address} "
            f"column={candidate.column} source={candidate.source} evidence={candidate.evidence!r}"
        )


def print_workbook_role_assessment(workbook: Any, candidates: Iterable[TriggerCandidate]) -> None:
    print_section("Workbook role assessment")
    sheet_names = workbook_sheet_names(workbook)
    extras = [name for name in sheet_names if name.upper() not in {sheet.upper() for sheet in SHEETS}]
    unique_candidates = dedupe_candidates(candidates)
    qrs_statuses = []
    for sheet in workbook_sheets(workbook):
        if sheet.name.upper() not in {sheet_name.upper() for sheet_name in SHEETS}:
            continue
        qrs_statuses.append((sheet.name, qrs_runner_cells_are_empty_and_plain(sheet)))

    print(f"extra_sheets={extras!r}")
    print(f"trigger_candidate_columns={[f'{item.sheet_name}!{item.column}' for item in unique_candidates]!r}")
    print(f"qrs_runner_cells_empty_plain={qrs_statuses!r}")
    if not extras and not unique_candidates and all(is_plain for _, is_plain in qrs_statuses):
        print(
            "Conclusion provisoire: ce workbook ressemble a une feuille Gruss de lecture marche "
            "sans zone d'ordre evidente; Q/R/S semblent libres/non surveillees."
        )
    elif extras:
        print("Conclusion provisoire: inspecter les feuilles hors WIN/PLACE ci-dessus pour une zone d'ordre possible.")
    elif unique_candidates:
        print("Conclusion provisoire: inspecter les colonnes candidates ci-dessus avant toute configuration de trigger.")
    else:
        print("Conclusion provisoire: aucun layout d'ordre certain detecte.")


def run_neutral_visible_write_tests(
    bridge: GrussExcelBridge,
    *,
    trigger_candidates: Iterable[TriggerCandidate],
    hold_visual_test: bool,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    print_section("Neutral visible write test")
    if neutral_value_is_forbidden(NEUTRAL_TEST_VALUE):
        raise PermissionError("neutral test value unexpectedly matches a trigger command")

    tests = [("PLACE", "Q5", "configured_Q_probe")]
    alternate = first_alternate_place_candidate(trigger_candidates)
    if alternate is not None:
        tests.append((alternate.sheet_name, alternate.address, f"candidate_from_{alternate.source}"))
    else:
        print("Aucune autre cellule trigger candidate PLACE detectee; seul PLACE!Q5 sera teste.")

    for sheet_name, address, reason in tests:
        run_one_neutral_visible_write(
            bridge,
            sheet_name=sheet_name,
            address=address,
            reason=reason,
            wait_seconds=DEFAULT_TEST_SECONDS,
            hold_visual_test=hold_visual_test,
            sleep_fn=sleep_fn,
        )


def run_one_neutral_visible_write(
    bridge: GrussExcelBridge,
    *,
    sheet_name: str,
    address: str,
    reason: str,
    wait_seconds: float = DEFAULT_TEST_SECONDS,
    hold_visual_test: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    if neutral_value_is_forbidden(NEUTRAL_TEST_VALUE):
        raise PermissionError("trigger command values are forbidden in layout diagnostics")

    before = bridge.read_cell(sheet_name, address)
    print(f"{sheet_name}!{address} neutral_test reason={reason} old_value={before!r}")
    can_hold_without_losing_value = before in (None, "")
    restored = False
    write_succeeded = False
    leave_neutral_visible = False
    try:
        bridge.write_cells(sheet_name, [(address, NEUTRAL_TEST_VALUE)], allow_write=True)
        write_succeeded = True
        post_write = bridge.read_cell(sheet_name, address)
        print(f"{sheet_name}!{address} post_write_value={post_write!r}")
        if hold_visual_test:
            print(
                f"{sheet_name}!{address} hold_visual_test=True "
                f"valeur neutre visible pendant {wait_seconds:g} secondes."
            )
        else:
            print(f"{sheet_name}!{address} valeur neutre visible pendant {wait_seconds:g} secondes.")
        sleep_fn(wait_seconds)
        leave_neutral_visible = hold_visual_test and can_hold_without_losing_value
    finally:
        if write_succeeded and leave_neutral_visible:
            print(
                f"{sheet_name}!{address} clear_skipped=True "
                "reason=hold_visual_test_neutral_value_only"
            )
            return True
        if write_succeeded and hold_visual_test and not can_hold_without_losing_value:
            print(
                f"{sheet_name}!{address} clear_skipped=False "
                "reason=old_value_was_not_empty_restore_required"
            )
        if write_succeeded:
            bridge.write_cells(sheet_name, [(address, before)], allow_write=True)
            restored = bridge.read_cell(sheet_name, address) == before
            print(f"{sheet_name}!{address} restored={restored}")
    if not restored:
        raise RuntimeError(f"restauration echouee: {sheet_name}!{address}")
    return restored


def first_alternate_place_candidate(candidates: Iterable[TriggerCandidate]) -> TriggerCandidate | None:
    for candidate in dedupe_candidates(candidates):
        if candidate.sheet_name.upper() == "PLACE" and candidate.column.upper() != "Q":
            return candidate
    return None


def dedupe_candidates(candidates: Iterable[TriggerCandidate]) -> list[TriggerCandidate]:
    seen: set[tuple[str, str]] = set()
    unique: list[TriggerCandidate] = []
    for candidate in candidates:
        key = (candidate.sheet_name.upper(), candidate.column.upper())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def collect_named_ranges(workbook: Any) -> list[tuple[str, str]]:
    names: list[tuple[str, str]] = []
    for name in safe_iter(lambda: workbook.names):
        names.append((str(safe_get(lambda name=name: name.name)), str(safe_get(lambda name=name: name.refers_to))))
    if names:
        return names

    api_names = safe_get(lambda: workbook.api.Names)
    count = safe_get(lambda: api_names.Count, 0) if api_names is not None else 0
    for index in range(1, int(count or 0) + 1):
        item = safe_get(lambda index=index: api_names.Item(index))
        if item is None:
            continue
        names.append((str(safe_get(lambda item=item: item.Name)), str(safe_get(lambda item=item: item.RefersTo))))
    return names


def workbook_sheets(workbook: Any) -> list[Any]:
    return list(safe_iter(lambda: workbook.sheets))


def workbook_sheet_names(workbook: Any) -> list[str]:
    return [str(safe_get(lambda sheet=sheet: sheet.name, "")) for sheet in workbook_sheets(workbook)]


def qrs_runner_cells_are_empty_and_plain(sheet: Any) -> bool:
    for row in RUNNER_ROWS:
        for column in QRS_COLUMNS:
            item = inspect_cell(sheet, f"{column}{row}")
            if any(value not in (None, "") for value in (item.value, item.formula, item.comment, item.validation)):
                return False
    return True


def inspect_cell(sheet: Any, address: str) -> CellDetails:
    cell = sheet.range(address)
    return CellDetails(
        address=address.upper(),
        value=safe_get(lambda: cell.value),
        formula=normalise_empty(safe_get(lambda: cell.formula)),
        comment=normalise_empty(read_comment(cell)),
        validation=normalise_empty(read_validation(cell)),
    )


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


def used_range_dimensions(sheet: Any) -> tuple[int, int, str]:
    used = sheet.used_range
    rows = int(safe_get(lambda: used.rows.count, 0) or 0)
    columns = int(safe_get(lambda: used.columns.count, 0) or 0)
    address = str(safe_get(lambda: used.address, ""))
    return rows, columns, address


def keyword_match(text: str) -> bool:
    if not text:
        return False
    for keyword in KEYWORDS:
        if keyword == "SP":
            if re.search(r"\bSP\b", text, flags=re.IGNORECASE):
                return True
            continue
        if re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE):
            return True
    return False


def neutral_value_is_forbidden(value: Any) -> bool:
    return str(value or "").strip().upper() in FORBIDDEN_TRIGGER_COMMANDS


def env_bool(name: str, env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return str(values.get(name, "")).strip().casefold() in {"1", "true", "yes", "on"}


def column_letter(column: int) -> str:
    letters = ""
    while column:
        column, remainder = divmod(column - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def split_address(address: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Z]+)([0-9]+)", address.upper())
    if not match:
        return address.upper(), 0
    return match.group(1), int(match.group(2))


def format_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", "\\r").replace("\n", "\\n").replace("\t", " ")
    if len(text) > 80:
        return text[:77] + "..."
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
