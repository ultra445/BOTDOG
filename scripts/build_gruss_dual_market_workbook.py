from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "gruss_bridge" / "Gruss Place_Bets.xls"
DEFAULT_OUTPUT = ROOT / "gruss_bridge" / "dogbot_gruss_WIN_PLACE.xls"
EXPECTED_SHEETS = (
    "WIN_Market",
    "WIN_Selections",
    "PLACE_Market",
    "PLACE_Selections",
    "README",
)
FORMULA_SCAN_RANGE = "A1:AZ120"


@dataclass(frozen=True)
class DiagnosticResult:
    ok: bool
    messages: tuple[str, ...]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = resolve_path(args.source)
    output = resolve_path(args.output)

    print("Gruss dual-market workbook builder")
    print(f"source={source}")
    print(f"output={output}")
    print("Aucune strategie n'est executee. Aucun ordre reel n'est envoye.")

    if args.diagnose_only:
        return 0 if diagnose_workbook(output, visible=args.visible).ok else 1

    if not source.exists():
        print(f"ERREUR: modele officiel introuvable: {source}")
        return 1

    try:
        build_workbook(source, output, visible=args.visible)
        result = diagnose_workbook(output, visible=args.visible)
    except Exception as exc:
        print(f"ERREUR: {exc}")
        return 1
    return 0 if result.ok else 1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the official Gruss Place_Bets.xls sheets into a WIN+PLACE workbook."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Official Gruss Place_Bets.xls path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output workbook path.")
    parser.add_argument("--visible", action="store_true", help="Show Excel while copying/diagnosing.")
    parser.add_argument("--diagnose-only", action="store_true", help="Only diagnose an existing output workbook.")
    return parser.parse_args(argv)


def build_workbook(source: Path, output: Path, *, visible: bool = False) -> None:
    import xlwings as xw

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    app = xw.App(visible=visible, add_book=False)
    previous_alerts = app.display_alerts
    app.display_alerts = False
    source_book = None
    output_book = None
    try:
        source_book = app.books.open(str(source), update_links=False, read_only=True)
        source_names = sheet_names(source_book)
        require_source_sheet(source_names, "Market")
        require_source_sheet(source_names, "Selections")

        output_book = create_output_from_first_sheet(source_book, "Market", "WIN_Market")
        copy_sheet(source_book, output_book, "Selections", "WIN_Selections")
        copy_sheet(source_book, output_book, "Market", "PLACE_Market")
        copy_sheet(source_book, output_book, "Selections", "PLACE_Selections")
        add_readme_sheet(output_book)

        # Only sheet references are adapted. Cell formulas themselves stay copied from the official model.
        rewrite_sheet_references(output_book.sheets["WIN_Market"], "Selections", "WIN_Selections")
        rewrite_sheet_references(output_book.sheets["PLACE_Market"], "Selections", "PLACE_Selections")

        output_book.api.SaveAs(str(output), FileFormat=56)
    finally:
        if output_book is not None:
            output_book.close()
        if source_book is not None:
            source_book.close()
        app.display_alerts = previous_alerts
        app.quit()


def diagnose_workbook(workbook_path: Path, *, visible: bool = False) -> DiagnosticResult:
    import xlwings as xw

    messages: list[str] = []
    if not workbook_path.exists():
        print(f"ERREUR: workbook introuvable: {workbook_path}")
        return DiagnosticResult(False, ("workbook_missing",))

    app = xw.App(visible=visible, add_book=False)
    previous_alerts = app.display_alerts
    app.display_alerts = False
    book = None
    try:
        book = app.books.open(str(workbook_path), update_links=False, read_only=True)
        try:
            book.app.api.CalculateFullRebuild()
        except Exception:
            book.app.api.Calculate()

        names = sheet_names(book)
        missing = [name for name in EXPECTED_SHEETS if name not in names]
        if missing:
            messages.append(f"missing_sheets={missing}")
        else:
            messages.append("sheets_ok=True")

        circular = safe_get(lambda: book.app.api.CircularReference)
        circular_address = safe_get(lambda: circular.Address(External=True)) if circular is not None else ""
        if circular_address:
            messages.append(f"circular_reference={circular_address}")
        else:
            messages.append("circular_reference=None")

        for sheet_name in ("WIN_Market", "PLACE_Market"):
            sheet = book.sheets[sheet_name]
            headers = nonempty_values(sheet.range("A1:AZ4").value)
            for header in ("Trigger", "Odds", "Stake"):
                messages.append(f"{sheet_name}.header_{header}={contains_text(headers, header)}")
            al5_formula = as_text(safe_get(lambda sheet=sheet: sheet.range("AL5").formula))
            al5_value = safe_get(lambda sheet=sheet: sheet.range("AL5").value)
            messages.append(f"{sheet_name}.AL5_formula={al5_formula!r}")
            messages.append(f"{sheet_name}.AL5_value={al5_value!r}")

        win_formulas = formula_texts(book.sheets["WIN_Market"], FORMULA_SCAN_RANGE)
        place_formulas = formula_texts(book.sheets["PLACE_Market"], FORMULA_SCAN_RANGE)
        messages.append(f"WIN_Market.references_WIN_Selections={any('WIN_Selections' in item for item in win_formulas)}")
        messages.append(f"WIN_Market.references_PLACE_Selections={any('PLACE_Selections' in item for item in win_formulas)}")
        messages.append(f"WIN_Market.references_source_workbook={any('[Gruss Place_Bets.xls]' in item for item in win_formulas)}")
        messages.append(f"PLACE_Market.references_PLACE_Selections={any('PLACE_Selections' in item for item in place_formulas)}")
        messages.append(f"PLACE_Market.references_WIN_Selections={any('WIN_Selections' in item for item in place_formulas)}")
        messages.append(f"PLACE_Market.references_source_workbook={any('[Gruss Place_Bets.xls]' in item for item in place_formulas)}")

        ok = (
            not missing
            and not circular_address
            and all("=True" in message for message in messages if ".header_" in message)
            and "WIN_Market.references_WIN_Selections=True" in messages
            and "WIN_Market.references_PLACE_Selections=False" in messages
            and "WIN_Market.references_source_workbook=False" in messages
            and "PLACE_Market.references_PLACE_Selections=True" in messages
            and "PLACE_Market.references_WIN_Selections=False" in messages
            and "PLACE_Market.references_source_workbook=False" in messages
        )
    finally:
        if book is not None:
            book.close()
        app.display_alerts = previous_alerts
        app.quit()

    print("")
    print("=== Diagnostic ===")
    for message in messages:
        print(message)
    print(f"diagnostic_ok={ok}")
    return DiagnosticResult(ok, tuple(messages))


def create_output_from_first_sheet(source_book: Any, source_sheet_name: str, target_sheet_name: str) -> Any:
    source_sheet = source_book.sheets[source_sheet_name]
    source_sheet.api.Copy()
    output_book = source_book.app.books.active
    output_book.sheets[0].name = target_sheet_name
    return output_book


def copy_sheet(source_book: Any, output_book: Any, source_sheet_name: str, target_sheet_name: str) -> None:
    source_sheet = source_book.sheets[source_sheet_name]
    source_sheet.api.Copy(After=output_book.sheets[-1].api)
    output_book.sheets[-1].name = target_sheet_name


def rewrite_sheet_references(sheet: Any, old_sheet_name: str, new_sheet_name: str) -> None:
    used = sheet.used_range
    formula_cells = safe_get(lambda: used.api.SpecialCells(-4123))  # xlCellTypeFormulas
    if formula_cells is None:
        return

    for cell in formula_cells.Cells:
        formula = as_text(safe_get(lambda cell=cell: cell.Formula))
        updated = replace_sheet_reference(formula, old_sheet_name, new_sheet_name)
        if updated != formula:
            cell.Formula = updated


def replace_sheet_reference(formula: str, old_sheet_name: str, new_sheet_name: str) -> str:
    """Retarget copied official formulas without changing their internal logic."""
    if not formula:
        return formula

    replacements = (
        (f"'{old_sheet_name}'!", f"'{new_sheet_name}'!"),
        (f"{old_sheet_name}!", f"'{new_sheet_name}'!"),
    )
    updated = formula
    for old, new in replacements:
        updated = updated.replace(old, new)

    # Excel copies formulas from the source .xls as external links:
    # 'C:\...\[Gruss Place_Bets.xls]Selections'!$A$2:$E$50
    external_marker = f"]{old_sheet_name}'!"
    marker_at = updated.find(external_marker)
    while marker_at != -1:
        prefix_start = updated.rfind("'", 0, marker_at)
        if prefix_start == -1:
            break
        updated = (
            updated[:prefix_start]
            + f"'{new_sheet_name}'!"
            + updated[marker_at + len(external_marker) :]
        )
        marker_at = updated.find(external_marker)
    return updated


def add_readme_sheet(book: Any) -> None:
    sheet = book.sheets.add("README", after=book.sheets[-1])
    rows = (
        ("dogbot_gruss_WIN_PLACE.xls", ""),
        ("Base", "Ce fichier est base sur Place_Bets.xls officiel."),
        ("WIN_Market", "Marche WIN alimente par Gruss."),
        ("PLACE_Market", "Marche PLACE alimente par Gruss."),
        ("WIN_Selections", "Ordres WIN a preparer par le bot."),
        ("PLACE_Selections", "Ordres PLACE a preparer par le bot."),
        ("Flux", "Selections -> Market formulas -> Trigger/Odds/Stake."),
        ("Securite", "Aucun ordre n'est envoye par ce script de generation."),
    )
    sheet.range("A1").value = rows
    sheet.autofit()


def require_source_sheet(sheet_names_: list[str], required: str) -> None:
    if required not in sheet_names_:
        raise RuntimeError(f"feuille officielle manquante: {required}; feuilles={sheet_names_}")


def formula_texts(sheet: Any, address: str) -> list[str]:
    formulas = sheet.range(address).formula
    result: list[str] = []
    for value in flatten(formulas):
        text = as_text(value)
        if text:
            result.append(text)
    return result


def nonempty_values(values: Any) -> list[str]:
    return [as_text(value) for value in flatten(values) if as_text(value)]


def contains_text(values: list[str], expected: str) -> bool:
    expected_lower = expected.casefold()
    return any(expected_lower in value.casefold() for value in values)


def flatten(values: Any) -> list[Any]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        return [values]
    flattened: list[Any] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened


def sheet_names(book: Any) -> list[str]:
    return [sheet.name for sheet in book.sheets]


def as_text(value: Any) -> str:
    return "" if value is None else str(value)


def safe_get(fn: Any, default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
