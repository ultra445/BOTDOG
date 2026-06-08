from __future__ import annotations

import argparse
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    ROOT / "gruss_bridge" / "dogbot_gruss_official_template.xlsx",
    ROOT / "dogbot_gruss_official_template.xlsx",
    ROOT / "gruss_bridge" / "dogbot_gruss.xlsx",
)
MAIN_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
QUOTED_REF_PATTERN = re.compile(
    r"'(?P<sheet>[^']+)'!(?P<cell>\$?[A-Z]{1,3}\$?[1-9][0-9]*)"
)
LOCAL_REF_PATTERN = re.compile(r"(?<![A-Za-z0-9_!])(?P<cell>\$?[A-Z]{1,3}\$?[1-9][0-9]*)")


@dataclass(frozen=True)
class FormulaCell:
    sheet: str
    cell: str
    formula: str

    @property
    def label(self) -> str:
        return f"{self.sheet}!{self.cell}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workbook_path = resolve_workbook_path(args.workbook)
    if workbook_path is None:
        print("ERREUR: workbook introuvable.")
        print("Chemins essayes:")
        for candidate in DEFAULT_CANDIDATES:
            print(f"- {candidate}")
        return 1

    print("Excel circular-reference diagnostic")
    print(f"workbook={workbook_path}")
    print("READ ONLY: aucune cellule n'est modifiee, aucun ordre Gruss n'est envoye.")

    excel_refs: list[str] = []
    if not args.no_excel:
        excel_refs = circular_refs_from_excel(workbook_path, visible=args.visible)
    else:
        print("Excel COM check skipped (--no-excel).")

    formulas = read_formula_cells(workbook_path)
    direct_refs = find_direct_self_references(formulas)
    graph_cycles = find_formula_graph_cycles(formulas)

    print("")
    print("=== Excel reported circular references ===")
    if excel_refs:
        for ref in excel_refs:
            print(ref)
    else:
        print("None reported by Excel COM.")

    print("")
    print("=== Static direct self-references ===")
    if direct_refs:
        for item in direct_refs:
            print(f"{item.label}: ={item.formula}")
    else:
        print("None.")

    print("")
    print("=== Static formula graph cycles ===")
    if graph_cycles:
        for cycle in graph_cycles:
            print(" -> ".join(cycle))
    else:
        print("None.")

    has_circular = bool(excel_refs or direct_refs or graph_cycles)
    print("")
    print(f"circular_refs_found={has_circular}")
    return 1 if has_circular else 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report Excel circular references in the Gruss workbook/template."
    )
    parser.add_argument(
        "workbook",
        nargs="?",
        help="Path to dogbot_gruss_official_template.xlsx or another Gruss workbook.",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Skip the Excel COM/xlwings check and run only static XLSX inspection.",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Open Excel visibly for the COM check. Default is hidden when the workbook is not already open.",
    )
    return parser.parse_args(argv)


def resolve_workbook_path(raw_path: str | None) -> Path | None:
    if raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve() if path.exists() else None
    for candidate in DEFAULT_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return None


def circular_refs_from_excel(workbook_path: Path, *, visible: bool = False) -> list[str]:
    try:
        import xlwings as xw
    except ImportError as exc:
        print(f"Excel COM check unavailable: xlwings missing ({exc}).")
        return []

    target = str(workbook_path.resolve()).casefold()
    opened_here = False
    app = None
    book = None
    previous_alerts = None
    try:
        for candidate_app in xw.apps:
            for candidate_book in candidate_app.books:
                full_name = safe_get(lambda candidate_book=candidate_book: candidate_book.fullname)
                if full_name and str(Path(full_name).resolve()).casefold() == target:
                    app = candidate_app
                    book = candidate_book
                    break
            if book is not None:
                break

        if book is None:
            app = xw.App(visible=visible, add_book=False)
            opened_here = True
            previous_alerts = safe_get(lambda: app.display_alerts, True)
            app.display_alerts = False
            book = app.books.open(str(workbook_path), update_links=False, read_only=True)

        try:
            book.app.api.CalculateFullRebuild()
        except Exception:
            try:
                book.app.api.Calculate()
            except Exception:
                pass

        circular = safe_get(lambda: book.app.api.CircularReference)
        if circular is None:
            return []
        address = safe_get(lambda: circular.Address(External=True))
        return [str(address)] if address else []
    except Exception as exc:
        print(f"Excel COM check failed: {exc}")
        return []
    finally:
        if opened_here and book is not None:
            try:
                book.close()
            except Exception:
                pass
        if opened_here and app is not None:
            if previous_alerts is not None:
                try:
                    app.display_alerts = previous_alerts
                except Exception:
                    pass
            try:
                app.quit()
            except Exception:
                pass


def read_formula_cells(workbook_path: Path) -> list[FormulaCell]:
    formulas: list[FormulaCell] = []
    with zipfile.ZipFile(workbook_path) as archive:
        sheet_paths = workbook_sheet_paths(archive)
        for sheet_name, sheet_path in sheet_paths:
            root = ET.fromstring(archive.read(sheet_path))
            for cell in root.findall(".//main:c", MAIN_NS):
                formula = cell.find("main:f", MAIN_NS)
                if formula is None or not formula.text:
                    continue
                formulas.append(
                    FormulaCell(
                        sheet=sheet_name,
                        cell=normal_cell_ref(cell.attrib.get("r", "")),
                        formula=formula.text,
                    )
                )
    return formulas


def workbook_sheet_paths(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    paths: list[tuple[str, str]] = []
    for sheet in workbook.find("main:sheets", MAIN_NS):
        target = rid_to_target[sheet.attrib[REL_ID]]
        sheet_path = "xl/" + target if not target.startswith("/") else target[1:]
        paths.append((sheet.attrib["name"], sheet_path))
    return paths


def find_direct_self_references(formulas: Iterable[FormulaCell]) -> list[FormulaCell]:
    result: list[FormulaCell] = []
    for item in formulas:
        refs = formula_refs(item.formula, default_sheet=item.sheet)
        if (item.sheet, item.cell) in refs:
            result.append(item)
    return result


def find_formula_graph_cycles(formulas: Iterable[FormulaCell]) -> list[list[str]]:
    formula_map = {(item.sheet, item.cell): item for item in formulas}
    graph: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for key, item in formula_map.items():
        graph[key] = {
            ref
            for ref in formula_refs(item.formula, default_sheet=item.sheet)
            if ref in formula_map
        }

    cycles: list[list[str]] = []
    seen_keys: set[tuple[tuple[str, str], ...]] = set()

    def visit(node: tuple[str, str], stack: list[tuple[str, str]], visiting: set[tuple[str, str]]) -> None:
        if node in visiting:
            start = stack.index(node)
            cycle = stack[start:] + [node]
            canonical = canonical_cycle(cycle)
            if canonical not in seen_keys:
                seen_keys.add(canonical)
                cycles.append([format_key(item) for item in cycle])
            return
        visiting.add(node)
        stack.append(node)
        for child in graph.get(node, ()):
            visit(child, stack, visiting)
        stack.pop()
        visiting.remove(node)

    for node in graph:
        visit(node, [], set())
    return cycles


def formula_refs(formula: str, *, default_sheet: str) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = set()
    formula_without_quoted_refs = formula
    for match in QUOTED_REF_PATTERN.finditer(formula):
        sheet = match.group("sheet")
        cell = normal_cell_ref(match.group("cell"))
        refs.add((sheet, cell))
        formula_without_quoted_refs = formula_without_quoted_refs.replace(match.group(0), "")

    for match in LOCAL_REF_PATTERN.finditer(formula_without_quoted_refs):
        refs.add((default_sheet, normal_cell_ref(match.group("cell"))))
    return refs


def canonical_cycle(cycle: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    body = cycle[:-1]
    rotations = [tuple(body[index:] + body[:index]) for index in range(len(body))]
    return min(rotations)


def normal_cell_ref(cell: str) -> str:
    return cell.replace("$", "").upper()


def format_key(key: tuple[str, str]) -> str:
    return f"{key[0]}!{key[1]}"


def safe_get(fn: Any, default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


if __name__ == "__main__":
    raise SystemExit(main())
