from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WORKBOOK_PATH = Path(r"C:\betfair-dogbot\gruss_bridge\dogbot_gruss.xlsx")


@dataclass(frozen=True)
class CsvDumpResult:
    sheet_name: str
    output_path: Path
    rows_dumped: int
    columns_dumped: int


class GrussExcelBridge:
    """Read-only access to the Excel workbook populated by Gruss."""

    def __init__(self, workbook_path: str | Path = DEFAULT_WORKBOOK_PATH) -> None:
        self.workbook_path = Path(workbook_path)
        self.workbook = None
        self.workbook_found_open = False

    def connect(self) -> bool:
        """Connect to an already opened workbook, or open it if needed.

        Returns True when an already opened workbook was found, False when the
        bridge had to open the workbook path itself.
        """
        xw = self._xlwings()
        target = self._normalise_path(self.workbook_path)

        for app in xw.apps:
            for book in app.books:
                book_path = self._book_full_name(book)
                if book_path and self._normalise_path(book_path) == target:
                    self.workbook = book
                    self.workbook_found_open = True
                    return True

        self.workbook = xw.Book(str(self.workbook_path))
        self.workbook_found_open = False
        return False

    def connect_open_workbook(self) -> bool:
        """Connect only to an already opened workbook.

        This is intended for live Gruss diagnostics where silently opening a
        workbook could hide a bad Excel/Gruss setup.
        """
        xw = self._xlwings()
        target = self._normalise_path(self.workbook_path)

        if len(xw.apps) == 0:
            raise RuntimeError("Microsoft Excel is not open or is not visible through COM")

        for app in xw.apps:
            for book in app.books:
                book_path = self._book_full_name(book)
                if book_path and self._normalise_path(book_path) == target:
                    self.workbook = book
                    self.workbook_found_open = True
                    return True

        raise RuntimeError(f"Workbook is not open: {self.workbook_path}")

    def has_sheet(self, sheet_name: str) -> bool:
        workbook = self._require_workbook()
        return any(sheet.name.upper() == sheet_name.upper() for sheet in workbook.sheets)

    def get_sheet(self, sheet_name: str) -> Any:
        workbook = self._require_workbook()
        for sheet in workbook.sheets:
            if sheet.name.upper() == sheet_name.upper():
                return sheet
        raise ValueError(f"Excel sheet not found: {sheet_name}")

    def read_sheet(self, sheet_name: str, rows: int = 80, columns: int = 50) -> list[list[Any]]:
        sheet = self.get_sheet(sheet_name)
        return self._read_rect(sheet, rows, columns)

    def read_range(self, sheet_name: str, address: str) -> list[list[Any]]:
        sheet = self.get_sheet(sheet_name)
        values = sheet.range(address).value
        return self._as_matrix(values)

    def export_csv_diagnostic(
        self,
        sheet_name: str,
        output_path: str | Path,
        rows: int = 80,
        columns: int = 50,
    ) -> CsvDumpResult:
        data = self.read_sheet(sheet_name, rows=rows, columns=columns)
        csv_path = Path(output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerows(data)

        return CsvDumpResult(
            sheet_name=sheet_name,
            output_path=csv_path,
            rows_dumped=len(data),
            columns_dumped=max((len(row) for row in data), default=0),
        )

    def _read_rect(self, sheet: Any, rows: int, columns: int) -> list[list[Any]]:
        if rows <= 0 or columns <= 0:
            return []
        values = sheet.range((1, 1), (rows, columns)).value
        matrix = self._as_matrix(values)
        return [self._pad_row(row, columns) for row in matrix[:rows]]

    def _require_workbook(self) -> Any:
        if self.workbook is None:
            self.connect()
        return self.workbook

    @staticmethod
    def _xlwings() -> Any:
        try:
            import xlwings as xw
        except ImportError as exc:
            raise RuntimeError(
                "xlwings is required to read Gruss Excel sheets. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        return xw

    @staticmethod
    def _book_full_name(book: Any) -> str | None:
        try:
            return str(book.fullname)
        except Exception:
            return None

    @staticmethod
    def _normalise_path(path: str | Path) -> str:
        return str(Path(path).resolve()).casefold()

    @staticmethod
    def _as_matrix(values: Any) -> list[list[Any]]:
        if values is None:
            return [[None]]
        if not isinstance(values, list):
            return [[values]]
        if not values:
            return []
        if all(not isinstance(item, list) for item in values):
            return [values]
        return [row if isinstance(row, list) else [row] for row in values]

    @staticmethod
    def _pad_row(row: Iterable[Any], columns: int) -> list[Any]:
        padded = list(row)
        if len(padded) < columns:
            padded.extend([None] * (columns - len(padded)))
        return padded[:columns]
