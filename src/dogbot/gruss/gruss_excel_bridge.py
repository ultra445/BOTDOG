from __future__ import annotations

import csv
import re
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
    """Access to the Excel workbook populated by Gruss.

    Reads are available by default. Writes require an explicit per-call
    ``allow_write=True`` guard and are intended only for the separately armed
    Gruss real-order provider.
    """

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

    def read_cell(self, sheet_name: str, address: str) -> Any:
        sheet = self.get_sheet(sheet_name)
        return sheet.range(address).value

    def is_workbook_visible(self) -> bool:
        workbook = self._require_workbook()
        try:
            return bool(workbook.app.visible)
        except Exception:
            return False

    def write_cells(
        self,
        sheet_name: str,
        cells: Iterable[tuple[str, Any]],
        *,
        allow_write: bool = False,
    ) -> list[str]:
        """Write cells sequentially only after an explicit safety opt-in."""
        if not allow_write:
            raise PermissionError("Excel writes require allow_write=True")
        sheet = self.get_sheet(sheet_name)
        written: list[str] = []
        for address, value in cells:
            sheet.range(address).value = value
            written.append(address)
        return written

    def write_cells_without_trigger(
        self,
        sheet_name: str,
        cells: Iterable[tuple[str, Any]],
        *,
        trigger_address: str,
        allow_write: bool = False,
    ) -> list[str]:
        """Write preparation cells while materially rejecting the trigger cell."""
        prepared = tuple(cells)
        forbidden = str(trigger_address).strip().upper()
        if any(str(address).strip().upper() == forbidden for address, _ in prepared):
            raise PermissionError(f"Trigger cell write forbidden: {trigger_address}")
        if not allow_write:
            raise PermissionError("Excel writes require allow_write=True")
        sheet = self.get_sheet(sheet_name)
        written: list[str] = []
        for address, value in prepared:
            if sheet.range(trigger_address).value not in (None, ""):
                raise PermissionError(f"Trigger cell is not empty: {trigger_address}")
            sheet.range(address).value = value
            written.append(address)
        return written

    def clear_trigger_cells(
        self,
        sheet_name: str,
        addresses: Iterable[str],
        *,
        trigger_column: str,
        allow_clear: bool = False,
    ) -> list[str]:
        """Clear only validated trigger-column cells after explicit opt-in."""
        prepared = tuple(str(address).strip().upper() for address in addresses)
        column = str(trigger_column).strip().upper()
        pattern = re.compile(rf"^{re.escape(column)}[1-9][0-9]*$")
        if not prepared:
            return []
        if not column.isalpha() or any(not pattern.fullmatch(address) for address in prepared):
            raise PermissionError("Only trigger-column cells may be cleared")
        if not allow_clear:
            raise PermissionError("Trigger clearing requires allow_clear=True")

        sheet = self.get_sheet(sheet_name)
        cleared: list[str] = []
        for address in prepared:
            sheet.range(address).value = None
            cleared.append(address)
        return cleared

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
