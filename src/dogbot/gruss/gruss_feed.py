from __future__ import annotations

from pathlib import Path
from typing import Any

from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH, GrussExcelBridge
from dogbot.gruss.gruss_mapper import (
    GrussSnapshot,
    parse_gruss_sheet,
    require_valid_win_place_pair,
)


class GrussFeed:
    """Minimal read-only feed facade for Gruss Excel sheets."""

    def __init__(self, workbook_path: str | Path = DEFAULT_WORKBOOK_PATH) -> None:
        self.bridge = GrussExcelBridge(workbook_path)

    def connect(self) -> bool:
        return self.bridge.connect()

    def connect_open_workbook(self) -> bool:
        return self.bridge.connect_open_workbook()

    def read_market_sheet(self, market_type: str, rows: int = 80, columns: int = 50) -> list[list[Any]]:
        sheet_name = market_type.upper()
        if sheet_name not in {"WIN", "PLACE"}:
            raise ValueError(f"Unsupported Gruss market sheet: {market_type}")
        return self.bridge.read_sheet(sheet_name, rows=rows, columns=columns)

    def read_snapshot(self, market_type: str, rows: int = 80, columns: int = 50) -> GrussSnapshot:
        sheet_name = market_type.upper()
        sheet_rows = self.read_market_sheet(sheet_name, rows=rows, columns=columns)
        return parse_gruss_sheet(sheet_rows, sheet_name)

    def read_validated_win_place_pair(
        self,
        rows: int = 80,
        columns: int = 50,
    ) -> tuple[GrussSnapshot, GrussSnapshot]:
        win_snapshot = self.read_snapshot("WIN", rows=rows, columns=columns)
        place_snapshot = self.read_snapshot("PLACE", rows=rows, columns=columns)
        require_valid_win_place_pair(win_snapshot, place_snapshot)
        return win_snapshot, place_snapshot
