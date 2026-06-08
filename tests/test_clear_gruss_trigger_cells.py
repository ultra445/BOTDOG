from __future__ import annotations

import csv
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dogbot.gruss.gruss_excel_bridge import GrussExcelBridge
from dogbot.gruss.gruss_real_orders import GrussTriggerLayout
from dogbot.gruss.gruss_trigger_clear import (
    GrussTriggerClearTarget,
    append_trigger_clear_log,
    clear_runner_trigger_cells,
    find_nonempty_runner_trigger_cells,
)
from dogbot.gruss.gruss_trigger_diagnostics import inspect_place_trigger_cells


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "clear_gruss_trigger_cells.py"
SPEC = importlib.util.spec_from_file_location("clear_gruss_trigger_cells", SCRIPT_PATH)
clear_gruss_trigger_cells = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(clear_gruss_trigger_cells)


class FakeBridge:
    def __init__(self) -> None:
        self.runner_values = {
            "WIN": [["1. Win One"], ["2. Win Two"], [None]],
            "PLACE": [["1. Place One"], ["2. Place Two"], [None]],
        }
        self.cells = {
            ("WIN", "Q5"): "BACK",
            ("WIN", "Q6"): None,
            ("PLACE", "Q5"): "LAYSP",
            ("PLACE", "Q6"): "BACKSP",
            ("WIN", "R5"): 2.5,
            ("WIN", "S5"): 2.0,
            ("PLACE", "R5"): 1.8,
            ("PLACE", "S5"): 2.0,
        }
        self.clear_calls = []
        self.write_calls = []

    def read_range(self, sheet_name, address):
        return self.runner_values[sheet_name]

    def read_cell(self, sheet_name, address):
        return self.cells.get((sheet_name, address))

    def clear_trigger_cells(self, sheet_name, addresses, *, trigger_column, allow_clear=False):
        prepared = tuple(addresses)
        self.clear_calls.append((sheet_name, prepared, trigger_column, allow_clear))
        for address in prepared:
            if not address.startswith("Q"):
                raise AssertionError("non-Q clear attempted")
            self.cells[(sheet_name, address)] = None
        return list(prepared)

    def write_cells(self, *args, **kwargs):
        self.write_calls.append((args, kwargs))
        raise AssertionError("ordinary Excel write must never be used")


class FakeCellRange:
    def __init__(self, cells, address):
        self.cells = cells
        self.address = address

    @property
    def value(self):
        return self.cells.get(self.address)

    @value.setter
    def value(self, value):
        self.cells[self.address] = value


class FakeSheet:
    name = "PLACE"

    def __init__(self):
        self.cells = {"Q5": "BACK", "R5": 2.5, "S5": 2.0}

    def range(self, address):
        return FakeCellRange(self.cells, address)


class FakeWorkbook:
    def __init__(self, sheet):
        self.sheets = [sheet]


class ClearGrussTriggerCellsTests(unittest.TestCase):
    def test_preview_is_default_and_never_clears(self) -> None:
        bridge = FakeBridge()
        targets = find_nonempty_runner_trigger_cells(bridge)

        results = clear_runner_trigger_cells(bridge, targets, allow_clear=False)

        self.assertEqual(len(targets), 3)
        self.assertTrue(all(result.status == "WOULD_CLEAR" for result in results))
        self.assertTrue(all(result.mode == "preview" for result in results))
        self.assertTrue(all(result.cleared is False for result in results))
        self.assertEqual(bridge.clear_calls, [])
        self.assertEqual(bridge.write_calls, [])
        self.assertEqual(bridge.cells[("PLACE", "Q6")], "BACKSP")

    def test_actual_clear_only_clears_nonempty_q_runner_cells(self) -> None:
        bridge = FakeBridge()
        targets = find_nonempty_runner_trigger_cells(bridge)

        results = clear_runner_trigger_cells(bridge, targets, allow_clear=True)

        self.assertTrue(all(result.status == "CLEARED" for result in results))
        self.assertTrue(all(result.mode == "real_clear" for result in results))
        self.assertTrue(all(result.cleared is True for result in results))
        self.assertEqual(
            bridge.clear_calls,
            [
                ("WIN", ("Q5",), "Q", True),
                ("PLACE", ("Q5", "Q6"), "Q", True),
            ],
        )
        self.assertEqual(bridge.cells[("WIN", "R5")], 2.5)
        self.assertEqual(bridge.cells[("WIN", "S5")], 2.0)
        self.assertEqual(bridge.cells[("PLACE", "R5")], 1.8)
        self.assertEqual(bridge.cells[("PLACE", "S5")], 2.0)
        self.assertEqual(bridge.write_calls, [])

        inspected = inspect_place_trigger_cells(bridge, GrussTriggerLayout())
        self.assertTrue(all(item.back_trigger_value is None for item in inspected))

    def test_bridge_clear_method_rejects_r_and_s_before_any_write(self) -> None:
        sheet = FakeSheet()
        bridge = GrussExcelBridge()
        bridge.workbook = FakeWorkbook(sheet)

        with self.assertRaisesRegex(PermissionError, "Only trigger-column cells"):
            bridge.clear_trigger_cells(
                "PLACE",
                ["Q5", "R5", "S5"],
                trigger_column="Q",
                allow_clear=True,
            )

        self.assertEqual(sheet.cells, {"Q5": "BACK", "R5": 2.5, "S5": 2.0})

    def test_clear_requires_exact_q_runner_address(self) -> None:
        bridge = FakeBridge()
        unsafe = GrussTriggerClearTarget(
            sheet="PLACE",
            row=5,
            runner="1. Place One",
            trigger_cell="R5",
            previous_value=1.8,
        )

        with self.assertRaisesRegex(PermissionError, "Unsafe trigger clear target"):
            clear_runner_trigger_cells(bridge, [unsafe], allow_clear=True)

        self.assertEqual(bridge.clear_calls, [])

    def test_clear_refuses_non_q_layout(self) -> None:
        bridge = FakeBridge()

        with self.assertRaisesRegex(PermissionError, "restricted to column Q"):
            find_nonempty_runner_trigger_cells(
                bridge,
                layout=GrussTriggerLayout(trigger_column="R"),
            )

    def test_environment_requires_explicit_true(self) -> None:
        self.assertFalse(clear_gruss_trigger_cells.clear_enabled({}))
        self.assertFalse(clear_gruss_trigger_cells.clear_enabled({"DOGBOT_GRUSS_CLEAR_TRIGGERS": "yes"}))
        self.assertTrue(clear_gruss_trigger_cells.clear_enabled({"DOGBOT_GRUSS_CLEAR_TRIGGERS": "true"}))

    def test_refuses_non_visible_workbook_and_missing_sheet(self) -> None:
        class UnsafeBridge:
            def __init__(self, *, visible, sheets):
                self.visible = visible
                self.sheets = set(sheets)

            def connect_open_workbook(self):
                return True

            def is_workbook_visible(self):
                return self.visible

            def has_sheet(self, sheet):
                return sheet in self.sheets

        with self.assertRaisesRegex(RuntimeError, "non visible"):
            clear_gruss_trigger_cells.ensure_open_visible_workbook_and_sheets(
                UnsafeBridge(visible=False, sheets={"WIN", "PLACE"})
            )
        with self.assertRaisesRegex(RuntimeError, "onglet PLACE manquant"):
            clear_gruss_trigger_cells.ensure_open_visible_workbook_and_sheets(
                UnsafeBridge(visible=True, sheets={"WIN"})
            )

    def test_log_records_preview_and_cleared_cells(self) -> None:
        bridge = FakeBridge()
        targets = find_nonempty_runner_trigger_cells(bridge)
        preview = clear_runner_trigger_cells(bridge, targets, allow_clear=False)
        cleared = clear_runner_trigger_cells(bridge, targets, allow_clear=True)

        with TemporaryDirectory() as tmp:
            path = append_trigger_clear_log(Path(tmp) / "clear.csv", [*preview, *cleared])
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["status"], "WOULD_CLEAR")
        self.assertEqual(rows[-1]["status"], "CLEARED")
        self.assertEqual(rows[0]["old_value"], "BACK")
        self.assertEqual(rows[0]["cleared"], "False")
        self.assertEqual(rows[0]["mode"], "preview")
        self.assertEqual(rows[-1]["cleared"], "True")
        self.assertEqual(rows[-1]["mode"], "real_clear")
        self.assertTrue(all(row["trigger_cell"].startswith("Q") for row in rows))


if __name__ == "__main__":
    unittest.main()
