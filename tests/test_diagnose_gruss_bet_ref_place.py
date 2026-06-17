from __future__ import annotations

import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_gruss_bet_ref_place.py"
SPEC = importlib.util.spec_from_file_location("diagnose_gruss_bet_ref_place", SCRIPT_PATH)
diagnose_gruss_bet_ref_place = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnose_gruss_bet_ref_place
SPEC.loader.exec_module(diagnose_gruss_bet_ref_place)


VALID_ENV = {
    "DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_PLACE": "true",
    "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
    "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
    "DOGBOT_GRUSS_REAL_TEST_MODE": "true",
    "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": "true",
    "DOGBOT_GRUSS_REAL_MAX_STAKE": "2",
    "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE": "2",
    "DOGBOT_GRUSS_REAL_PREVIEW": "false",
    "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
}


class FakeBridge:
    def __init__(self) -> None:
        self.visible = True
        self.sheets = {"PLACE", "PLACE_Selections", "WIN_Selections"}
        self.cells = {
            ("PLACE", "A5"): "Diagnostic Runner",
            ("PLACE", "Q5"): None,
            ("PLACE", "R5"): None,
            ("PLACE", "S5"): None,
            ("PLACE", "T5"): None,
        }
        self.sheet_data = {
            "PLACE_Selections": [
                ["Selection", "Bet Ref", "Bet type", "Amount", "Average Odds", "Result"],
                ["Old Runner", "123", "B", 2.0, 3.0, "LAPSED"],
            ],
            "WIN_Selections": [
                ["Selection", "Bet Ref", "Bet type"],
                ["Win Old Runner", "456", "L"],
            ],
        }
        self.writes: list[tuple[str, tuple[tuple[str, object], ...], bool]] = []

    def connect_open_workbook(self) -> bool:
        return True

    def is_workbook_visible(self) -> bool:
        return self.visible

    def has_sheet(self, sheet_name: str) -> bool:
        return sheet_name in self.sheets

    def sheet_names(self) -> list[str]:
        return sorted(self.sheets)

    def read_cell(self, sheet_name: str, address: str):
        return self.cells.get((sheet_name, address.upper()))

    def write_cells(self, sheet_name: str, cells, *, allow_write: bool = False):
        plan = tuple((address.upper(), value) for address, value in cells)
        self.writes.append((sheet_name, plan, allow_write))
        if not allow_write:
            raise PermissionError("writes require allow_write=True")
        for address, value in plan:
            self.cells[(sheet_name, address)] = value
        return [address for address, _ in plan]

    def read_sheet(self, sheet_name: str, rows: int = 80, columns: int = 50):
        values = self.sheet_data.get(sheet_name, [])
        padded = []
        for row in values[:rows]:
            padded.append([*row[:columns], *([None] * max(0, columns - len(row)))])
        while len(padded) < rows:
            padded.append([None] * columns)
        return padded


class FakeComRejected(Exception):
    def __init__(self) -> None:
        super().__init__(-2147418111, "L'appel a ete rejete par l'appele.")


class FlakyBetRefBridge(FakeBridge):
    def __init__(self) -> None:
        super().__init__()
        self.reject_t5_once = True

    def read_cell(self, sheet_name: str, address: str):
        if sheet_name == "PLACE" and address.upper() == "T5" and self.writes and self.reject_t5_once:
            self.reject_t5_once = False
            self.cells[("PLACE", "T5")] = "432049736284"
            raise FakeComRejected()
        return super().read_cell(sheet_name, address)


class DiagnoseGrussBetRefPlaceTests(TestCase):
    def test_environment_refuses_unarmed_real_diagnostic(self) -> None:
        env = dict(VALID_ENV)
        env["DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_PLACE"] = "false"
        with self.assertRaisesRegex(RuntimeError, "BET_REF_DIAGNOSTIC_PLACE=true"):
            diagnose_gruss_bet_ref_place.validate_environment(env)

    def test_environment_requires_safe_real_test_caps(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_MAX_STAKE="3")
        with self.assertRaisesRegex(RuntimeError, "REAL_MAX_STAKE"):
            diagnose_gruss_bet_ref_place.validate_environment(env)

    def test_config_rejects_update_and_non_two_euro_stake(self) -> None:
        with self.assertRaisesRegex(ValueError, "BACK ou LAY"):
            diagnose_gruss_bet_ref_place.config_from_args(
                _args(side="UPDATE", price=2.2, stake=2.0)
            )
        with self.assertRaisesRegex(ValueError, "exactement 2"):
            diagnose_gruss_bet_ref_place.config_from_args(
                _args(side="BACK", price=2.2, stake=1.0)
            )

    def test_writes_one_place_order_and_dumps_tracking_files(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp:
            config = diagnose_gruss_bet_ref_place.DiagnosticConfig(
                row=5,
                side="BACK",
                price=2.2,
                stake=2.0,
                seconds=1,
                interval_seconds=1.0,
                selection_rows=200,
                selection_columns=8,
                output_dir=Path(tmp),
                manual_note="open bet not visible in manual check",
                no_manual_prompt=True,
            )
            session_dir = diagnose_gruss_bet_ref_place.run_diagnostic(
                bridge,
                config,
                sleep_fn=lambda _: None,
                now_fn=lambda: datetime(2026, 6, 12, 18, 0, 0),
            )
            self.assertEqual(
                bridge.writes,
                [
                    (
                        "PLACE",
                        (("R5", 2.2), ("S5", 2.0), ("Q5", "BACK")),
                        True,
                    )
                ],
            )
            self.assertEqual(bridge.cells[("PLACE", "Q5")], "BACK")
            self.assertEqual(bridge.cells[("PLACE", "R5")], 2.2)
            self.assertEqual(bridge.cells[("PLACE", "S5")], 2.0)
            self.assertTrue((session_dir / "ticks.csv").exists())
            self.assertEqual(
                (session_dir / "manual_open_bets_note.txt").read_text(encoding="utf-8"),
                "open bet not visible in manual check",
            )

            with (session_dir / "ticks.csv").open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["place_q"], "BACK")
            self.assertEqual(rows[0]["place_r"], "2.2")
            self.assertEqual(rows[0]["place_s"], "2.0")
            self.assertIn("PLACE_Selections", rows[0]["workbook_sheet_names"])
            self.assertTrue(Path(rows[0]["all_sheet_dumps_dir"]).exists())
            self.assertTrue(Path(rows[0]["place_selections_csv"]).exists())
            self.assertTrue(Path(rows[0]["win_selections_csv"]).exists())
            summary = json.loads((session_dir / "summary_final.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["bet_ref_found"])
            self.assertEqual(summary["ticks_written"], 2)

    def test_com_rejected_tick_is_logged_and_next_tick_can_find_bet_ref(self) -> None:
        bridge = FlakyBetRefBridge()
        with TemporaryDirectory() as tmp:
            config = diagnose_gruss_bet_ref_place.DiagnosticConfig(
                row=5,
                side="BACK",
                price=20.0,
                stake=2.0,
                seconds=1,
                interval_seconds=1.0,
                selection_rows=200,
                selection_columns=8,
                output_dir=Path(tmp),
                manual_note="open bet visible manually",
                no_manual_prompt=True,
            )
            session_dir = diagnose_gruss_bet_ref_place.run_diagnostic(
                bridge,
                config,
                sleep_fn=lambda _: None,
                now_fn=lambda: datetime(2026, 6, 12, 18, 1, 0),
            )

            with (session_dir / "ticks.csv").open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("COM_REJECTED", rows[0]["read_errors"])
            self.assertEqual(rows[0]["com_errors_count"], "1")
            self.assertEqual(rows[1]["place_t"], "432049736284")
            self.assertEqual(rows[1]["bet_ref_found"], "True")
            self.assertEqual(rows[1]["bet_ref_source"], "ROW_T")

            summary = json.loads((session_dir / "summary_final.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["bet_ref_found"])
            self.assertEqual(summary["first_tick_with_bet_ref"], 1)
            self.assertEqual(summary["bet_ref_source"], "ROW_T")
            self.assertEqual(summary["bet_ref_value"], "432049736284")
            self.assertEqual(summary["com_errors_count"], 1)


def _args(*, side: str, price: float, stake: float):
    class Args:
        runner_row = 5
        seconds = 20
        interval = 1.0
        selection_rows = 200
        selection_columns = 80
        output_dir = Path("data")
        manual_note = ""
        no_manual_prompt = True

    args = Args()
    args.side = side
    args.price = price
    args.stake = stake
    return args
