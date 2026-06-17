from __future__ import annotations

import csv
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_gruss_backr_layr_place.py"
SPEC = importlib.util.spec_from_file_location("diagnose_gruss_backr_layr_place", SCRIPT_PATH)
diagnose_gruss_backr_layr_place = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnose_gruss_backr_layr_place
assert SPEC.loader is not None
SPEC.loader.exec_module(diagnose_gruss_backr_layr_place)


VALID_ENV = {
    "DOGBOT_GRUSS_BACKR_LAYR_DIAGNOSTIC_PLACE": "true",
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
    def __init__(self, *, matched_stake=0.0, suffix_after_initial: bool = False) -> None:
        self.visible = True
        self.sheets = {"PLACE", "PLACE_Selections", "WIN_Selections"}
        self.cells = {
            ("PLACE", "A5"): "Diagnostic Runner",
            ("PLACE", "Q5"): None,
            ("PLACE", "R5"): None,
            ("PLACE", "S5"): None,
            ("PLACE", "T5"): None,
            ("PLACE", "W5"): matched_stake,
            ("PLACE", "D2"): "00:00:20",
            ("PLACE", "E2"): "Active",
            ("PLACE", "F2"): "",
        }
        self.suffix_after_initial = suffix_after_initial
        self.sheet_data = {
            "PLACE_Selections": [["Selection", "Bet Ref"], ["Diagnostic Runner", "432049736284"]],
            "WIN_Selections": [["Selection", "Bet Ref"]],
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
            raise PermissionError("allow_write required")
        for address, value in plan:
            self.cells[(sheet_name, address)] = value
        trigger = dict(plan).get("Q5")
        if trigger in {"BACK", "LAY"}:
            self.cells[(sheet_name, "T5")] = "432049736284N" if self.suffix_after_initial else "432049736284"
        elif trigger in {"BACKR", "LAYR"}:
            self.cells[(sheet_name, "T5")] = "432049736285N"
        return [address for address, _ in plan]

    def read_sheet(self, sheet_name: str, rows: int = 80, columns: int = 50):
        values = self.sheet_data.get(sheet_name, [])
        padded = []
        for row in values[:rows]:
            padded.append([*row[:columns], *([None] * max(0, columns - len(row)))])
        while len(padded) < rows:
            padded.append([None] * columns)
        return padded


class DiagnoseGrussBackrLayrPlaceTests(TestCase):
    def test_environment_requires_dedicated_arm_flag(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_BACKR_LAYR_DIAGNOSTIC_PLACE="false")
        with self.assertRaisesRegex(RuntimeError, "BACKR_LAYR_DIAGNOSTIC_PLACE=true"):
            diagnose_gruss_backr_layr_place.validate_environment(env)

    def test_back_order_waits_for_bet_ref_then_writes_backr(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp:
            config = diagnose_gruss_backr_layr_place.BackrLayrDiagnosticConfig(
                row=5,
                side="BACK",
                initial_price=3.0,
                replace_price=3.2,
                stake=2.0,
                seconds=1,
                interval_seconds=1.0,
                after_replace_ticks=1,
                selection_rows=200,
                selection_columns=8,
                output_dir=Path(tmp),
                manual_note="one open order, odds changed",
                no_manual_prompt=True,
            )
            session_dir = diagnose_gruss_backr_layr_place.run_diagnostic(
                bridge,
                config,
                sleep_fn=lambda _: None,
                now_fn=lambda: datetime(2026, 6, 12, 18, 2, 0),
            )
            with (session_dir / "summary.csv").open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(bridge.writes[0][1], (("R5", 3.0), ("S5", 2.0), ("Q5", "BACK")))
        self.assertEqual(bridge.writes[1][1], (("R5", 3.2), ("S5", 2.0), ("Q5", "BACKR")))
        self.assertEqual(rows[1]["trigger"], "BACKR")
        self.assertEqual(rows[1]["reason"], "replace_written")
        self.assertEqual(rows[1]["result"], "written")
        self.assertEqual(rows[1]["replace_attempted"], "True")

    def test_float_bet_ref_from_excel_allows_backr(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "T5")] = 432237671913.0
        decision = diagnose_gruss_backr_layr_place.maybe_write_replace(
            bridge,
            _config(),
            432237671913.0,
            48.0,
        )

        self.assertEqual(decision["result"], "written")
        self.assertEqual(decision["reason"], "replace_written")
        self.assertEqual(decision["bet_ref_before"], "432237671913")
        self.assertEqual(bridge.writes[0][1], (("R5", 48.0), ("S5", 2.0), ("Q5", "BACKR")))

    def test_float_bet_ref_with_n_suffix_is_stripped_before_backr(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "T5")] = "432237671913.0N"
        decision = diagnose_gruss_backr_layr_place.maybe_write_replace(
            bridge,
            _config(),
            "432237671913.0N",
            48.0,
        )

        self.assertEqual(decision["result"], "written")
        self.assertEqual(decision["bet_ref_before"], "432237671913N")
        self.assertEqual(decision["bet_ref_suffix_n_handled"], True)
        self.assertEqual(
            bridge.writes[0][1],
            (("T5", "432237671913"), ("R5", 48.0), ("S5", 2.0), ("Q5", "BACKR")),
        )

    def test_initial_write_is_refused_when_market_is_suspended(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "F2")] = "Suspended"
        with TemporaryDirectory() as tmp:
            config = _config(output_dir=Path(tmp))

            with self.assertRaisesRegex(RuntimeError, "marche suspendu avant ordre initial"):
                diagnose_gruss_backr_layr_place.run_diagnostic(
                    bridge,
                    config,
                    sleep_fn=lambda _: None,
                    now_fn=lambda: datetime(2026, 6, 12, 18, 2, 0),
                )

        self.assertEqual(bridge.writes, [])

    def test_initial_write_is_refused_when_countdown_is_unreadable(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "D2")] = ""
        with TemporaryDirectory() as tmp:
            config = _config(output_dir=Path(tmp))

            with self.assertRaisesRegex(RuntimeError, "countdown illisible avant ordre initial"):
                diagnose_gruss_backr_layr_place.run_diagnostic(
                    bridge,
                    config,
                    sleep_fn=lambda _: None,
                    now_fn=lambda: datetime(2026, 6, 12, 18, 2, 0),
                )

        self.assertEqual(bridge.writes, [])

    def test_initial_write_is_refused_when_countdown_lte_ten(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "D2")] = "00:00:10"
        with TemporaryDirectory() as tmp:
            config = _config(output_dir=Path(tmp))

            with self.assertRaisesRegex(RuntimeError, "countdown trop bas avant ordre initial"):
                diagnose_gruss_backr_layr_place.run_diagnostic(
                    bridge,
                    config,
                    sleep_fn=lambda _: None,
                    now_fn=lambda: datetime(2026, 6, 12, 18, 2, 0),
                )

        self.assertEqual(bridge.writes, [])

    def test_suspended_market_blocks_replace_even_with_numeric_bet_ref(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "T5")] = "432049736284"
        bridge.cells[("PLACE", "F2")] = "Suspended"
        decision = diagnose_gruss_backr_layr_place.maybe_write_replace(
            bridge,
            _config(),
            "432049736284",
            3.2,
        )

        self.assertEqual(decision["result"], "skipped")
        self.assertEqual(decision["reason"], "market_suspended_no_replace")
        self.assertEqual(decision["replace_attempted"], False)
        self.assertEqual(
            diagnose_gruss_backr_layr_place.build_final_summary([], [decision])["replace_attempted"],
            False,
        )
        self.assertEqual(bridge.writes, [])

    def test_countdown_lte_ten_blocks_replace(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "T5")] = "432049736284"
        bridge.cells[("PLACE", "D2")] = "00:00:10"
        decision = diagnose_gruss_backr_layr_place.maybe_write_replace(
            bridge,
            _config(),
            "432049736284",
            3.2,
        )

        self.assertEqual(decision["result"], "skipped")
        self.assertEqual(decision["reason"], "seconds_until_start_lte_10_no_replace")
        self.assertEqual(decision["replace_attempted"], False)
        self.assertEqual(bridge.writes, [])

    def test_cancelled_t_cell_blocks_replace_even_if_scanned_ref_is_numeric(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "T5")] = "CANCELLED"
        decision = diagnose_gruss_backr_layr_place.maybe_write_replace(
            bridge,
            _config(),
            "432049736284",
            3.2,
        )

        self.assertEqual(decision["result"], "skipped")
        self.assertEqual(decision["reason"], "row_status_not_replaceable")
        self.assertEqual(decision["replace_attempted"], False)
        self.assertEqual(
            diagnose_gruss_backr_layr_place.build_final_summary([], [decision])["replace_attempted"],
            False,
        )
        self.assertEqual(bridge.writes, [])

    def test_cancelled_t_cell_keeps_terminal_reason_even_if_market_suspended(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "T5")] = "CANCELLED"
        bridge.cells[("PLACE", "F2")] = "Suspended"
        decision = diagnose_gruss_backr_layr_place.maybe_write_replace(
            bridge,
            _config(),
            "432049736284",
            3.2,
        )

        self.assertEqual(decision["result"], "skipped")
        self.assertEqual(decision["reason"], "row_status_not_replaceable")
        self.assertEqual(decision["replace_attempted"], False)
        self.assertEqual(bridge.writes, [])

    def test_blank_t_cell_blocks_replace_even_if_scanned_ref_is_numeric(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "T5")] = ""
        decision = diagnose_gruss_backr_layr_place.maybe_write_replace(
            bridge,
            _config(),
            "432049736284",
            3.2,
        )

        self.assertEqual(decision["result"], "skipped")
        self.assertEqual(decision["reason"], "bet_ref_not_ready")
        self.assertEqual(decision["replace_attempted"], False)
        self.assertEqual(bridge.writes, [])

    def test_lay_side_uses_layr(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "T5")] = "432049736284"
        decision = diagnose_gruss_backr_layr_place.maybe_write_replace(
            bridge,
            diagnose_gruss_backr_layr_place.BackrLayrDiagnosticConfig(
                row=5,
                side="LAY",
                initial_price=6.0,
                replace_price=6.2,
                stake=2.0,
                seconds=1,
                interval_seconds=1.0,
                after_replace_ticks=0,
                selection_rows=200,
                selection_columns=8,
                output_dir=Path("."),
            ),
            "432049736284",
            6.2,
        )

        self.assertEqual(decision["trigger"], "LAYR")
        self.assertEqual(bridge.writes[0][1], (("R5", 6.2), ("S5", 2.0), ("Q5", "LAYR")))

    def test_matched_stake_positive_blocks_replace(self) -> None:
        bridge = FakeBridge(matched_stake=1.0)
        bridge.cells[("PLACE", "T5")] = "432049736284"
        decision = diagnose_gruss_backr_layr_place.maybe_write_replace(
            bridge,
            _config(),
            "432049736284",
            3.2,
        )

        self.assertEqual(decision["result"], "skipped")
        self.assertEqual(decision["reason"], "matched_stake_positive_no_replace")
        self.assertEqual(bridge.writes, [])

    def test_suffix_n_is_stripped_before_next_replace(self) -> None:
        bridge = FakeBridge()
        bridge.cells[("PLACE", "T5")] = "432049736284N"
        decision = diagnose_gruss_backr_layr_place.maybe_write_replace(
            bridge,
            _config(),
            "432049736284N",
            3.2,
        )

        self.assertEqual(decision["result"], "written")
        self.assertEqual(decision["bet_ref_suffix_n_handled"], True)
        self.assertEqual(
            bridge.writes[0][1],
            (("T5", "432049736284"), ("R5", 3.2), ("S5", 2.0), ("Q5", "BACKR")),
        )


def _config(*, output_dir=Path(".")):
    return diagnose_gruss_backr_layr_place.BackrLayrDiagnosticConfig(
        row=5,
        side="BACK",
        initial_price=3.0,
        replace_price=3.2,
        stake=2.0,
        seconds=1,
        interval_seconds=1.0,
        after_replace_ticks=0,
        selection_rows=200,
        selection_columns=8,
        output_dir=output_dir,
    )
