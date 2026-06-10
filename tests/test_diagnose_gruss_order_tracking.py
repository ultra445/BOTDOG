from __future__ import annotations

import importlib.util
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_gruss_order_tracking.py"
SPEC = importlib.util.spec_from_file_location("diagnose_gruss_order_tracking", SCRIPT_PATH)
diagnose_gruss_order_tracking = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = diagnose_gruss_order_tracking
SPEC.loader.exec_module(diagnose_gruss_order_tracking)


class DiagnoseGrussOrderTrackingTests(unittest.TestCase):
    def test_bet_type_and_result_are_mapped_to_side_and_status(self) -> None:
        headers = [
            diagnose_gruss_order_tracking.HeaderCell(
                sheet="PLACE_Selections",
                row=1,
                column=3,
                address="C1",
                value="Bet type",
            ),
            diagnose_gruss_order_tracking.HeaderCell(
                sheet="PLACE_Selections",
                row=1,
                column=6,
                address="F1",
                value="Result",
            ),
        ]

        candidates = diagnose_gruss_order_tracking.candidate_columns_from_headers(headers)

        self.assertIn("side", {candidate.field for candidate in candidates})
        self.assertIn("status", {candidate.field for candidate in candidates})

    def test_missing_unmatched_and_cancel_keeps_real_ladder_blocked(self) -> None:
        headers = [
            diagnose_gruss_order_tracking.HeaderCell("PLACE_Selections", 1, 1, "A1", "Selection"),
            diagnose_gruss_order_tracking.HeaderCell("PLACE_Selections", 1, 2, "B1", "Bet ref"),
            diagnose_gruss_order_tracking.HeaderCell("PLACE_Selections", 1, 3, "C1", "Bet type"),
            diagnose_gruss_order_tracking.HeaderCell("PLACE_Selections", 1, 4, "D1", "Odds"),
            diagnose_gruss_order_tracking.HeaderCell("PLACE_Selections", 1, 5, "E1", "Stake"),
            diagnose_gruss_order_tracking.HeaderCell("PLACE_Selections", 1, 6, "F1", "Result"),
            diagnose_gruss_order_tracking.HeaderCell("PLACE_Selections", 1, 7, "G1", "Matched"),
        ]
        candidates = diagnose_gruss_order_tracking.candidate_columns_from_headers(headers)
        output = StringIO()

        with redirect_stdout(output):
            diagnose_gruss_order_tracking.print_order_tracking_assessment(candidates, [])

        text = output.getvalue()
        self.assertIn("tracking_possible=False", text)
        self.assertIn("cancel_possible=False", text)
        self.assertIn("unmatched_stake", text)
        self.assertIn("cancel_command", text)


if __name__ == "__main__":
    unittest.main()
