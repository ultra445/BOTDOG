from __future__ import annotations

import unittest

from dogbot.gruss.gruss_real_orders import GrussTriggerLayout
from dogbot.gruss.gruss_trigger_diagnostics import inspect_place_trigger_cells


class ReadOnlyBridge:
    def __init__(self) -> None:
        self.read_calls = []
        self.write_calls = []

    def read_range(self, sheet_name, address):
        self.read_calls.append((sheet_name, address))
        return [["1. First Runner"], ["2. Second Runner"], [None]]

    def read_cell(self, sheet_name, address):
        self.read_calls.append((sheet_name, address))
        return {"Q5": None, "Q6": "BACKSP"}[address]

    def write_cells(self, *args, **kwargs):
        self.write_calls.append((args, kwargs))
        raise AssertionError("diagnostic must never write Excel")


class InspectGrussTriggerCellsTests(unittest.TestCase):
    def test_backsp_mapping_returns_expected_address(self) -> None:
        layout = GrussTriggerLayout()

        self.assertEqual(layout.trigger_mapping_name("BACK", "SP_MOC"), "BACKSP")
        self.assertEqual(layout.trigger_address(6), "Q6")
        self.assertEqual(layout.odds_address(6), "R6")
        self.assertEqual(layout.stake_address(6), "S6")

    def test_diagnostic_reads_configured_cells_and_never_writes(self) -> None:
        bridge = ReadOnlyBridge()

        rows = inspect_place_trigger_cells(bridge, GrussTriggerLayout())

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1].runner, "2. Second Runner")
        self.assertEqual(rows[1].back_trigger_cell, "Q6")
        self.assertEqual(rows[1].lay_trigger_cell, "Q6")
        self.assertEqual(rows[1].backsp_trigger_cell, "Q6")
        self.assertEqual(rows[1].laysp_trigger_cell, "Q6")
        self.assertEqual(rows[1].backsp_trigger_value, "BACKSP")
        self.assertEqual(bridge.write_calls, [])


if __name__ == "__main__":
    unittest.main()
