from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gruss_manual_visible_write.py"
SPEC = importlib.util.spec_from_file_location("gruss_manual_visible_write_script", SCRIPT_PATH)
test_gruss_manual_visible_write = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(test_gruss_manual_visible_write)


class FakeBridge:
    def __init__(self, *, trigger_value=None) -> None:
        self.cells = {"Q5": trigger_value, "R5": 2.5, "S5": 3.0}
        self.write_calls = []

    def read_cell(self, sheet_name, address):
        return self.cells[address]

    def write_cells_without_trigger(
        self,
        sheet_name,
        cells,
        *,
        trigger_address,
        allow_write=False,
    ):
        plan = tuple(cells)
        if self.cells[trigger_address] not in (None, ""):
            raise PermissionError("Trigger cell is not empty")
        self.write_calls.append((sheet_name, plan, trigger_address, allow_write))
        for address, value in plan:
            self.cells[address] = value
        return [address for address, _ in plan]


class GrussManualVisibleWriteTests(unittest.TestCase):
    def test_writes_visible_values_waits_then_restores_without_touching_q(self) -> None:
        bridge = FakeBridge()
        sleep = Mock()

        restored = test_gruss_manual_visible_write.run_visible_write_test(
            bridge,
            sleep_fn=sleep,
        )

        self.assertTrue(restored)
        sleep.assert_called_once_with(10)
        self.assertEqual(bridge.cells, {"Q5": None, "R5": 2.5, "S5": 3.0})
        self.assertEqual(
            bridge.write_calls,
            [
                ("PLACE", (("R5", 9.99), ("S5", 1)), "Q5", True),
                ("PLACE", (("R5", 2.5), ("S5", 3.0)), "Q5", True),
            ],
        )
        self.assertTrue(
            all(
                address != "Q5" and value not in test_gruss_manual_visible_write.TRIGGER_COMMANDS
                for _, plan, _, _ in bridge.write_calls
                for address, value in plan
            )
        )

    def test_refuses_if_q5_is_not_empty(self) -> None:
        bridge = FakeBridge(trigger_value="BACK")

        with self.assertRaisesRegex(RuntimeError, "PLACE!Q5 non vide"):
            test_gruss_manual_visible_write.run_visible_write_test(
                bridge,
                sleep_fn=Mock(),
            )

        self.assertEqual(bridge.write_calls, [])
        self.assertEqual(bridge.cells, {"Q5": "BACK", "R5": 2.5, "S5": 3.0})

    def test_rejects_any_plan_outside_r5_s5_or_containing_trigger_command(self) -> None:
        bridge = FakeBridge()

        with self.assertRaisesRegex(PermissionError, "restricted exactly"):
            test_gruss_manual_visible_write._write_rs_only(bridge, [("Q5", None)])
        with self.assertRaisesRegex(PermissionError, "trigger command values are forbidden"):
            test_gruss_manual_visible_write._write_rs_only(
                bridge,
                [("R5", "BACK"), ("S5", 1)],
            )

        self.assertEqual(bridge.write_calls, [])


if __name__ == "__main__":
    unittest.main()
