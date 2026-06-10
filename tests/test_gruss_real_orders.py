from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_REAL
from dogbot.gruss.gruss_excel_bridge import GrussExcelBridge
from dogbot.gruss.gruss_orders import make_order_intent
from dogbot.gruss.gruss_real_orders import (
    GrussExcelOrderProvider,
    GrussRealOrderContext,
    GrussTriggerLayout,
)

_SAME_AS_WRITTEN = object()


class FakeBridge:
    def __init__(
        self,
        runner_values=None,
        *,
        visible: bool = True,
        sheets=None,
        market_ids=None,
        trigger_values=None,
        trigger_value_after_write=_SAME_AS_WRITTEN,
        trigger_value_before_clear=_SAME_AS_WRITTEN,
        readback_overrides=None,
    ) -> None:
        self.runner_values = runner_values or [["1. Test Runner"], ["2. Other Runner"]]
        self.visible = visible
        self.sheets = set(sheets or {"WIN", "PLACE"})
        self.market_ids = market_ids or {"WIN": "258836707", "PLACE": "258836708"}
        self.trigger_values = trigger_values or {}
        self.trigger_value_after_write = trigger_value_after_write
        self.trigger_value_before_clear = trigger_value_before_clear
        self.readback_overrides = readback_overrides or {}
        self.cells = dict(self.trigger_values)
        self.write_happened = False
        self.post_write_trigger_reads = 0
        self.connect_calls = 0
        self.write_calls: list[tuple[str, tuple[tuple[str, object], ...], bool]] = []
        self.clear_calls: list[tuple[str, tuple[str, ...], str, bool]] = []

    def connect_open_workbook(self) -> bool:
        self.connect_calls += 1
        return True

    def is_workbook_visible(self) -> bool:
        return self.visible

    def has_sheet(self, sheet_name: str) -> bool:
        return sheet_name in self.sheets

    def read_range(self, sheet_name: str, address: str):
        return self.runner_values

    def read_cell(self, sheet_name: str, address: str):
        if address == "N3":
            return self.market_ids.get(sheet_name)
        if address == "F2":
            return "Active"
        if (sheet_name, address) in self.readback_overrides:
            return self.readback_overrides[(sheet_name, address)]
        if address.startswith("Q"):
            if self.write_happened:
                self.post_write_trigger_reads += 1
                if (
                    self.post_write_trigger_reads >= 2
                    and self.trigger_value_before_clear is not _SAME_AS_WRITTEN
                ):
                    self.cells[(sheet_name, address)] = self.trigger_value_before_clear
            return self.cells.get((sheet_name, address))
        return self.cells.get((sheet_name, address))

    def write_cells(self, sheet_name: str, cells, *, allow_write: bool = False):
        plan = tuple(cells)
        self.write_calls.append((sheet_name, plan, allow_write))
        for address, value in plan:
            if address.startswith("Q"):
                self.cells[(sheet_name, address)] = (
                    value
                    if self.trigger_value_after_write is _SAME_AS_WRITTEN
                    else self.trigger_value_after_write
                )
            else:
                self.cells[(sheet_name, address)] = value
        self.write_happened = any(address.startswith("Q") for address, _ in plan)
        return [address for address, _ in plan]

    def write_cells_without_trigger(
        self,
        sheet_name: str,
        cells,
        *,
        trigger_address: str,
        allow_write: bool = False,
    ):
        plan = tuple(cells)
        if any(address == trigger_address for address, _ in plan):
            raise AssertionError("trigger address reached write_cells_without_trigger")
        return self.write_cells(sheet_name, plan, allow_write=allow_write)

    def clear_trigger_cells(self, sheet_name, addresses, *, trigger_column, allow_clear=False):
        prepared = tuple(addresses)
        self.clear_calls.append((sheet_name, prepared, trigger_column, allow_clear))
        for address in prepared:
            if not address.startswith(trigger_column):
                raise AssertionError("non-trigger clear attempted")
            self.cells[(sheet_name, address)] = None
        return list(prepared)


class FakeCellRange:
    def __init__(self, cells: dict[str, object], address: str) -> None:
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

    def __init__(self) -> None:
        self.cells: dict[str, object] = {}

    def range(self, address: str) -> FakeCellRange:
        return FakeCellRange(self.cells, address)


class FakeWorkbook:
    def __init__(self, sheet: FakeSheet) -> None:
        self.sheets = [sheet]


def _intent(**overrides):
    data = {
        "provider": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
        "market_type": "PLACE",
        "market_id": "258836708",
        "parent_id": "35678301",
        "runner_name": "Test Runner",
        "trap": 1,
        "side": "BACK",
        "order_type": "LIMIT",
        "price": 3.2,
        "stake": 2.0,
        "strategy_id": "BACK_PLACE_101",
        "course_id": "parent:35678301",
        "timestamp": "2026-06-04T10:00:00Z",
        "dry_run": False,
    }
    data.update(overrides)
    return make_order_intent(**data)


def _context(**overrides):
    data = {
        "validation_ok": True,
        "tradable": True,
        "region": "UK",
        "countdown_seconds": 2,
        "course": "Greyhound Racing\\PGR\\Romford 4th Jun",
        "market_already_processed": False,
        "win_market_id": "258836707",
        "place_market_id": "258836708",
    }
    data.update(overrides)
    return GrussRealOrderContext(**data)


class GrussRealOrdersTests(unittest.TestCase):
    def test_bridge_write_without_trigger_materially_rejects_trigger_address(self) -> None:
        sheet = FakeSheet()
        bridge = GrussExcelBridge()
        bridge.workbook = FakeWorkbook(sheet)

        with self.assertRaisesRegex(PermissionError, "Trigger cell write forbidden"):
            bridge.write_cells_without_trigger(
                "PLACE",
                [("R5", 3.2), ("Q5", "BACK")],
                trigger_address="Q5",
                allow_write=True,
            )

        self.assertEqual(sheet.cells, {})

    def test_bridge_write_without_trigger_rechecks_trigger_before_each_write(self) -> None:
        sheet = FakeSheet()
        sheet.cells["Q5"] = "LAY"
        bridge = GrussExcelBridge()
        bridge.workbook = FakeWorkbook(sheet)

        with self.assertRaisesRegex(PermissionError, "Trigger cell is not empty"):
            bridge.write_cells_without_trigger(
                "PLACE",
                [("R5", 3.2), ("S5", 2.0)],
                trigger_address="Q5",
                allow_write=True,
            )

        self.assertNotIn("R5", sheet.cells)
        self.assertNotIn("S5", sheet.cells)

    def test_real_provider_refuses_when_enable_env_is_absent(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {"DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL},
            clear=True,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("real_orders_not_enabled", result.reason)
        self.assertEqual(bridge.connect_calls, 0)
        self.assertEqual(bridge.write_calls, [])

    def test_preview_never_writes_to_excel(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_PREVIEW")
        self.assertEqual(bridge.write_calls, [])
        self.assertEqual(result.write_plan[-1], ("Q5", "BACK"))
        self.assertEqual(rows[0]["excel_cells_written"], "")
        self.assertFalse(result.trigger_clear_attempted)
        self.assertEqual(bridge.clear_calls, [])

    def test_preview_allows_real_orders_disabled_and_never_writes(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "false",
                "DOGBOT_GRUSS_REAL_PREVIEW": "true",
                "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
                "DOGBOT_GRUSS_REAL_TEST_MODE": "false",
            },
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_REAL_PREVIEW")
        self.assertEqual(result.reason, "preview_only_no_excel_write")
        self.assertEqual(bridge.write_calls, [])
        self.assertEqual(result.write_plan[-1], ("Q5", "BACK"))

    def test_preview_ignores_hold_trigger_visual_test_environment(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "false",
                "DOGBOT_GRUSS_REAL_PREVIEW": "true",
                "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
                "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST": "true",
            },
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_REAL_PREVIEW")
        self.assertEqual(bridge.write_calls, [])
        self.assertFalse(result.trigger_written)
        self.assertFalse(result.hold_trigger_for_visual_test)

    def test_sp_moc_preview_uses_gruss_sp_trigger_and_price(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                _intent(side="LAY", order_type="SP_MOC", price=6.0),
                _context(),
            )

        self.assertEqual(result.status, "GRUSS_REAL_PREVIEW")
        self.assertEqual(result.write_plan, (("R5", 6.0), ("S5", 2.0), ("Q5", "LAYSP")))
        self.assertEqual(bridge.write_calls, [])

    def test_provider_rechecks_enable_flag_for_each_attempt(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)

        with patch.dict("os.environ", {"DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "false"}, clear=False):
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("real_orders_not_enabled", result.reason)
        self.assertEqual(bridge.write_calls, [])

    def test_preview_only_guard_allows_unarmed_preview_and_never_writes(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "false",
                "DOGBOT_GRUSS_REAL_PREVIEW": "true",
                "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
            },
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, preview_only_guard=True)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_REAL_PREVIEW")
        self.assertEqual(bridge.write_calls, [])

    def test_preview_only_guard_refuses_if_real_orders_become_enabled(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
                "DOGBOT_GRUSS_REAL_PREVIEW": "true",
                "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
            },
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, preview_only_guard=True)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("preview_only_refuses_real_orders_enabled", result.reason)
        self.assertEqual(bridge.connect_calls, 0)
        self.assertEqual(bridge.write_calls, [])

    def test_preview_only_guard_refuses_if_preview_becomes_disabled(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "false",
                "DOGBOT_GRUSS_REAL_PREVIEW": "false",
                "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
                "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": "true",
            },
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, preview_only_guard=True)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("preview_only_requires_preview", result.reason)
        self.assertEqual(bridge.connect_calls, 0)
        self.assertEqual(bridge.write_calls, [])

    def test_write_no_trigger_writes_preparation_cells_without_real_enable(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(side="LAY"), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_WRITE_NO_TRIGGER")
        self.assertEqual(result.reason, "no_trigger_written")
        self.assertFalse(result.trigger_written)
        self.assertEqual(
            bridge.write_calls,
            [("PLACE", (("R5", 3.2), ("S5", 2.0)), True)],
        )
        self.assertNotIn("Q5", result.excel_cells_written)
        self.assertEqual(rows[0]["cells_written"], "R5;S5")
        self.assertEqual(rows[0]["trigger_written"], "False")
        self.assertFalse(result.trigger_clear_attempted)
        self.assertEqual(bridge.clear_calls, [])

    def test_write_no_trigger_allows_multiple_intents_in_same_batch_course(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            results = [
                provider.place_order(
                    _intent(strategy_id=f"BACK_PLACE_{index}", price=3.0 + index / 10),
                    _context(),
                )
                for index in range(1, 4)
            ]

        self.assertEqual([result.status for result in results], ["GRUSS_WRITE_NO_TRIGGER"] * 3)
        self.assertTrue(all(result.trigger_written is False for result in results))
        self.assertEqual(len(bridge.write_calls), 3)
        self.assertTrue(
            all(
                all(address != "Q5" for address, _ in plan)
                for _, plan, _ in bridge.write_calls
            )
        )

    def test_write_no_trigger_accepts_real_preview_absent(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False, preview=None):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_WRITE_NO_TRIGGER")
        self.assertFalse(result.trigger_written)

    def test_write_no_trigger_accepts_real_preview_empty(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False, preview=""):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_WRITE_NO_TRIGGER")
        self.assertFalse(result.trigger_written)

    def test_write_no_trigger_accepts_real_preview_false(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False, preview="false"):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_WRITE_NO_TRIGGER")
        self.assertFalse(result.trigger_written)

    def test_write_no_trigger_refuses_real_preview_true(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False, preview="true"):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("write_no_trigger_requires_real_preview_false", result.reason)
        self.assertEqual(bridge.write_calls, [])

    def test_write_no_trigger_never_writes_trigger_even_when_real_enabled(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(order_type="SP_MOC", side="BACK"), _context())

        self.assertEqual(result.status, "GRUSS_WRITE_NO_TRIGGER")
        self.assertEqual(result.write_plan, (("R5", 3.2), ("S5", 2.0)))
        self.assertEqual(result.intended_trigger, "BACKSP")
        self.assertTrue(all(address != "Q5" for address, _ in bridge.write_calls[0][1]))
        self.assertFalse(result.trigger_written)

    def test_regular_real_provider_refuses_when_write_no_trigger_flag_is_set(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("write_no_trigger_requires_guarded_provider", result.reason)
        self.assertEqual(bridge.connect_calls, 0)
        self.assertEqual(bridge.write_calls, [])

    def test_write_no_trigger_refuses_dangerous_context(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(
                _intent(),
                _context(validation_ok=False, tradable=False, region="UNKNOWN", countdown_seconds=3),
            )

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("win_place_validation_failed", result.reason)
        self.assertIn("market_not_tradable", result.reason)
        self.assertIn("unknown_region", result.reason)
        self.assertIn("countdown_above_2_seconds", result.reason)
        self.assertEqual(bridge.write_calls, [])

    def test_write_no_trigger_refuses_invalid_price_and_stake(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(price=1.01, stake=0), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("invalid_price", result.reason)
        self.assertIn("invalid_stake", result.reason)
        self.assertEqual(bridge.connect_calls, 0)
        self.assertEqual(bridge.write_calls, [])

    def test_write_no_trigger_refuses_changed_market(self) -> None:
        bridge = FakeBridge(market_ids={"WIN": "999", "PLACE": "258836708"})
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertEqual(result.reason, "current_market_id_mismatch=WIN:999")
        self.assertEqual(bridge.write_calls, [])

    def test_write_no_trigger_refuses_nonempty_trigger_cell(self) -> None:
        bridge = FakeBridge(trigger_values={("PLACE", "Q5"): "LAY"})
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertEqual(result.reason, "trigger_cell_not_empty")
        self.assertEqual(bridge.write_calls, [])

    def test_real_provider_refuses_invalid_price(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(price=1.01), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("invalid_price", result.reason)
        self.assertEqual(bridge.connect_calls, 0)

    def test_real_provider_refuses_invalid_stake(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(stake=0), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("invalid_stake", result.reason)
        self.assertEqual(bridge.connect_calls, 0)

    def test_real_provider_refuses_missing_runner_row(self) -> None:
        bridge = FakeBridge(runner_values=[["2. Other Runner"]])
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertEqual(result.reason, "runner_row_not_found")
        self.assertEqual(bridge.write_calls, [])

    def test_real_provider_refuses_when_excel_has_changed_market(self) -> None:
        bridge = FakeBridge(market_ids={"WIN": "999", "PLACE": "258836708"})
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertEqual(result.reason, "current_market_id_mismatch=WIN:999")
        self.assertEqual(bridge.write_calls, [])

    def test_real_provider_refuses_unsafe_market_context(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                _intent(),
                _context(
                    validation_ok=False,
                    tradable=False,
                    region="UNKNOWN",
                    countdown_seconds=None,
                    market_already_processed=True,
                ),
            )

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("win_place_validation_failed", result.reason)
        self.assertIn("market_not_tradable", result.reason)
        self.assertIn("unknown_region", result.reason)
        self.assertIn("countdown_seconds_unavailable", result.reason)
        self.assertIn("market_already_processed", result.reason)
        self.assertEqual(bridge.connect_calls, 0)

    def test_real_provider_refuses_without_layout_confirmation(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=False):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertEqual(result.reason, "trigger_layout_not_confirmed")
        self.assertEqual(bridge.write_calls, [])

    def test_fake_real_order_writes_only_when_fully_armed(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(side="LAY"), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(
            bridge.write_calls,
            [("PLACE", (("R5", 3.2), ("S5", 2.0), ("Q5", "LAY")), True)],
        )
        self.assertEqual(rows[0]["status"], "GRUSS_REAL_WRITTEN")
        self.assertEqual(rows[0]["dry_run_or_real"], "REAL")
        self.assertTrue(result.trigger_clear_attempted)
        self.assertTrue(result.trigger_cleared)
        self.assertEqual(result.trigger_clear_reason, "trigger_cleared")
        self.assertEqual(result.trigger_cell_value_before_clear, "LAY")
        self.assertEqual(
            bridge.clear_calls,
            [("PLACE", ("Q5",), "Q", True)],
        )
        self.assertEqual(rows[0]["trigger_value_written"], "LAY")
        self.assertEqual(rows[0]["trigger_clear_attempted"], "True")
        self.assertEqual(rows[0]["trigger_cleared"], "True")
        self.assertEqual(rows[0]["trigger_clear_reason"], "trigger_cleared")
        self.assertEqual(rows[0]["trigger_cell_value_before_clear"], "LAY")
        self.assertEqual(rows[0]["trigger_clear_delay_ms"], "0")

    def test_real_back_order_caps_stake_before_excel_write(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_REAL_MAX_STAKE": "5"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(side="BACK", stake=8.0), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.2), ("S5", 5.0), ("Q5", "BACK")))
        self.assertEqual(result.stake_original, 8.0)
        self.assertEqual(result.stake_used, 5.0)
        self.assertTrue(result.stake_capped)
        self.assertEqual(result.stake_cap_value, 5.0)
        self.assertEqual(rows[0]["stake_original"], "8.0")
        self.assertEqual(rows[0]["stake_used"], "5.0")
        self.assertEqual(rows[0]["stake_capped"], "True")
        self.assertEqual(rows[0]["stake_cap_value"], "5.0")

    def test_real_lay_order_caps_stake_before_excel_write(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_REAL_MAX_STAKE": "5"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(side="LAY", stake=8.0), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.2), ("S5", 5.0), ("Q5", "LAY")))
        self.assertEqual(result.stake_original, 8.0)
        self.assertEqual(result.stake_used, 5.0)
        self.assertTrue(result.stake_capped)
        self.assertEqual(result.stake_cap_value, 5.0)
        self.assertEqual(rows[0]["stake_capped"], "True")

    def test_real_order_below_stake_cap_keeps_original_stake(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_REAL_MAX_STAKE": "5"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(stake=3.0), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1][1], ("S5", 3.0))
        self.assertEqual(result.stake_original, 3.0)
        self.assertEqual(result.stake_used, 3.0)
        self.assertFalse(result.stake_capped)
        self.assertEqual(result.stake_cap_value, 5.0)
        self.assertEqual(rows[0]["stake_capped"], "False")
        self.assertEqual(rows[0]["stake_cap_value"], "5.0")

    def test_real_order_without_stake_cap_keeps_original_stake(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(stake=8.0), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1][1], ("S5", 8.0))
        self.assertEqual(result.stake_original, 8.0)
        self.assertEqual(result.stake_used, 8.0)
        self.assertFalse(result.stake_capped)
        self.assertIsNone(result.stake_cap_value)
        self.assertEqual(rows[0]["stake_capped"], "False")
        self.assertEqual(rows[0]["stake_cap_value"], "")

    def test_preview_does_not_apply_real_stake_cap_or_write_excel(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=True), patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_REAL_MAX_STAKE": "5"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(stake=8.0), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_PREVIEW")
        self.assertEqual(result.write_plan[1], ("S5", 8.0))
        self.assertEqual(result.stake_original, 8.0)
        self.assertEqual(result.stake_used, 8.0)
        self.assertFalse(result.stake_capped)
        self.assertIsNone(result.stake_cap_value)
        self.assertEqual(bridge.write_calls, [])
        self.assertEqual(rows[0]["stake_used"], "8.0")
        self.assertEqual(rows[0]["stake_capped"], "False")
        self.assertEqual(rows[0]["stake_cap_value"], "")

    def test_post_trigger_does_not_clear_if_value_changed(self) -> None:
        bridge = FakeBridge(trigger_value_before_clear="CHANGED_BY_GRUSS")
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertTrue(result.trigger_written)
        self.assertTrue(result.trigger_clear_attempted)
        self.assertFalse(result.trigger_cleared)
        self.assertEqual(result.trigger_clear_reason, "trigger_clear_skipped_value_changed")
        self.assertEqual(result.trigger_cell_value_before_clear, "CHANGED_BY_GRUSS")
        self.assertEqual(bridge.clear_calls, [])
        self.assertEqual(rows[0]["trigger_clear_reason"], "trigger_clear_skipped_value_changed")

    def test_post_write_readback_match_is_verified_and_logged(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertTrue(result.post_write_verified)
        self.assertEqual(result.post_write_odds_cell_address, "R5")
        self.assertEqual(result.post_write_odds_value, 3.2)
        self.assertEqual(result.post_write_stake_cell_address, "S5")
        self.assertEqual(result.post_write_stake_value, 2.0)
        self.assertEqual(result.post_write_trigger_cell_address, "Q5")
        self.assertEqual(result.post_write_trigger_value, "BACK")
        self.assertEqual(rows[0]["post_write_verified"], "True")

    def test_post_write_readback_mismatch_fails_and_does_not_count_order(self) -> None:
        bridge = FakeBridge(readback_overrides={("PLACE", "R5"): 99.0})
        counts = {}
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, real_order_counts=counts)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_WRITE_FAILED")
        self.assertEqual(result.reason, "post_write_verification_failed")
        self.assertFalse(result.post_write_verified)
        self.assertTrue(result.trigger_written)
        self.assertEqual(counts, {})

    def test_post_write_missing_trigger_is_not_marked_written(self) -> None:
        bridge = FakeBridge(trigger_value_after_write=None)
        counts = {}
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, real_order_counts=counts)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_WRITE_FAILED")
        self.assertFalse(result.post_write_verified)
        self.assertFalse(result.trigger_written)
        self.assertFalse(result.trigger_clear_attempted)
        self.assertEqual(counts, {})

    def test_hold_trigger_for_visual_test_waits_after_verified_readback(self) -> None:
        bridge = FakeBridge()
        trigger_values_during_sleep = []

        def observe_trigger_during_sleep(seconds):
            trigger_values_during_sleep.append((seconds, bridge.cells.get(("PLACE", "Q5"))))

        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1), patch.dict(
            "os.environ",
            {
                "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST": "true",
                "DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS": "3000",
            },
            clear=False,
        ), patch("dogbot.gruss.gruss_real_orders.time.sleep", side_effect=observe_trigger_during_sleep) as sleep:
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(stake=1.0), _context())

        self.assertTrue(result.post_write_verified)
        self.assertTrue(result.hold_trigger_for_visual_test)
        self.assertTrue(result.trigger_cleared)
        sleep.assert_called_once_with(3.0)
        self.assertEqual(trigger_values_during_sleep, [(3.0, "BACK")])

    def test_post_trigger_clear_failure_is_logged_without_retrying_order(self) -> None:
        class ClearFailingBridge(FakeBridge):
            def clear_trigger_cells(self, sheet_name, addresses, *, trigger_column, allow_clear=False):
                raise RuntimeError("mock clear failure")

        bridge = ClearFailingBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertTrue(result.trigger_written)
        self.assertTrue(result.trigger_clear_attempted)
        self.assertFalse(result.trigger_cleared)
        self.assertIn("trigger_clear_failed: mock clear failure", result.trigger_clear_reason)
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertIn("trigger_clear_failed: mock clear failure", rows[0]["trigger_clear_reason"])

    def test_configured_trigger_clear_delay_is_used(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS": "3000"},
            clear=False,
        ), patch("dogbot.gruss.gruss_real_orders.time.sleep") as sleep:
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(result.trigger_clear_delay_ms, 3000)
        self.assertTrue(result.trigger_cleared)
        sleep.assert_called_once_with(3.0)

    def test_default_trigger_clear_delay_is_1500_ms(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            provider = GrussExcelOrderProvider(bridge=FakeBridge())

        self.assertEqual(provider.trigger_clear_delay_ms, 1500)

    def test_rejected_order_never_attempts_trigger_clear(self) -> None:
        bridge = FakeBridge(trigger_values={("PLACE", "Q5"): "BACK"})
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertFalse(result.trigger_written)
        self.assertFalse(result.trigger_clear_attempted)
        self.assertFalse(result.trigger_cleared)
        self.assertEqual(bridge.clear_calls, [])

    def test_real_provider_refuses_nonempty_trigger_cell(self) -> None:
        bridge = FakeBridge(trigger_values={("PLACE", "Q5"): "BACK"})
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertEqual(result.reason, "trigger_cell_not_empty")
        self.assertFalse(result.trigger_written)
        self.assertEqual(result.trigger_cell_address, "Q5")
        self.assertEqual(result.trigger_cell_current_value, "BACK")
        self.assertFalse(result.trigger_cell_expected_empty)
        self.assertEqual(result.trigger_mapping_name, "BACK")
        self.assertEqual(rows[0]["trigger_cell_address"], "Q5")
        self.assertEqual(rows[0]["trigger_cell_current_value"], "BACK")
        self.assertEqual(rows[0]["trigger_cell_expected_empty"], "False")
        self.assertEqual(rows[0]["trigger_mapping_name"], "BACK")
        self.assertEqual(bridge.write_calls, [])

    def test_backsp_nonempty_trigger_logs_exact_address_value_and_mapping(self) -> None:
        bridge = FakeBridge(trigger_values={("PLACE", "Q5"): "formula-or-old-trigger"})
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                _intent(
                    side="BACK",
                    order_type="SP_MOC",
                    price=1.37,
                    stake=1.0,
                    strategy_id="GRUSS_FORCE_TEST_BSP_PLACE",
                    stake_original=1.0,
                    stake_forced=True,
                    force_test_bsp_place=True,
                ),
                _context(),
            )

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertEqual(result.reason, "trigger_cell_not_empty")
        self.assertEqual(result.trigger_cell_address, "Q5")
        self.assertEqual(result.trigger_cell_current_value, "formula-or-old-trigger")
        self.assertFalse(result.trigger_cell_expected_empty)
        self.assertEqual(result.trigger_mapping_name, "BACKSP")
        self.assertFalse(result.trigger_written)
        self.assertEqual(bridge.write_calls, [])

    def test_real_test_mode_limits_ten_signals_to_one_order(self) -> None:
        bridge = FakeBridge(runner_values=[[f"{index}. Runner {index}"] for index in range(1, 11)])
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results = [
                provider.place_order(
                    _intent(
                        strategy_id=f"BACK_PLACE_{index}",
                        stake=1.0,
                        selection_id=f"runner-{index}",
                        trap=index,
                        runner_name=f"Runner {index}",
                    ),
                    _context(),
                )
                for index in range(1, 11)
            ]
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(
            [result.reason for result in results[1:]],
            ["max_orders_reached"] * 9,
        )
        self.assertEqual([result.trigger_written for result in results], [True] + [False] * 9)
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual(len(rows), 10)
        self.assertEqual([row["reason"] for row in rows[1:]], ["max_orders_reached"] * 9)
        self.assertEqual([row["trigger_written"] for row in rows], ["True"] + ["False"] * 9)

    def test_processed_pre_does_not_block_processed_post(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            pre = provider.place_order(
                _intent(stake=1.0, execution_phase="PRE", selection_id="runner-1"),
                _context(),
            )
            post = provider.place_order(
                _intent(stake=1.0, execution_phase="POST", selection_id="runner-1"),
                _context(),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(pre.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(post.status, "GRUSS_REAL_WRITTEN")
        self.assertIn("|PRE", pre.processed_key)
        self.assertIn("|POST", post.processed_key)
        self.assertNotEqual(pre.processed_key, post.processed_key)
        self.assertEqual([row["execution_phase"] for row in rows], ["PRE", "POST"])
        self.assertEqual([row["status"] for row in rows], ["GRUSS_REAL_WRITTEN", "GRUSS_REAL_WRITTEN"])

    def test_processed_post_does_not_block_processed_pre(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            post = provider.place_order(
                _intent(stake=1.0, execution_phase="POST", selection_id="runner-1"),
                _context(),
            )
            pre = provider.place_order(
                _intent(stake=1.0, execution_phase="PRE", selection_id="runner-1"),
                _context(),
            )

        self.assertEqual(post.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(pre.status, "GRUSS_REAL_WRITTEN")
        self.assertNotEqual(post.processed_key, pre.processed_key)

    def test_duplicate_processed_key_blocks_same_phase_only(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=2, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first = provider.place_order(
                _intent(stake=1.0, execution_phase="PRE", selection_id="runner-1"),
                _context(),
            )
            duplicate = provider.place_order(
                _intent(stake=1.0, execution_phase="PRE", selection_id="runner-1"),
                _context(),
            )
            post = provider.place_order(
                _intent(stake=1.0, execution_phase="POST", selection_id="runner-1"),
                _context(),
            )

        self.assertEqual(first.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(duplicate.reason, "market_already_processed")
        self.assertEqual(post.status, "GRUSS_REAL_WRITTEN")

    def test_max_orders_pre_does_not_block_post(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            pre = provider.place_order(
                _intent(stake=1.0, execution_phase="PRE", selection_id="runner-1"),
                _context(),
            )
            blocked_pre = provider.place_order(
                _intent(stake=1.0, execution_phase="PRE", selection_id="runner-2", trap=2, runner_name="Other Runner"),
                _context(),
            )
            post = provider.place_order(
                _intent(stake=1.0, execution_phase="POST", selection_id="runner-2", trap=2, runner_name="Other Runner"),
                _context(),
            )

        self.assertEqual(pre.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(blocked_pre.reason, "max_orders_reached")
        self.assertEqual(post.status, "GRUSS_REAL_WRITTEN")

    def test_max_orders_post_does_not_block_pre(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            post = provider.place_order(
                _intent(stake=1.0, execution_phase="POST", selection_id="runner-1"),
                _context(),
            )
            blocked_post = provider.place_order(
                _intent(stake=1.0, execution_phase="POST", selection_id="runner-2", trap=2, runner_name="Other Runner"),
                _context(),
            )
            pre = provider.place_order(
                _intent(stake=1.0, execution_phase="PRE", selection_id="runner-2", trap=2, runner_name="Other Runner"),
                _context(),
            )

        self.assertEqual(post.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(blocked_post.reason, "max_orders_reached")
        self.assertEqual(pre.status, "GRUSS_REAL_WRITTEN")

    def test_phase_specific_max_order_environment_overrides_default(self) -> None:
        bridge = FakeBridge(runner_values=[[f"{index}. Runner {index}"] for index in range(1, 5)])
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1), patch.dict(
            "os.environ",
            {
                "DOGBOT_GRUSS_REAL_MAX_ORDERS_PRE": "2",
                "DOGBOT_GRUSS_REAL_MAX_ORDERS_POST": "1",
            },
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            pre_1 = provider.place_order(
                _intent(stake=1.0, execution_phase="PRE", selection_id="runner-1", runner_name="Runner 1"),
                _context(),
            )
            pre_2 = provider.place_order(
                _intent(stake=1.0, execution_phase="PRE", selection_id="runner-2", trap=2, runner_name="Runner 2"),
                _context(),
            )
            post_1 = provider.place_order(
                _intent(stake=1.0, execution_phase="POST", selection_id="runner-3", trap=3, runner_name="Runner 3"),
                _context(),
            )
            post_2 = provider.place_order(
                _intent(stake=1.0, execution_phase="POST", selection_id="runner-4", trap=4, runner_name="Runner 4"),
                _context(),
            )

        self.assertEqual([pre_1.status, pre_2.status, post_1.status], ["GRUSS_REAL_WRITTEN"] * 3)
        self.assertEqual(post_2.reason, "max_orders_reached")

    def test_real_test_mode_defaults_to_one_order_and_one_stake_unit(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=None, max_stake=None):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)

        self.assertEqual(provider.real_max_orders, 1)
        self.assertEqual(provider.real_max_stake, 1.0)

    def test_real_test_mode_caps_stake_above_limit(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=10, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(stake=2.0), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(result.reason, "excel_trigger_written")
        self.assertEqual(bridge.write_calls[0][1][1], ("S5", 1.0))
        self.assertEqual(result.stake_original, 2.0)
        self.assertEqual(result.stake_used, 1.0)
        self.assertTrue(result.stake_capped)
        self.assertEqual(result.stake_cap_value, 1.0)
        self.assertEqual(rows[0]["stake_original"], "2.0")
        self.assertEqual(rows[0]["stake_used"], "1.0")
        self.assertEqual(rows[0]["stake_capped"], "True")
        self.assertEqual(rows[0]["stake_cap_value"], "1.0")

    def test_real_test_mode_allows_positive_stake_below_dryrun_minimum(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(stake=1.0), _context())

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertTrue(result.trigger_written)
        self.assertEqual(len(bridge.write_calls), 1)

    def test_configured_real_limit_applies_outside_test_mode(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(
            real_test_mode=False,
            max_orders=1,
            max_stake=10,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first = provider.place_order(_intent(strategy_id="BACK_PLACE_101"), _context())
            second = provider.place_order(
                _intent(strategy_id="BACK_PLACE_102", selection_id="runner-2", trap=2, runner_name="Runner 2"),
                _context(),
            )

        self.assertEqual(first.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(second.reason, "max_orders_reached")
        self.assertEqual(len(bridge.write_calls), 1)

    def test_preview_ignores_real_test_limits(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(
            preview=True,
            max_orders=0,
            max_stake=0,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(stake=2.0), _context())

        self.assertEqual(result.status, "GRUSS_REAL_PREVIEW")
        self.assertEqual(bridge.write_calls, [])

    def test_preview_ignores_force_stake_environment(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(preview=True, max_orders=1, max_stake=1), patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE": "1"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(stake=2.0), _context())

        self.assertEqual(result.status, "GRUSS_REAL_PREVIEW")
        self.assertEqual(result.write_plan[1], ("S5", 2.0))
        self.assertEqual(result.stake_used, 2.0)
        self.assertFalse(result.stake_forced)

    def test_write_no_trigger_ignores_real_test_limits(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(
            enabled=False,
            real_test_mode=True,
            max_orders=0,
            max_stake=0,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(stake=2.0), _context())

        self.assertEqual(result.status, "GRUSS_WRITE_NO_TRIGGER")
        self.assertFalse(result.trigger_written)
        self.assertEqual(len(bridge.write_calls), 1)

    def test_write_no_trigger_ignores_force_stake_environment(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _write_no_trigger_env(enabled=False), patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE": "1"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, write_no_trigger_guard=True)
            result = provider.place_order(_intent(stake=2.0), _context())

        self.assertEqual(result.status, "GRUSS_WRITE_NO_TRIGGER")
        self.assertEqual(result.write_plan[1], ("S5", 2.0))
        self.assertEqual(result.stake_used, 2.0)
        self.assertFalse(result.stake_forced)

    def test_forced_stake_diagnostic_is_rejected_outside_true_real_test_mode(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                _intent(stake=2.0, stake_original=5.0, stake_forced=True),
                _context(),
            )

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("forced_stake_requires_real_test_mode", result.reason)
        self.assertEqual(bridge.write_calls, [])

    def test_force_bsp_place_logs_selection_diagnostics_and_backsp_trigger(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                _intent(
                    side="BACK",
                    order_type="SP_MOC",
                    price=2.4,
                    stake=1.0,
                    strategy_id="GRUSS_FORCE_TEST_BSP_PLACE",
                    stake_original=1.0,
                    stake_forced=True,
                    force_test_bsp_place=True,
                    selected_reason="lowest_place_odds",
                    selected_runner="Test Runner",
                    selected_trap=1,
                    selected_place_odds=2.4,
                ),
                _context(),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(result.intended_trigger, "BACKSP")
        self.assertTrue(result.trigger_written)
        self.assertEqual(rows[0]["force_test_bsp_place"], "True")
        self.assertEqual(rows[0]["selected_reason"], "lowest_place_odds")
        self.assertEqual(rows[0]["selected_runner"], "Test Runner")
        self.assertEqual(rows[0]["selected_trap"], "1")
        self.assertEqual(rows[0]["selected_place_odds"], "2.4")
        self.assertEqual(rows[0]["stake_used"], "1.0")

    def test_force_bsp_place_refuses_missing_backsp_mapping(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(
                tmp,
                bridge=bridge,
                layout=GrussTriggerLayout(back_sp_moc_trigger=""),
            )
            result = provider.place_order(
                _intent(
                    side="BACK",
                    order_type="SP_MOC",
                    price=2.4,
                    stake=1.0,
                    strategy_id="GRUSS_FORCE_TEST_BSP_PLACE",
                    stake_original=1.0,
                    stake_forced=True,
                    force_test_bsp_place=True,
                ),
                _context(),
            )

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("back_sp_mapping_unavailable", result.reason)
        self.assertEqual(bridge.write_calls, [])

    def test_force_back_place_limit_uses_back_trigger_and_logs_selected_prices(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                _intent(
                    side="BACK",
                    order_type="LIMIT",
                    price=2.5,
                    stake=1.0,
                    strategy_id="GRUSS_FORCE_TEST_BACK_PLACE_LIMIT",
                    stake_original=1.0,
                    stake_forced=True,
                    force_test_back_place_limit=True,
                    selected_reason="lowest_place_odds_best_lay_price",
                    selected_runner="Test Runner",
                    selected_trap=1,
                    selected_place_back_odds=2.4,
                    selected_place_lay_odds=2.5,
                    price_used=2.5,
                ),
                _context(),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(result.intended_trigger, "BACK")
        self.assertEqual(
            bridge.write_calls,
            [("PLACE", (("R5", 2.5), ("S5", 1.0), ("Q5", "BACK")), True)],
        )
        self.assertTrue(result.trigger_written)
        self.assertTrue(result.trigger_cleared)
        self.assertEqual(rows[0]["force_test_back_place_limit"], "True")
        self.assertEqual(rows[0]["selected_place_back_odds"], "2.4")
        self.assertEqual(rows[0]["selected_place_lay_odds"], "2.5")
        self.assertEqual(rows[0]["price_used"], "2.5")
        self.assertEqual(rows[0]["trigger"], "BACK")
        self.assertEqual(rows[0]["trigger_written"], "True")
        self.assertEqual(rows[0]["trigger_cleared"], "True")

    def test_force_back_place_limit_allows_two_euro_real_test_stake(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=2):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                _intent(
                    side="BACK",
                    order_type="LIMIT",
                    price=2.5,
                    stake=2.0,
                    strategy_id="GRUSS_FORCE_TEST_BACK_PLACE_LIMIT",
                    stake_original=2.0,
                    stake_forced=True,
                    force_test_back_place_limit=True,
                    selected_reason="lowest_place_odds_best_lay_price",
                    selected_runner="Test Runner",
                    selected_trap=1,
                    selected_place_back_odds=2.4,
                    selected_place_lay_odds=2.5,
                    price_used=2.5,
                ),
                _context(),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(
            bridge.write_calls,
            [("PLACE", (("R5", 2.5), ("S5", 2.0), ("Q5", "BACK")), True)],
        )
        self.assertEqual(rows[0]["stake_used"], "2.0")
        self.assertEqual(rows[0]["stake_forced"], "True")

    def test_force_back_place_limit_refuses_missing_best_lay(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                _intent(
                    price=2.5,
                    stake=1.0,
                    strategy_id="GRUSS_FORCE_TEST_BACK_PLACE_LIMIT",
                    stake_original=1.0,
                    stake_forced=True,
                    force_test_back_place_limit=True,
                ),
                _context(),
            )

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertIn("missing_place_best_lay", result.reason)
        self.assertEqual(bridge.write_calls, [])

    def test_real_provider_treats_real_preview_absent_as_false(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
                "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
                "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": "true",
            },
            clear=True,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")

    def test_real_provider_treats_empty_real_preview_as_false(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
                "DOGBOT_GRUSS_REAL_PREVIEW": "",
                "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
                "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": "true",
            },
            clear=True,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")

def _provider_env(*, preview: bool, layout_confirmed: bool = False):
    return patch.dict(
        "os.environ",
        {
            "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
            "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
            "DOGBOT_GRUSS_REAL_PREVIEW": str(preview).lower(),
            "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
            "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": str(layout_confirmed).lower(),
            "DOGBOT_GRUSS_REAL_TEST_MODE": "false",
            "DOGBOT_GRUSS_REAL_MAX_ORDERS": "",
            "DOGBOT_GRUSS_REAL_MAX_STAKE": "",
            "DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS": "0",
            "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST": "false",
        },
        clear=False,
    )


def _write_no_trigger_env(
    *,
    enabled: bool,
    preview: str | None = "false",
    real_test_mode: bool = False,
    max_orders: int | None = None,
    max_stake: float | None = None,
):
    values = {
        "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
        "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": str(enabled).lower(),
        "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "true",
        "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": "true",
        "DOGBOT_GRUSS_REAL_TEST_MODE": str(real_test_mode).lower(),
        "DOGBOT_GRUSS_REAL_MAX_ORDERS": "" if max_orders is None else str(max_orders),
        "DOGBOT_GRUSS_REAL_MAX_STAKE": "" if max_stake is None else str(max_stake),
        "DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS": "0",
        "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST": "false",
    }
    if preview is not None:
        values["DOGBOT_GRUSS_REAL_PREVIEW"] = preview
    return patch.dict("os.environ", values, clear=preview is None)


def _real_test_env(
    *,
    real_test_mode: bool = True,
    preview: bool = False,
    max_orders: int | None = None,
    max_stake: float | None = None,
):
    return patch.dict(
        "os.environ",
        {
            "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
            "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
            "DOGBOT_GRUSS_REAL_PREVIEW": str(preview).lower(),
            "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
            "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": "true",
            "DOGBOT_GRUSS_REAL_TEST_MODE": str(real_test_mode).lower(),
            "DOGBOT_GRUSS_REAL_MAX_ORDERS": "" if max_orders is None else str(max_orders),
            "DOGBOT_GRUSS_REAL_MAX_STAKE": "" if max_stake is None else str(max_stake),
            "DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS": "0",
            "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST": "false",
        },
        clear=False,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
