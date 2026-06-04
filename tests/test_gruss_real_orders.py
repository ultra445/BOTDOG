from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_REAL
from dogbot.gruss.gruss_excel_bridge import GrussExcelBridge
from dogbot.gruss.gruss_orders import make_order_intent
from dogbot.gruss.gruss_real_orders import GrussExcelOrderProvider, GrussRealOrderContext


class FakeBridge:
    def __init__(
        self,
        runner_values=None,
        *,
        visible: bool = True,
        sheets=None,
        market_ids=None,
        trigger_values=None,
    ) -> None:
        self.runner_values = runner_values or [["1. Test Runner"], ["2. Other Runner"]]
        self.visible = visible
        self.sheets = set(sheets or {"WIN", "PLACE"})
        self.market_ids = market_ids or {"WIN": "258836707", "PLACE": "258836708"}
        self.trigger_values = trigger_values or {}
        self.connect_calls = 0
        self.write_calls: list[tuple[str, tuple[tuple[str, object], ...], bool]] = []

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
        if address.startswith("Q"):
            return self.trigger_values.get((sheet_name, address))
        return None

    def write_cells(self, sheet_name: str, cells, *, allow_write: bool = False):
        plan = tuple(cells)
        self.write_calls.append((sheet_name, plan, allow_write))
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

    def test_real_test_mode_limits_ten_signals_to_one_order(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=1, max_stake=10):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results = [
                provider.place_order(_intent(strategy_id=f"BACK_PLACE_{index}"), _context())
                for index in range(10)
            ]
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(
            [result.reason for result in results[1:]],
            ["max_orders_reached"] * 9,
        )
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual([row["reason"] for row in rows[1:]], ["max_orders_reached"] * 9)

    def test_real_test_mode_defaults_to_one_order_and_one_stake_unit(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=None, max_stake=None):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)

        self.assertEqual(provider.real_max_orders, 1)
        self.assertEqual(provider.real_max_stake, 1.0)

    def test_real_test_mode_rejects_stake_above_limit(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(max_orders=10, max_stake=1):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(stake=2.0), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "REJECTED_REAL")
        self.assertEqual(result.reason, "stake_above_real_test_limit")
        self.assertEqual(rows[0]["reason"], "stake_above_real_test_limit")
        self.assertEqual(bridge.connect_calls, 0)
        self.assertEqual(bridge.write_calls, [])

    def test_configured_real_limit_applies_outside_test_mode(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(
            real_test_mode=False,
            max_orders=1,
            max_stake=10,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            first = provider.place_order(_intent(strategy_id="BACK_PLACE_101"), _context())
            second = provider.place_order(_intent(strategy_id="BACK_PLACE_102"), _context())

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
        },
        clear=False,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
