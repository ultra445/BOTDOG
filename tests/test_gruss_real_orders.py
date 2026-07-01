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
        write_bet_ref_after_post: bool = True,
        bet_refs_after_write=None,
        selection_rows_by_sheet=None,
    ) -> None:
        self.runner_values = runner_values or [["1. Test Runner"], ["2. Other Runner"]]
        self.visible = visible
        self.sheets = set(sheets or {"WIN", "PLACE"})
        self.market_ids = market_ids or {"WIN": "258836707", "PLACE": "258836708"}
        self.trigger_values = trigger_values or {}
        self.trigger_value_after_write = trigger_value_after_write
        self.trigger_value_before_clear = trigger_value_before_clear
        self.readback_overrides = readback_overrides or {}
        self.write_bet_ref_after_post = write_bet_ref_after_post
        self.bet_refs_after_write = list(bet_refs_after_write or [])
        self.selection_rows_by_sheet = selection_rows_by_sheet or {}
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
        return sheet_name in self.sheets or sheet_name in self.selection_rows_by_sheet

    def read_range(self, sheet_name: str, address: str):
        return self.runner_values

    def read_sheet(self, sheet_name: str, rows: int = 80, columns: int = 80):
        return self.selection_rows_by_sheet.get(sheet_name, [])

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
                if self.write_bet_ref_after_post and str(value).upper() in {"BACK", "LAY", "BACKSP", "LAYSP"}:
                    row = "".join(ch for ch in str(address) if ch.isdigit())
                    if self.bet_refs_after_write:
                        bet_ref = self.bet_refs_after_write.pop(0)
                    else:
                        bet_ref = f"4325000000{int(row):02d}"
                    self.cells[(sheet_name, f"T{row}")] = bet_ref
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

    def clear_trigger_cells(self, sheet_name, addresses, *, trigger_column, command_columns=None, allow_clear=False):
        prepared = tuple(addresses)
        self.clear_calls.append((sheet_name, prepared, trigger_column, allow_clear))
        allowed_columns = tuple(command_columns or (trigger_column,))
        for address in prepared:
            if not any(address.startswith(column) for column in allowed_columns):
                raise AssertionError("non-command clear attempted")
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
        "countdown_seconds": 1,
        "course": "Greyhound Racing\\PGR\\Romford 4th Jun",
        "market_already_processed": False,
        "win_market_id": "258836707",
        "place_market_id": "258836708",
    }
    data.update(overrides)
    return GrussRealOrderContext(**data)


def _post_selection_rows(**overrides):
    row = {
        "timestamp": "2099-01-01T00:00:00Z",
        "market_id": "258836708",
        "market_type": "PLACE",
        "selection_id": "sel-1",
        "selection": "1. Test Runner",
        "bet_ref": "432500000006",
        "side": "BACK",
        "req_odds": 3.2,
        "req_stake": 2.0,
        "matched_odds": "",
        "matched_stake": "",
        "market_name": "Romford 4th Jun To Be Placed",
    }
    row.update(overrides)
    return [
        [
            "Timestamp",
            "Market ID",
            "Market Type",
            "Selection ID",
            "Selection",
            "Bet Ref",
            "Bet type",
            "Req Odds",
            "Req Stake",
            "Matched Odds",
            "Matched Stake",
            "Market Name",
        ],
        [
            row["timestamp"],
            row["market_id"],
            row["market_type"],
            row["selection_id"],
            row["selection"],
            row["bet_ref"],
            row["side"],
            row["req_odds"],
            row["req_stake"],
            row["matched_odds"],
            row["matched_stake"],
            row["market_name"],
        ],
    ]


class GrussRealOrdersTests(unittest.TestCase):
    def test_stale_command_cells_cleanup_clears_backsp_laysp_and_reports_done(self) -> None:
        bridge = FakeBridge(
            runner_values=[
                ["1. Runner One"],
                ["2. Runner Two"],
            ],
            trigger_values={
                ("WIN", "Q5"): "BACKSP",
                ("WIN", "R5"): 3.5,
                ("WIN", "S5"): 2.0,
                ("PLACE", "Q6"): "LAYSP",
                ("PLACE", "R6"): 4.0,
                ("PLACE", "S6"): 2.0,
            },
        )

        with TemporaryDirectory() as tmp:
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            cleanup = provider.cleanup_stale_command_cells(reason="startup")
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertTrue(cleanup["attempted"])
        self.assertTrue(cleanup["done"])
        self.assertFalse(cleanup["failed"])
        self.assertEqual(bridge.cells[("WIN", "Q5")], None)
        self.assertEqual(bridge.cells[("WIN", "R5")], None)
        self.assertEqual(bridge.cells[("WIN", "S5")], None)
        self.assertEqual(bridge.cells[("PLACE", "Q6")], None)
        self.assertEqual(bridge.cells[("PLACE", "R6")], None)
        self.assertEqual(bridge.cells[("PLACE", "S6")], None)
        self.assertEqual(rows[0]["startup_command_cells_cleanup_attempted"], "True")
        self.assertEqual(rows[0]["startup_command_cells_cleanup_done"], "True")
        self.assertIn("WIN!Q5", rows[0]["stale_command_cells_cleanup_addresses"])
        self.assertIn("PLACE!Q6", rows[0]["stale_command_cells_cleanup_addresses"])

    def test_startup_cleanup_without_stale_triggers_still_reports_done(self) -> None:
        bridge = FakeBridge()

        with TemporaryDirectory() as tmp:
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            cleanup = provider.cleanup_stale_command_cells(reason="startup")
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertTrue(cleanup["attempted"])
        self.assertTrue(cleanup["done"])
        self.assertFalse(cleanup["failed"])
        self.assertEqual(cleanup["reason"], "no_stale_command_cells")
        self.assertEqual(rows[0]["startup_command_cells_cleanup_done"], "True")

    def test_stale_command_cells_cleanup_failure_is_reported_unsafe(self) -> None:
        class FailingClearBridge(FakeBridge):
            def clear_trigger_cells(self, *args, **kwargs):
                raise RuntimeError("clear failed")

        bridge = FailingClearBridge(trigger_values={("PLACE", "Q5"): "CANCEL"})

        with TemporaryDirectory() as tmp:
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            cleanup = provider.cleanup_stale_command_cells(reason="startup")
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertTrue(cleanup["attempted"])
        self.assertFalse(cleanup["done"])
        self.assertTrue(cleanup["failed"])
        self.assertEqual(cleanup["unsafe_stop_reason"], "unsafe_confirmed_stale_gruss_triggers_cleanup_failed")
        self.assertIn("unsafe_confirmed_stale_gruss_triggers_cleanup_failed", cleanup["reason"])
        self.assertEqual(rows[0]["startup_command_cells_cleanup_attempted"], "True")
        self.assertEqual(rows[0]["startup_command_cells_cleanup_done"], "False")
        self.assertIn("clear failed", rows[0]["stale_command_cells_cleanup_reason"])
        self.assertEqual(rows[0]["stale_triggers_confirmed"], "True")
        self.assertEqual(rows[0]["unsafe_stop_reason"], "unsafe_confirmed_stale_gruss_triggers_cleanup_failed")

    def test_stale_cleanup_recovers_from_temporary_scan_com_error_without_unsafe_stop(self) -> None:
        class FlakyScanBridge(FakeBridge):
            def __init__(self) -> None:
                super().__init__()
                self.read_range_calls = 0

            def read_range(self, sheet_name: str, address: str):
                self.read_range_calls += 1
                if self.read_range_calls <= 4:
                    raise RuntimeError("This object does not support enumeration")
                return super().read_range(sheet_name, address)

        bridge = FlakyScanBridge()

        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_EXCEL_COM_RETRY_BACKOFF_MS": "0,0,0"},
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            cleanup = provider.cleanup_stale_command_cells(reason="periodic:tick=419")
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(cleanup["failed"])
        self.assertTrue(cleanup["done"])
        self.assertIn("no_stale_command_cells", cleanup["reason"])
        self.assertNotEqual(rows[0]["stale_scan_retry_count"], "0")
        self.assertEqual(rows[0]["stale_scan_recovered"], "True")
        self.assertEqual(rows[0]["stale_triggers_confirmed"], "False")
        self.assertEqual(rows[0]["unsafe_stop_reason"], "")

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

    def test_bridge_clear_trigger_cells_accepts_semicolon_address_string(self) -> None:
        sheet = FakeSheet()
        sheet.cells.update({"Q5": "CANCEL", "R5": 3.0, "S5": 2.0})
        bridge = GrussExcelBridge()
        bridge.workbook = FakeWorkbook(sheet)

        cleared = bridge.clear_trigger_cells(
            "PLACE",
            "Q5;R5;S5",
            trigger_column="Q",
            command_columns=("Q", "R", "S"),
            allow_clear=True,
        )

        self.assertEqual(cleared, ["Q5", "R5", "S5"])
        self.assertIsNone(sheet.cells["Q5"])
        self.assertIsNone(sheet.cells["R5"])
        self.assertIsNone(sheet.cells["S5"])

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
            result = provider.place_order(_intent(price=1.0, stake=0), _context())

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
            result = provider.place_order(_intent(price=1.0), _context())

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
            [("PLACE", ("Q5", "R5", "S5"), "Q", True)],
        )
        self.assertIsNone(bridge.cells[("PLACE", "Q5")])
        self.assertIsNone(bridge.cells[("PLACE", "R5")])
        self.assertIsNone(bridge.cells[("PLACE", "S5")])
        self.assertEqual(rows[0]["trigger_value_written"], "LAY")
        self.assertEqual(rows[0]["trigger_clear_attempted"], "True")
        self.assertEqual(rows[0]["trigger_cleared"], "True")
        self.assertEqual(rows[0]["trigger_clear_reason"], "trigger_cleared")
        self.assertEqual(rows[0]["trigger_cell_value_before_clear"], "LAY")
        self.assertEqual(rows[0]["trigger_clear_delay_ms"], "0")
        self.assertEqual(rows[0]["command_cells_clear_attempted"], "True")
        self.assertEqual(rows[0]["command_cells_cleared"], "True")
        self.assertEqual(rows[0]["command_cells_clear_reason"], "command_cells_cleared")
        self.assertEqual(rows[0]["command_cells_clear_addresses"], "Q5;R5;S5")
        self.assertEqual(rows[0]["command_cells_clear_delay_ms"], "0")

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

    def test_real_back_order_floors_positive_stake_below_one_before_excel_write(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_REAL_MAX_STAKE": "5"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(side="BACK", stake=0.5), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.2), ("S5", 1.0), ("Q5", "BACK")))
        self.assertEqual(result.stake_original, 0.5)
        self.assertEqual(result.stake_used, 1.0)
        self.assertTrue(result.stake_min_floor_applied)
        self.assertEqual(result.stake_before_min_floor, 0.5)
        self.assertEqual(result.stake_after_min_floor, 1.0)
        self.assertEqual(result.stake_final, 1.0)
        self.assertFalse(result.stake_capped)
        self.assertEqual(rows[0]["stake_min_floor_applied"], "True")
        self.assertEqual(rows[0]["stake_before_min_floor"], "0.5")
        self.assertEqual(rows[0]["stake_after_min_floor"], "1.0")
        self.assertEqual(rows[0]["stake_final"], "1.0")

    def test_real_lay_order_floors_positive_stake_below_one_before_excel_write(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_REAL_MAX_STAKE": "5"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(side="LAY", stake=0.5), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.2), ("S5", 1.0), ("Q5", "LAY")))
        self.assertEqual(result.stake_original, 0.5)
        self.assertEqual(result.stake_used, 1.0)
        self.assertTrue(result.stake_min_floor_applied)
        self.assertEqual(rows[0]["stake_min_floor_applied"], "True")
        self.assertEqual(rows[0]["stake_final"], "1.0")

    def test_real_order_rejects_zero_negative_or_invalid_stake_without_excel_write(self) -> None:
        for stake in (0.0, -0.5, float("nan")):
            with self.subTest(stake=stake):
                bridge = FakeBridge()
                with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
                    provider = GrussExcelOrderProvider(tmp, bridge=bridge)
                    result = provider.place_order(_intent(stake=stake), _context())
                    rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

                self.assertEqual(result.status, "REJECTED_REAL")
                self.assertEqual(result.reason, "invalid_stake")
                self.assertEqual(bridge.write_calls, [])
                self.assertEqual(rows[0]["reason"], "invalid_stake")

    def test_real_post_back_limit_price_is_rounded_up_to_betfair_tick_before_excel_write(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(side="BACK", price=3.478), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.5), ("S5", 2.0), ("Q5", "BACK")))
        self.assertEqual(result.price_raw_before_tick, 3.478)
        self.assertEqual(result.price_tick_rounded, 3.5)
        self.assertEqual(result.price_tick_rounding_side, "BACK_CEIL")
        self.assertTrue(result.price_is_valid_betfair_tick)
        self.assertEqual(rows[0]["price_raw_before_tick"], "3.478")
        self.assertEqual(rows[0]["price_tick_rounded"], "3.5")
        self.assertEqual(rows[0]["price_tick_rounding_side"], "BACK_CEIL")
        self.assertEqual(rows[0]["price_is_valid_betfair_tick"], "true")

    def test_real_post_lay_limit_price_is_rounded_down_to_betfair_tick_before_excel_write(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(side="LAY", price=3.478), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.45), ("S5", 2.0), ("Q5", "LAY")))
        self.assertEqual(result.price_raw_before_tick, 3.478)
        self.assertEqual(result.price_tick_rounded, 3.45)
        self.assertEqual(result.price_tick_rounding_side, "LAY_FLOOR")
        self.assertTrue(result.price_is_valid_betfair_tick)
        self.assertEqual(rows[0]["price_raw_before_tick"], "3.478")
        self.assertEqual(rows[0]["price_tick_rounded"], "3.45")
        self.assertEqual(rows[0]["price_tick_rounding_side"], "LAY_FLOOR")
        self.assertEqual(rows[0]["price_is_valid_betfair_tick"], "true")

    def test_real_pre_ladder_clamps_back_price_to_value_target_before_excel_write(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _real_test_env(preview=False, max_stake=2.0), patch.dict(
            "os.environ",
            {
                "DOGBOT_PRE_LADDER_ENABLED": "true",
                "DOGBOT_PRE_LADDER_PREVIEW": "false",
                "DOGBOT_PRE_LADDER_STEPS": "52,38,26,16",
                "DOGBOT_PRE_LADDER_REAL_REQUIRE_BET_REF_FOR_REPLACE": "true",
                "DOGBOT_PRE_LADDER_REAL_STOP_IF_NO_BET_REF": "true",
                "DOGBOT_PRE_LADDER_REAL_NO_STACKING": "true",
                "DOGBOT_PRE_LADDER_REAL_MAX_LADDERS": "1",
            },
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(
                _intent(
                    price=2.92,
                    pre_ladder=True,
                    ladder_id="BACK_PLACE_902:market:1:PLACE:BACK:PRE",
                    ladder_step="1/4",
                    execution_phase="PRE",
                    strategy_id="BACK_PLACE_902",
                    stake_forced=True,
                    computed_limit_price_raw=3.088524,
                    computed_limit_price_effective=3.088524,
                    pre_value_target_price=3.088524,
                    ladder_planned_price=2.92,
                    sent_price_before_value_clamp=2.92,
                    value_clamp_applied=True,
                    value_limit_breached=True,
                    tick_rounding_direction="BACK_CEIL",
                    signal_countdown_seconds=52,
                ),
                _context(countdown_seconds=52, milestone_seen=52),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_PRE_LADDER_WRITTEN")
        self.assertEqual(result.price_tick_rounded, 3.1)
        self.assertEqual(result.sent_price_after_value_clamp, 3.1)
        self.assertEqual(bridge.write_calls[0][1][0], ("R5", 3.1))
        self.assertEqual(rows[0]["computed_limit_price_raw"], "3.088524")
        self.assertEqual(rows[0]["computed_limit_price_effective"], "3.088524")
        self.assertEqual(rows[0]["pre_value_target_price"], "3.088524")
        self.assertEqual(rows[0]["sent_price_before_value_clamp"], "2.92")
        self.assertEqual(rows[0]["sent_price_after_value_clamp"], "3.1")
        self.assertEqual(rows[0]["value_clamp_applied"], "True")
        self.assertEqual(rows[0]["value_limit_breached"], "True")
        self.assertEqual(rows[0]["tick_rounding_direction"], "BACK_CEIL")

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
            result = provider.place_order(_intent(), _context(countdown_seconds=1))
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertTrue(result.post_write_verified)
        self.assertEqual(result.post_write_odds_cell_address, "R5")
        self.assertEqual(result.post_write_odds_value, 3.2)
        self.assertEqual(result.post_write_stake_cell_address, "S5")
        self.assertEqual(result.post_write_stake_value, 2.0)
        self.assertEqual(result.post_write_trigger_cell_address, "Q5")
        self.assertEqual(result.post_write_trigger_value, "BACK")
        self.assertEqual(result.post_send_seconds_before_off, 1)
        self.assertTrue(result.post_trigger_window_hit)
        self.assertTrue(result.post_write_attempted)
        self.assertEqual(result.post_write_status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(result.post_write_reason, "excel_trigger_written")
        self.assertTrue(result.post_bet_ref_required)
        self.assertTrue(result.post_bet_ref_wait_attempted)
        self.assertEqual(result.post_bet_ref_after, "432500000005")
        self.assertTrue(result.post_order_confirmed)
        self.assertEqual(result.post_order_confirmation_source, "post_excel_row_poll:T5")
        self.assertEqual(result.countdown_seconds_at_post_write, 1)
        self.assertEqual(result.market_status_at_post_write, "Active")
        self.assertEqual(rows[0]["post_write_verified"], "True")
        self.assertEqual(rows[0]["post_send_seconds_before_off"], "1")
        self.assertEqual(rows[0]["post_trigger_window_hit"], "True")
        self.assertEqual(rows[0]["post_write_attempted"], "True")
        self.assertEqual(rows[0]["post_write_status"], "GRUSS_REAL_WRITTEN")
        self.assertEqual(rows[0]["post_write_reason"], "excel_trigger_written")
        self.assertEqual(rows[0]["post_bet_ref_required"], "True")
        self.assertEqual(rows[0]["post_bet_ref_wait_attempted"], "True")
        self.assertEqual(rows[0]["post_bet_ref_after"], "432500000005")
        self.assertEqual(rows[0]["post_order_confirmed"], "True")
        self.assertEqual(rows[0]["countdown_seconds_at_post_write"], "1")
        self.assertEqual(rows[0]["market_status_at_post_write"], "Active")

    def test_post_without_bet_ref_is_unconfirmed_and_logged(self) -> None:
        bridge = FakeBridge(write_bet_ref_after_post=False)
        counts = {}
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_POST_BET_REF_WAIT_MS": "0"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, real_order_counts=counts)
            result = provider.place_order(_intent(), _context(countdown_seconds=1))
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "POST_WRITE_UNCONFIRMED")
        self.assertEqual(result.reason, "POST_WRITE_ATTEMPTED_BUT_NO_NEW_ORDER_EVIDENCE")
        self.assertTrue(result.trigger_written)
        self.assertTrue(result.post_write_verified)
        self.assertTrue(result.post_bet_ref_wait_attempted)
        self.assertFalse(result.post_order_confirmed)
        self.assertEqual(result.post_write_unconfirmed_reason, "POST_WRITE_ATTEMPTED_BUT_NO_NEW_ORDER_EVIDENCE")
        self.assertEqual(result.post_bet_ref_poll_attempts, 1)
        self.assertEqual(result.post_bet_ref_after, "")
        self.assertEqual(counts, {})
        self.assertEqual(rows[0]["status"], "POST_WRITE_UNCONFIRMED")
        self.assertEqual(rows[0]["post_write_status"], "POST_WRITE_UNCONFIRMED")
        self.assertEqual(rows[0]["post_write_unconfirmed_reason"], "POST_WRITE_ATTEMPTED_BUT_NO_NEW_ORDER_EVIDENCE")
        self.assertEqual(rows[0]["post_order_confirmed"], "False")

    def test_post_existing_pre_bet_ref_is_not_accepted_as_new_confirmation(self) -> None:
        bridge = FakeBridge(write_bet_ref_after_post=True)
        bridge.cells[("PLACE", "T5")] = "432500000005"
        counts = {}
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_POST_BET_REF_WAIT_MS": "0"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, real_order_counts=counts)
            result = provider.place_order(_intent(), _context(countdown_seconds=1))
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "POST_WRITE_UNCONFIRMED_EXISTING_PRE_BETREF")
        self.assertEqual(result.reason, "POST_BET_REF_NOT_NEW_AND_NO_STAKE_DELTA")
        self.assertEqual(result.post_existing_bet_ref_before, "432500000005")
        self.assertEqual(result.post_bet_ref_after, "432500000005")
        self.assertFalse(result.post_bet_ref_changed)
        self.assertFalse(result.post_bet_ref_confirmed_new)
        self.assertFalse(result.post_order_confirmed)
        self.assertTrue(result.post_selections_lookup_attempted)
        self.assertFalse(result.post_selections_match_found)
        self.assertEqual(result.post_unconfirmed_reason, "POST_BET_REF_NOT_NEW_AND_NO_STAKE_DELTA")
        self.assertEqual(counts, {})
        self.assertEqual(rows[0]["post_existing_bet_ref_before"], "432500000005")
        self.assertEqual(rows[0]["post_bet_ref_changed"], "False")
        self.assertEqual(rows[0]["post_bet_ref_confirmed_new"], "False")
        self.assertEqual(rows[0]["post_selections_lookup_attempted"], "True")
        self.assertEqual(rows[0]["post_selections_match_found"], "False")
        self.assertEqual(rows[0]["post_unconfirmed_reason"], "POST_BET_REF_NOT_NEW_AND_NO_STAKE_DELTA")

    def test_post_independent_clears_pre_bet_ref_before_write_and_confirms_new_ref(self) -> None:
        bridge = FakeBridge(
            write_bet_ref_after_post=True,
            bet_refs_after_write=["432500000006"],
        )
        bridge.cells[("PLACE", "T5")] = "432500000005"
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_PRE_POST_INDEPENDENT": "true", "DOGBOT_POST_BET_REF_WAIT_MS": "0"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context(countdown_seconds=1))
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(bridge.write_calls[0][1], (("T5", ""), ("R5", 3.2), ("S5", 2.0), ("Q5", "BACK")))
        self.assertEqual(result.post_existing_pre_bet_ref, "432500000005")
        self.assertTrue(result.post_independent_mode_enabled)
        self.assertTrue(result.post_row_prepared_for_new_order)
        self.assertTrue(result.post_pre_bet_ref_preserved_in_state)
        self.assertTrue(result.post_pre_bet_ref_cleared_for_write)
        self.assertTrue(result.post_new_bet_ref_expected)
        self.assertTrue(result.post_new_bet_ref_found)
        self.assertEqual(result.post_new_bet_ref, "432500000006")
        self.assertEqual(result.post_bet_ref_after, "432500000006")
        self.assertTrue(result.post_order_confirmed)
        self.assertEqual(rows[0]["post_existing_pre_bet_ref"], "432500000005")
        self.assertEqual(rows[0]["post_pre_bet_ref_cleared_for_write"], "True")
        self.assertEqual(rows[0]["post_new_bet_ref"], "432500000006")

    def test_post_matched_stake_delta_confirms_without_reusing_pre_bet_ref(self) -> None:
        class MatchedDeltaBridge(FakeBridge):
            def write_cells(self, sheet_name: str, cells, *, allow_write: bool = False):
                written = super().write_cells(sheet_name, cells, allow_write=allow_write)
                if any(address.startswith("Q") for address, value in cells if str(value).upper() in {"BACK", "LAY"}):
                    self.cells[(sheet_name, "T5")] = "432500000005"
                    self.cells[(sheet_name, "W5")] = 4.0
                return written

        bridge = MatchedDeltaBridge(write_bet_ref_after_post=False)
        bridge.cells[("PLACE", "T5")] = "432500000005"
        bridge.cells[("PLACE", "W5")] = 2.0
        counts = {}
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_POST_BET_REF_WAIT_MS": "0"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, real_order_counts=counts)
            result = provider.place_order(_intent(), _context(countdown_seconds=1))
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(result.post_bet_ref_after, "432500000005")
        self.assertFalse(result.post_bet_ref_changed)
        self.assertFalse(result.post_bet_ref_confirmed_new)
        self.assertTrue(result.post_added_stake_confirmed)
        self.assertEqual(result.post_added_stake_amount, 2.0)
        self.assertEqual(result.post_total_matched_before, 2.0)
        self.assertEqual(result.post_total_matched_after, 4.0)
        self.assertEqual(result.post_total_matched_delta, 2.0)
        self.assertTrue(result.post_order_confirmed)
        self.assertEqual(result.post_confirmation_source, "post_matched_stake_delta")
        self.assertEqual(sum(counts.values()), 2)
        self.assertEqual(rows[0]["post_added_stake_confirmed"], "True")
        self.assertEqual(rows[0]["post_total_matched_delta"], "2.0")
        self.assertEqual(rows[0]["post_confirmation_source"], "post_matched_stake_delta")

    def test_post_after_pre_confirms_when_row_bet_ref_changes(self) -> None:
        for strategy_id in ("BACK_PLACE_903", "BACK_PLACE_913"):
            with self.subTest(strategy_id=strategy_id):
                bridge = FakeBridge(bet_refs_after_write=["432500000005", "432500000006"])
                with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True):
                    provider = GrussExcelOrderProvider(tmp, bridge=bridge)
                    pre = provider.place_order(
                        _intent(strategy_id=strategy_id, execution_phase="PRE", selection_id="sel-1"),
                        _context(countdown_seconds=1),
                    )
                    post = provider.place_order(
                        _intent(strategy_id=strategy_id, execution_phase="POST", selection_id="sel-1"),
                        _context(countdown_seconds=1),
                    )
                    rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

                self.assertEqual(pre.status, "GRUSS_REAL_WRITTEN")
                self.assertEqual(post.status, "GRUSS_REAL_WRITTEN")
                self.assertEqual(post.post_existing_bet_ref_before, "432500000005")
                self.assertEqual(post.post_bet_ref_after, "432500000006")
                self.assertTrue(post.post_bet_ref_changed)
                self.assertTrue(post.post_order_confirmed)
                self.assertEqual(post.post_confirmation_source, "post_excel_row_poll:T5")
                self.assertEqual(rows[-1]["post_existing_bet_ref_before"], "432500000005")
                self.assertEqual(rows[-1]["post_bet_ref_after"], "432500000006")
                self.assertEqual(rows[-1]["post_order_confirmed"], "True")

    def test_post_strict_selections_match_confirms_new_bet_ref(self) -> None:
        bridge = FakeBridge(
            write_bet_ref_after_post=False,
            selection_rows_by_sheet={"PLACE_Selections": _post_selection_rows()},
        )
        bridge.cells[("PLACE", "T5")] = "432500000005"
        counts = {}
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_POST_BET_REF_WAIT_MS": "0"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge, real_order_counts=counts)
            result = provider.place_order(
                _intent(strategy_id="BACK_PLACE_903", selection_id="sel-1"),
                _context(countdown_seconds=1),
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(result.post_bet_ref_after, "432500000006")
        self.assertTrue(result.post_bet_ref_changed)
        self.assertTrue(result.post_order_confirmed)
        self.assertTrue(result.post_selections_lookup_attempted)
        self.assertTrue(result.post_selections_match_found)
        self.assertEqual(result.post_selections_match_reason, "post_selections_strict_match")
        self.assertEqual(result.post_confirmation_source, "post_selections_sheet:PLACE_Selections!row2")
        self.assertEqual(sum(counts.values()), 2)
        self.assertEqual(rows[0]["post_bet_ref_after"], "432500000006")
        self.assertEqual(rows[0]["post_selections_match_found"], "True")
        self.assertEqual(rows[0]["post_confirmation_source"], "post_selections_sheet:PLACE_Selections!row2")

    def test_post_strict_selections_rejects_wrong_market_runner_or_course(self) -> None:
        cases = [
            ("other_market_id", _post_selection_rows(market_id="999999999"), "market_id_mismatch"),
            (
                "other_runner",
                _post_selection_rows(selection_id="", selection="2. Other Runner"),
                "runner_mismatch",
            ),
            (
                "old_course_without_market_id",
                _post_selection_rows(market_id="", market_name="Harlow 4th Jun To Be Placed"),
                "market_id_missing",
            ),
        ]
        for name, selection_rows, expected_reason in cases:
            with self.subTest(name=name):
                bridge = FakeBridge(
                    write_bet_ref_after_post=False,
                    selection_rows_by_sheet={"PLACE_Selections": selection_rows},
                )
                bridge.cells[("PLACE", "T5")] = "432500000005"
                counts = {}
                with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
                    "os.environ",
                    {"DOGBOT_POST_BET_REF_WAIT_MS": "0"},
                    clear=False,
                ):
                    provider = GrussExcelOrderProvider(tmp, bridge=bridge, real_order_counts=counts)
                    result = provider.place_order(
                        _intent(strategy_id="BACK_PLACE_903", selection_id="sel-1"),
                        _context(countdown_seconds=1),
                    )

                self.assertEqual(result.status, "POST_WRITE_UNCONFIRMED_EXISTING_PRE_BETREF")
                self.assertFalse(result.post_order_confirmed)
                self.assertTrue(result.post_selections_lookup_attempted)
                self.assertFalse(result.post_selections_match_found)
                self.assertIn(expected_reason, result.post_selections_reject_reason)
                self.assertEqual(counts, {})

    def test_post_bet_ref_poll_waits_for_changed_value(self) -> None:
        bridge = FakeBridge(write_bet_ref_after_post=True)
        bridge.cells[("PLACE", "T5")] = "432500000005"
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {
                "DOGBOT_POST_BET_REF_WAIT_MS": "35",
                "DOGBOT_POST_BET_REF_POLL_MS": "10",
            },
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context(countdown_seconds=1))

        self.assertEqual(result.status, "POST_WRITE_UNCONFIRMED_EXISTING_PRE_BETREF")
        self.assertGreaterEqual(result.post_bet_ref_poll_attempts, 2)
        self.assertGreaterEqual(result.post_bet_ref_poll_duration_ms, 20)

    def test_post_unconfirmed_cleanup_is_logged_separately(self) -> None:
        bridge = FakeBridge(write_bet_ref_after_post=True)
        bridge.cells[("PLACE", "T5")] = "432500000005"
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_POST_BET_REF_WAIT_MS": "0"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context(countdown_seconds=1))
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "POST_WRITE_UNCONFIRMED_EXISTING_PRE_BETREF")
        self.assertFalse(result.post_cells_cleared_after_confirmation)
        self.assertTrue(result.post_cells_cleared_after_unconfirmed)
        self.assertEqual(result.post_clear_reason, "unconfirmed_post_cleanup")
        self.assertEqual(rows[0]["post_cells_cleared_after_confirmation"], "False")
        self.assertEqual(rows[0]["post_cells_cleared_after_unconfirmed"], "True")
        self.assertEqual(rows[0]["post_clear_reason"], "unconfirmed_post_cleanup")

    def test_post_clear_after_bet_ref_uses_post_delay(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {"DOGBOT_POST_COMMAND_CELLS_CLEAR_DELAY_MS": "1234"},
            clear=False,
        ):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context(countdown_seconds=1))
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertTrue(result.post_order_confirmed)
        self.assertTrue(result.post_clear_after_bet_ref)
        self.assertEqual(result.post_cells_clear_delay_ms, 1234)
        self.assertTrue(result.post_cells_cleared_after_confirmation)
        self.assertEqual(result.command_cells_clear_delay_ms, 1234)
        self.assertEqual(rows[0]["post_cells_clear_delay_ms"], "1234")
        self.assertEqual(rows[0]["post_cells_cleared_after_confirmation"], "True")

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
                "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS": "3000",
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
            def clear_trigger_cells(self, sheet_name, addresses, *, trigger_column, command_columns=None, allow_clear=False):
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
            {
                "DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS": "3000",
                "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS": "3000",
            },
            clear=False,
        ), patch("dogbot.gruss.gruss_real_orders.time.sleep") as sleep:
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())

        self.assertEqual(result.status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(result.trigger_clear_delay_ms, 3000)

    def test_command_cells_clear_delay_uses_dedicated_env_over_trigger_delay(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, _provider_env(preview=False, layout_confirmed=True), patch.dict(
            "os.environ",
            {
                "DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS": "1000",
                "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS": "5000",
                "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_NON_BLOCKING": "true",
            },
            clear=False,
        ), patch("dogbot.gruss.gruss_real_orders.time.sleep") as sleep:
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(), _context())
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")
            provider._pending_command_cell_clears[0].due_monotonic = 0
            provider._drain_due_command_cell_clears()
            drained_rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(provider.command_cells_clear_delay_ms, 5000)
        self.assertEqual(result.command_cells_clear_delay_ms, 5000)
        self.assertEqual(rows[0]["command_cells_clear_delay_ms"], "5000")
        self.assertEqual(rows[0]["command_cells_clear_scheduled"], "True")
        self.assertEqual(rows[0]["command_cells_clear_non_blocking"], "True")
        self.assertEqual(rows[0]["command_cells_clear_executed"], "False")
        self.assertEqual(rows[0]["command_cells_clear_reason"], "command_cells_clear_scheduled")
        self.assertEqual(rows[0]["command_cells_clear_due_time"], result.command_cells_clear_due_time)
        self.assertTrue(rows[0]["command_cells_clear_due_time"])
        sleep.assert_not_called()
        self.assertFalse(result.trigger_cleared)
        self.assertEqual(bridge.clear_calls, [("PLACE", ("Q5", "R5", "S5"), "Q", True)])
        self.assertEqual(drained_rows[0]["command_cells_clear_executed"], "True")
        self.assertEqual(drained_rows[0]["command_cells_clear_reason"], "command_cells_cleared")
        self.assertEqual(drained_rows[0]["command_cells_cleared"], "True")
        self.assertEqual(drained_rows[0]["trigger_cleared"], "True")
        self.assertEqual(drained_rows[0]["trigger_clear_reason"], "trigger_cleared")

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
        self.assertEqual(post.status, "POST_WRITE_UNCONFIRMED_EXISTING_PRE_BETREF")
        self.assertEqual(post.reason, "POST_BET_REF_NOT_NEW_AND_NO_STAKE_DELTA")
        self.assertIn("|PRE", pre.processed_key)
        self.assertIn("|POST", post.processed_key)
        self.assertNotEqual(pre.processed_key, post.processed_key)
        self.assertEqual([row["execution_phase"] for row in rows], ["PRE", "POST"])
        self.assertEqual(
            [row["status"] for row in rows],
            ["GRUSS_REAL_WRITTEN", "POST_WRITE_UNCONFIRMED_EXISTING_PRE_BETREF"],
        )

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
        self.assertEqual(post.status, "POST_WRITE_UNCONFIRMED_EXISTING_PRE_BETREF")
        self.assertEqual(post.reason, "POST_BET_REF_NOT_NEW_AND_NO_STAKE_DELTA")

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
            "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS": "0",
            "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST": "false",
            "DOGBOT_PRE_POST_INDEPENDENT": "false",
            "DOGBOT_POST_BET_REF_WAIT_MS": "0",
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
        "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS": "0",
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
            "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS": "0",
            "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST": "false",
            "DOGBOT_PRE_POST_INDEPENDENT": "false",
            "DOGBOT_POST_BET_REF_WAIT_MS": "0",
        },
        clear=False,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
