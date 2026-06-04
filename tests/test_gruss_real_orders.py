from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_REAL
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
    ) -> None:
        self.runner_values = runner_values or [["1. Test Runner"], ["2. Other Runner"]]
        self.visible = visible
        self.sheets = set(sheets or {"WIN", "PLACE"})
        self.market_ids = market_ids or {"WIN": "258836707", "PLACE": "258836708"}
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
        return None

    def write_cells(self, sheet_name: str, cells, *, allow_write: bool = False):
        plan = tuple(cells)
        self.write_calls.append((sheet_name, plan, allow_write))
        return [address for address, _ in plan]


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

def _provider_env(*, preview: bool, layout_confirmed: bool = False):
    return patch.dict(
        "os.environ",
        {
            "DOGBOT_ORDER_PROVIDER": ORDER_PROVIDER_GRUSS_EXCEL_REAL,
            "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
            "DOGBOT_GRUSS_REAL_PREVIEW": str(preview).lower(),
            "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": str(layout_confirmed).lower(),
        },
        clear=False,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
