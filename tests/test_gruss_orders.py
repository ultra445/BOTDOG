from __future__ import annotations

import builtins
import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dogbot.gruss.gruss_orders import GrussOrderProvider, make_order_intent


def _intent(**overrides):
    data = {
        "provider": "gruss_excel",
        "market_type": "WIN",
        "market_id": "258836707",
        "parent_id": "35678301",
        "runner_name": "Test Runner",
        "trap": 1,
        "side": "BACK",
        "order_type": "LIMIT",
        "price": 3.2,
        "stake": 2.0,
        "strategy_id": "BACK_WIN_1",
        "course_id": "course-1",
        "timestamp": "2026-06-03T18:00:00Z",
        "dry_run": True,
    }
    data.update(overrides)
    return make_order_intent(**data)


class GrussOrdersTests(unittest.TestCase):
    def test_valid_back_limit_order_is_logged(self) -> None:
        with TemporaryDirectory() as tmp:
            provider = GrussOrderProvider(tmp)

            result = provider.place_order(_intent(side="BACK"))

            self.assertEqual(result.status, "GRUSS_DRYRUN")
            rows = _read_rows(Path(tmp) / "orders_gruss_dryrun.csv")
            self.assertEqual(rows[0]["side"], "BACK")
            self.assertEqual(rows[0]["status"], "GRUSS_DRYRUN")

    def test_valid_lay_limit_order_is_logged(self) -> None:
        with TemporaryDirectory() as tmp:
            provider = GrussOrderProvider(tmp)

            result = provider.place_order(_intent(side="LAY"))

            self.assertEqual(result.status, "GRUSS_DRYRUN")
            rows = _read_rows(Path(tmp) / "orders_gruss_dryrun.csv")
            self.assertEqual(rows[0]["side"], "LAY")

    def test_rejects_stake_below_minimum(self) -> None:
        with TemporaryDirectory() as tmp:
            provider = GrussOrderProvider(tmp)

            result = provider.place_order(_intent(stake=1.99))

            self.assertEqual(result.status, "REJECTED_DRYRUN")
            self.assertIn("stake_below_minimum", result.reason)

    def test_rejects_invalid_limit_price(self) -> None:
        with TemporaryDirectory() as tmp:
            provider = GrussOrderProvider(tmp)

            result = provider.place_order(_intent(price=1.0))

            self.assertEqual(result.status, "REJECTED_DRYRUN")
            self.assertIn("invalid_limit_price", result.reason)

    def test_rejects_invalid_side(self) -> None:
        with TemporaryDirectory() as tmp:
            provider = GrussOrderProvider(tmp)

            result = provider.place_order(_intent(side="CANCEL"))

            self.assertEqual(result.status, "REJECTED_DRYRUN")
            self.assertIn("invalid_side", result.reason)

    def test_dryrun_provider_does_not_import_excel(self) -> None:
        real_import = builtins.__import__

        def guard_import(name, *args, **kwargs):
            if name in {"xlwings", "win32com", "win32com.client"}:
                raise AssertionError(f"unexpected Excel import: {name}")
            return real_import(name, *args, **kwargs)

        with TemporaryDirectory() as tmp, patch("builtins.__import__", guard_import):
            provider = GrussOrderProvider(tmp)
            result = provider.place_order(_intent())

        self.assertEqual(result.status, "GRUSS_DRYRUN")

    def test_dryrun_provider_ignores_real_stake_cap_environment(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {"DOGBOT_GRUSS_REAL_MAX_STAKE": "5"},
            clear=False,
        ):
            provider = GrussOrderProvider(tmp)
            result = provider.place_order(_intent(stake=8.0))
            rows = _read_rows(Path(tmp) / "orders_gruss_dryrun.csv")

        self.assertEqual(result.status, "GRUSS_DRYRUN")
        self.assertEqual(rows[0]["stake"], "8.0")
        self.assertNotIn("stake_capped", rows[0])


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
