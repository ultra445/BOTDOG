from __future__ import annotations

import csv
import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_REAL
from dogbot.gruss.gruss_orders import make_order_intent
from dogbot.gruss.gruss_real_orders import GrussExcelOrderProvider, GrussRealOrderContext


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_gruss_real_strategy_test.py"
SPEC = importlib.util.spec_from_file_location("watch_gruss_real_strategy_test", SCRIPT_PATH)
watch_gruss_real_strategy_test = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watch_gruss_real_strategy_test)


VALID_ENV = {
    "DOGBOT_DATA_PROVIDER": "gruss_excel",
    "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
    "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
    "DOGBOT_GRUSS_REAL_TEST_MODE": "true",
    "DOGBOT_GRUSS_REAL_MAX_ORDERS": "1",
    "DOGBOT_GRUSS_REAL_MAX_STAKE": "2",
    "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE": "2",
    "DOGBOT_GRUSS_REAL_PREVIEW": "false",
    "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
    "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": "true",
    "DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS": "0",
    "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST": "false",
}


class FakeBridge:
    def __init__(self) -> None:
        self.write_calls = []
        self.clear_calls = []
        self.cells = {}

    def connect_open_workbook(self):
        return True

    def is_workbook_visible(self):
        return True

    def has_sheet(self, sheet_name):
        return sheet_name in {"WIN", "PLACE"}

    def read_cell(self, sheet_name, address):
        if address == "N3":
            return {"WIN": "win-1", "PLACE": "place-1"}[sheet_name]
        if address == "F2":
            return "Active"
        return self.cells.get((sheet_name, address))

    def read_range(self, sheet_name, address):
        return [[f"{index}. Runner {index}"] for index in range(1, 11)]

    def write_cells(self, sheet_name, cells, *, allow_write=False):
        plan = tuple(cells)
        self.write_calls.append((sheet_name, plan, allow_write))
        for address, value in plan:
            self.cells[(sheet_name, address)] = value
        return [address for address, _ in plan]

    def clear_trigger_cells(self, sheet_name, addresses, *, trigger_column, allow_clear=False):
        prepared = tuple(addresses)
        self.clear_calls.append((sheet_name, prepared, trigger_column, allow_clear))
        for address in prepared:
            self.cells[(sheet_name, address)] = None
        return list(prepared)


class FakeProcessedStore:
    def __init__(self) -> None:
        self.seen = set()
        self.mark_calls = []

    def has_seen(self, key):
        return key in self.seen

    def mark_seen(self, key, win_market_id, place_market_id):
        self.mark_calls.append((key, win_market_id, place_market_id))
        self.seen.add(key)


def _intent(index: int, *, stake: float = 5.0):
    return make_order_intent(
        provider="gruss_excel",
        market_type="PLACE",
        market_id="place-1",
        parent_id="1",
        runner_name=f"Runner {index}",
        trap=index,
        side="BACK",
        order_type="LIMIT",
        price=3.0,
        stake=stake,
        strategy_id=f"BACK_PLACE_{100 + index}",
        course_id="parent:1",
        dry_run=True,
    )


def _context():
    return GrussRealOrderContext(
        validation_ok=True,
        tradable=True,
        region="UK",
        countdown_seconds=2,
        course="parent:1",
        win_market_id="win-1",
        place_market_id="place-1",
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class WatchGrussRealStrategyTestTests(unittest.TestCase):
    def test_valid_environment_is_accepted(self) -> None:
        self.assertEqual(
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(VALID_ENV),
            (1, 2.0, 2.0),
        )

    def test_force_test_modes_are_forbidden(self) -> None:
        for forbidden in (
            "DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT",
            "DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE",
        ):
            with self.subTest(forbidden=forbidden):
                env = dict(VALID_ENV, **{forbidden: "true"})
                with self.assertRaisesRegex(RuntimeError, f"{forbidden}=true est interdit"):
                    watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_preview_and_write_no_trigger_are_forbidden(self) -> None:
        for forbidden in ("DOGBOT_GRUSS_REAL_PREVIEW", "DOGBOT_GRUSS_WRITE_NO_TRIGGER"):
            with self.subTest(forbidden=forbidden):
                env = dict(VALID_ENV, **{forbidden: "true"})
                with self.assertRaisesRegex(RuntimeError, f"{forbidden}=true est interdit"):
                    watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_force_stake_two_requires_real_test_mode(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_TEST_MODE="false")

        with self.assertRaisesRegex(RuntimeError, "REAL_TEST_MODE=true est obligatoire"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_force_stake_must_be_exactly_two(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1")

        with self.assertRaisesRegex(RuntimeError, "REAL_TEST_FORCE_STAKE doit etre exactement 2"):
            watch_gruss_real_strategy_test.validate_real_strategy_test_environment(env)

    def test_real_strategies_multiple_signals_write_one_and_reject_rest(self) -> None:
        bridge = FakeBridge()
        store = FakeProcessedStore()
        intents = [_intent(index, stake=5.0) for index in range(1, 5)]

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=_context(),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertFalse(skipped)
        self.assertEqual([result.status for result in results].count("GRUSS_REAL_WRITTEN"), 1)
        self.assertEqual([result.reason for result in results[1:]], ["max_orders_reached"] * 3)
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual(bridge.write_calls[0][1], (("R5", 3.0), ("S5", 2.0), ("Q5", "BACK")))
        self.assertEqual(store.mark_calls, [("parent:1|POST", "win-1", "place-1")])
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["status"], "GRUSS_REAL_WRITTEN")
        self.assertEqual([row["reason"] for row in rows[1:]], ["max_orders_reached"] * 3)

    def test_processed_pre_batch_does_not_block_post_batch(self) -> None:
        bridge = FakeBridge()
        store = FakeProcessedStore()
        pre_intents = [_intent(1, stake=5.0)]
        post_intents = [_intent(1, stake=5.0)]
        pre_intents[0] = replace(pre_intents[0], execution_phase="PRE", selection_id="runner-1")
        post_intents[0] = replace(post_intents[0], execution_phase="POST", selection_id="runner-1")

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            pre_results, pre_skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=pre_intents,
                context=_context(),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            post_results, post_skipped = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=post_intents,
                context=_context(),
                processed_store=store,
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

        self.assertFalse(pre_skipped)
        self.assertFalse(post_skipped)
        self.assertEqual(pre_results[0].status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(post_results[0].status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(
            store.mark_calls,
            [("parent:1|PRE", "win-1", "place-1"), ("parent:1|POST", "win-1", "place-1")],
        )

    def test_force_stake_two_is_logged_as_original_and_used(self) -> None:
        bridge = FakeBridge()

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_intent(1, stake=5.0)],
                context=_context(),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )
            rows = _read_rows(Path(tmp) / "gruss_real_order_attempts.csv")

        self.assertEqual(results[0].stake_original, 5.0)
        self.assertEqual(results[0].stake_used, 2.0)
        self.assertTrue(results[0].stake_forced)
        self.assertEqual(rows[0]["stake_original"], "5.0")
        self.assertEqual(rows[0]["stake_used"], "2.0")
        self.assertEqual(rows[0]["stake_forced"], "True")

    def test_post_write_verification_and_trigger_cleanup_are_preserved(self) -> None:
        bridge = FakeBridge()

        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_strategy_test.process_real_strategy_test_batch(
                provider=provider,
                intents=[_intent(1, stake=5.0)],
                context=_context(),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=2.0,
            )

        result = results[0]
        self.assertTrue(result.post_write_verified)
        self.assertEqual(result.post_write_odds_cell_address, "R5")
        self.assertEqual(result.post_write_stake_cell_address, "S5")
        self.assertEqual(result.post_write_trigger_cell_address, "Q5")
        self.assertTrue(result.trigger_clear_attempted)
        self.assertTrue(result.trigger_cleared)
        self.assertEqual(result.trigger_clear_delay_ms, 0)
        self.assertEqual(bridge.clear_calls, [("PLACE", ("Q5",), "Q", True)])
        self.assertIsNone(bridge.cells[("PLACE", "Q5")])


if __name__ == "__main__":
    unittest.main()
