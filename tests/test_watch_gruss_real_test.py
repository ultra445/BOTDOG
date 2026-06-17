from __future__ import annotations

import csv
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_REAL
from dogbot.gruss.gruss_orders import make_order_intent
from dogbot.gruss.gruss_real_orders import GrussExcelOrderProvider, GrussRealOrderContext


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_gruss_real_test.py"
SPEC = importlib.util.spec_from_file_location("watch_gruss_real_test", SCRIPT_PATH)
watch_gruss_real_test = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watch_gruss_real_test)


VALID_ENV = {
    "DOGBOT_DATA_PROVIDER": "gruss_excel",
    "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
    "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
    "DOGBOT_GRUSS_REAL_TEST_MODE": "true",
    "DOGBOT_GRUSS_REAL_MAX_ORDERS": "1",
    "DOGBOT_GRUSS_REAL_MAX_STAKE": "1",
    "DOGBOT_GRUSS_REAL_PREVIEW": "false",
    "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
    "DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED": "true",
    "DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS": "0",
    "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS": "0",
    "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST": "false",
}


def _place_snapshot():
    return SimpleNamespace(
        sheet_name="PLACE",
        metadata=SimpleNamespace(
            market_id="place-1",
            parent_id="1",
            event_path=r"Greyhound Racing\PGR\Test",
        ),
        runners=[
            SimpleNamespace(runner_name="Runner 1", trap=1, best_back=4.2, best_lay=4.4),
            SimpleNamespace(runner_name="Runner 2", trap=2, best_back=2.4, best_lay=2.5),
            SimpleNamespace(runner_name="Runner 3", trap=3, best_back=3.1, best_lay=3.2),
        ],
    )


def _intent(index: int, *, stake: float = 1.0):
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
        strategy_id=f"BACK_PLACE_{index}",
        course_id="parent:1",
        dry_run=True,
    )


class FakeBridge:
    def __init__(self) -> None:
        self.write_calls = []
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

    def clear_trigger_cells(self, sheet_name, addresses, *, trigger_column, command_columns=None, allow_clear=False):
        for address in addresses:
            self.cells[(sheet_name, address)] = None
        return list(addresses)


class FakeProcessedStore:
    def __init__(self) -> None:
        self.seen = set()
        self.mark_calls = []

    def has_seen(self, key):
        return key in self.seen

    def mark_seen(self, key, win_market_id, place_market_id):
        self.mark_calls.append((key, win_market_id, place_market_id))
        self.seen.add(key)


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


class WatchGrussRealTestTests(unittest.TestCase):
    def test_refuses_if_enable_real_is_absent(self) -> None:
        env = dict(VALID_ENV)
        env.pop("DOGBOT_GRUSS_ENABLE_REAL_ORDERS")

        with self.assertRaisesRegex(RuntimeError, "ENABLE_REAL_ORDERS=true est obligatoire"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_refuses_if_test_mode_is_absent(self) -> None:
        env = dict(VALID_ENV)
        env.pop("DOGBOT_GRUSS_REAL_TEST_MODE")

        with self.assertRaisesRegex(RuntimeError, "REAL_TEST_MODE=true est obligatoire"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_requires_explicit_true_for_real_enable(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_ENABLE_REAL_ORDERS="yes")

        with self.assertRaisesRegex(RuntimeError, "ENABLE_REAL_ORDERS=true est obligatoire"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_refuses_if_max_orders_is_greater_than_one(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_MAX_ORDERS="2")

        with self.assertRaisesRegex(RuntimeError, "MAX_ORDERS doit etre exactement 1"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_accepts_two_euro_max_stake_and_two_euro_force_stake(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_MAX_STAKE="2",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="2",
            DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT="true",
        )

        self.assertEqual(
            watch_gruss_real_test.validate_real_test_environment(env),
            (1, 2.0, 2.0, False),
        )

    def test_refuses_if_max_stake_is_greater_than_two(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_MAX_STAKE="3")

        with self.assertRaisesRegex(RuntimeError, "MAX_STAKE doit etre > 0 et <= 2"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_refuses_non_finite_max_stake(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_MAX_STAKE="nan")

        with self.assertRaisesRegex(RuntimeError, "MAX_STAKE doit etre > 0 et <= 2"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_preview_is_incompatible(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_PREVIEW="true")

        with self.assertRaisesRegex(RuntimeError, "REAL_PREVIEW=true est incompatible"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_write_no_trigger_is_incompatible(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_WRITE_NO_TRIGGER="true")

        with self.assertRaisesRegex(RuntimeError, "WRITE_NO_TRIGGER=true est incompatible"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_hold_trigger_for_visual_test_is_rejected_outside_real_test_mode(self) -> None:
        bridge = FakeBridge()
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_TEST_MODE="false",
            DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST="true",
        )
        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            result = provider.place_order(_intent(1), _context())

        self.assertIn("hold_trigger_for_visual_test_requires_real_test_mode", result.reason)
        self.assertEqual(bridge.write_calls, [])

    def test_force_stake_is_impossible_outside_real_test_mode(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_TEST_MODE="false",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="2",
        )

        with self.assertRaisesRegex(RuntimeError, "FORCE_STAKE exige.*REAL_TEST_MODE=true"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_force_stake_above_two_is_refused(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_MAX_STAKE="2",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="3",
        )

        with self.assertRaisesRegex(RuntimeError, "FORCE_STAKE doit etre <= .*MAX_STAKE"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_force_stake_allows_large_strategy_stake_and_logs_diagnostics(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1")
        _, _, force_stake, _ = watch_gruss_real_test.validate_real_test_environment(env)
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, patch.dict("os.environ", env, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_test.process_real_test_batch(
                provider=provider,
                intents=[_intent(index, stake=5) for index in range(1, 11)],
                context=_context(),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
                force_stake=force_stake,
            )
            with (Path(tmp) / "gruss_real_order_attempts.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([result.status for result in results].count("GRUSS_REAL_WRITTEN"), 1)
        self.assertEqual([result.reason for result in results[1:]], ["max_orders_reached"] * 9)
        self.assertEqual(results[0].stake_original, 5)
        self.assertEqual(results[0].stake_used, 1)
        self.assertTrue(results[0].stake_forced)
        self.assertTrue(all(result.stake_original == 5 for result in results))
        self.assertTrue(all(result.stake_used == 1 for result in results))
        self.assertTrue(all(result.stake_forced for result in results))
        self.assertEqual(bridge.write_calls[0][1][1], ("S5", 1.0))
        self.assertEqual(len(bridge.write_calls), 1)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["stake_original"], "5")
        self.assertEqual(rows[0]["stake_used"], "1.0")
        self.assertEqual(rows[0]["stake_forced"], "True")
        self.assertEqual([row["reason"] for row in rows[1:]], ["max_orders_reached"] * 9)

    def test_without_force_large_strategy_stake_is_capped(self) -> None:
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp, patch.dict("os.environ", VALID_ENV, clear=True):
            provider = GrussExcelOrderProvider(tmp, bridge=bridge)
            results, _ = watch_gruss_real_test.process_real_test_batch(
                provider=provider,
                intents=[_intent(1, stake=5)],
                context=_context(),
                processed_store=FakeProcessedStore(),
                key="parent:1",
                win_market_id="win-1",
                place_market_id="place-1",
            )

        self.assertEqual(results[0].status, "GRUSS_REAL_WRITTEN")
        self.assertEqual(results[0].reason, "excel_trigger_written")
        self.assertEqual(results[0].stake_original, 5)
        self.assertEqual(results[0].stake_used, 1)
        self.assertFalse(results[0].stake_forced)
        self.assertTrue(results[0].stake_capped)
        self.assertEqual(results[0].stake_cap_value, 1.0)
        self.assertEqual(bridge.write_calls[0][1][1], ("S5", 1.0))

    def test_force_bsp_place_generates_exactly_one_intent_and_skips_strategies(self) -> None:
        normal_calls = []

        intents = watch_gruss_real_test.build_real_test_intents(
            force_test_bsp_place=True,
            place_snapshot=_place_snapshot(),
            normal_intents_factory=lambda: normal_calls.append(True),
            layout=watch_gruss_real_test.GrussTriggerLayout(),
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(normal_calls, [])
        intent = intents[0]
        self.assertEqual(intent.strategy_id, "GRUSS_FORCE_TEST_BSP_PLACE")
        self.assertEqual(intent.market_type, "PLACE")
        self.assertEqual(intent.side, "BACK")
        self.assertEqual(intent.order_type, "SP_MOC")
        self.assertEqual(intent.stake, 1.0)
        self.assertEqual(intent.runner_name, "Runner 2")
        self.assertEqual(intent.trap, 2)
        self.assertEqual(intent.price, 2.4)
        self.assertEqual(intent.selected_reason, "lowest_place_odds")
        self.assertTrue(intent.force_test_bsp_place)

    def test_force_bsp_place_refuses_absent_place_market(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "forced_bsp_place_market_absent"):
            watch_gruss_real_test.build_force_test_bsp_place_intent(
                None,
                watch_gruss_real_test.GrussTriggerLayout(),
            )

    def test_force_bsp_place_refuses_missing_backsp_mapping(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "back_sp_mapping_unavailable"):
            watch_gruss_real_test.build_force_test_bsp_place_intent(
                _place_snapshot(),
                watch_gruss_real_test.GrussTriggerLayout(back_sp_moc_trigger=""),
            )

    def test_force_bsp_place_requires_real_test_mode(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_TEST_MODE="false",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1",
            DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE="true",
        )

        with self.assertRaisesRegex(RuntimeError, "REAL_TEST_MODE=true"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_force_bsp_place_requires_max_orders_one(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1",
            DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE="true",
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
        )

        with self.assertRaisesRegex(RuntimeError, "MAX_ORDERS doit etre exactement 1"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_force_bsp_place_accepts_two_euro_cap_and_force_stake(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="2",
            DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="2",
        )

        self.assertEqual(
            watch_gruss_real_test.validate_real_test_environment(env),
            (1, 2.0, 2.0, True),
        )

    def test_force_bsp_place_requires_force_stake(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE="true",
        )

        with self.assertRaisesRegex(RuntimeError, "FORCE_TEST_BSP_PLACE exige.*FORCE_STAKE"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_force_back_place_limit_generates_one_intent_at_best_lay_and_skips_strategies(self) -> None:
        normal_calls = []

        intents = watch_gruss_real_test.build_real_test_intents(
            force_test_bsp_place=False,
            force_test_back_place_limit=True,
            place_snapshot=_place_snapshot(),
            normal_intents_factory=lambda: normal_calls.append(True),
            layout=watch_gruss_real_test.GrussTriggerLayout(),
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(normal_calls, [])
        intent = intents[0]
        self.assertEqual(intent.strategy_id, "GRUSS_FORCE_TEST_BACK_PLACE_LIMIT")
        self.assertEqual(intent.market_type, "PLACE")
        self.assertEqual(intent.side, "BACK")
        self.assertEqual(intent.order_type, "LIMIT")
        self.assertEqual(intent.stake, 1.0)
        self.assertTrue(intent.stake_forced)
        self.assertTrue(intent.force_test_back_place_limit)
        self.assertEqual(intent.runner_name, "Runner 2")
        self.assertEqual(intent.trap, 2)
        self.assertEqual(intent.selected_place_back_odds, 2.4)
        self.assertEqual(intent.selected_place_lay_odds, 2.5)
        self.assertEqual(intent.price, 2.5)
        self.assertEqual(intent.price_used, 2.5)
        self.assertEqual(intent.selected_reason, "lowest_place_odds_best_lay_price")

    def test_force_back_place_limit_refuses_missing_best_lay(self) -> None:
        snapshot = _place_snapshot()
        snapshot.runners[1].best_lay = None

        with self.assertRaisesRegex(RuntimeError, "missing_place_best_lay"):
            watch_gruss_real_test.build_force_test_back_place_limit_intent(snapshot)

    def test_force_back_place_limit_and_bsp_are_mutually_exclusive(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1",
            DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE="true",
            DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT="true",
        )

        with self.assertRaisesRegex(RuntimeError, "sont incompatibles"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_force_back_place_limit_requires_force_stake(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT="true",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "FORCE_TEST_BACK_PLACE_LIMIT exige.*FORCE_STAKE",
        ):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_force_back_place_limit_requires_real_test_mode(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_TEST_MODE="false",
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1",
            DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT="true",
        )

        with self.assertRaisesRegex(RuntimeError, "REAL_TEST_MODE=true"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_force_back_place_limit_requires_max_orders_one(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1",
            DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT="true",
            DOGBOT_GRUSS_REAL_MAX_ORDERS="2",
        )

        with self.assertRaisesRegex(RuntimeError, "MAX_ORDERS doit etre exactement 1"):
            watch_gruss_real_test.validate_real_test_environment(env)

    def test_force_back_place_limit_accepts_two_euro_cap_and_force_stake(self) -> None:
        env = dict(
            VALID_ENV,
            DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="2",
            DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT="true",
            DOGBOT_GRUSS_REAL_MAX_STAKE="2",
        )

        self.assertEqual(
            watch_gruss_real_test.validate_real_test_environment(env),
            (1, 2.0, 2.0, False),
        )

    def test_batch_of_ten_writes_one_trigger_and_rejects_nine(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.count = 0
                self.intents = []

            def place_order(self, intent, context):
                self.intents.append(intent)
                if self.count >= 1:
                    return SimpleNamespace(
                        status="REJECTED_REAL",
                        reason="max_orders_reached",
                        trigger_written=False,
                    )
                self.count += 1
                return SimpleNamespace(
                    status="GRUSS_REAL_WRITTEN",
                    reason="excel_trigger_written",
                    trigger_written=True,
                )

        provider = FakeProvider()
        store = FakeProcessedStore()
        context = _context()

        results, skipped = watch_gruss_real_test.process_real_test_batch(
            provider=provider,
            intents=[_intent(index) for index in range(1, 11)],
            context=context,
            processed_store=store,
            key="parent:1",
            win_market_id="win-1",
            place_market_id="place-1",
            force_stake=1,
        )

        self.assertFalse(skipped)
        self.assertEqual([result.status for result in results].count("GRUSS_REAL_WRITTEN"), 1)
        self.assertEqual([result.reason for result in results[1:]], ["max_orders_reached"] * 9)
        self.assertEqual([result.trigger_written for result in results], [True] + [False] * 9)
        self.assertTrue(all(intent.stake == 1 for intent in provider.intents))
        self.assertTrue(all(intent.stake_forced for intent in provider.intents))
        self.assertEqual(store.mark_calls, [("parent:1", "win-1", "place-1")])

    def test_refuses_non_visible_workbook(self) -> None:
        class HiddenBridge:
            def connect_open_workbook(self):
                return True

            def is_workbook_visible(self):
                return False

        with self.assertRaisesRegex(RuntimeError, "ouvert mais non visible"):
            watch_gruss_real_test.ensure_open_visible_workbook(HiddenBridge())

    def test_waits_only_outside_active_pre_post_milestones(self) -> None:
        for seconds in (45, 32, 20, 14, 1):
            with self.subTest(seconds=seconds):
                self.assertIsNone(watch_gruss_real_test.countdown_wait_reason(seconds, seconds))
        self.assertEqual(
            watch_gruss_real_test.countdown_wait_reason(3, 3),
            "wait: countdown_seconds=3 next_milestone=1 execution_phase=POST",
        )

    def test_valid_environment_is_accepted(self) -> None:
        self.assertEqual(
            watch_gruss_real_test.validate_real_test_environment(VALID_ENV),
            (1, 1.0, None, False),
        )


if __name__ == "__main__":
    unittest.main()

