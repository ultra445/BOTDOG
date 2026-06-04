from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from dogbot.gruss.gruss_orders import make_order_intent
from dogbot.gruss.gruss_real_orders import GrussRealOrderContext


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_gruss_write_no_trigger.py"
SPEC = importlib.util.spec_from_file_location("watch_gruss_write_no_trigger", SCRIPT_PATH)
watch_gruss_write_no_trigger = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watch_gruss_write_no_trigger)


VALID_ENV = {
    "DOGBOT_DATA_PROVIDER": "gruss_excel",
    "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
    "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "false",
    "DOGBOT_GRUSS_REAL_PREVIEW": "false",
    "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "true",
}


def _intent(index: int):
    return make_order_intent(
        provider="gruss_excel",
        market_type="PLACE",
        market_id="place-1",
        parent_id="1",
        runner_name=f"Runner {index}",
        trap=index,
        side="BACK",
        order_type="LIMIT",
        price=3.0 + index / 10,
        stake=2.0,
        strategy_id=f"BACK_PLACE_{index}",
        course_id="parent:1",
        dry_run=True,
    )


class WatchGrussWriteNoTriggerTests(unittest.TestCase):
    def test_batch_processes_all_intents_then_skips_next_tick(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.intents = []

            def place_order(self, intent, context):
                self.intents.append(intent)
                return SimpleNamespace(status="GRUSS_WRITE_NO_TRIGGER", trigger_written=False)

        class FakeProcessedStore:
            def __init__(self) -> None:
                self.seen = set()
                self.mark_calls = []

            def has_seen(self, key):
                return key in self.seen

            def mark_seen(self, key, win_market_id, place_market_id):
                self.mark_calls.append((key, win_market_id, place_market_id))
                self.seen.add(key)

        intents = [_intent(index) for index in range(1, 4)]
        provider = FakeProvider()
        store = FakeProcessedStore()
        context = GrussRealOrderContext(
            validation_ok=True,
            tradable=True,
            region="UK",
            countdown_seconds=2,
            course="parent:1",
        )

        results, skipped = watch_gruss_write_no_trigger.process_write_no_trigger_batch(
            provider=provider,
            intents=intents,
            context=context,
            processed_store=store,
            key="parent:1",
            win_market_id="win-1",
            place_market_id="place-1",
        )
        second_results, second_skipped = watch_gruss_write_no_trigger.process_write_no_trigger_batch(
            provider=provider,
            intents=intents,
            context=context,
            processed_store=store,
            key="parent:1",
            win_market_id="win-1",
            place_market_id="place-1",
        )

        self.assertFalse(skipped)
        self.assertEqual([result.status for result in results], ["GRUSS_WRITE_NO_TRIGGER"] * 3)
        self.assertTrue(all(result.trigger_written is False for result in results))
        self.assertEqual(len(provider.intents), 3)
        self.assertEqual(store.mark_calls, [("parent:1", "win-1", "place-1")])
        self.assertTrue(second_skipped)
        self.assertEqual(second_results, [])
        self.assertEqual(len(provider.intents), 3)

    def test_requires_write_no_trigger_flag(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_WRITE_NO_TRIGGER="false")

        with self.assertRaisesRegex(RuntimeError, "WRITE_NO_TRIGGER=true est obligatoire"):
            watch_gruss_write_no_trigger.validate_write_no_trigger_environment(env)

    def test_does_not_require_real_orders_enable(self) -> None:
        watch_gruss_write_no_trigger.validate_write_no_trigger_environment(VALID_ENV)

    def test_requires_real_preview_false(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_PREVIEW="true")

        with self.assertRaisesRegex(RuntimeError, "REAL_PREVIEW=false est obligatoire"):
            watch_gruss_write_no_trigger.validate_write_no_trigger_environment(env)

    def test_accepts_real_preview_absent(self) -> None:
        env = dict(VALID_ENV)
        env.pop("DOGBOT_GRUSS_REAL_PREVIEW")
        watch_gruss_write_no_trigger.validate_write_no_trigger_environment(env)

    def test_accepts_real_preview_empty(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_PREVIEW="")
        watch_gruss_write_no_trigger.validate_write_no_trigger_environment(env)

    def test_accepts_real_preview_false(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_REAL_PREVIEW="false")
        watch_gruss_write_no_trigger.validate_write_no_trigger_environment(env)

    def test_accepts_real_enable_but_provider_still_omits_trigger(self) -> None:
        env = dict(VALID_ENV, DOGBOT_GRUSS_ENABLE_REAL_ORDERS="true")
        watch_gruss_write_no_trigger.validate_write_no_trigger_environment(env)

    def test_waits_until_countdown_is_at_most_two_seconds(self) -> None:
        self.assertEqual(
            watch_gruss_write_no_trigger.countdown_wait_reason(3, 3),
            "wait: countdown_seconds=3 > trigger=2",
        )
        self.assertIsNone(watch_gruss_write_no_trigger.countdown_wait_reason(2, 2))


if __name__ == "__main__":
    unittest.main()
