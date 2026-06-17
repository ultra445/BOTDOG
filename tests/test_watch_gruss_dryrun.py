from __future__ import annotations

import importlib.util
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_gruss_dryrun.py"
SPEC = importlib.util.spec_from_file_location("watch_gruss_dryrun", SCRIPT_PATH)
watch_gruss_dryrun = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watch_gruss_dryrun)


class FakeProcessedStore:
    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.mark_calls: list[tuple[str, str, str]] = []

    def has_seen(self, key: str) -> bool:
        return key in self.seen

    def mark_seen(self, key: str, win_market_id: str | None, place_market_id: str | None) -> None:
        assert win_market_id is not None
        assert place_market_id is not None
        self.seen.add(key)
        self.mark_calls.append((key, win_market_id, place_market_id))


class FakeRunner:
    instances: list["FakeRunner"] = []

    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.processed_store = FakeProcessedStore()
        self.evaluate_calls: list[tuple[int | None, int | None, bool]] = []

        FakeRunner.instances.append(self)

    def trade_row_count(self) -> int:
        return 0

    def evaluate(self, win_snapshot, place_snapshot, *, debug_strategies: bool, momentum_buffer) -> None:
        self.evaluate_calls.append(
            (
                win_snapshot.metadata.countdown_seconds,
                place_snapshot.metadata.countdown_seconds,
                debug_strategies,
            )
        )

    def trade_rows_since(self, row_count_before: int) -> list[dict[str, str]]:
        return []

    def trade_path(self) -> Path:
        return Path("fake_trades.csv")

    def log_gruss_order_intents(self, trade_rows, win_snapshot, place_snapshot) -> list[object]:
        return []


def _snapshot(sheet_name: str, countdown_seconds: int) -> SimpleNamespace:
    return SimpleNamespace(
        sheet_name=sheet_name,
        metadata=SimpleNamespace(
            countdown_seconds=countdown_seconds,
            countdown_display=f"T-{countdown_seconds}",
            market_id=f"{sheet_name.lower()}-1",
            parent_id="race-1",
            event_path=r"Greyhound Racing\PGR\Test",
            market_title="Test - 20:00",
        ),
        runners=[],
    )


def _state(countdown_seconds: int) -> SimpleNamespace:
    return SimpleNamespace(
        win_snapshot=_snapshot("WIN", countdown_seconds),
        place_snapshot=_snapshot("PLACE", countdown_seconds),
        validation_warnings=[],
        tradable=True,
        skip_reason=None,
    )


def _run_main_with_countdowns(countdowns: list[int]) -> tuple[int, str, FakeRunner]:
    FakeRunner.instances = []
    states = [_state(seconds) for seconds in countdowns]
    reads = iter(states)

    with (
        patch.object(watch_gruss_dryrun, "validate_gruss_dryrun_provider_config", return_value=SimpleNamespace(
            data_provider="gruss_excel",
            order_provider="gruss_excel_dryrun",
        )),
        patch.object(watch_gruss_dryrun, "create_configured_gruss_feed", return_value=object()),
        patch.object(watch_gruss_dryrun, "read_gruss_dryrun_state", side_effect=lambda feed: next(reads)),
        patch.object(watch_gruss_dryrun, "GrussDryRunRunner", FakeRunner),
        patch.object(watch_gruss_dryrun, "print_strategy_registry_diagnostics"),
        patch.object(watch_gruss_dryrun, "_print_mom45_status"),
        patch.object(watch_gruss_dryrun.time, "sleep"),
    ):
        output = StringIO()
        with redirect_stdout(output):
            result = watch_gruss_dryrun.main(
                ["--interval", "0.001", "--max-ticks", str(len(countdowns)), "--debug-strategies"]
            )

    assert FakeRunner.instances
    return result, output.getvalue(), FakeRunner.instances[-1]


class WatchGrussDryRunTests(unittest.TestCase):
    def test_pre_ladder_milestones_do_not_wait_on_legacy_trigger(self) -> None:
        for seconds in (45, 32, 20, 14):
            with self.subTest(seconds=seconds):
                result, output, runner = _run_main_with_countdowns([seconds])

                self.assertEqual(result, 0)
                self.assertNotIn(f"countdown_seconds={seconds} > trigger=2", output)
                self.assertIn(f"milestone={seconds} execution_phase=PRE", output)
                self.assertIn("evaluating PRE systems", output)
                self.assertEqual(runner.evaluate_calls, [(seconds, seconds, True)])

    def test_post_milestone_evaluates_post_phase(self) -> None:
        result, output, runner = _run_main_with_countdowns([1])

        self.assertEqual(result, 0)
        self.assertIn("milestone=1 execution_phase=POST", output)
        self.assertIn("evaluating POST systems", output)
        self.assertEqual(runner.evaluate_calls, [(1, 1, True)])

    def test_processed_key_includes_execution_phase_and_milestone(self) -> None:
        result, output, runner = _run_main_with_countdowns([45, 32])

        self.assertEqual(result, 0)
        self.assertIn("active_milestones=[45, 32, 20, 14, 1]", output)
        self.assertEqual(runner.evaluate_calls, [(45, 45, True), (32, 32, True)])
        self.assertEqual(
            [call[0] for call in runner.processed_store.mark_calls],
            [
                "parent:race-1|milestone=45|phase=PRE",
                "parent:race-1|milestone=32|phase=PRE",
            ],
        )

    def test_pre_fourteen_does_not_block_post(self) -> None:
        result, output, runner = _run_main_with_countdowns([14, 1])

        self.assertEqual(result, 0)
        self.assertIn("milestone=14 execution_phase=PRE", output)
        self.assertIn("milestone=1 execution_phase=POST", output)
        self.assertEqual(runner.evaluate_calls, [(14, 14, True), (1, 1, True)])
        self.assertEqual(
            [call[0] for call in runner.processed_store.mark_calls],
            [
                "parent:race-1|milestone=14|phase=PRE",
                "parent:race-1|milestone=1|phase=POST",
            ],
        )


if __name__ == "__main__":
    unittest.main()

