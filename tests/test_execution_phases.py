from __future__ import annotations

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from dogbot.executor import (
    Executor,
    POST_SEND_SECONDS_BEFORE_OFF,
    PRE_SEND_SECONDS_BEFORE_OFF,
    _StrategyOrderCandidate,
    _execution_phase_for_milestone,
    _merge_order_candidates,
)
from dogbot.gruss.gruss_dryrun_engine import strategy_registry_diagnostics
from dogbot.gruss.gruss_dryrun_engine import (
    active_strategy_milestones,
    current_strategy_milestone,
    describe_current_strategy_milestone,
)
from dogbot.pre_ladder import round_final_lim_to_ladder_tick
from dogbot.strategies import EXECUTION_PHASE_POST, EXECUTION_PHASE_PRE, ExecMode, Slot, build_registry
from dogbot.staking import Side


def _candidate(
    system: str,
    *,
    phase: str,
    side: str = "BACK",
    price: float = 3.0,
    size: float = 2.0,
    phase_seconds: int | None = None,
    best_unmatched_back_offer: float | None = None,
    best_unmatched_lay_offer: float | None = None,
) -> _StrategyOrderCandidate:
    slot = SimpleNamespace(
        tag=system,
        family=system.rsplit("_", 1)[0],
        slot=int(system.rsplit("_", 1)[1]),
        market_family="PLACE",
    )
    return _StrategyOrderCandidate(
        slot=slot,
        market_id="market-1",
        market_type="PLACE",
        selection_id=1,
        course_id="course-1",
        side=side,
        price=price,
        size=size,
        liability=round(size * max(0.01, price - 1.0), 2),
        reason="cond_ok",
        exec_mode=ExecMode.LIMIT_LTP,
        sp_limit=None,
        execution_phase=phase,
        triggered_systems=[system],
        triggered_prices=[price],
        bet_per_market_key=(slot.family, slot.slot, "market-1"),
        phase_send_seconds_before_off=phase_seconds,
        best_unmatched_back_offer=best_unmatched_back_offer,
        best_unmatched_lay_offer=best_unmatched_lay_offer,
    )


class ExecutionPhaseTests(unittest.TestCase):
    EXPECTED_PRE_SYSTEMS = {
        "BACK_PLACE_101",
        "BACK_PLACE_102",
        "BACK_PLACE_104",
        "BACK_PLACE_201",
        "BACK_PLACE_202",
        "BACK_PLACE_203",
        "LAY_PLACE_301",
        "LAY_PLACE_302",
        "LAY_PLACE_303",
        "LAY_PLACE_304",
        "LAY_PLACE_305",
        "LAY_PLACE_351",
    }

    def test_slots_default_to_post_phase(self) -> None:
        slot = Slot(
            family="BACK_PLACE",
            slot=999,
            side=Side.BACK,
            condition=lambda ctx: True,
        )

        self.assertEqual(slot.execution_phase, EXECUTION_PHASE_POST)

    def test_phase_timing_constants_map_to_expected_phases(self) -> None:
        self.assertEqual(_execution_phase_for_milestone(PRE_SEND_SECONDS_BEFORE_OFF), EXECUTION_PHASE_PRE)
        self.assertEqual(_execution_phase_for_milestone(20), EXECUTION_PHASE_PRE)
        self.assertEqual(_execution_phase_for_milestone(15), EXECUTION_PHASE_PRE)
        self.assertEqual(_execution_phase_for_milestone(10), EXECUTION_PHASE_PRE)
        self.assertEqual(_execution_phase_for_milestone(5), EXECUTION_PHASE_PRE)
        self.assertEqual(_execution_phase_for_milestone(POST_SEND_SECONDS_BEFORE_OFF), EXECUTION_PHASE_POST)
        self.assertIsNone(_execution_phase_for_milestone(2))

    def test_multiple_systems_merge_only_inside_same_phase(self) -> None:
        merged = _merge_order_candidates(
            [
                _candidate("BACK_PLACE_101", phase=EXECUTION_PHASE_PRE, price=3.0, size=2.0),
                _candidate("BACK_PLACE_103", phase=EXECUTION_PHASE_PRE, price=3.4, size=4.0),
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].execution_phase, EXECUTION_PHASE_PRE)
        self.assertEqual(merged[0].triggered_systems, ["BACK_PLACE_101", "BACK_PLACE_103"])
        self.assertEqual(merged[0].triggered_prices, [3.0, 3.4])
        self.assertEqual(merged[0].price, 3.0)
        self.assertEqual(merged[0].size, 6.0)

    def test_pre_and_post_same_runner_side_are_not_duplicates(self) -> None:
        merged = _merge_order_candidates(
            [
                _candidate("BACK_PLACE_101", phase=EXECUTION_PHASE_PRE, price=3.0, size=2.0),
                _candidate("BACK_PLACE_103", phase=EXECUTION_PHASE_POST, price=3.4, size=4.0),
            ]
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            {candidate.execution_phase for candidate in merged},
            {EXECUTION_PHASE_PRE, EXECUTION_PHASE_POST},
        )

    def test_two_post_systems_merge_inside_post_phase(self) -> None:
        merged = _merge_order_candidates(
            [
                _candidate("BACK_PLACE_101", phase=EXECUTION_PHASE_POST, price=3.0, size=2.0),
                _candidate("BACK_PLACE_103", phase=EXECUTION_PHASE_POST, price=3.4, size=4.0),
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].execution_phase, EXECUTION_PHASE_POST)
        self.assertEqual(merged[0].triggered_systems, ["BACK_PLACE_101", "BACK_PLACE_103"])
        self.assertEqual(merged[0].price, 3.0)
        self.assertEqual(merged[0].size, 6.0)

    def test_lay_merge_keeps_highest_limit_price(self) -> None:
        merged = _merge_order_candidates(
            [
                _candidate("LAY_PLACE_301", phase=EXECUTION_PHASE_POST, side="LAY", price=4.2, size=2.0),
                _candidate("LAY_PLACE_302", phase=EXECUTION_PHASE_POST, side="LAY", price=3.8, size=4.0),
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].price, 4.2)
        self.assertEqual(merged[0].size, 6.0)

    def test_registry_execution_phase_assignment_is_exact(self) -> None:
        phase_by_id = {slot.tag: slot.execution_phase for slot in build_registry()}
        missing = self.EXPECTED_PRE_SYSTEMS.difference(phase_by_id)

        self.assertEqual(missing, set())
        self.assertEqual(
            {strategy_id for strategy_id, phase in phase_by_id.items() if phase == EXECUTION_PHASE_PRE},
            self.EXPECTED_PRE_SYSTEMS,
        )
        for strategy_id, phase in phase_by_id.items():
            expected_phase = (
                EXECUTION_PHASE_PRE
                if strategy_id in self.EXPECTED_PRE_SYSTEMS
                else EXECUTION_PHASE_POST
            )
            self.assertEqual(phase, expected_phase, strategy_id)

    def test_registry_diagnostics_reports_phase_counts_and_lists(self) -> None:
        diagnostics = strategy_registry_diagnostics()

        self.assertEqual(set(diagnostics["pre_ids"]), self.EXPECTED_PRE_SYSTEMS)
        self.assertEqual(diagnostics["by_execution_phase"], {"PRE": 12, "POST": 18})
        self.assertEqual(len(diagnostics["post_ids"]), 18)

    def test_pre_ladder_missing_same_side_offer_falls_back_only_at_last_step(self) -> None:
        executor = object.__new__(Executor)
        executor._pre_ladder_steps = (20, 15, 10, 5)
        executor._phase_stakes_by_runner_side = defaultdict(
            lambda: {EXECUTION_PHASE_PRE: 0.0, EXECUTION_PHASE_POST: 0.0}
        )
        early = _candidate("BACK_PLACE_101", phase=EXECUTION_PHASE_PRE, phase_seconds=20)
        final = _candidate("BACK_PLACE_101", phase=EXECUTION_PHASE_PRE, phase_seconds=5)

        self.assertIsNone(executor._pre_ladder_step_payload(early))
        payload = executor._pre_ladder_step_payload(final)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["fallback_reason"], "no_same_side_offer")
        self.assertEqual(payload["current_ladder_price"], 3.0)
        self.assertEqual(payload["ladder_step"], "4/4")

    def test_pre_ladder_preview_logs_step_price_and_stake_in_main_columns(self) -> None:
        executor = object.__new__(Executor)
        executor._pre_ladder_steps = (20, 15, 10, 5)
        executor._phase_stakes_by_runner_side = defaultdict(
            lambda: {EXECUTION_PHASE_PRE: 0.0, EXECUTION_PHASE_POST: 0.0}
        )
        rows = []
        executor._log_trade_row = rows.append
        order = _candidate(
            "BACK_PLACE_101",
            phase=EXECUTION_PHASE_PRE,
            price=5.0,
            size=2.0,
            phase_seconds=20,
            best_unmatched_back_offer=7.0,
        )

        with patch.dict(
            "os.environ",
            {
                "DOGBOT_PRE_LADDER_ENABLED": "false",
                "DOGBOT_PRE_LADDER_PREVIEW": "true",
            },
        ):
            handled = executor._handle_pre_ladder_order(order)

        self.assertTrue(handled)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "PRE_LADDER_PREVIEW")
        self.assertEqual(row["price_req"], row["current_ladder_price"])
        self.assertEqual(row["size_req"], row["current_step_stake"])
        self.assertEqual(row["final_price"], row["current_ladder_price"])
        self.assertEqual(row["final_stake"], row["current_step_stake"])
        self.assertEqual(row["final_lim_price"], 5.0)
        self.assertEqual(row["final_lim_price_raw"], 5.0)
        self.assertEqual(row["final_lim_price_tick"], 5.0)
        self.assertEqual(row["start_price_raw"], 7.0)
        self.assertEqual(row["start_price_tick"], 6.8)
        self.assertEqual(row["tick_rounding_mode"], "BACK_CEIL")
        self.assertEqual(row["best_same_side_back_offer"], 7.0)
        self.assertEqual(row["best_same_side_lay_offer"], "")
        self.assertEqual(row["no_better_ladder_range_reason"], "")
        self.assertIn("runner.ex.available_to_back[0].price", row["source_fields_used"])
        self.assertIn("ignored_executable_against_back", row["source_fields_used"])
        self.assertEqual(row["ladder_step"], "1/4")
        self.assertIn("5.0", row["ladder_prices"])
        self.assertEqual(row["gruss_planned_trigger"], "BACK")
        self.assertEqual(row["gruss_trigger_allowed"], "True")
        self.assertEqual(row["gruss_bet_ref_required"], "False")
        self.assertEqual(row["gruss_no_stack"], "False")

    def test_pre_ladder_preview_after_first_step_requires_bet_ref_and_does_not_stack(self) -> None:
        executor = object.__new__(Executor)
        executor._pre_ladder_steps = (20, 15, 10, 5)
        executor._phase_stakes_by_runner_side = defaultdict(
            lambda: {EXECUTION_PHASE_PRE: 0.0, EXECUTION_PHASE_POST: 0.0}
        )
        rows = []
        executor._log_trade_row = rows.append
        order = _candidate(
            "BACK_PLACE_101",
            phase=EXECUTION_PHASE_PRE,
            price=5.0,
            size=2.0,
            phase_seconds=15,
            best_unmatched_back_offer=7.0,
        )

        with patch.dict(
            "os.environ",
            {
                "DOGBOT_PRE_LADDER_ENABLED": "false",
                "DOGBOT_PRE_LADDER_PREVIEW": "true",
            },
        ):
            handled = executor._handle_pre_ladder_order(order)

        self.assertTrue(handled)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "PRE_LADDER_PREVIEW")
        self.assertEqual(row["reason"], "missing_bet_ref_do_not_stack")
        self.assertEqual(row["gruss_planned_trigger"], "")
        self.assertEqual(row["gruss_trigger_allowed"], "False")
        self.assertEqual(row["gruss_bet_ref_required"], "True")
        self.assertEqual(row["gruss_bet_ref_present"], "False")
        self.assertEqual(row["gruss_no_stack"], "True")
        self.assertEqual(row["gruss_update_confirmed"], "False")

    def test_pre_ladder_preview_invalid_back_range_logs_direct_limit_price(self) -> None:
        executor = object.__new__(Executor)
        executor._pre_ladder_steps = (20, 15, 10, 5)
        executor._phase_stakes_by_runner_side = defaultdict(
            lambda: {EXECUTION_PHASE_PRE: 0.0, EXECUTION_PHASE_POST: 0.0}
        )
        rows = []
        executor._log_trade_row = rows.append
        order = _candidate(
            "BACK_PLACE_101",
            phase=EXECUTION_PHASE_PRE,
            price=5.0,
            size=2.0,
            phase_seconds=20,
            best_unmatched_back_offer=4.0,
        )

        with patch.dict(
            "os.environ",
            {
                "DOGBOT_PRE_LADDER_ENABLED": "false",
                "DOGBOT_PRE_LADDER_PREVIEW": "true",
            },
        ):
            self.assertTrue(executor._handle_pre_ladder_order(order))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["reason"], "no_better_back_ladder_range")
        self.assertEqual(row["current_ladder_price"], 5.0)
        self.assertEqual(row["final_price"], 5.0)
        self.assertEqual(row["final_lim_price"], 5.0)
        self.assertEqual(row["final_lim_price_raw"], 5.0)
        self.assertEqual(row["final_lim_price_tick"], 5.0)
        self.assertEqual(row["start_price_raw"], 4.0)
        self.assertEqual(row["start_price_tick"], "")
        self.assertEqual(row["tick_rounding_mode"], "BACK_CEIL")
        self.assertEqual(row["best_same_side_back_offer"], 4.0)
        self.assertEqual(row["best_same_side_lay_offer"], "")
        self.assertEqual(row["no_better_ladder_range_reason"], "no_better_back_ladder_range")
        self.assertIn("runner.ex.available_to_back[0].price", row["source_fields_used"])
        self.assertEqual(row["ladder_prices"], "5.0")
        self.assertEqual(row["ladder_step"], "1/4")

    def test_pre_ladder_preview_invalid_lay_range_logs_direct_limit_price(self) -> None:
        executor = object.__new__(Executor)
        executor._pre_ladder_steps = (20, 15, 10, 5)
        executor._phase_stakes_by_runner_side = defaultdict(
            lambda: {EXECUTION_PHASE_PRE: 0.0, EXECUTION_PHASE_POST: 0.0}
        )
        rows = []
        executor._log_trade_row = rows.append
        order = _candidate(
            "LAY_PLACE_301",
            phase=EXECUTION_PHASE_PRE,
            side="LAY",
            price=8.0,
            size=2.0,
            phase_seconds=20,
            best_unmatched_lay_offer=9.0,
        )

        with patch.dict(
            "os.environ",
            {
                "DOGBOT_PRE_LADDER_ENABLED": "false",
                "DOGBOT_PRE_LADDER_PREVIEW": "true",
            },
        ):
            self.assertTrue(executor._handle_pre_ladder_order(order))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["reason"], "no_better_lay_ladder_range")
        self.assertEqual(row["current_ladder_price"], 8.0)
        self.assertEqual(row["final_price"], 8.0)
        self.assertEqual(row["final_lim_price_raw"], 8.0)
        self.assertEqual(row["final_lim_price_tick"], 8.0)
        self.assertEqual(row["start_price_raw"], 9.0)
        self.assertEqual(row["start_price_tick"], "")
        self.assertEqual(row["tick_rounding_mode"], "LAY_FLOOR")
        self.assertEqual(row["best_same_side_back_offer"], "")
        self.assertEqual(row["best_same_side_lay_offer"], 9.0)
        self.assertEqual(row["no_better_ladder_range_reason"], "no_better_lay_ladder_range")
        self.assertIn("runner.ex.available_to_lay[0].price", row["source_fields_used"])
        self.assertEqual(row["ladder_prices"], "8.0")
        self.assertEqual(row["ladder_step"], "1/4")

    def test_pre_ladder_uses_same_side_offer_not_immediately_executable_price(self) -> None:
        for order, expected_current, expected_start, expected_source in (
            (
                _candidate(
                    "BACK_PLACE_101",
                    phase=EXECUTION_PHASE_PRE,
                    side="BACK",
                    price=5.0,
                    size=2.0,
                    phase_seconds=20,
                    best_unmatched_back_offer=7.0,
                    best_unmatched_lay_offer=4.0,
                ),
                6.8,
                6.8,
                "best_same_side_back_offer=runner.ex.available_to_back[0].price",
            ),
            (
                _candidate(
                    "LAY_PLACE_301",
                    phase=EXECUTION_PHASE_PRE,
                    side="LAY",
                    price=8.0,
                    size=2.0,
                    phase_seconds=20,
                    best_unmatched_back_offer=9.0,
                    best_unmatched_lay_offer=6.0,
                ),
                6.2,
                6.2,
                "best_same_side_lay_offer=runner.ex.available_to_lay[0].price",
            ),
        ):
            with self.subTest(side=order.side):
                executor = object.__new__(Executor)
                executor._pre_ladder_steps = (20, 15, 10, 5)
                executor._phase_stakes_by_runner_side = defaultdict(
                    lambda: {EXECUTION_PHASE_PRE: 0.0, EXECUTION_PHASE_POST: 0.0}
                )
                rows = []
                executor._log_trade_row = rows.append

                with patch.dict(
                    "os.environ",
                    {
                        "DOGBOT_PRE_LADDER_ENABLED": "false",
                        "DOGBOT_PRE_LADDER_PREVIEW": "true",
                    },
                ):
                    self.assertTrue(executor._handle_pre_ladder_order(order))

                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["current_ladder_price"], expected_current)
                self.assertEqual(row["start_price_tick"], expected_start)
                self.assertEqual(row["reason"], "initial_ladder_trigger_preview_only")
                self.assertIn(expected_source, row["source_fields_used"])
                if order.side == "BACK":
                    self.assertEqual(row["best_same_side_back_offer"], 7.0)
                    self.assertEqual(row["best_unmatched_lay_offer"], 4.0)
                    self.assertIn("ignored_executable_against_back", row["source_fields_used"])
                else:
                    self.assertEqual(row["best_same_side_lay_offer"], 6.0)
                    self.assertEqual(row["best_unmatched_back_offer"], 9.0)
                    self.assertIn("ignored_executable_against_lay", row["source_fields_used"])

    def test_pre_ladder_preview_prices_never_cross_final_limit_tick(self) -> None:
        for side, system, final_lim_price, same_side_offer, phase_seconds in (
            ("BACK", "BACK_PLACE_101", 8.393, 20.0, 20),
            ("BACK", "BACK_PLACE_101", 8.214, 6.8, 20),
            ("BACK", "BACK_PLACE_101", 12.155, 20.0, 20),
            ("BACK", "BACK_PLACE_101", 11.218, 11.0, 20),
            ("BACK", "BACK_PLACE_101", 8.393, 20.0, 5),
            ("LAY", "LAY_PLACE_301", 8.393, 4.0, 20),
            ("LAY", "LAY_PLACE_301", 8.214, 9.0, 20),
            ("LAY", "LAY_PLACE_301", 8.393, 4.0, 5),
        ):
            with self.subTest(
                side=side,
                final_lim_price=final_lim_price,
                same_side_offer=same_side_offer,
                phase_seconds=phase_seconds,
            ):
                executor = object.__new__(Executor)
                executor._pre_ladder_steps = (20, 15, 10, 5)
                executor._phase_stakes_by_runner_side = defaultdict(
                    lambda: {EXECUTION_PHASE_PRE: 0.0, EXECUTION_PHASE_POST: 0.0}
                )
                rows = []
                executor._log_trade_row = rows.append
                order = _candidate(
                    system,
                    phase=EXECUTION_PHASE_PRE,
                    side=side,
                    price=final_lim_price,
                    size=2.0,
                    phase_seconds=phase_seconds,
                    best_unmatched_back_offer=same_side_offer if side == "BACK" else None,
                    best_unmatched_lay_offer=same_side_offer if side == "LAY" else None,
                )

                with patch.dict(
                    "os.environ",
                    {
                        "DOGBOT_PRE_LADDER_ENABLED": "false",
                        "DOGBOT_PRE_LADDER_PREVIEW": "true",
                    },
                ):
                    self.assertTrue(executor._handle_pre_ladder_order(order))

                self.assertEqual(len(rows), 1)
                row = rows[0]
                final_tick = round_final_lim_to_ladder_tick(side, final_lim_price)
                self.assertEqual(row["final_lim_price_raw"], final_lim_price)
                self.assertEqual(row["final_lim_price_tick"], final_tick)
                self.assertEqual(row["tick_rounding_mode"], "BACK_CEIL" if side == "BACK" else "LAY_FLOOR")
                current = float(row["current_ladder_price"])
                if side == "BACK":
                    self.assertGreaterEqual(current, final_tick)
                    if row["reason"] == "no_better_back_ladder_range":
                        self.assertEqual(current, final_tick)
                else:
                    self.assertLessEqual(current, final_tick)
                    if row["reason"] == "no_better_lay_ladder_range":
                        self.assertEqual(current, final_tick)
                if row["ladder_step"] == "4/4":
                    self.assertEqual(current, final_tick)

    def test_generated_pre_ladder_preview_rows_respect_global_invariants(self) -> None:
        scenarios = (
            _candidate(
                "BACK_PLACE_101",
                phase=EXECUTION_PHASE_PRE,
                side="BACK",
                price=8.393,
                size=2.0,
                phase_seconds=20,
                best_unmatched_back_offer=20.0,
            ),
            _candidate(
                "BACK_PLACE_101",
                phase=EXECUTION_PHASE_PRE,
                side="BACK",
                price=8.393,
                size=2.0,
                phase_seconds=5,
                best_unmatched_back_offer=20.0,
            ),
            _candidate(
                "BACK_PLACE_101",
                phase=EXECUTION_PHASE_PRE,
                side="BACK",
                price=8.393,
                size=2.0,
                phase_seconds=20,
                best_unmatched_back_offer=6.8,
            ),
            _candidate(
                "LAY_PLACE_301",
                phase=EXECUTION_PHASE_PRE,
                side="LAY",
                price=8.393,
                size=2.0,
                phase_seconds=20,
                best_unmatched_lay_offer=4.0,
            ),
            _candidate(
                "LAY_PLACE_301",
                phase=EXECUTION_PHASE_PRE,
                side="LAY",
                price=8.393,
                size=2.0,
                phase_seconds=5,
                best_unmatched_lay_offer=4.0,
            ),
            _candidate(
                "LAY_PLACE_301",
                phase=EXECUTION_PHASE_PRE,
                side="LAY",
                price=8.393,
                size=2.0,
                phase_seconds=20,
                best_unmatched_lay_offer=9.0,
            ),
        )
        rows = []
        for order in scenarios:
            executor = object.__new__(Executor)
            executor._pre_ladder_steps = (20, 15, 10, 5)
            executor._phase_stakes_by_runner_side = defaultdict(
                lambda: {EXECUTION_PHASE_PRE: 0.0, EXECUTION_PHASE_POST: 0.0}
            )
            executor._log_trade_row = rows.append
            with patch.dict(
                "os.environ",
                {
                    "DOGBOT_PRE_LADDER_ENABLED": "false",
                    "DOGBOT_PRE_LADDER_PREVIEW": "true",
                },
            ):
                self.assertTrue(executor._handle_pre_ladder_order(order))

        self.assertTrue(rows)
        for row in rows:
            with self.subTest(side=row["side"], ladder_step=row["ladder_step"], reason=row["reason"]):
                self.assertEqual(row["status"], "PRE_LADDER_PREVIEW")
                final_tick = float(row["final_lim_price_tick"])
                current = float(row["current_ladder_price"])
                if row["side"] == "BACK":
                    self.assertGreaterEqual(current, final_tick)
                else:
                    self.assertLessEqual(current, final_tick)
                if row["ladder_step"] == "4/4":
                    self.assertEqual(current, final_tick)
                if row["no_better_ladder_range_reason"] in {
                    "no_better_back_ladder_range",
                    "no_better_lay_ladder_range",
                }:
                    self.assertEqual(current, final_tick)

    def test_pre_ladder_defaults_to_preview_without_real_update_trigger(self) -> None:
        executor = object.__new__(Executor)
        executor._pre_ladder_steps = (20, 15, 10, 5)
        executor._phase_stakes_by_runner_side = defaultdict(
            lambda: {EXECUTION_PHASE_PRE: 0.0, EXECUTION_PHASE_POST: 0.0}
        )
        rows = []
        executor._log_trade_row = rows.append
        order = _candidate(
            "BACK_PLACE_101",
            phase=EXECUTION_PHASE_PRE,
            side="BACK",
            price=5.0,
            size=2.0,
            phase_seconds=15,
            best_unmatched_back_offer=7.0,
        )

        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(executor._handle_pre_ladder_order(order))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["ladder_enabled"], "False")
        self.assertEqual(row["ladder_preview"], "True")
        self.assertEqual(row["status"], "PRE_LADDER_PREVIEW")
        self.assertNotEqual(row["gruss_planned_trigger"], "UPDATE")
        self.assertEqual(row["gruss_update_confirmed"], "False")

    def test_pre_ladder_preview_does_not_change_post_phase_mapping(self) -> None:
        self.assertEqual(_execution_phase_for_milestone(0), EXECUTION_PHASE_POST)

    def test_gruss_active_milestones_include_pre_ladder_steps_and_post(self) -> None:
        self.assertEqual(active_strategy_milestones(), (20, 15, 10, 5, 0))
        for seconds in (20, 15, 10, 5):
            with self.subTest(seconds=seconds):
                self.assertEqual(current_strategy_milestone(seconds, seconds), seconds)
                self.assertIn("execution_phase=PRE", describe_current_strategy_milestone(seconds, seconds))
        self.assertIn("execution_phase=POST", describe_current_strategy_milestone(0, 0))

    def test_phase_filter_selects_only_pre_or_post_strategy_slots(self) -> None:
        phase_by_id = {slot.tag: slot.execution_phase for slot in build_registry()}
        pre_phase = _execution_phase_for_milestone(20)
        post_phase = _execution_phase_for_milestone(0)

        self.assertEqual(
            {strategy_id for strategy_id, phase in phase_by_id.items() if phase == pre_phase},
            self.EXPECTED_PRE_SYSTEMS,
        )
        self.assertEqual(
            {strategy_id for strategy_id, phase in phase_by_id.items() if phase == post_phase},
            set(phase_by_id).difference(self.EXPECTED_PRE_SYSTEMS),
        )


if __name__ == "__main__":
    unittest.main()
