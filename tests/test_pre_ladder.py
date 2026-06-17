from __future__ import annotations

import unittest

from dogbot.pre_ladder import (
    PreLadderOrderState,
    build_pre_ladder_from_same_side_offer,
    build_pre_ladder_prices,
    decide_pre_ladder_step,
    next_tick,
    plan_gruss_pre_ladder_trigger,
    previous_tick,
    round_final_lim_to_ladder_tick,
    round_to_betfair_tick,
)


class PreLadderTests(unittest.TestCase):
    def test_back_start_price_is_one_tick_below_same_side_back_offer(self) -> None:
        self.assertEqual(previous_tick(7.0), 6.8)

    def test_back_ladder_descends_to_final_lim_price(self) -> None:
        prices = build_pre_ladder_prices("BACK", start_price=6.8, final_lim_price=5.0, steps=4)

        self.assertEqual(prices, [6.8, 6.2, 5.6, 5.0])
        self.assertEqual(prices, sorted(prices, reverse=True))

    def test_back_ladder_refuses_range_below_final_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "no_better_back_ladder_range"):
            build_pre_ladder_prices("BACK", start_price=1.98, final_lim_price=2.82, steps=4)

    def test_lay_start_price_is_one_tick_above_same_side_lay_offer(self) -> None:
        self.assertEqual(next_tick(6.0), 6.2)

    def test_lay_ladder_mounts_to_final_lim_price(self) -> None:
        prices = build_pre_ladder_prices("LAY", start_price=6.2, final_lim_price=8.0, steps=4)

        self.assertEqual(prices, [6.2, 6.8, 7.4, 8.0])
        self.assertEqual(prices, sorted(prices))

    def test_lay_ladder_refuses_range_above_final_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "no_better_lay_ladder_range"):
            build_pre_ladder_prices("LAY", start_price=8.5, final_lim_price=8.0, steps=4)

    def test_prices_respect_betfair_ticks(self) -> None:
        prices = build_pre_ladder_prices("BACK", start_price=6.8, final_lim_price=5.0, steps=4)

        self.assertEqual([round_to_betfair_tick(price) for price in prices], prices)

    def test_final_step_is_always_final_limit_rounded_to_betfair_tick(self) -> None:
        self.assertEqual(
            build_pre_ladder_prices("BACK", start_price=6.8, final_lim_price=5.04, steps=4)[-1],
            round_final_lim_to_ladder_tick("BACK", 5.04),
        )
        self.assertEqual(
            build_pre_ladder_prices("LAY", start_price=6.2, final_lim_price=8.11, steps=4)[-1],
            round_final_lim_to_ladder_tick("LAY", 8.11),
        )

    def test_back_final_limit_uses_upper_tick_so_prices_never_go_below_limit(self) -> None:
        for final_lim_price in (8.393, 8.214, 12.155, 11.218):
            with self.subTest(final_lim_price=final_lim_price):
                final_tick = round_final_lim_to_ladder_tick("BACK", final_lim_price)
                plan = build_pre_ladder_from_same_side_offer(
                    "BACK",
                    best_same_side_offer=20.0,
                    final_lim_price=final_lim_price,
                    steps=4,
                )

                self.assertEqual(plan.prices[-1], final_tick)
                self.assertTrue(all(price >= final_tick for price in plan.prices), plan.prices)

    def test_lay_final_limit_uses_lower_tick_so_prices_never_go_above_limit(self) -> None:
        for final_lim_price in (8.393, 8.214, 12.155, 11.218):
            with self.subTest(final_lim_price=final_lim_price):
                final_tick = round_final_lim_to_ladder_tick("LAY", final_lim_price)
                plan = build_pre_ladder_from_same_side_offer(
                    "LAY",
                    best_same_side_offer=4.0,
                    final_lim_price=final_lim_price,
                    steps=4,
                )

                self.assertEqual(plan.prices[-1], final_tick)
                self.assertTrue(all(price <= final_tick for price in plan.prices), plan.prices)

    def test_back_no_range_uses_upper_tick_direct_price(self) -> None:
        plan = build_pre_ladder_from_same_side_offer(
            "BACK",
            best_same_side_offer=6.8,
            final_lim_price=8.393,
            steps=4,
        )

        self.assertEqual(plan.prices, [8.4])
        self.assertEqual(plan.reason, "no_better_back_ladder_range")

    def test_back_same_side_offer_above_final_limit_builds_descending_ladder(self) -> None:
        plan = build_pre_ladder_from_same_side_offer(
            "BACK",
            best_same_side_offer=7.0,
            final_lim_price=5.0,
            steps=4,
        )

        self.assertEqual(plan.start_price, 6.8)
        self.assertEqual(plan.prices, [6.8, 6.2, 5.6, 5.0])
        self.assertEqual(plan.reason, "")

    def test_back_same_side_offer_below_final_limit_uses_direct_limit_price(self) -> None:
        plan = build_pre_ladder_from_same_side_offer(
            "BACK",
            best_same_side_offer=4.0,
            final_lim_price=5.0,
            steps=4,
        )

        self.assertIsNone(plan.start_price)
        self.assertEqual(plan.prices, [5.0])
        self.assertEqual(plan.reason, "no_better_back_ladder_range")

    def test_back_same_side_offer_equal_final_limit_uses_direct_limit_price(self) -> None:
        plan = build_pre_ladder_from_same_side_offer(
            "BACK",
            best_same_side_offer=5.0,
            final_lim_price=5.0,
            steps=4,
        )

        self.assertIsNone(plan.start_price)
        self.assertEqual(plan.prices, [5.0])
        self.assertEqual(plan.reason, "no_better_back_ladder_range")

    def test_lay_same_side_offer_below_final_limit_builds_ascending_ladder(self) -> None:
        plan = build_pre_ladder_from_same_side_offer(
            "LAY",
            best_same_side_offer=6.0,
            final_lim_price=8.0,
            steps=4,
        )

        self.assertEqual(plan.start_price, 6.2)
        self.assertEqual(plan.prices, [6.2, 6.8, 7.4, 8.0])
        self.assertEqual(plan.reason, "")

    def test_lay_same_side_offer_above_final_limit_uses_direct_limit_price(self) -> None:
        plan = build_pre_ladder_from_same_side_offer(
            "LAY",
            best_same_side_offer=9.0,
            final_lim_price=8.0,
            steps=4,
        )

        self.assertIsNone(plan.start_price)
        self.assertEqual(plan.prices, [8.0])
        self.assertEqual(plan.reason, "no_better_lay_ladder_range")

    def test_lay_same_side_offer_equal_final_limit_uses_direct_limit_price(self) -> None:
        plan = build_pre_ladder_from_same_side_offer(
            "LAY",
            best_same_side_offer=8.0,
            final_lim_price=8.0,
            steps=4,
        )

        self.assertIsNone(plan.start_price)
        self.assertEqual(plan.prices, [8.0])
        self.assertEqual(plan.reason, "no_better_lay_ladder_range")

    def test_fully_matched_previous_order_stops_following_steps(self) -> None:
        decision = decide_pre_ladder_step(
            PreLadderOrderState(status="FULLY_MATCHED", requested_stake=2.0, matched_stake=2.0),
            full_stake=2.0,
        )

        self.assertFalse(decision.send_new_order)
        self.assertEqual(decision.stop_reason, "fully_matched")

    def test_partially_matched_previous_order_replaces_only_remaining_stake(self) -> None:
        decision = decide_pre_ladder_step(
            PreLadderOrderState(status="PARTIALLY_MATCHED", requested_stake=2.0, matched_stake=0.75),
            full_stake=2.0,
        )

        self.assertTrue(decision.send_new_order)
        self.assertTrue(decision.cancelled_previous)
        self.assertEqual(decision.current_step_stake, 1.25)
        self.assertEqual(decision.remaining_stake, 1.25)

    def test_cancel_failure_prevents_new_order_to_avoid_stacking(self) -> None:
        decision = decide_pre_ladder_step(
            PreLadderOrderState(
                status="UNMATCHED",
                requested_stake=2.0,
                matched_stake=0.0,
                cancel_succeeded=False,
            ),
            full_stake=2.0,
        )

        self.assertFalse(decision.send_new_order)
        self.assertTrue(decision.cancel_failed)
        self.assertEqual(decision.stop_reason, "cancel_failed_do_not_stack")

    def test_gruss_step_one_without_bet_ref_plans_back_or_lay_preview_trigger(self) -> None:
        back_plan = plan_gruss_pre_ladder_trigger(side="BACK", step_index=0, bet_ref=None)
        lay_plan = plan_gruss_pre_ladder_trigger(side="LAY", step_index=0, bet_ref=None)

        self.assertTrue(back_plan.allowed)
        self.assertEqual(back_plan.trigger, "BACK")
        self.assertFalse(back_plan.bet_ref_required)
        self.assertFalse(lay_plan.real_confirmed)
        self.assertEqual(lay_plan.trigger, "LAY")

    def test_gruss_following_step_with_bet_ref_plans_replace_preview_only(self) -> None:
        plan = plan_gruss_pre_ladder_trigger(side="BACK", step_index=1, bet_ref="123456789")

        self.assertTrue(plan.allowed)
        self.assertEqual(plan.trigger, "BACKR")
        self.assertTrue(plan.bet_ref_required)
        self.assertTrue(plan.bet_ref_present)
        self.assertFalse(plan.no_stack)
        self.assertFalse(plan.real_confirmed)
        self.assertEqual(plan.reason, "replace_trigger_not_confirmed_preview_only")

    def test_gruss_following_lay_step_with_bet_ref_plans_layr_preview_only(self) -> None:
        plan = plan_gruss_pre_ladder_trigger(side="LAY", step_index=1, bet_ref="123456789")

        self.assertTrue(plan.allowed)
        self.assertEqual(plan.trigger, "LAYR")
        self.assertTrue(plan.bet_ref_required)
        self.assertTrue(plan.bet_ref_present)

    def test_gruss_following_step_without_bet_ref_never_replaces_or_stacks(self) -> None:
        plan = plan_gruss_pre_ladder_trigger(side="BACK", step_index=1, bet_ref="")

        self.assertFalse(plan.allowed)
        self.assertEqual(plan.trigger, "")
        self.assertTrue(plan.bet_ref_required)
        self.assertFalse(plan.bet_ref_present)
        self.assertTrue(plan.no_stack)
        self.assertFalse(plan.real_confirmed)
        self.assertEqual(plan.reason, "bet_ref_not_ready")


if __name__ == "__main__":
    unittest.main()
