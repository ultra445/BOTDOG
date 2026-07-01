from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from dogbot.executor import Executor


def _load_analyzer_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_pre_ladder_preview.py"
    spec = importlib.util.spec_from_file_location("analyze_pre_ladder_preview", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PreLadderConsolidationTests(unittest.TestCase):
    REQUIRED_CSV_COLUMNS = {
        "run_id",
        "evaluation_id",
        "parent_market_id",
        "milestone",
        "execution_phase",
        "final_lim_price_raw",
        "final_lim_price_tick",
        "start_price_raw",
        "start_price_tick",
        "current_ladder_price",
        "tick_rounding_mode",
        "computed_limit_price_raw",
        "computed_limit_price_effective",
        "min_price_floor_applied",
        "pre_value_target_price",
        "ladder_planned_price",
        "sent_price_before_value_clamp",
        "sent_price_after_value_clamp",
        "value_clamp_applied",
        "value_limit_breached",
        "value_limit_skip_reason",
        "tick_rounding_direction",
        "best_same_side_back_offer",
        "best_same_side_lay_offer",
        "best_back_displayed",
        "best_lay_displayed",
        "start_price_source",
        "ladder_prices",
        "ladder_prices_frozen",
        "ladder_step",
        "ladder_id",
        "ladder_plan_frozen",
        "current_ladder_price_from_frozen_plan",
        "direct_lim_order_planned",
        "direct_lim_order_written",
        "no_replace_steps_for_direct_lim",
        "no_better_ladder_range_reason",
        "post_checked",
        "post_signal_count",
        "post_evaluated",
        "post_missing_reason",
    }

    def test_pre_ladder_trade_header_contains_required_columns(self) -> None:
        self.assertTrue(self.REQUIRED_CSV_COLUMNS.issubset(set(Executor.TRADE_HEADER)))

    def test_env_example_keeps_real_ladder_disabled_and_preview_enabled(self) -> None:
        env_path = Path(__file__).resolve().parents[1] / ".env.example"
        values = {}
        for line in env_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            values[key.strip()] = value.strip()

        self.assertEqual(values.get("DOGBOT_PRE_LADDER_ENABLED"), "false")
        self.assertEqual(values.get("DOGBOT_PRE_LADDER_PREVIEW"), "true")
        self.assertEqual(values.get("DOGBOT_PRE_LADDER_STEPS"), "52,38,26,16")
        self.assertEqual(values.get("DOGBOT_PRE_INITIAL_BATCH_WRITE_GRACE_SECONDS"), "10")
        self.assertEqual(values.get("DOGBOT_PRE_CANCEL_BEFORE_POST"), "true")
        self.assertEqual(values.get("DOGBOT_PRE_CANCEL_SECONDS_BEFORE_OFF"), "1")
        self.assertEqual(values.get("DOGBOT_PRE_IGNORE_STALE_PRICE_BEFORE_WRITE"), "true")
        self.assertEqual(values.get("DOGBOT_POST_SEND_SECONDS_BEFORE_OFF"), "1")
        self.assertEqual(values.get("DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS"), "500")
        self.assertEqual(values.get("DOGBOT_GRUSS_REAL_VARIABLE_STAKES"), "false")

    def test_analyzer_reports_clean_group_and_post_presence(self) -> None:
        analyzer = _load_analyzer_module()
        rows = [
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "ladder-back",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "1/4",
                "milestone": "20",
                "current_ladder_price": "6.8",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "6.8|6.2|5.6|5.0",
                "reason": "initial_ladder_trigger_preview_only",
                "no_better_ladder_range_reason": "",
            },
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "ladder-back",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "4/4",
                "milestone": "5",
                "current_ladder_price": "5.0",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "6.8|6.2|5.6|5.0",
                "reason": "missing_bet_ref_do_not_stack",
                "no_better_ladder_range_reason": "",
            },
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "DRYRUN",
                "execution_phase": "POST",
                "milestone": "0",
                "complete_after_post": "True",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
            },
        ]

        lines, warnings, issues = analyzer.analyze_rows(rows)

        self.assertTrue(any("ladder_id=ladder-back" in line for line in lines))
        self.assertTrue(any("post_checked=True" in line for line in lines))
        self.assertTrue(any("post_evaluated=True" in line for line in lines))
        self.assertTrue(any("post_signal_count=1" in line for line in lines))
        self.assertEqual(warnings, [])
        self.assertEqual(issues, [])

    def test_analyzer_flags_price_direction_limit_and_missing_post(self) -> None:
        analyzer = _load_analyzer_module()
        rows = [
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "bad-back",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "1/4",
                "milestone": "20",
                "current_ladder_price": "5.2",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "5.2|5.4|4.8",
                "reason": "initial_ladder_trigger_preview_only",
                "no_better_ladder_range_reason": "",
            },
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "bad-back",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "2/4",
                "milestone": "15",
                "current_ladder_price": "5.4",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "5.2|5.4|4.8",
                "reason": "missing_bet_ref_do_not_stack",
                "no_better_ladder_range_reason": "",
            },
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "bad-back",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "3/4",
                "milestone": "10",
                "current_ladder_price": "4.8",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "5.2|5.4|4.8",
                "reason": "missing_bet_ref_do_not_stack",
                "no_better_ladder_range_reason": "",
            },
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "other-ladder-reached-final-pre",
                "market_id": "m2",
                "selection_id": "12",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "4/4",
                "milestone": "5",
                "current_ladder_price": "5.0",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "6.8|6.2|5.6|5.0",
                "reason": "missing_bet_ref_do_not_stack",
                "no_better_ladder_range_reason": "",
            },
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "DRYRUN",
                "execution_phase": "POST",
                "milestone": "0",
                "complete_after_post": "True",
                "market_id": "not-m1",
                "selection_id": "99",
                "market_type": "PLACE",
                "side": "BACK",
            },
        ]

        _, warnings, issues = analyzer.analyze_rows(rows)

        joined = "\n".join(issues)
        self.assertIn("BACK ladder non decroissant", joined)
        self.assertIn("BACK prix hors limite", joined)
        self.assertIn("step 4 manquant", joined)
        self.assertEqual(warnings, [])
        self.assertIn("steps_seen=1/4,2/4,3/4", joined)
        self.assertIn("current_ladder_price_by_step=5.2,5.4,4.8", joined)
        self.assertIn("final_lim_price_tick=5.0", joined)

    def test_analyzer_treats_no_better_back_range_as_direct_limit_not_ladder(self) -> None:
        analyzer = _load_analyzer_module()
        rows = [
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "direct-back",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": step,
                "milestone": milestone,
                "current_ladder_price": "5.0",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "5.0",
                "reason": "no_better_back_ladder_range",
                "no_better_ladder_range_reason": "no_better_back_ladder_range",
            }
            for step, milestone in (("1/4", "20"), ("3/4", "10"), ("4/4", "5"))
        ]

        _, _, issues = analyzer.analyze_rows(rows)

        self.assertEqual(issues, [])

    def test_analyzer_does_not_report_missing_post_before_post_milestone_reached(self) -> None:
        analyzer = _load_analyzer_module()
        rows = [
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "ladder-back",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "4/4",
                "milestone": "5",
                "current_ladder_price": "5.0",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "6.8|6.2|5.6|5.0",
                "reason": "missing_bet_ref_do_not_stack",
                "no_better_ladder_range_reason": "",
            }
        ]

        _, warnings, issues = analyzer.analyze_rows(rows)

        self.assertEqual(issues, [])
        self.assertTrue(any("POST non verifie apres PRE" in warning for warning in warnings))
        self.assertTrue(any("post_milestone_not_reached" in warning for warning in warnings))

    def test_analyzer_does_not_mix_two_runs_with_same_ladder_id(self) -> None:
        analyzer = _load_analyzer_module()
        rows = [
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "same-ladder",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "1/4",
                "milestone": "20",
                "current_ladder_price": "6.8",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "6.8|6.2|5.6|5.0",
                "reason": "initial_ladder_trigger_preview_only",
                "no_better_ladder_range_reason": "",
            },
            {
                "run_id": "run-2",
                "evaluation_id": "run-2|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "same-ladder",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "4/4",
                "milestone": "5",
                "current_ladder_price": "5.0",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "6.8|6.2|5.6|5.0",
                "reason": "missing_bet_ref_do_not_stack",
                "no_better_ladder_range_reason": "",
            },
        ]

        lines, _, issues = analyzer.analyze_rows(rows)

        self.assertEqual(issues, [])
        self.assertEqual(sum(1 for line in lines if "ladder_id=same-ladder" in line), 2)

    def test_analyzer_warns_not_issues_when_post_reached_but_no_post_evaluation_logged(self) -> None:
        analyzer = _load_analyzer_module()
        rows = [
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "ladder-back",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "4/4",
                "milestone": "5",
                "current_ladder_price": "5.0",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "6.8|6.2|5.6|5.0",
                "reason": "missing_bet_ref_do_not_stack",
                "post_checked": "true",
                "post_evaluated": "false",
                "post_signal_count": "0",
                "post_missing_reason": "post_not_logged_or_no_signal",
            }
        ]

        lines, warnings, issues = analyzer.analyze_rows(rows)

        self.assertTrue(any("post_checked=True" in line for line in lines))
        self.assertTrue(any("post_evaluated=False" in line for line in lines))
        self.assertTrue(any("post_signal_count=0" in line for line in lines))
        self.assertTrue(any("post_not_logged_or_no_signal" in line for line in lines))
        self.assertTrue(any("POST evaluation absente apres PRE" in warning for warning in warnings))
        self.assertEqual(issues, [])

    def test_analyzer_does_not_issue_when_post_evaluated_with_zero_signals(self) -> None:
        analyzer = _load_analyzer_module()
        rows = [
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "ladder-back",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "4/4",
                "milestone": "5",
                "current_ladder_price": "5.0",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "6.8|6.2|5.6|5.0",
                "reason": "missing_bet_ref_do_not_stack",
                "post_checked": "true",
                "post_evaluated": "true",
                "post_signal_count": "0",
                "post_missing_reason": "no_post_signal",
            }
        ]

        lines, warnings, issues = analyzer.analyze_rows(rows)

        self.assertTrue(any("post_evaluated=True" in line for line in lines))
        self.assertTrue(any("post_signal_count=0" in line for line in lines))
        self.assertEqual(warnings, [])
        self.assertEqual(issues, [])

    def test_analyzer_matches_post_by_run_id_and_parent_not_ladder_id(self) -> None:
        analyzer = _load_analyzer_module()
        rows = [
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "ladder-back-1",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "4/4",
                "milestone": "5",
                "current_ladder_price": "5.0",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "6.8|6.2|5.6|5.0",
                "reason": "missing_bet_ref_do_not_stack",
            },
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "ladder-back-2",
                "market_id": "m1",
                "selection_id": "12",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "4/4",
                "milestone": "5",
                "current_ladder_price": "6.0",
                "final_lim_price_tick": "6.0",
                "ladder_prices": "7.0|6.6|6.2|6.0",
                "reason": "missing_bet_ref_do_not_stack",
            },
            {
                "run_id": "run-1",
                "evaluation_id": "",
                "parent_market_id": "parent-1",
                "status": "DRYRUN",
                "execution_phase": "POST",
                "milestone": "0",
                "complete_after_post": "True",
                "market_id": "different-post-market",
                "selection_id": "99",
                "market_type": "WIN",
                "side": "LAY",
            },
        ]

        lines, warnings, issues = analyzer.analyze_rows(rows)

        self.assertEqual(sum(1 for line in lines if "ladder_id=ladder-back" in line), 2)
        self.assertTrue(all("post_evaluated=True" in line for line in lines if "post_evaluated=" in line))
        self.assertEqual(warnings, [])
        self.assertEqual(issues, [])

    def test_analyzer_issues_only_when_expected_post_evaluation_missing(self) -> None:
        analyzer = _load_analyzer_module()
        rows = [
            {
                "run_id": "run-1",
                "evaluation_id": "run-1|parent-1",
                "parent_market_id": "parent-1",
                "status": "PRE_LADDER_PREVIEW",
                "ladder_id": "ladder-back",
                "market_id": "m1",
                "selection_id": "11",
                "market_type": "PLACE",
                "side": "BACK",
                "strategy": "BACK_PLACE_101",
                "final_system": "BACK_PLACE_101",
                "ladder_step": "4/4",
                "milestone": "5",
                "current_ladder_price": "5.0",
                "final_lim_price_tick": "5.0",
                "ladder_prices": "6.8|6.2|5.6|5.0",
                "reason": "missing_bet_ref_do_not_stack",
                "post_checked": "true",
                "post_evaluated": "false",
                "post_signal_count": "0",
                "post_missing_reason": "expected_post_evaluation_missing",
            }
        ]

        _, warnings, issues = analyzer.analyze_rows(rows)

        self.assertEqual(warnings, [])
        self.assertTrue(any("POST evaluation absente apres PRE" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
