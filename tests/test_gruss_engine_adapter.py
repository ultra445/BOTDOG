from __future__ import annotations

import csv
import unittest
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dogbot.gruss.gruss_dryrun_engine import (
    GRUSS_TRADE_DIAGNOSTIC_FIELDS,
    GrussDryRunRunner,
    ProcessedRaceStore,
    build_order_intents_from_trade_rows,
    build_gruss_trade_diagnostics,
    get_skip_reason,
    gruss_region_for_snapshots,
    install_gruss_trade_diagnostics,
    is_result_screen_for_snapshots,
    race_key,
    strategy_registry_diagnostics,
)
from dogbot.gruss.gruss_engine_adapter import build_engine_bundle
from dogbot.gruss.gruss_mapper import parse_gruss_sheet
from dogbot.gruss.gruss_momentum import GrussMomentumBuffer
from tests.test_gruss_mapper import _sample_sheet


class GrussEngineAdapterTests(unittest.TestCase):
    def test_build_engine_bundle_maps_snapshots_to_market_books(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        win_rows[1][3] = "00:00:02"
        place_rows[1][3] = "00:00:02"
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        bundle = build_engine_bundle(
            win,
            place,
            now_utc=datetime(2026, 6, 3, 17, 0, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(bundle.win_book.market_id, "258835465")
        self.assertEqual(bundle.place_book.market_id, "258835466")
        self.assertEqual(bundle.win_book.market_definition.market_type, "WIN")
        self.assertEqual(bundle.place_book.market_definition.market_type, "PLACE")
        self.assertEqual(bundle.win_book.runners[0].selection_id, 1)
        self.assertEqual(bundle.win_book.runners[0].last_price_traded, 9.2)
        self.assertEqual(bundle.win_book.runners[0].ex.available_to_back[0].price, 9.4)
        self.assertEqual(bundle.win_book.market_definition.country_code, "GB")
        self.assertEqual(bundle.win_book.market_definition.normalized_region, "UK")
        self.assertEqual(bundle.place_book.market_definition.number_of_winners, 2)
        self.assertEqual(bundle.market_index.get("258835466").n_places, 2)
        self.assertEqual(bundle.market_index.get("258835465").place_market_id, "258835466")
        self.assertEqual(bundle.market_index.get("258835465").runners[0].runner_name, "Gingers Layla")

    def test_engine_bundle_maps_australia_to_row_country_code(self) -> None:
        win_rows = _sample_sheet("The Meadows WIN", 258835465.0)
        place_rows = _sample_sheet("The Meadows PLACE", 258835466.0, winners=2.0)
        win_rows[0][5] = r"Greyhound Racing\Australia\The Meadows"
        place_rows[0][5] = r"Greyhound Racing\Australia\The Meadows"
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        bundle = build_engine_bundle(win, place)

        self.assertEqual(gruss_region_for_snapshots(win, place), "ROW")
        self.assertEqual(bundle.win_book.market_definition.country_code, "AU")
        self.assertEqual(bundle.win_book.market_definition.normalized_region, "ROW")

    def test_skip_reason_blocks_unknown_region(self) -> None:
        win_rows = _sample_sheet("Mystery WIN", 258835465.0)
        place_rows = _sample_sheet("Mystery PLACE", 258835466.0, winners=2.0)
        win_rows[0][5] = r"Greyhound Racing\PGR\Mystery Track"
        place_rows[0][5] = r"Greyhound Racing\PGR\Mystery Track"
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        self.assertEqual(gruss_region_for_snapshots(win, place), "UNKNOWN")
        self.assertEqual(get_skip_reason(win, place, [], True), "unknown_gruss_region")

    def test_result_screen_skips_without_exception(self) -> None:
        win_rows = _sample_sheet("Bet ref", 258835465.0)
        place_rows = _sample_sheet("Bet ref", 258835466.0, winners=2.0)
        win_rows[0][5] = "Result"
        place_rows[0][5] = "Result"
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        self.assertTrue(is_result_screen_for_snapshots(win, place))
        self.assertEqual(gruss_region_for_snapshots(win, place), "UNKNOWN")
        self.assertEqual(get_skip_reason(win, place, [], True), "result_screen")

    def test_skip_reason_blocks_validation_and_missing_countdown(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        win_rows[1][3] = ""
        place_rows[1][3] = ""
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        self.assertEqual(race_key(win, place), "parent:35678242")
        self.assertEqual(get_skip_reason(win, place, [], True), "countdown_seconds_unavailable")

    def test_missing_mom45_is_never_a_global_skip_reason(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        win_rows[1][3] = "00:00:20"
        place_rows[1][3] = "00:00:20"
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        self.assertIsNone(get_skip_reason(win, place, [], True))

    def test_processed_race_store_persists_seen_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/processed.csv"
            store = ProcessedRaceStore(path)
            self.assertFalse(store.has_seen("parent:1"))

            store.mark_seen("parent:1", "win", "place")

            self.assertTrue(store.has_seen("parent:1"))
            reloaded = ProcessedRaceStore(path)
            self.assertTrue(reloaded.has_seen("parent:1"))

    def test_gruss_trade_diagnostics_enrich_trade_rows(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        win_rows[1][3] = "00:00:02"
        place_rows[1][3] = "00:00:02"
        win_rows[1][5] = ""
        place_rows[1][5] = ""
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")
        logged_rows = []

        class FakeExecutor:
            TRADE_HEADER = ["ts", "selection_id"]

            def __init__(self) -> None:
                self._last_place_theo_by_market = {"258835465": {1: 2.34}}
                self._last_ev_place_by_market = {"258835466": {1: 0.12}}
                self._k_place_used_by_market = {"258835466": 2}
                self._fallback_k_place_used_by_market = {"258835466": False}

            def _log_trade_row(self, row: dict) -> None:
                logged_rows.append(row)

        executor = FakeExecutor()
        momentum_buffer = GrussMomentumBuffer()
        momentum_buffer.add_snapshot_pair(win, place)
        momentum_values = momentum_buffer.momentum_by_trap(win, place)
        momentum_status = momentum_buffer.course_status(win, place)
        install_gruss_trade_diagnostics(executor, win, place, momentum_values, momentum_status)

        self.assertIn("data_provider", executor.TRADE_HEADER)
        self.assertTrue(all(field in executor.TRADE_HEADER for field in GRUSS_TRADE_DIAGNOSTIC_FIELDS))

        diagnostics = build_gruss_trade_diagnostics(executor, {"selection_id": 1})
        self.assertEqual(diagnostics["data_provider"], "gruss_excel")
        self.assertEqual(diagnostics["order_provider"], "gruss_excel_dryrun")
        self.assertEqual(diagnostics["win_market_id"], "258835465")
        self.assertEqual(diagnostics["place_market_id"], "258835466")
        self.assertEqual(diagnostics["parent_id"], "35678242")
        self.assertEqual(diagnostics["countdown_seconds"], 2)
        self.assertEqual(diagnostics["countdown_display"], "00:02")
        self.assertEqual(diagnostics["tradable"], "1")
        self.assertFalse(diagnostics["has_mom45"])
        self.assertEqual(diagnostics["mom45_reason"], "watcher_started_after_t45")
        self.assertEqual(diagnostics["first_seen_countdown"], 2)
        self.assertFalse(diagnostics["t45_anchor_found"])
        self.assertEqual(diagnostics["win_best_back"], 9.4)
        self.assertEqual(diagnostics["win_best_lay"], 9.8)
        self.assertEqual(diagnostics["place_best_back"], 9.4)
        self.assertEqual(diagnostics["place_best_lay"], 9.8)
        self.assertEqual(diagnostics["place_winners"], 2)
        self.assertEqual(diagnostics["k_place_used"], 2)
        self.assertFalse(diagnostics["fallback_k_place_used"])
        self.assertEqual(diagnostics["place_theorique"], 2.34)
        self.assertEqual(diagnostics["ev_place"], 0.12)
        self.assertEqual(diagnostics["gruss_event_path"], r"Greyhound Racing\PGR\Hove 3rd Jun")

        executor._log_trade_row({"ts": "now", "selection_id": 1})
        self.assertEqual(logged_rows[0]["data_provider"], "gruss_excel")

    def test_gruss_trade_diagnostics_marks_real_ready_pre_ladder_rows_as_real_provider(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        class FakeExecutor:
            TRADE_HEADER = ["ts", "selection_id", "status"]

            def _log_trade_row(self, row: dict) -> None:
                pass

        executor = FakeExecutor()
        install_gruss_trade_diagnostics(executor, win, place)

        with patch.dict(
            "os.environ",
            {
                "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
                "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
            },
            clear=True,
        ):
            diagnostics = build_gruss_trade_diagnostics(
                executor,
                {"selection_id": 1, "status": "PRE_LADDER_REAL_READY"},
            )

        self.assertEqual(diagnostics["order_provider"], "gruss_excel_real")

    def test_gruss_trade_diagnostics_marks_armed_post_rows_as_real_provider(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        class FakeExecutor:
            TRADE_HEADER = ["ts", "selection_id", "status"]

            def _log_trade_row(self, row: dict) -> None:
                pass

        executor = FakeExecutor()
        install_gruss_trade_diagnostics(executor, win, place)

        with patch.dict(
            "os.environ",
            {
                "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
                "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
                "DOGBOT_GRUSS_REAL_PREVIEW": "false",
                "DOGBOT_GRUSS_WRITE_NO_TRIGGER": "false",
            },
            clear=True,
        ):
            diagnostics = build_gruss_trade_diagnostics(
                executor,
                {"selection_id": 1, "status": "DRYRUN", "execution_phase": "POST"},
            )

        self.assertEqual(diagnostics["order_provider"], "gruss_excel_real")

    def test_gruss_trade_diagnostics_keeps_preview_and_write_no_trigger_rows_dryrun(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        class FakeExecutor:
            TRADE_HEADER = ["ts", "selection_id", "status"]

            def _log_trade_row(self, row: dict) -> None:
                pass

        executor = FakeExecutor()
        install_gruss_trade_diagnostics(executor, win, place)

        for flag in ("DOGBOT_GRUSS_REAL_PREVIEW", "DOGBOT_GRUSS_WRITE_NO_TRIGGER"):
            with self.subTest(flag=flag), patch.dict(
                "os.environ",
                {
                    "DOGBOT_ORDER_PROVIDER": "gruss_excel_real",
                    "DOGBOT_GRUSS_ENABLE_REAL_ORDERS": "true",
                    flag: "true",
                },
                clear=True,
            ):
                diagnostics = build_gruss_trade_diagnostics(
                    executor,
                    {"selection_id": 1, "status": "DRYRUN", "execution_phase": "POST"},
                )

            self.assertEqual(diagnostics["order_provider"], "gruss_excel_dryrun")

    def test_build_order_intents_from_trade_rows(self) -> None:
        win = parse_gruss_sheet(_sample_sheet("Hove WIN", 258835465.0), "WIN")
        place = parse_gruss_sheet(_sample_sheet("Hove PLACE", 258835466.0, winners=2.0), "PLACE")

        intents = build_order_intents_from_trade_rows(
            [
                {
                    "ts": "2026-06-03T18:00:00Z",
                    "market_type": "PLACE",
                    "market_id": "258835466",
                    "selection_id": "1",
                    "side": "BACK",
                    "price_req": "3.2",
                    "size_req": "2.0",
                    "strategy": "BACK_PLACE_101",
                    "course_id": "course-1",
                    "status": "DRYRUN",
                    "parent_id": "35678242",
                }
            ],
            win,
            place,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].provider, "gruss_excel")
        self.assertEqual(intents[0].market_type, "PLACE")
        self.assertEqual(intents[0].market_id, "258835466")
        self.assertEqual(intents[0].runner_name, "Gingers Layla")
        self.assertEqual(intents[0].trap, 1)
        self.assertEqual(intents[0].price, 3.2)
        self.assertEqual(intents[0].stake, 2.0)

    def test_build_order_intents_accepts_post_real_ready_rows(self) -> None:
        win = parse_gruss_sheet(_sample_sheet("Hove WIN", 258835465.0), "WIN")
        place = parse_gruss_sheet(_sample_sheet("Hove PLACE", 258835466.0, winners=2.0), "PLACE")

        intents = build_order_intents_from_trade_rows(
            [
                {
                    "ts": "2026-06-03T18:00:00Z",
                    "market_type": "PLACE",
                    "market_id": "258835466",
                    "selection_id": "1",
                    "side": "BACK",
                    "price_req": "3.2",
                    "size_req": "2.0",
                    "strategy": "BACK_PLACE_999",
                    "course_id": "course-1",
                    "status": "REAL_READY",
                    "parent_id": "35678242",
                    "execution_phase": "POST",
                }
            ],
            win,
            place,
        )

        self.assertEqual(len(intents), 1)
        self.assertFalse(intents[0].pre_ladder)
        self.assertEqual(intents[0].execution_phase, "POST")
        self.assertEqual(intents[0].price, 3.2)
        self.assertEqual(intents[0].stake, 2.0)

    def test_build_order_intents_maps_pre_ladder_real_ready_rows(self) -> None:
        win = parse_gruss_sheet(_sample_sheet("Hove WIN", 258835465.0), "WIN")
        place = parse_gruss_sheet(_sample_sheet("Hove PLACE", 258835466.0, winners=2.0), "PLACE")

        intents = build_order_intents_from_trade_rows(
            [
                {
                    "ts": "2026-06-03T18:00:00Z",
                    "market_type": "PLACE",
                    "market_id": "258835466",
                    "selection_id": "1",
                    "side": "BACK",
                    "price_req": "5.0",
                    "size_req": "99.0",
                    "current_ladder_price": "6.8",
                    "current_step_stake": "2.0",
                    "strategy": "BACK_PLACE_101",
                    "course_id": "course-1",
                    "status": "PRE_LADDER_REAL_READY",
                    "parent_id": "35678242",
                    "execution_phase": "PRE",
                    "ladder_id": "ladder-1",
                    "ladder_step": "1/4",
                    "ladder_tracking_key": "ladder-1:tracking",
                    "gruss_planned_trigger": "BACK",
                    "matched_stake": "0.0",
                    "ladder_plan_frozen": "True",
                    "ladder_plan_created_step": "1",
                    "ladder_prices_frozen": "6.8|6.2|5.6|5.0",
                    "current_ladder_price_from_frozen_plan": "True",
                    "best_same_side_offer_at_creation": "7.0",
                    "ladder_direction": "BACK_DESCENDING",
                    "ladder_disabled_lim_not_in_ladder_direction": "False",
                }
            ],
            win,
            place,
        )

        self.assertEqual(len(intents), 1)
        self.assertTrue(intents[0].pre_ladder)
        self.assertEqual(intents[0].execution_phase, "PRE")
        self.assertEqual(intents[0].price, 6.8)
        self.assertEqual(intents[0].stake, 2.0)
        self.assertEqual(intents[0].ladder_id, "ladder-1")
        self.assertEqual(intents[0].ladder_step, "1/4")
        self.assertEqual(intents[0].gruss_planned_trigger, "BACK")
        self.assertEqual(intents[0].matched_stake, 0.0)
        self.assertTrue(intents[0].ladder_plan_frozen)
        self.assertEqual(intents[0].ladder_prices_frozen, "6.8|6.2|5.6|5.0")
        self.assertEqual(intents[0].best_same_side_offer_at_creation, 7.0)
        self.assertEqual(intents[0].ladder_direction, "BACK_DESCENDING")

    def test_build_order_intents_treats_legacy_direct_lim_written_as_planned_only(self) -> None:
        win = parse_gruss_sheet(_sample_sheet("Hove WIN", 258835465.0), "WIN")
        place = parse_gruss_sheet(_sample_sheet("Hove PLACE", 258835466.0, winners=2.0), "PLACE")

        intents = build_order_intents_from_trade_rows(
            [
                {
                    "ts": "2026-06-03T18:00:00Z",
                    "market_type": "PLACE",
                    "market_id": "258835466",
                    "selection_id": "1",
                    "side": "BACK",
                    "price_req": "5.0",
                    "size_req": "99.0",
                    "final_lim_price": "5.0",
                    "final_stake": "2.0",
                    "strategy": "BACK_PLACE_101",
                    "course_id": "course-1",
                    "status": "PRE_LADDER_REAL_READY",
                    "parent_id": "35678242",
                    "execution_phase": "PRE",
                    "ladder_id": "ladder-1",
                    "ladder_step": "1/4",
                    "ladder_tracking_key": "ladder-1:tracking",
                    "gruss_planned_trigger": "BACK",
                    "ladder_disabled_lim_not_in_ladder_direction": "True",
                    "direct_lim_order_written": "True",
                    "no_replace_steps_for_direct_lim": "True",
                }
            ],
            win,
            place,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].price, 5.0)
        self.assertEqual(intents[0].stake, 2.0)
        self.assertTrue(intents[0].direct_lim_order_planned)
        self.assertFalse(intents[0].direct_lim_order_written)
        self.assertTrue(intents[0].no_replace_steps_for_direct_lim)

    def test_build_order_intents_accepts_lay_place_signal(self) -> None:
        win = parse_gruss_sheet(_sample_sheet("Hove WIN", 258835465.0), "WIN")
        place = parse_gruss_sheet(_sample_sheet("Hove PLACE", 258835466.0, winners=2.0), "PLACE")

        intents = build_order_intents_from_trade_rows(
            [
                {
                    "ts": "2026-06-03T18:00:00Z",
                    "market_type": "PLACE",
                    "market_id": "258835466",
                    "selection_id": "1",
                    "side": "LAY",
                    "price_req": "3.2",
                    "size_req": "2.0",
                    "strategy": "LAY_PLACE_301",
                    "course_id": "course-1",
                    "status": "DRYRUN",
                    "parent_id": "35678242",
                }
            ],
            win,
            place,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].side, "LAY")
        self.assertEqual(intents[0].strategy_id, "LAY_PLACE_301")
        self.assertEqual(intents[0].order_type, "LIMIT")

    def test_build_order_intents_maps_static_sp_moc_strategy(self) -> None:
        win = parse_gruss_sheet(_sample_sheet("Hove WIN", 258835465.0), "WIN")
        place = parse_gruss_sheet(_sample_sheet("Hove PLACE", 258835466.0, winners=2.0), "PLACE")

        intents = build_order_intents_from_trade_rows(
            [
                {
                    "ts": "2026-06-03T18:00:00Z",
                    "market_type": "PLACE",
                    "market_id": "258835466",
                    "selection_id": "1",
                    "side": "LAY",
                    "price_req": "9.4",
                    "size_req": "2.0",
                    "strategy": "LAY_PLACE_502",
                    "course_id": "course-1",
                    "status": "DRYRUN",
                    "parent_id": "35678242",
                }
            ],
            win,
            place,
        )

        self.assertEqual(intents[0].order_type, "SP_MOC")

    def test_lay_place_trade_row_is_logged_to_gruss_orders_dryrun(self) -> None:
        win = parse_gruss_sheet(_sample_sheet("Hove WIN", 258835465.0), "WIN")
        place = parse_gruss_sheet(_sample_sheet("Hove PLACE", 258835466.0, winners=2.0), "PLACE")
        trade_rows = [
            {
                "ts": "2026-06-03T18:00:00Z",
                "market_type": "PLACE",
                "market_id": "258835466",
                "selection_id": "1",
                "side": "LAY",
                "price_req": "3.2",
                "size_req": "2.0",
                "strategy": "LAY_PLACE_301",
                "course_id": "course-1",
                "status": "DRYRUN",
                "parent_id": "35678242",
            }
        ]

        with TemporaryDirectory() as tmp:
            runner = GrussDryRunRunner(tmp)
            results = runner.log_gruss_order_intents(trade_rows, win, place)
            order_path = f"{tmp}/orders_gruss_dryrun.csv"

            self.assertEqual(results[0].status, "GRUSS_DRYRUN")
            with open(order_path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["side"], "LAY")
        self.assertEqual(rows[0]["strategy_id"], "LAY_PLACE_301")
        self.assertEqual(rows[0]["status"], "GRUSS_DRYRUN")

    def test_registry_diagnostics_reports_lay_place_slots(self) -> None:
        diagnostics = strategy_registry_diagnostics()

        self.assertGreater(diagnostics["total"], 0)
        self.assertGreater(diagnostics["by_side"].get("LAY", 0), 0)
        self.assertGreater(diagnostics["by_market_type"].get("PLACE", 0), 0)
        self.assertTrue(diagnostics["lay_place_ids"])
        self.assertIn("LAY_PLACE_301", diagnostics["lay_place_ids"])
        self.assertNotIn("LAY_PLACE_501", diagnostics["lay_place_ids"])
        self.assertNotIn("LAY_PLACE_503", diagnostics["lay_place_ids"])
        details = {detail["strategy_id"]: detail for detail in diagnostics["details"]}
        self.assertFalse(details["LAY_PLACE_301"]["requires_mom45"])
        self.assertTrue(details["LAY_PLACE_541"]["requires_mom45"])


if __name__ == "__main__":
    unittest.main()
