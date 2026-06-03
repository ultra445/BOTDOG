from __future__ import annotations

import unittest
from datetime import datetime, timezone
from tempfile import TemporaryDirectory

from dogbot.gruss.gruss_dryrun_engine import (
    GRUSS_TRADE_DIAGNOSTIC_FIELDS,
    ProcessedRaceStore,
    build_gruss_trade_diagnostics,
    get_skip_reason,
    install_gruss_trade_diagnostics,
    race_key,
)
from dogbot.gruss.gruss_engine_adapter import build_engine_bundle
from dogbot.gruss.gruss_mapper import parse_gruss_sheet
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
        self.assertEqual(bundle.market_index.get("258835465").place_market_id, "258835466")
        self.assertEqual(bundle.market_index.get("258835465").runners[0].runner_name, "Gingers Layla")

    def test_skip_reason_blocks_validation_and_missing_countdown(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        win_rows[1][3] = ""
        place_rows[1][3] = ""
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        self.assertEqual(race_key(win, place), "parent:35678242")
        self.assertEqual(get_skip_reason(win, place, [], True), "countdown_seconds_unavailable")

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

            def _log_trade_row(self, row: dict) -> None:
                logged_rows.append(row)

        executor = FakeExecutor()
        install_gruss_trade_diagnostics(executor, win, place)

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
        self.assertEqual(diagnostics["win_best_back"], 9.4)
        self.assertEqual(diagnostics["win_best_lay"], 9.8)
        self.assertEqual(diagnostics["place_best_back"], 9.4)
        self.assertEqual(diagnostics["place_best_lay"], 9.8)
        self.assertEqual(diagnostics["place_theorique"], 2.34)
        self.assertEqual(diagnostics["ev_place"], 0.12)
        self.assertEqual(diagnostics["gruss_event_path"], r"Greyhound Racing\PGR\Hove 3rd Jun")

        executor._log_trade_row({"ts": "now", "selection_id": 1})
        self.assertEqual(logged_rows[0]["data_provider"], "gruss_excel")


if __name__ == "__main__":
    unittest.main()
