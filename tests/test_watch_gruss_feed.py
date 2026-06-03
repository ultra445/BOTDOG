from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from dogbot.gruss.gruss_mapper import parse_gruss_sheet
from tests.test_gruss_mapper import _sample_sheet


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_gruss_feed.py"
SPEC = importlib.util.spec_from_file_location("watch_gruss_feed", SCRIPT_PATH)
watch_gruss_feed = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watch_gruss_feed)


class WatchGrussFeedTests(unittest.TestCase):
    def test_build_csv_rows_pairs_runners_by_trap(self) -> None:
        win = parse_gruss_sheet(_sample_sheet("Hove WIN", 258835465.0), "WIN")
        place = parse_gruss_sheet(_sample_sheet("Hove PLACE", 258835466.0, winners=2.0), "PLACE")

        rows = watch_gruss_feed.build_csv_rows("2026-06-03T17:00:00", win, place)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["timestamp"], "2026-06-03T17:00:00")
        self.assertEqual(rows[0]["parent_id"], "35678242")
        self.assertEqual(rows[0]["win_market_id"], "258835465")
        self.assertEqual(rows[0]["place_market_id"], "258835466")
        self.assertEqual(rows[0]["countdown"], "-00:58")
        self.assertEqual(rows[0]["trap"], 1)
        self.assertEqual(rows[0]["runner_name"], "Gingers Layla")
        self.assertEqual(rows[0]["win_best_back"], 9.4)
        self.assertEqual(rows[0]["win_best_lay"], 9.8)
        self.assertEqual(rows[0]["win_ltp"], 9.2)
        self.assertEqual(rows[0]["win_total_matched"], 11274.07)
        self.assertEqual(rows[0]["place_best_back"], 9.4)
        self.assertEqual(rows[0]["place_best_lay"], 9.8)
        self.assertEqual(rows[0]["place_ltp"], 9.2)
        self.assertEqual(rows[0]["place_total_matched"], 11274.07)

    def test_is_tradable_requires_validation_and_unsuspended_available_odds(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        win_rows[1][5] = ""
        place_rows[1][5] = ""
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        self.assertTrue(watch_gruss_feed.is_tradable(win, place, validation_ok=True))
        self.assertFalse(watch_gruss_feed.is_tradable(win, place, validation_ok=False))

    def test_is_tradable_rejects_suspended_market(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        win_rows[1][5] = "Suspended"
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        self.assertFalse(watch_gruss_feed.is_tradable(win, place, validation_ok=True))

    def test_is_tradable_rejects_missing_odds(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
        for row in (win_rows[4], win_rows[5], place_rows[4], place_rows[5]):
            row[5] = 0.0
            row[7] = 0.0
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        self.assertFalse(watch_gruss_feed.is_tradable(win, place, validation_ok=True))


if __name__ == "__main__":
    unittest.main()
