from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from dogbot.executor import Executor
from dogbot.gruss.gruss_engine_adapter import build_engine_bundle
from dogbot.gruss.gruss_mapper import parse_gruss_sheet
from dogbot.indexer import MarketIndex
from dogbot.types import MarketIndexEntry
from tests.test_gruss_mapper import _sample_sheet


class KPlaceUsedTests(unittest.TestCase):
    def test_gruss_place_winners_2_sets_k_place_used_2_for_win_calculation(self) -> None:
        executor = _executor_with_index_from_gruss(place_winners=2)
        win_entry = executor.market_index.get("258835465")
        win_md = SimpleNamespace(market_type="WIN", number_of_winners=1)

        k_place_used, place_winners, fallback = executor._resolve_k_place_used(
            "258835465", "258835466", win_entry, win_md, "WIN", 6
        )

        self.assertEqual(k_place_used, 2)
        self.assertEqual(place_winners, 2)
        self.assertFalse(fallback)

    def test_gruss_place_winners_3_sets_k_place_used_3_for_win_calculation(self) -> None:
        executor = _executor_with_index_from_gruss(place_winners=3)
        win_entry = executor.market_index.get("258835465")
        win_md = SimpleNamespace(market_type="WIN", number_of_winners=1)

        k_place_used, place_winners, fallback = executor._resolve_k_place_used(
            "258835465", "258835466", win_entry, win_md, "WIN", 8
        )

        self.assertEqual(k_place_used, 3)
        self.assertEqual(place_winners, 3)
        self.assertFalse(fallback)

    def test_betfair_place_market_definition_number_of_winners_sets_k_place_used(self) -> None:
        executor = _executor_with_market_index([])
        place_md = SimpleNamespace(market_type="PLACE", numberOfWinners=2)

        k_place_used, place_winners, fallback = executor._resolve_k_place_used(
            "place-1", "place-1", None, place_md, "PLACE", 6
        )

        self.assertEqual(k_place_used, 2)
        self.assertEqual(place_winners, 2)
        self.assertFalse(fallback)

    def test_fallback_is_used_only_when_official_place_winners_absent(self) -> None:
        executor = _executor_with_market_index(
            [
                MarketIndexEntry(
                    market_id="win-1",
                    market_type="WIN",
                    event_id="event-1",
                    event_name="Race",
                    event_open_utc=datetime(2026, 6, 3, tzinfo=timezone.utc),
                    venue="Hove",
                    country_code="GB",
                    event_local_date=None,
                    race_number=None,
                    course_id="course-1",
                    win_market_id="win-1",
                    place_market_id="place-1",
                    n_places=None,
                )
            ]
        )
        win_entry = executor.market_index.get("win-1")
        win_md = SimpleNamespace(market_type="WIN", number_of_winners=1)

        k_place_used, place_winners, fallback = executor._resolve_k_place_used(
            "win-1", "place-1", win_entry, win_md, "WIN", 8
        )

        self.assertEqual(k_place_used, 3)
        self.assertIsNone(place_winners)
        self.assertTrue(fallback)


def _executor_with_index_from_gruss(place_winners: int):
    win_rows = _sample_sheet("Hove WIN", 258835465.0, winners=1.0)
    place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=float(place_winners))
    win = parse_gruss_sheet(win_rows, "WIN")
    place = parse_gruss_sheet(place_rows, "PLACE")
    bundle = build_engine_bundle(win, place)
    return _executor_with_market_index(list(bundle.market_index.values()))


def _executor_with_market_index(entries):
    executor = Executor.__new__(Executor)
    executor.market_index = MarketIndex(entries)
    executor._place_winners_by_market = {}
    executor._k_place_used_by_market = {}
    executor._fallback_k_place_used_by_market = {}
    executor._k_place_fallback_logged = set()
    return executor


if __name__ == "__main__":
    unittest.main()
