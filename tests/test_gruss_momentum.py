from __future__ import annotations

import unittest
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace

from dogbot.gruss.gruss_dryrun_engine import seed_gruss_momentum_into_executor
from dogbot.gruss.gruss_engine_adapter import build_engine_bundle
from dogbot.gruss.gruss_mapper import parse_gruss_sheet
from dogbot.gruss.gruss_momentum import GrussMomentumBuffer, gruss_win_base_price
from tests.test_gruss_mapper import _sample_sheet


def _snapshots_at(
    seconds: int,
    *,
    win_runner_1: tuple[float, float, float] = (4.0, 4.4, 4.2),
    place_runner_1: tuple[float, float, float] = (2.0, 2.2, 2.1),
):
    win_rows = _sample_sheet("Hove WIN", 258835465.0)
    place_rows = _sample_sheet("Hove PLACE", 258835466.0, winners=2.0)
    countdown = f"00:00:{seconds:02d}" if seconds >= 0 else f"-00:00:{abs(seconds):02d}"
    win_rows[1][3] = countdown
    place_rows[1][3] = countdown
    win_rows[1][5] = ""
    place_rows[1][5] = ""
    win_rows[4][5], win_rows[4][7], win_rows[4][14] = win_runner_1
    place_rows[4][5], place_rows[4][7], place_rows[4][14] = place_runner_1
    return parse_gruss_sheet(win_rows, "WIN"), parse_gruss_sheet(place_rows, "PLACE")


class GrussMomentumTests(unittest.TestCase):
    def test_base_win_matches_gruss_executor_fallback_definition(self) -> None:
        win, _ = _snapshots_at(45, win_runner_1=(4.0, 4.4, 4.2))

        self.assertEqual(gruss_win_base_price(win.runners[0]), 4.2)

    def test_buffer_uses_closest_t45_anchor_and_calculates_mom45(self) -> None:
        buffer = GrussMomentumBuffer()
        timestamp = datetime(2026, 6, 3, 18, 0, 0, tzinfo=timezone.utc)
        win_60, place_60 = _snapshots_at(60, win_runner_1=(5.0, 5.4, 5.2))
        win_45, place_45 = _snapshots_at(45, win_runner_1=(4.0, 4.4, 4.2))
        win_2, place_2 = _snapshots_at(2, win_runner_1=(3.0, 3.4, 3.2))

        buffer.add_snapshot_pair(win_60, place_60, timestamp=timestamp)
        captured, anchor = buffer.add_snapshot_pair(win_45, place_45, timestamp=timestamp)
        buffer.add_snapshot_pair(win_2, place_2, timestamp=timestamp)

        self.assertTrue(captured)
        self.assertIsNotNone(anchor)
        values = buffer.momentum_by_trap(win_2, place_2)

        self.assertTrue(values[1].has_mom45)
        self.assertEqual(values[1].source_countdown_seconds, 45)
        self.assertAlmostEqual(values[1].anchor_value or 0.0, 4.2)
        self.assertAlmostEqual(values[1].current_value or 0.0, 3.2)
        self.assertAlmostEqual(values[1].mom45 or 0.0, (3.2 / 4.2) - 1.0)

    def test_buffer_returns_missing_mom45_without_t45_anchor(self) -> None:
        buffer = GrussMomentumBuffer()
        win_60, place_60 = _snapshots_at(60)
        win_2, place_2 = _snapshots_at(2)

        buffer.add_snapshot_pair(win_60, place_60)
        values = buffer.momentum_by_trap(win_2, place_2)

        self.assertFalse(values[1].has_mom45)
        self.assertEqual(values[1].reason, "missing_mom45")

    def test_buffer_does_not_crash_when_runner_missing_at_t45(self) -> None:
        buffer = GrussMomentumBuffer()
        _, place_45 = _snapshots_at(45)
        win_2, place_2 = _snapshots_at(2)
        win_45_rows = _remove_runner_1(_sample_sheet("Hove WIN", 258835465.0))
        win_45_rows[1][3] = "00:00:45"
        win_45_rows[1][5] = ""
        win_45 = parse_gruss_sheet(win_45_rows, "WIN")

        buffer.add_snapshot_pair(win_45, place_45)
        values = buffer.momentum_by_trap(win_2, place_2)

        self.assertFalse(values[1].has_mom45)
        self.assertEqual(values[1].reason, "runner_missing_t45")

    def test_seed_gruss_momentum_into_executor_populates_t45_caches(self) -> None:
        buffer = GrussMomentumBuffer()
        win_45, place_45 = _snapshots_at(45, win_runner_1=(4.0, 4.4, 4.2), place_runner_1=(2.0, 2.2, 2.1))
        win_2, place_2 = _snapshots_at(2, win_runner_1=(3.0, 3.4, 3.2), place_runner_1=(1.8, 2.0, 1.9))
        buffer.add_snapshot_pair(win_45, place_45)
        bundle = build_engine_bundle(win_2, place_2)
        fake_executor = SimpleNamespace(
            _base_win_ms=defaultdict(lambda: defaultdict(dict)),
            _ltp_place_ms=defaultdict(lambda: defaultdict(dict)),
        )

        values = seed_gruss_momentum_into_executor(fake_executor, bundle, win_2, place_2, buffer)

        self.assertTrue(values[1].has_mom45)
        self.assertEqual(fake_executor._base_win_ms["258835465"][1][45], 4.2)
        self.assertEqual(fake_executor._ltp_place_ms["258835466"][1][45], 2.1)


def _remove_runner_1(rows: list[list[object]]) -> list[list[object]]:
    rows[4][0] = ""
    return rows


if __name__ == "__main__":
    unittest.main()
