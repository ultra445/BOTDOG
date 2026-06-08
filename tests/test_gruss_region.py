from __future__ import annotations

import unittest

from dogbot.gruss.gruss_region import (
    is_gruss_result_screen,
    normalize_gruss_meeting_name,
    normalize_gruss_region,
)


class GrussRegionTests(unittest.TestCase):
    def test_pgr_dunstall_park_is_uk(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\PGR\Dunstall Park 3rd Jun"), "UK")

    def test_pgr_romford_is_uk(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\PGR\Romford 3rd Jun"), "UK")

    def test_sis_doncaster_is_uk(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\SIS\Doncaster 3rd Jun"), "UK")

    def test_sis_newcastle_is_uk(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\SIS\Newcastle 3rd Jun"), "UK")

    def test_sis_star_pelaw_is_uk(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\SIS\Star Pelaw 7th Jun"), "UK")

    def test_star_pelaw_meeting_name_is_uk(self) -> None:
        self.assertEqual(normalize_gruss_region(meeting_name="Star Pelaw"), "UK")

    def test_australia_the_meadows_is_row(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\Australia\The Meadows"), "ROW")

    def test_australia_taree_is_row(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\Australia\Taree"), "ROW")

    def test_new_zealand_addington_is_row(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\New Zealand\Addington"), "ROW")

    def test_aus_casino_is_row(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\AUS\Casino (AUS) 4th Jun"), "ROW")

    def test_aus_warragul_is_row(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\AUS\Warragul (AUS) 4th Jun"), "ROW")

    def test_aus_bendigo_is_row(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\AUS\Bendigo (AUS)"), "ROW")

    def test_aus_ballarat_is_row(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\AUS\Ballarat (AUS)"), "ROW")

    def test_nzl_addington_is_row(self) -> None:
        self.assertEqual(normalize_gruss_region(r"Greyhound Racing\NZL\Addington (NZL)"), "ROW")

    def test_aus_and_nzl_markers_are_row_when_passed_as_meeting_name(self) -> None:
        self.assertEqual(normalize_gruss_region(meeting_name="Casino (AUS)"), "ROW")
        self.assertEqual(normalize_gruss_region(meeting_name="Addington (NZL)"), "ROW")

    def test_meeting_name_normalization_strips_date(self) -> None:
        self.assertEqual(normalize_gruss_meeting_name(r"Greyhound Racing\PGR\Dunstall Park 3rd Jun"), "Dunstall Park")

    def test_result_screen_is_unknown_without_warning(self) -> None:
        self.assertTrue(is_gruss_result_screen(event_path="Result", market_title="Bet ref"))
        with self.assertNoLogs("dogbot.gruss.gruss_region", level="WARNING"):
            self.assertEqual(normalize_gruss_region(event_path="Result", market_title="Bet ref"), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
