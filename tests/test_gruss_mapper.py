from __future__ import annotations

import unittest

from dogbot.gruss.gruss_mapper import (
    GrussMapper,
    extract_trap,
    format_countdown_display,
    normalize_runner_name,
    parse_countdown_seconds,
    parse_gruss_sheet,
    validate_win_place_pair,
)


def _blank_rows(row_count: int = 12, column_count: int = 50) -> list[list[object]]:
    return [["" for _ in range(column_count)] for _ in range(row_count)]


def _sample_sheet(title: str, market_id: object, winners: object = 1.0) -> list[list[object]]:
    rows = _blank_rows()
    rows[0][0] = title
    rows[0][5] = r"Greyhound Racing\PGR\Hove 3rd Jun"
    rows[0][12] = "Parent id"
    rows[0][13] = 35678242.0
    rows[1][0] = "Last updated:"
    rows[1][1] = 46176.70692924769
    rows[1][3] = "-00:00:58"
    rows[1][4] = "Not In Play"
    rows[1][5] = "Suspended"
    rows[2][0] = "Total matched:"
    rows[2][1] = 25293.47
    rows[2][2] = "A4 500m"
    rows[2][3] = "Winners"
    rows[2][4] = winners
    rows[2][12] = "Market id"
    rows[2][13] = market_id
    rows[3][0] = "Selection name"
    rows[3][1] = "Back Odds 3"
    rows[3][3] = "Back Odds 2"
    rows[3][5] = "Back Odds 1"
    rows[3][7] = "Lay Odds 1"
    rows[3][14] = "Last price matched"
    rows[3][15] = "Total amount matched"
    rows[4][0] = "1. Gingers Layla"
    rows[4][1] = 9.8
    rows[4][2] = 12.0
    rows[4][3] = 9.6
    rows[4][4] = 20.0
    rows[4][5] = 9.4
    rows[4][6] = 33.0
    rows[4][7] = 9.8
    rows[4][8] = 14.0
    rows[4][9] = 10.0
    rows[4][10] = 16.0
    rows[4][11] = 11.0
    rows[4][12] = 18.0
    rows[4][13] = 2.5
    rows[4][14] = 9.2
    rows[4][15] = 11274.07
    rows[5][0] = "2. Coppeen Class"
    rows[5][5] = 4.5
    rows[5][7] = 4.7
    rows[5][14] = 4.6
    rows[5][15] = 2613.39
    return rows


class GrussMapperTests(unittest.TestCase):
    def test_parse_gruss_sheet_metadata_and_runners(self) -> None:
        snapshot = parse_gruss_sheet(_sample_sheet("Hove WIN", 258835465.0), "WIN")

        self.assertEqual(snapshot.sheet_name, "WIN")
        self.assertEqual(snapshot.metadata.market_title, "Hove WIN")
        self.assertEqual(snapshot.metadata.event_path, r"Greyhound Racing\PGR\Hove 3rd Jun")
        self.assertEqual(snapshot.metadata.parent_id, "35678242")
        self.assertEqual(snapshot.metadata.last_updated, 46176.70692924769)
        self.assertEqual(snapshot.metadata.countdown, "-00:00:58")
        self.assertEqual(snapshot.metadata.countdown_seconds, -58)
        self.assertEqual(snapshot.metadata.countdown_display, "-00:58")
        self.assertEqual(snapshot.metadata.market_status, "Not In Play")
        self.assertEqual(snapshot.metadata.suspend_status, "Suspended")
        self.assertEqual(snapshot.metadata.total_matched, 25293.47)
        self.assertEqual(snapshot.metadata.market_id, "258835465")
        self.assertEqual(snapshot.metadata.winners, 1)

        runner = snapshot.runners[0]
        self.assertEqual(runner.selection_raw, "1. Gingers Layla")
        self.assertEqual(runner.trap, 1)
        self.assertEqual(runner.runner_name, "Gingers Layla")
        self.assertEqual(runner.back_odds_3, 9.8)
        self.assertEqual(runner.back_stake_3, 12.0)
        self.assertEqual(runner.back_odds_2, 9.6)
        self.assertEqual(runner.back_stake_2, 20.0)
        self.assertEqual(runner.back_odds_1, 9.4)
        self.assertEqual(runner.back_stake_1, 33.0)
        self.assertEqual(runner.lay_odds_1, 9.8)
        self.assertEqual(runner.lay_stake_1, 14.0)
        self.assertEqual(runner.lay_odds_2, 10.0)
        self.assertEqual(runner.lay_stake_2, 16.0)
        self.assertEqual(runner.lay_odds_3, 11.0)
        self.assertEqual(runner.lay_stake_3, 18.0)
        self.assertEqual(runner.reduction_factor, 2.5)
        self.assertEqual(runner.last_price_matched, 9.2)
        self.assertEqual(runner.total_amount_matched, 11274.07)
        self.assertEqual(runner.best_back, runner.back_odds_1)
        self.assertEqual(runner.best_lay, runner.lay_odds_1)
        self.assertEqual(runner.ltp, runner.last_price_matched)

    def test_runner_name_helpers(self) -> None:
        self.assertEqual(extract_trap("1. Gingers Layla"), 1)
        self.assertEqual(extract_trap("7. Late Runner"), 7)
        self.assertEqual(extract_trap("Trap 8 Wide Runner"), 8)
        self.assertEqual(normalize_runner_name("1. Ginger's  Layla"), "ginger s layla")
        self.assertEqual(GrussMapper.extract_trap("2. Coppeen Class"), 2)

    def test_parse_gruss_sheet_keeps_traps_seven_and_eight_after_blank_row(self) -> None:
        rows = _sample_sheet("Hove WIN", 258835465.0)
        rows[6][0] = ""
        rows[7][0] = "7. Seven Runner"
        rows[7][5] = 6.0
        rows[7][7] = 6.2
        rows[8][0] = "8. Eight Runner"
        rows[8][5] = 7.0
        rows[8][7] = 7.2

        snapshot = parse_gruss_sheet(rows, "WIN")

        self.assertIn(7, [runner.trap for runner in snapshot.runners])
        self.assertIn(8, [runner.trap for runner in snapshot.runners])
        self.assertEqual(snapshot.runners[-2].runner_name, "Seven Runner")
        self.assertEqual(snapshot.runners[-1].runner_name, "Eight Runner")

    def test_countdown_helpers_parse_excel_fraction_and_text(self) -> None:
        self.assertEqual(parse_countdown_seconds(0.00047453703703703704), 41)
        self.assertEqual(format_countdown_display(41), "00:41")
        self.assertEqual(parse_countdown_seconds(0.000462962962962963), 40)
        self.assertEqual(format_countdown_display(40), "00:40")
        self.assertEqual(parse_countdown_seconds("00:00:41"), 41)
        self.assertEqual(parse_countdown_seconds("-00:00:13"), -13)
        self.assertEqual(format_countdown_display(-13), "-00:13")

    def test_validate_win_place_pair_accepts_matching_race(self) -> None:
        win = parse_gruss_sheet(_sample_sheet("Hove WIN", 258835465.0), "WIN")
        place = parse_gruss_sheet(_sample_sheet("Hove PLACE", 258835466.0, winners=2.0), "PLACE")

        self.assertTrue(validate_win_place_pair(win, place))

    def test_validate_win_place_pair_rejects_mismatch(self) -> None:
        win_rows = _sample_sheet("Hove WIN", 258835465.0)
        place_rows = _sample_sheet("Romford PLACE", 258835466.0, winners=2.0)
        place_rows[0][5] = r"Greyhound Racing\PGR\Romford 3rd Jun"
        place_rows[0][13] = 999
        place_rows[4][0] = "5. Different Name"
        place_rows[5][0] = "6. Other Runner"
        win = parse_gruss_sheet(win_rows, "WIN")
        place = parse_gruss_sheet(place_rows, "PLACE")

        with self.assertLogs("dogbot.gruss.gruss_mapper", level="WARNING") as logs:
            self.assertFalse(validate_win_place_pair(win, place))

        self.assertIn("Gruss WIN/PLACE validation failed", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
