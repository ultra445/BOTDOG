import math
import unittest
from unittest.mock import patch

from dogbot.staking import Side, StakingEngine


BASE_ENV = {
    "CAPITAL": "500",
    "MIN_STAKE": "2",
    "MAX_MARKET_STAKE": "1000",
    "MAX_RUNNER_STAKE": "1000",
    "DOGBOT_MAX_LAY_LIABILITY_PER_ORDER": "50",
}


class StakingEngineTests(unittest.TestCase):
    def test_back_uses_capital_edge_over_odds_power(self) -> None:
        env = dict(BASE_ENV, DOGBOT_STAKE_BACK_ODDS_DECAY_ALPHA="0.60")
        with patch.dict("os.environ", env, clear=True):
            result = StakingEngine().quote(Side.BACK, price_ltp=10.0, edge=0.02)

        expected = 500 * 0.02 / (10.0**0.60)
        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "back_capital_edge_over_odds_power")
        self.assertAlmostEqual(result.size, round(expected, 2))
        self.assertIsNone(result.liability)
        self.assertEqual(result.staking_formula, "capital_edge_over_odds_power")
        self.assertEqual(result.staking_alpha, 0.60)
        self.assertAlmostEqual(result.stake_raw_before_caps, expected)

    def test_lay_uses_stake_formula_then_liability_as_consequence(self) -> None:
        env = dict(BASE_ENV, DOGBOT_STAKE_LAY_ODDS_DECAY_ALPHA="0.70")
        with patch.dict("os.environ", env, clear=True):
            result = StakingEngine().quote(Side.LAY, price_ltp=10.0, edge=0.02)

        expected_stake = 500 * 0.02 / (10.0**0.70)
        expected_liability = round(round(expected_stake, 2) * 9.0, 2)
        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "lay_capital_edge_over_odds_power")
        self.assertAlmostEqual(result.size, round(expected_stake, 2))
        self.assertEqual(result.liability, expected_liability)
        self.assertFalse(result.lay_liability_cap_hit)
        self.assertEqual(result.lay_liability_cap, 50.0)

    def test_lay_liability_cap_reduces_stake_without_using_min_liability(self) -> None:
        env = dict(
            BASE_ENV,
            DOGBOT_STAKE_LAY_ODDS_DECAY_ALPHA="0.70",
            DOGBOT_MAX_LAY_LIABILITY_PER_ORDER="30",
            MIN_LIABILITY="999",
        )
        with patch.dict("os.environ", env, clear=True):
            result = StakingEngine().quote(Side.LAY, price_ltp=10.0, edge=0.20)

        self.assertTrue(result.ok)
        self.assertTrue(result.lay_liability_cap_hit)
        self.assertAlmostEqual(result.size, round(30.0 / 9.0, 2))
        self.assertEqual(result.liability, round(result.size * 9.0, 2))
        self.assertLess(result.liability, 999)

    def test_lay_liability_cap_below_min_stake_rejects(self) -> None:
        env = dict(
            BASE_ENV,
            DOGBOT_STAKE_LAY_ODDS_DECAY_ALPHA="0.70",
            DOGBOT_MAX_LAY_LIABILITY_PER_ORDER="1",
        )
        with patch.dict("os.environ", env, clear=True):
            result = StakingEngine().quote(Side.LAY, price_ltp=10.0, edge=0.20)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "lay_liability_cap_below_min_stake")
        self.assertTrue(result.lay_liability_cap_hit)
        self.assertLess(result.size, 2.0)

    def test_lay_min_stake_is_not_raised_when_it_would_break_liability_cap(self) -> None:
        env = dict(
            BASE_ENV,
            DOGBOT_STAKE_LAY_ODDS_DECAY_ALPHA="0.70",
            DOGBOT_MAX_LAY_LIABILITY_PER_ORDER="5",
        )
        with patch.dict("os.environ", env, clear=True):
            result = StakingEngine().quote(Side.LAY, price_ltp=4.0, edge=0.007916)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "lay_liability_cap_below_min_stake")
        self.assertLess(result.size, 2.0)

    def test_side_alpha_falls_back_to_global_alpha(self) -> None:
        env = dict(BASE_ENV, DOGBOT_STAKE_ODDS_DECAY_ALPHA="0.55")
        with patch.dict("os.environ", env, clear=True):
            back = StakingEngine().quote(Side.BACK, price_ltp=10.0, edge=0.02)
            lay = StakingEngine().quote(Side.LAY, price_ltp=10.0, edge=0.02)

        self.assertTrue(math.isclose(back.staking_alpha, 0.55))
        self.assertTrue(math.isclose(lay.staking_alpha, 0.55))

    def test_default_side_alphas_are_back_point_six_and_lay_point_seven(self) -> None:
        with patch.dict("os.environ", BASE_ENV, clear=True):
            back = StakingEngine().quote(Side.BACK, price_ltp=10.0, edge=0.02)
            lay = StakingEngine().quote(Side.LAY, price_ltp=10.0, edge=0.02)

        self.assertTrue(math.isclose(back.staking_alpha, 0.60))
        self.assertTrue(math.isclose(lay.staking_alpha, 0.70))


if __name__ == "__main__":
    unittest.main()
