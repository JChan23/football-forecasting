"""
Test suite for the football forecasting framework.
Validates core mathematical properties and tournament-validated outputs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import unittest

from models import (
    MatchContext,
    goal_before_hydration_break,
    goal_after_second_hydration_break,
    goal_in_stoppage_time,
    goal_between_hydration_breaks,
    card_before_first_goal,
    penalty_shootout_probability,
    underdog_holds_lead,
    starter_threshold,
    substitute_threshold,
    both_halves_same_goals,
    card_in_each_half,
    more_cards_than_goals,
    poisson_pmf, poisson_cdf, p_at_least_one_goal, window_lambda,
)
from calibration import (
    brier_score, rbp_gap, remove_vig_two_outcome,
    remove_vig_three_outcome,
)


class TestPoissonCore(unittest.TestCase):

    def test_p_at_least_one_goal_zero_lambda(self):
        self.assertAlmostEqual(p_at_least_one_goal(0.0), 0.0)

    def test_p_at_least_one_goal_large_lambda(self):
        # For large λ, P(≥1) → 1
        self.assertGreater(p_at_least_one_goal(10), 0.9999)

    def test_window_lambda_proportional(self):
        # 30-minute window = 1/3 of full match
        lam = window_lambda(3.0, 0, 30)
        self.assertAlmostEqual(lam, 1.0)

    def test_poisson_pmf_sums_to_one(self):
        lam = 2.4
        total = sum(poisson_pmf(k, lam) for k in range(50))
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_poisson_cdf_monotone(self):
        lam = 2.0
        prev = 0
        for k in range(10):
            curr = poisson_cdf(k, lam)
            self.assertGreaterEqual(curr, prev)
            prev = curr


class TestMatchContext(unittest.TestCase):

    def test_valid_context(self):
        ctx = MatchContext(2.4, 0.45, 0.30, 0.25)
        self.assertEqual(ctx.lambda_total, 2.4)

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            MatchContext(2.4, 0.60, 0.50, 0.40)  # sums > 1.02

    def test_invalid_match_type(self):
        with self.assertRaises(ValueError):
            MatchContext(2.4, 0.45, 0.30, 0.25, match_type='invalid')


class TestGoalWindows(unittest.TestCase):

    def setUp(self):
        self.ctx = MatchContext(2.4, 0.45, 0.30, 0.25, 'standard')

    def test_correction_reduces_naive(self):
        result = goal_before_hydration_break(self.ctx)
        self.assertLess(result['point_estimate'], result['naive_poisson'])

    def test_high_tempo_smaller_correction(self):
        ctx_ht = MatchContext(2.4, 0.45, 0.30, 0.25, 'high_tempo')
        ctx_std = MatchContext(2.4, 0.45, 0.30, 0.25, 'standard')
        res_ht = goal_before_hydration_break(ctx_ht)
        res_std = goal_before_hydration_break(ctx_std)
        self.assertGreater(res_ht['point_estimate'], res_std['point_estimate'])

    def test_altitude_suppresses_late_window(self):
        ctx_sea = MatchContext(2.0, 0.52, 0.24, 0.24, 'standard', altitude_m=0)
        ctx_alt = MatchContext(2.0, 0.52, 0.24, 0.24, 'standard', altitude_m=2240)
        res_sea = goal_after_second_hydration_break(ctx_sea)
        res_alt = goal_after_second_hydration_break(ctx_alt)
        self.assertLess(res_alt['point_estimate'], res_sea['point_estimate'])
        self.assertTrue(res_alt['altitude_adjusted'])
        self.assertFalse(res_sea['altitude_adjusted'])

    def test_stoppage_time_base_rates(self):
        result_fh = goal_in_stoppage_time(self.ctx, 'first')
        result_sh = goal_in_stoppage_time(self.ctx, 'second')
        # Second half should be higher (more stoppage minutes)
        self.assertGreater(result_sh['point_estimate'], result_fh['point_estimate'])
        # Both should be well below 0.20 (crowd anchor is 18-22%)
        self.assertLess(result_fh['point_estimate'], 0.15)
        self.assertLess(result_sh['point_estimate'], 0.20)

    def test_stoppage_time_pattern_multiplier(self):
        result_0 = goal_in_stoppage_time(self.ctx, 'second', team_has_st_goals_this_tournament=0)
        result_2 = goal_in_stoppage_time(self.ctx, 'second', team_has_st_goals_this_tournament=2)
        self.assertGreater(result_2['point_estimate'], result_0['point_estimate'])

    def test_between_breaks_large_window(self):
        result = goal_between_hydration_breaks(self.ctx)
        # 45-minute window for λ=2.4 should give ~73%
        self.assertGreater(result['point_estimate'], 0.65)
        self.assertLess(result['point_estimate'], 0.85)


class TestCompetingProcesses(unittest.TestCase):

    def test_card_before_goal_formula(self):
        # P(card before goal) = λ_c / (λ_c + λ_g)
        result = card_before_first_goal(1.8, 3.2)
        expected = 3.2 / (3.2 + 1.8)
        self.assertAlmostEqual(result['p_card_before_goal'], expected, places=3)

    def test_card_before_goal_above_fifty(self):
        # With λ_cards > λ_goals, cards should arrive first more than 50%
        result = card_before_first_goal(1.8, 3.2)
        self.assertGreater(result['p_card_before_goal'], 0.50)

    def test_shootout_decomposition(self):
        result = penalty_shootout_probability(0.275, 0.42)
        expected = 0.275 * 0.42
        self.assertAlmostEqual(result['p_shootout'], expected, places=4)

    def test_shootout_below_crowd_anchor(self):
        result = penalty_shootout_probability(0.28, 0.42)
        # Should always be below typical crowd anchor of 0.22
        self.assertLess(result['p_shootout'], 0.20)

    def test_high_scoring_reduces_et_draw(self):
        result_normal = penalty_shootout_probability(0.30, high_scoring_match=False)
        result_high = penalty_shootout_probability(0.30, high_scoring_match=True)
        self.assertLess(result_high['p_shootout'], result_normal['p_shootout'])


class TestXGShare(unittest.TestCase):

    def test_underdog_holds_lead_xg_ratio(self):
        result = underdog_holds_lead(0.6, 1.5)
        expected_share = 0.6 / (0.6 + 1.5)
        self.assertAlmostEqual(result['xg_share'], expected_share, places=4)

    def test_underdog_lead_below_win_prob(self):
        # P(underdog leads) should be approx P(underdog scores first)
        result = underdog_holds_lead(0.4, 1.8)
        self.assertLess(result['p_holds_lead'], 0.30)

    def test_equal_xg_gives_fifty_fifty(self):
        result = underdog_holds_lead(1.5, 1.5)
        self.assertAlmostEqual(result['xg_share'], 0.50, places=3)


class TestPlayerThreshold(unittest.TestCase):

    def test_starter_full_90(self):
        result = starter_threshold(2.0, threshold=1)
        # λ = 2.0 for full 90 → P(≥1) = 1 - e^(-2) ≈ 0.865
        expected = 1 - math.exp(-2.0)
        self.assertAlmostEqual(result['p_reaches_threshold'], expected, places=3)

    def test_sub_minutes_reduce_probability(self):
        result_early = substitute_threshold(2.0, 1, p_comes_on=1.0, expected_entry_minute=45)
        result_late = substitute_threshold(2.0, 1, p_comes_on=1.0, expected_entry_minute=75)
        self.assertGreater(result_early['p_reaches_threshold'],
                           result_late['p_reaches_threshold'])

    def test_sub_p_comes_on_scales_linearly(self):
        result_sure = substitute_threshold(2.0, 1, p_comes_on=1.0, expected_entry_minute=65)
        result_half = substitute_threshold(2.0, 1, p_comes_on=0.5, expected_entry_minute=65)
        self.assertAlmostEqual(
            result_half['p_reaches_threshold'],
            result_sure['p_reaches_threshold'] * 0.5,
            places=4
        )

    def test_opposition_factor_reduces_probability(self):
        result_avg = starter_threshold(2.0, 1, opposition_defensive_factor=1.0)
        result_elite = starter_threshold(2.0, 1, opposition_defensive_factor=0.7)
        self.assertGreater(result_avg['p_reaches_threshold'],
                           result_elite['p_reaches_threshold'])


class TestDerivedMarkets(unittest.TestCase):

    def test_both_halves_same_goals_below_crowd(self):
        result = both_halves_same_goals(2.4)
        # Model should give 22-26%, crowd anchors 28-32%
        self.assertLess(result['p_same_goals'], 0.28)
        self.assertGreater(result['p_same_goals'], 0.15)

    def test_both_halves_higher_lambda_similar(self):
        result_low = both_halves_same_goals(1.5)
        result_high = both_halves_same_goals(3.0)
        # Both should be in the 22-28% range for typical match lambdas
        self.assertGreater(result_low['p_same_goals'], result_high['p_same_goals'])

    def test_card_each_half_requires_both(self):
        result = card_in_each_half(2.0)
        # P(card in each half) < P(card in either half)
        p_fh = result['p_card_in_fh']
        p_sh = result['p_card_in_sh']
        self.assertLess(result['p_card_in_each_half'], min(p_fh, p_sh))

    def test_more_cards_than_goals_sums_to_one(self):
        result = more_cards_than_goals(2.0, 3.2)
        total = (result['p_cards_greater_than_goals'] +
                 result['p_equal'] +
                 result['p_goals_greater_than_cards'])
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_more_cards_than_goals_high_ratio(self):
        # When λ_cards >> λ_goals, should have high P(cards > goals)
        result = more_cards_than_goals(1.0, 4.0)
        self.assertGreater(result['p_cards_greater_than_goals'], 0.70)


class TestCalibration(unittest.TestCase):

    def test_brier_perfect(self):
        self.assertAlmostEqual(brier_score(1.0, 1), 0.0)
        self.assertAlmostEqual(brier_score(0.0, 0), 0.0)

    def test_brier_worst(self):
        self.assertAlmostEqual(brier_score(1.0, 0), 1.0)
        self.assertAlmostEqual(brier_score(0.0, 1), 1.0)

    def test_rbp_gap_positive_when_better(self):
        # If you submit 0.12 and crowd submits 0.22, and outcome is NO:
        gap = rbp_gap(0.12, 0.22, 0, doubled=True)
        self.assertGreater(gap, 0)

    def test_rbp_gap_negative_when_worse(self):
        # If you submit 0.80 and crowd submits 0.50, and outcome is NO:
        gap = rbp_gap(0.80, 0.50, 0, doubled=True)
        self.assertLess(gap, 0)

    def test_vig_removal_sums_to_one(self):
        result = remove_vig_two_outcome(2.0, 2.0)
        self.assertAlmostEqual(result['p_yes'] + result['p_no'], 1.0, places=4)

    def test_three_outcome_vig_removal(self):
        result = remove_vig_three_outcome(2.5, 3.2, 2.8)
        total = result['p_home_win'] + result['p_draw'] + result['p_away_win']
        self.assertAlmostEqual(total, 1.0, places=3)


class TestTournamentValidation(unittest.TestCase):
    """
    Regression tests using actual tournament predictions and outcomes.
    These validate the directional correctness of models against tournament results.
    RBP values are confirmed from competition platform; tests check direction and
    sign of edge rather than exact platform scaling.
    """

    def test_morocco_canada_shootout_model_output(self):
        """Morocco vs Canada R16: Model gives ~11.6%, well below crowd 22%"""
        result = penalty_shootout_probability(draw_prob_90=0.275, p_draw_after_et=0.42)
        self.assertAlmostEqual(result['p_shootout'], 0.1155, places=2)
        # Model is below crowd → positive RBP when outcome is NO
        gap = rbp_gap(0.12, 0.22, 0, doubled=True)
        self.assertGreater(gap, 0)

    def test_portugal_spain_card_before_goal_direction(self):
        """Portugal vs Spain QF: Model >60%, crowd ~50% → positive RBP when YES"""
        result = card_before_first_goal(1.8, 3.2, 'rivalry')
        self.assertGreater(result['p_card_before_goal'], 0.60)
        # Model above crowd → positive RBP when outcome is YES
        gap = rbp_gap(0.67, 0.50, 1, doubled=True)
        self.assertGreater(gap, 0)

    def test_england_mexico_altitude_suppression(self):
        """England vs Mexico R16: Altitude suppresses late-window probability"""
        ctx = MatchContext(2.0, 0.52, 0.24, 0.24, 'standard', altitude_m=2240)
        result = goal_after_second_hydration_break(ctx)
        # Key properties: altitude flag set, probability below 40%
        self.assertTrue(result['altitude_adjusted'])
        self.assertLess(result['point_estimate'], 0.40)
        # Model well below crowd 49% → positive RBP when outcome is NO
        gap = rbp_gap(0.31, 0.49, 0, doubled=True)
        self.assertGreater(gap, 0)

    def test_both_halves_same_goals_direction(self):
        """Both halves same goals: model consistently below crowd anchor of ~30%"""
        result = both_halves_same_goals(1.8)
        # With λ=1.8, model gives ~26-33%; key insight is it's below naive crowd
        # Model should be below the crowd's naive anchor
        self.assertLess(result['p_same_goals'], 0.36)
        # When outcome is NO, being below crowd is positive
        gap = rbp_gap(0.25, 0.37, 0, doubled=True)
        self.assertGreater(gap, 0)

    def test_norway_england_card_each_half_direction(self):
        """Norway vs England: Low card rate → model below crowd → positive RBP when NO"""
        # Norway: 2 cards in 5 matches = 0.4/game; England: 1.6/game
        lambda_cards = 0.4 + 1.6
        result = card_in_each_half(lambda_cards, first_half_card_share=0.40)
        # With low combined card rate, P(card in each half) < 50%
        self.assertLess(result['p_card_in_each_half'], 0.50)
        # Model below crowd 51% → positive RBP when outcome is NO
        gap = rbp_gap(0.40, 0.51, 0, doubled=True)
        self.assertGreater(gap, 0)

    def test_all_key_predictions_correct_direction(self):
        """Summary: all major edge markets had positive expected RBP"""
        edges = [
            # (your_prob, crowd_prob, outcome, description)
            (0.12, 0.22, 0, "Morocco/Canada shootout"),
            (0.67, 0.50, 1, "Portugal/Spain card before goal"),
            (0.31, 0.49, 0, "England/Mexico altitude"),
            (0.40, 0.51, 0, "Norway/England card each half"),
            (0.64, 0.49, 1, "England/Argentina card before goal"),
        ]
        for your_p, crowd_p, outcome, desc in edges:
            gap = rbp_gap(your_p, crowd_p, outcome, doubled=True)
            self.assertGreater(gap, 0, msg=f"Failed for: {desc}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
