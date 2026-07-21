"""
derived_markets.py
------------------
Models for compound and derived markets that combine multiple Poisson
distributions or require joint probability calculations.

Markets covered:
- Both halves same number of goals
- Card shown in each half
- More total cards than total goals
- Either team wins both halves
- N+ different players attempt a shot (coupon collector model)
- 9+ total substitutions
"""

import math
from models.poisson_goals import poisson_pmf, poisson_cdf


def both_halves_same_goals(lambda_total: float,
                            first_half_share: float = 0.45) -> dict:
    """
    P(first-half goals == second-half goals) in regulation.

    Crowds anchor at 28-32% without properly computing the joint Poisson
    distribution. Model consistently gives 22-26%. Validated: 3 instances,
    3 wins, average +11.6 RBP.

    Parameters
    ----------
    lambda_total : float
        Expected total goals in regulation.
    first_half_share : float
        Proportion of goals expected in first half (default 0.45).

    Returns
    -------
    dict
    """
    lam_fh = lambda_total * first_half_share
    lam_sh = lambda_total * (1 - first_half_share)

    # P(FH = k AND SH = k) for k = 0, 1, 2, 3, ...
    p_same = 0.0
    max_k = 8
    breakdown = {}

    for k in range(max_k + 1):
        p_k = poisson_pmf(k, lam_fh) * poisson_pmf(k, lam_sh)
        p_same += p_k
        if p_k > 0.001:
            breakdown[f'both_{k}_goals'] = round(p_k, 4)

    return {
        'lambda_total': round(lambda_total, 4),
        'lambda_fh': round(lam_fh, 4),
        'lambda_sh': round(lam_sh, 4),
        'p_same_goals': round(p_same, 4),
        'scenario_breakdown': breakdown,
        'crowd_anchor': '0.28-0.32',
        'model_typical_range': '0.22-0.26',
        'note': (
            'Key driver: P(0 FH goals) × P(0 SH goals) alone contributes '
            f'{round(poisson_pmf(0, lam_fh) * poisson_pmf(0, lam_sh), 4):.1%} '
            'which crowds ignore.'
        )
    }


def card_in_each_half(lambda_cards_total: float,
                       first_half_card_share: float = 0.40) -> dict:
    """
    P(at least one card in each half of regulation).

    P(card in FH AND card in SH) = P(≥1 in FH) × P(≥1 in SH)
    These are approximately independent.

    Crowds anchor at 48-55%; model gives 35-45% depending on card rates.
    Validated: Norway vs England (40% vs crowd 51%), NO → +25.90 RBP.

    Critical input: use actual team card rates, not tournament average.
    Some teams (e.g. Norway 2026: 1 card per 24 fouls) are dramatically
    cleaner than average.

    Parameters
    ----------
    lambda_cards_total : float
        Expected total cards in regulation.
    first_half_card_share : float
        Proportion of cards expected in first half (default 0.40).

    Returns
    -------
    dict
    """
    lam_fh = lambda_cards_total * first_half_card_share
    lam_sh = lambda_cards_total * (1 - first_half_card_share)

    p_card_fh = 1 - math.exp(-lam_fh)
    p_card_sh = 1 - math.exp(-lam_sh)
    p_card_each_half = p_card_fh * p_card_sh  # approximately independent

    return {
        'lambda_cards_total': round(lambda_cards_total, 4),
        'lambda_cards_fh': round(lam_fh, 4),
        'lambda_cards_sh': round(lam_sh, 4),
        'p_card_in_fh': round(p_card_fh, 4),
        'p_card_in_sh': round(p_card_sh, 4),
        'p_card_in_each_half': round(p_card_each_half, 4),
        'crowd_anchor': '0.48-0.55',
        'note': (
            'Always check actual team card rates, not tournament average. '
            'Norway 2026: 2 cards in 5 matches = 0.4/game vs tournament avg 3.2/game. '
            'This alone drops the estimate dramatically.'
        )
    }


def more_cards_than_goals(lambda_goals: float,
                           lambda_cards: float) -> dict:
    """
    P(total cards > total goals) in regulation.

    Computed via joint Poisson distribution enumeration.
    Profitable when lambda_goals < 2.0 in high-intensity matches.

    Parameters
    ----------
    lambda_goals : float
        Expected total goals in regulation.
    lambda_cards : float
        Expected total cards in regulation.

    Returns
    -------
    dict
    """
    max_k = 12
    p_cards_greater = 0.0
    p_equal = 0.0
    p_goals_greater = 0.0

    for g in range(max_k + 1):
        for c in range(max_k + 1):
            p_joint = poisson_pmf(g, lambda_goals) * poisson_pmf(c, lambda_cards)
            if c > g:
                p_cards_greater += p_joint
            elif c == g:
                p_equal += p_joint
            else:
                p_goals_greater += p_joint

    return {
        'lambda_goals': round(lambda_goals, 4),
        'lambda_cards': round(lambda_cards, 4),
        'p_cards_greater_than_goals': round(p_cards_greater, 4),
        'p_equal': round(p_equal, 4),
        'p_goals_greater_than_cards': round(p_goals_greater, 4),
        'note': (
            f'Ratio λ_cards/λ_goals = {lambda_cards/lambda_goals:.2f}. '
            'Typically profitable when λ_goals < 2.0 and λ_cards > 2.5.'
        )
    }


def either_team_wins_both_halves(lambda_home: float,
                                  lambda_away: float,
                                  p_home_win: float,
                                  p_away_win: float) -> dict:
    """
    P(one team leads at halftime AND scores more in second half).

    Rarely exceeds 25% in evenly matched knockout matches. Crowds anchor
    at 22-28%; model gives 16-22%.

    Approximation method: model each team's conditional probability of
    winning both halves independently, then combine.

    Parameters
    ----------
    lambda_home : float
        Expected goals for home team.
    lambda_away : float
        Expected goals for away team.
    p_home_win : float
        Market-implied probability of home team winning in 90 min.
    p_away_win : float
        Market-implied probability of away team winning in 90 min.

    Returns
    -------
    dict
    """
    # P(team wins both halves | wins match) × P(wins match)
    # Empirical estimate: teams win both halves in ~40-45% of their wins
    P_WINS_BOTH_HALVES_GIVEN_WIN = 0.42

    p_home_both = p_home_win * P_WINS_BOTH_HALVES_GIVEN_WIN
    p_away_both = p_away_win * P_WINS_BOTH_HALVES_GIVEN_WIN

    # These are mutually exclusive
    p_either_both = p_home_both + p_away_both

    return {
        'lambda_home': round(lambda_home, 4),
        'lambda_away': round(lambda_away, 4),
        'p_home_wins_both_halves': round(p_home_both, 4),
        'p_away_wins_both_halves': round(p_away_both, 4),
        'p_either_wins_both_halves': round(p_either_both, 4),
        'crowd_anchor': '0.22-0.28',
        'note': (
            'Crowd overprices by treating match win probability as sufficient. '
            'Winning both halves requires sustained dominance across 90 minutes.'
        )
    }


def n_plus_distinct_shooters(lambda_shots_total: float,
                               n_regular_shooters: int,
                               threshold: int) -> dict:
    """
    P(≥ threshold different players attempt at least one shot).

    Uses a simplified coupon collector / occupancy model.
    Primarily used for "5+ different Spain players attempt a shot" type markets.

    Parameters
    ----------
    lambda_shots_total : float
        Expected total shots in the match for the team.
    n_regular_shooters : int
        Number of players in the team who regularly attempt shots.
    threshold : int
        Minimum number of distinct shooters required.

    Returns
    -------
    dict
    """
    # Expected shots per player
    expected_per_player = lambda_shots_total / n_regular_shooters

    # P(player attempts 0 shots) ~ Poisson
    p_no_shot_per_player = math.exp(-expected_per_player)
    p_shot_per_player = 1 - p_no_shot_per_player

    # P(exactly k players attempt a shot) ~ Binomial(n, p)
    p_at_least_threshold = 0.0
    for k in range(threshold, n_regular_shooters + 1):
        binom_coef = math.comb(n_regular_shooters, k)
        p_k = binom_coef * (p_shot_per_player ** k) * (p_no_shot_per_player ** (n_regular_shooters - k))
        p_at_least_threshold += p_k

    return {
        'lambda_shots_total': round(lambda_shots_total, 4),
        'n_regular_shooters': n_regular_shooters,
        'threshold': threshold,
        'expected_shots_per_player': round(expected_per_player, 4),
        'p_each_player_shoots': round(p_shot_per_player, 4),
        'p_at_least_threshold_distinct_shooters': round(p_at_least_threshold, 4),
        'note': (
            'Possession-dominant teams (Spain, France) distribute shots across '
            'more players. Assumes uniform shot distribution — adjust for '
            'teams with one dominant striker.'
        )
    }


def substitution_count_probability(p_each_team_makes_5: float,
                                    p_each_team_makes_4: float,
                                    threshold: int = 9) -> dict:
    """
    P(total substitutions ≥ threshold) in regulation (max 10).

    Parameters
    ----------
    p_each_team_makes_5 : float
        Probability each team uses all 5 substitutions.
    p_each_team_makes_4 : float
        Probability each team uses exactly 4 substitutions.
    threshold : int
        Minimum total subs required (default 9).

    Returns
    -------
    dict
    """
    # P(total ≥ 9) = P(5+4) + P(4+5) + P(5+5)
    p_team_3_or_fewer = max(0, 1 - p_each_team_makes_5 - p_each_team_makes_4)

    p_ten = p_each_team_makes_5 ** 2
    p_nine = 2 * p_each_team_makes_5 * p_each_team_makes_4
    p_at_least_nine = p_ten + p_nine

    p_eight = p_each_team_makes_4 ** 2
    p_seven_or_fewer = 1 - p_at_least_nine - p_eight

    return {
        'p_each_team_makes_5': round(p_each_team_makes_5, 4),
        'p_each_team_makes_4': round(p_each_team_makes_4, 4),
        'p_total_10_subs': round(p_ten, 4),
        'p_total_9_subs': round(p_nine, 4),
        f'p_total_gte_{threshold}_subs': round(p_at_least_nine, 4),
        'p_total_8_subs': round(p_eight, 4),
        'note': (
            'ET probability reduces regulation subs (managers save one slot). '
            'Apply ~10ppt downward adjustment if P(ET) > 0.25.'
        )
    }
