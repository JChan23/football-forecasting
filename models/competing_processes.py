"""
competing_processes.py
----------------------
Competing exponential (Poisson) process models for race-style markets.

Core mathematical result: When two independent Poisson processes race to
fire first, with rates λ_A and λ_B respectively:

    P(A fires before B) = λ_A / (λ_A + λ_B)

This is the fundamental result for any "which happens first" market in
football. The crowd typically anchors these markets at ~50% treating them
as coin flips, while the rate-based model systematically diverges.

Validated markets:
- Will a card be shown before the first goal? (+32.29 RBP, Portugal/Spain)
- Will a card be shown before the first goal? (+30.21 RBP, England/Argentina)

Tournament results: 2 instances, 2 wins, average +31.25 RBP per instance.
"""

import math
from typing import Optional
from models.poisson_goals import poisson_pmf, poisson_cdf


# Tournament-average card rates (cards per 90 minutes)
TOURNAMENT_AVG_CARDS_PER_90 = 3.2

# Card rate multipliers for match context
CARD_RATE_MULTIPLIERS = {
    'standard':       1.0,
    'rivalry':        1.15,   # Iberian derby, historical tension
    'high_stakes':    1.10,   # World Cup final / semi-final
    'physical':       1.20,   # Both teams press aggressively
    'low_card':       0.75,   # One or both teams very disciplined
}


def card_before_first_goal(lambda_goals: float,
                            lambda_cards: float = TOURNAMENT_AVG_CARDS_PER_90,
                            card_context: str = 'standard') -> dict:
    """
    P(first card shown before first goal) using competing Poisson processes.

    Mathematical derivation:
        Let T_goal ~ Exp(λ_goals/90) and T_card ~ Exp(λ_cards/90).
        P(T_card < T_goal) = λ_cards / (λ_cards + λ_goals)

    This is exact for exponential (memoryless) inter-event times, which is
    the natural assumption for Poisson-distributed events.

    Parameters
    ----------
    lambda_goals : float
        Expected total goals in the match (from market over/under).
    lambda_cards : float
        Expected total cards in the match.
    card_context : str
        Match context modifier for card rate. One of CARD_RATE_MULTIPLIERS.

    Returns
    -------
    dict with keys:
        'lambda_goals': float
        'lambda_cards_adjusted': float
        'p_card_before_goal': float — the market probability
        'crowd_anchor': str — typical crowd pricing for reference
        'edge_vs_crowd': float — estimated RBP edge per instance
    """
    if card_context not in CARD_RATE_MULTIPLIERS:
        raise ValueError(f"card_context must be one of {list(CARD_RATE_MULTIPLIERS.keys())}")

    lam_cards_adj = lambda_cards * CARD_RATE_MULTIPLIERS[card_context]
    p = lam_cards_adj / (lam_cards_adj + lambda_goals)

    return {
        'lambda_goals': round(lambda_goals, 4),
        'lambda_cards_adjusted': round(lam_cards_adj, 4),
        'card_context': card_context,
        'p_card_before_goal': round(p, 4),
        'crowd_anchor': '~0.49-0.52 (treated as coin flip)',
        'estimated_edge_vs_crowd_ppt': round((p - 0.50) * 100, 1),
        'formula': 'λ_cards / (λ_cards + λ_goals)',
    }


def first_event_from_distribution(rates: dict) -> dict:
    """
    P(each event type fires first) when multiple Poisson processes race.

    Generalisation of the two-process case to n competing processes.
    The probability that process i fires first is:
        P(i first) = λ_i / Σ(λ_j for all j)

    Use case: "Will first goal be scored by a player wearing shirt #1-9?"
    Build a distribution of first-scorer probabilities and map to attribute.

    Parameters
    ----------
    rates : dict
        {event_label: rate} mapping. Rates are proportional to probability
        of each event firing first (e.g. expected goals per game per player).

    Returns
    -------
    dict
        {event_label: p_fires_first} with probabilities summing to 1.
    """
    total_rate = sum(rates.values())
    if total_rate == 0:
        raise ValueError("At least one rate must be positive.")

    return {
        label: round(rate / total_rate, 4)
        for label, rate in rates.items()
    }


def first_goal_by_attribute(scorer_rates: dict,
                             attribute_map: dict,
                             target_attribute: str) -> dict:
    """
    P(first goal has a specific attribute) given per-player goal rates.

    Validated markets:
    - First goal by player wearing shirt #1-9 (crowd ~35%, model ~22%, YES resolved NO — +21 RBP)
    - First goal by player other than [named players] (crowd ~55%, model ~65%, YES resolved YES — +22 RBP)

    Core insight: crowds apply naive base rates (e.g. "roughly 1/3 of shirt
    numbers are 1-9") without weighting by the actual goal-scoring distribution,
    where high-numbered attackers (#10, #11, #17, #19) dominate.

    Parameters
    ----------
    scorer_rates : dict
        {player_name: expected_goals_per_game} for all likely scorers.
        Include 'no_goal' and 'own_goal' as special entries.
    attribute_map : dict
        {player_name: attribute_value} mapping each player to their attribute.
        Use 'no_goal' -> None, 'own_goal' -> 'own'.
    target_attribute : str
        The attribute value to compute probability for.

    Returns
    -------
    dict with keys:
        'p_target_attribute': float — P(first goal has target attribute)
        'p_no_goal': float — probability of 0-0 result (settles NO)
        'p_other_attribute': float — probability of first goal with different attribute
        'contributing_players': dict — players contributing to target probability
    """
    # First, compute first-scorer probabilities using competing processes
    first_scorer_probs = first_event_from_distribution(scorer_rates)

    # Map to attribute
    p_target = 0.0
    p_no_goal = 0.0
    contributing = {}

    for player, p_first in first_scorer_probs.items():
        attr = attribute_map.get(player)
        if player == 'no_goal' or attr is None:
            p_no_goal += p_first
        elif attr == target_attribute:
            p_target += p_first
            contributing[player] = round(p_first, 4)

    p_other = 1.0 - p_target - p_no_goal

    return {
        'target_attribute': target_attribute,
        'p_target_attribute': round(p_target, 4),
        'p_no_goal_settles_no': round(p_no_goal, 4),
        'p_other_attribute': round(p_other, 4),
        'contributing_players': contributing,
        'note': (
            'Crowd anchors on naive base rates. Model weights by actual '
            'goal-scoring rate distribution. Edge typically 10-15ppt.'
        )
    }


def penalty_shootout_probability(draw_prob_90: float,
                                  p_draw_after_et: float = 0.42,
                                  high_scoring_match: bool = False) -> dict:
    """
    P(match decided by penalty shootout) via two-stage decomposition.

    P(shootout) = P(draw after 90) × P(draw after ET | drew after 90)

    Validated: 4 instances in tournament, all 4 resolved NO.
    Average crowd anchor: ~22%. Model range: 10-14%.
    Average RBP per instance: ~+12.

    The crowd anchors on "knockout matches often go to penalties" without
    performing the proper conditional probability decomposition.

    Parameters
    ----------
    draw_prob_90 : float
        Market-implied probability of draw after 90 minutes (vig-removed).
    p_draw_after_et : float
        Historical probability of match remaining level after 30 min ET.
        Default 0.42 (historical World Cup knockout average).
        Reduce to ~0.38 for high-scoring matches (more likely to score in ET).
    high_scoring_match : bool
        If True, adjusts p_draw_after_et downward for open, attacking matches.

    Returns
    -------
    dict
    """
    if high_scoring_match:
        p_draw_after_et = min(p_draw_after_et, 0.38)

    p_shootout = draw_prob_90 * p_draw_after_et

    return {
        'draw_prob_90': round(draw_prob_90, 4),
        'p_draw_after_et': round(p_draw_after_et, 4),
        'p_shootout': round(p_shootout, 4),
        'p_extra_time': round(draw_prob_90, 4),  # same as draw at 90
        'crowd_anchor': '~0.20-0.24',
        'model_typical_range': '0.10-0.14',
        'formula': 'P(draw_90) × P(draw_ET | drew_90)',
        'note': (
            'Crowd overprices by ~10ppt by ignoring conditional probability. '
            'Cross-check with bookie direct shootout odds if available; '
            'take midpoint if gap exists.'
        )
    }


def match_goes_to_et(draw_prob_90: float) -> dict:
    """
    P(match goes to extra time) = P(draw after 90 minutes).

    Trivially equal to the market-implied draw probability after vig removal.
    Crowd anchors ~7-10ppt above the market-implied draw probability.

    Parameters
    ----------
    draw_prob_90 : float
        Market-implied probability of draw after 90 minutes (vig-removed).

    Returns
    -------
    dict
    """
    return {
        'p_extra_time': round(draw_prob_90, 4),
        'crowd_anchor': round(draw_prob_90 + 0.07, 4),
        'note': 'Submit draw_prob_90 directly. Crowd adds ~7ppt of noise.'
    }
