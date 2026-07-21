"""
player_threshold.py
-------------------
Named player threshold models using Poisson CDF.

Two market structures require different approaches:

1. STARTER: Player plays full 90 minutes. Use their per-90 rate directly.
   Adjust downward for strong opposition defensive records.

2. SUBSTITUTE: Player enters mid-game. Must model:
   P(comes on) × P(reaches threshold | comes on in X minutes)

The sub minutes-constraint chain is the critical insight crowds miss.
Crowds price "quality player = shots" without accounting for the limited
time window available to a substitute.

Validated markets:
- De Bruyne 1+ SoT (starter): 53% (crowd 47%), YES → +18.39 RBP
- Marmoush 1+ SoT (sub, confirmed not starting): 28% (crowd 37%), NO → +20.63 RBP
- Lukaku 1+ SoT (sub): model 32% → submitted 55% (diverged from model) → -11.93 RBP
"""

import math
from models.poisson_goals import poisson_pmf, poisson_cdf


def p_reaches_threshold_poisson(rate_per_90: float,
                                  minutes_available: float,
                                  threshold: int,
                                  opposition_defensive_factor: float = 1.0) -> float:
    """
    P(player reaches threshold in available minutes) via Poisson CDF.

    Parameters
    ----------
    rate_per_90 : float
        Player's per-90-minute rate for the metric (e.g. SoT/90, goals/90).
    minutes_available : float
        Minutes available to the player in the match.
    threshold : int
        Minimum count to reach (e.g. 1 for "1+ SoT", 2 for "2+ goals").
    opposition_defensive_factor : float
        Multiplier for opposition defensive quality.
        1.0 = average defence. 0.7 = elite defence (e.g. Spain conceding 1 goal in 7 matches).
        Values < 1.0 reduce effective rate; values > 1.0 increase it.

    Returns
    -------
    float
        P(player metric ≥ threshold in available minutes).
    """
    if minutes_available <= 0:
        return 0.0

    # Scale rate to available minutes, adjusted for opposition
    effective_rate = rate_per_90 * (minutes_available / 90.0) * opposition_defensive_factor

    # P(X >= threshold) = 1 - P(X <= threshold - 1)
    p_below_threshold = poisson_cdf(threshold - 1, effective_rate)
    return round(1.0 - p_below_threshold, 4)


def starter_threshold(rate_per_90: float,
                       threshold: int,
                       opposition_defensive_factor: float = 1.0,
                       minutes_expected: float = 90.0) -> dict:
    """
    P(starter reaches threshold in regulation).

    Parameters
    ----------
    rate_per_90 : float
        Player's per-90-minute rate for the metric.
    threshold : int
        Minimum count to reach.
    opposition_defensive_factor : float
        Defensive quality adjustment (1.0 = average, <1.0 = strong defence).
    minutes_expected : float
        Expected minutes played (default 90, reduce if tactical sub likely).

    Returns
    -------
    dict
    """
    p = p_reaches_threshold_poisson(
        rate_per_90, minutes_expected, threshold, opposition_defensive_factor
    )

    effective_lambda = rate_per_90 * (minutes_expected / 90) * opposition_defensive_factor

    return {
        'player_type': 'starter',
        'rate_per_90': round(rate_per_90, 4),
        'opposition_defensive_factor': round(opposition_defensive_factor, 4),
        'effective_lambda': round(effective_lambda, 4),
        'minutes_expected': minutes_expected,
        'threshold': threshold,
        'p_reaches_threshold': p,
        'note': (
            'For elite defensive opposition (e.g. Spain 2026), '
            'apply factor ~0.7. Average opposition: 1.0. '
            'Leaky defence: 1.2-1.4.'
        )
    }


def substitute_threshold(rate_per_90: float,
                           threshold: int,
                           p_comes_on: float,
                           expected_entry_minute: float = 65.0,
                           match_minutes: float = 90.0,
                           opposition_defensive_factor: float = 1.0) -> dict:
    """
    P(substitute reaches threshold) via probability chain.

    P(threshold) = P(comes on) × P(threshold | comes on with X minutes remaining)

    Critical insight: Crowds ignore the minutes constraint. A player
    with 2.0 SoT/90 entering at the 65th minute has only ~25 minutes,
    giving effective λ ≈ 2.0 × 25/90 ≈ 0.56, not 2.0.

    Parameters
    ----------
    rate_per_90 : float
        Player's per-90-minute rate for the metric.
    threshold : int
        Minimum count to reach.
    p_comes_on : float
        Probability the player is substituted on during the match.
    expected_entry_minute : float
        Expected minute of substitution (default 65).
    match_minutes : float
        End of regulation (default 90).
    opposition_defensive_factor : float
        Defensive quality adjustment.

    Returns
    -------
    dict
    """
    minutes_available = match_minutes - expected_entry_minute

    p_threshold_given_on = p_reaches_threshold_poisson(
        rate_per_90, minutes_available, threshold, opposition_defensive_factor
    )

    p_overall = p_comes_on * p_threshold_given_on

    effective_lambda = rate_per_90 * (minutes_available / 90) * opposition_defensive_factor

    return {
        'player_type': 'substitute',
        'rate_per_90': round(rate_per_90, 4),
        'p_comes_on': round(p_comes_on, 4),
        'expected_entry_minute': expected_entry_minute,
        'minutes_available': round(minutes_available, 1),
        'opposition_defensive_factor': round(opposition_defensive_factor, 4),
        'effective_lambda': round(effective_lambda, 4),
        'p_threshold_given_on': round(p_threshold_given_on, 4),
        'p_reaches_threshold': round(p_overall, 4),
        'threshold': threshold,
        'crowd_error': (
            'Crowds anchor on "quality player = metric" without '
            'accounting for limited substitute minutes. '
            'Typical crowd overestimate: +8 to +15ppt for SoT markets.'
        )
    }


def head_to_head_sot_comparison(rate_a: float, rate_b: float,
                                  opposition_factor_a: float = 1.0,
                                  opposition_factor_b: float = 1.0,
                                  minutes_a: float = 90.0,
                                  minutes_b: float = 90.0) -> dict:
    """
    P(Player A records more SoT than Player B) via Poisson comparison.

    Used for markets like "Will Yamal record more SoT than Messi?"

    Method: Enumerate joint distribution P(A=i, B=j) for i,j in 0..N
    and sum where i > j.

    Parameters
    ----------
    rate_a : float
        Player A's SoT rate per 90 minutes.
    rate_b : float
        Player B's SoT rate per 90 minutes.
    opposition_factor_a : float
        Defensive quality factor for Player A's shots.
    opposition_factor_b : float
        Defensive quality factor for Player B's shots.
    minutes_a : float
        Minutes available to Player A.
    minutes_b : float
        Minutes available to Player B.

    Returns
    -------
    dict
    """
    lam_a = rate_a * (minutes_a / 90) * opposition_factor_a
    lam_b = rate_b * (minutes_b / 90) * opposition_factor_b

    # Enumerate joint distribution up to reasonable max
    max_k = 15
    p_a_greater = 0.0
    p_equal = 0.0
    p_b_greater = 0.0

    for i in range(max_k + 1):
        for j in range(max_k + 1):
            p_joint = poisson_pmf(i, lam_a) * poisson_pmf(j, lam_b)
            if i > j:
                p_a_greater += p_joint
            elif i == j:
                p_equal += p_joint
            else:
                p_b_greater += p_joint

    return {
        'lambda_a': round(lam_a, 4),
        'lambda_b': round(lam_b, 4),
        'p_a_greater': round(p_a_greater, 4),
        'p_equal': round(p_equal, 4),
        'p_b_greater': round(p_b_greater, 4),
        'note': (
            f'λ_A={lam_a:.2f} vs λ_B={lam_b:.2f}. '
            f'When λ_A < λ_B, P(A>B) < 0.50.'
        )
    }


def brace_probability_any_player(players: dict,
                                   opposition_defensive_factor: float = 1.0) -> dict:
    """
    P(at least one player scores 2+ goals in regulation).

    Parameters
    ----------
    players : dict
        {player_name: {'rate_per_90': float, 'minutes': float, 'type': 'starter'|'sub'}}
    opposition_defensive_factor : float
        Applied uniformly to all players.

    Returns
    -------
    dict
    """
    individual_probs = {}

    for name, info in players.items():
        p = p_reaches_threshold_poisson(
            info['rate_per_90'],
            info.get('minutes', 90),
            2,
            opposition_defensive_factor
        )
        individual_probs[name] = round(p, 4)

    # P(at least one player braces) = 1 - P(no player braces)
    p_none = 1.0
    for p in individual_probs.values():
        p_none *= (1.0 - p)

    p_at_least_one = 1.0 - p_none

    return {
        'individual_probabilities': individual_probs,
        'opposition_defensive_factor': round(opposition_defensive_factor, 4),
        'p_no_player_braces': round(p_none, 4),
        'p_at_least_one_braces': round(p_at_least_one, 4),
        'note': (
            'Cap at 0.32 for standard matches unless named player has '
            'elite tournament rate (≥1.5 goals/game) or opposition is '
            'demonstrably leaky (factor > 1.3).'
        )
    }
