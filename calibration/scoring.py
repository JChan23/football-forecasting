"""
scoring.py
----------
Brier score computation and calibration analysis.

The Brier score is the scoring rule used in the Jump Trading Probability Cup.
It rewards well-calibrated probabilities and penalises overconfidence.

Brier Score: BS = (p - o)²
where p is the predicted probability and o is the outcome (1 or 0).
Lower is better. Range: [0, 1].

Key property: The Brier score is a STRICTLY PROPER scoring rule.
This means it is uniquely minimised in expectation when you report
your true belief. There is no strategic incentive to misreport.

RBP (Relative Brier Points): The platform's scoring metric.
RBP = (crowd_BS - your_BS) × scale_factor
Positive = you outperformed the crowd on that question.
"""

from typing import List, Tuple, Optional
import math


def brier_score(predicted_prob: float, outcome: int) -> float:
    """
    Compute the Brier score for a single forecast.

    Parameters
    ----------
    predicted_prob : float
        Predicted probability of the event occurring (0 to 1).
    outcome : int
        Actual outcome: 1 if event occurred, 0 if not.

    Returns
    -------
    float
        Brier score (lower is better, range [0, 1]).
    """
    if not (0.0 <= predicted_prob <= 1.0):
        raise ValueError("predicted_prob must be between 0 and 1.")
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1.")

    return round((predicted_prob - outcome) ** 2, 6)


def rbp_gap(your_prob: float, crowd_prob: float, outcome: int,
             doubled: bool = False) -> float:
    """
    Compute your RBP gap vs crowd for a single question.

    RBP gap = crowd_BS - your_BS
    Positive means you outperformed the crowd.

    Parameters
    ----------
    your_prob : float
        Your predicted probability.
    crowd_prob : float
        Crowd consensus probability.
    outcome : int
        Actual outcome (0 or 1).
    doubled : bool
        Whether this question has doubled points (knockout stage).

    Returns
    -------
    float
        RBP gap. Positive = beat crowd.
    """
    your_bs = brier_score(your_prob, outcome)
    crowd_bs = brier_score(crowd_prob, outcome)
    gap = crowd_bs - your_bs

    if doubled:
        gap *= 2

    return round(gap * 100, 2)  # Scale to match platform display


def expected_rbp_gap(your_prob: float, crowd_prob: float,
                      true_prob: float, doubled: bool = False) -> float:
    """
    Expected RBP gap given true probability (before outcome is known).

    Useful for pre-match evaluation of whether a deviation from crowd
    is worth taking.

    E[RBP gap] = E[crowd_BS] - E[your_BS]
               = [(p_c - p*)² × p* + p_c² × (1-p*)] - [(p_y - p*)² × p* + p_y² × (1-p*)]

    where p* = true probability, p_c = crowd, p_y = your estimate.

    A positive expected RBP gap means deviating from the crowd is
    mathematically justified given your true probability estimate.

    Parameters
    ----------
    your_prob : float
        Your predicted probability.
    crowd_prob : float
        Crowd consensus probability.
    true_prob : float
        Your estimate of the true probability.
    doubled : bool
        Whether this question has doubled points.

    Returns
    -------
    float
        Expected RBP gap. Positive = deviation from crowd is EV-positive.
    """
    def expected_bs(p_pred: float, p_true: float) -> float:
        return (p_pred - p_true) ** 2 * p_true + p_pred ** 2 * (1 - p_true)

    e_crowd_bs = expected_bs(crowd_prob, true_prob)
    e_your_bs = expected_bs(your_prob, true_prob)
    gap = (e_crowd_bs - e_your_bs) * 100

    if doubled:
        gap *= 2

    return round(gap, 3)


def optimal_submission(true_prob: float) -> float:
    """
    The optimal probability to submit under a proper scoring rule.

    For a strictly proper scoring rule like the Brier score, the optimal
    submission is simply your true belief. This function exists as a
    reminder and sanity check.

    Parameters
    ----------
    true_prob : float
        Your honest belief about the probability of the event.

    Returns
    -------
    float
        The optimal submission = true_prob.
    """
    return true_prob


def calibration_by_bucket(forecasts: List[Tuple[float, int]],
                           n_buckets: int = 10) -> dict:
    """
    Compute calibration statistics across probability buckets.

    A perfectly calibrated forecaster sees actual outcome rates match
    their stated probabilities. E.g. of all questions you answered
    with 70%, roughly 70% should have resolved YES.

    Parameters
    ----------
    forecasts : list of (predicted_prob, outcome) tuples
    n_buckets : int
        Number of equal-width probability buckets.

    Returns
    -------
    dict with 'buckets' list, each containing:
        'range': (low, high)
        'count': int
        'mean_predicted': float
        'actual_rate': float
        'calibration_error': float (|mean_predicted - actual_rate|)
    """
    bucket_size = 1.0 / n_buckets
    buckets = []

    for i in range(n_buckets):
        low = i * bucket_size
        high = (i + 1) * bucket_size

        bucket_forecasts = [
            (p, o) for p, o in forecasts
            if low <= p < high or (i == n_buckets - 1 and p == 1.0)
        ]

        if not bucket_forecasts:
            continue

        mean_pred = sum(p for p, _ in bucket_forecasts) / len(bucket_forecasts)
        actual_rate = sum(o for _, o in bucket_forecasts) / len(bucket_forecasts)

        buckets.append({
            'range': (round(low, 2), round(high, 2)),
            'count': len(bucket_forecasts),
            'mean_predicted': round(mean_pred, 4),
            'actual_rate': round(actual_rate, 4),
            'calibration_error': round(abs(mean_pred - actual_rate), 4),
        })

    mean_calibration_error = (
        sum(b['calibration_error'] for b in buckets) / len(buckets)
        if buckets else 0
    )

    return {
        'buckets': buckets,
        'n_forecasts': len(forecasts),
        'mean_calibration_error': round(mean_calibration_error, 4),
        'note': (
            'Jump Trading Probability Cup 2026 result: '
            '"You\'re tracking the line — your confidence consistently '
            'matches reality. That\'s elite calibration."'
        )
    }


def why_not_100_percent(true_p: float = 0.90) -> dict:
    """
    Illustrate why you should never submit 100% (or 0%) even when very confident.

    Demonstrates the asymmetric Brier penalty for extreme probabilities.

    Parameters
    ----------
    true_p : float
        Your true belief probability.

    Returns
    -------
    dict comparing Brier scores at 100% vs true_p.
    """
    bs_at_100_if_yes = brier_score(1.0, 1)
    bs_at_100_if_no = brier_score(1.0, 0)

    bs_at_true_if_yes = brier_score(true_p, 1)
    bs_at_true_if_no = brier_score(true_p, 0)

    return {
        'your_true_belief': true_p,
        'if_submit_100pct': {
            'bs_if_yes': bs_at_100_if_yes,
            'bs_if_no': bs_at_100_if_no,
            'penalty_if_wrong': bs_at_100_if_no,
        },
        'if_submit_true_belief': {
            'bs_if_yes': bs_at_true_if_yes,
            'bs_if_no': bs_at_true_if_no,
            'penalty_if_wrong': bs_at_true_if_no,
        },
        'cost_of_overclaiming_when_wrong': round(bs_at_100_if_no - bs_at_true_if_no, 6),
        'gain_from_overclaiming_when_right': round(bs_at_true_if_yes - bs_at_100_if_yes, 6),
        'verdict': (
            f'Submitting 100% instead of {true_p:.0%} gains only '
            f'{bs_at_true_if_yes - bs_at_100_if_yes:.4f} Brier points when right, '
            f'but costs {bs_at_100_if_no - bs_at_true_if_no:.4f} when wrong. '
            f'The downside is {(bs_at_100_if_no - bs_at_true_if_no)/(bs_at_true_if_yes - bs_at_100_if_yes):.0f}x larger.'
        )
    }
