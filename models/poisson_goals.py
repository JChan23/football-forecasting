"""
poisson_goals.py
----------------
Poisson-based goal probability model for time-window markets.

Core insight: Goals in football approximate a Poisson process with rate λ
(expected goals per 90 minutes). Given λ, we can compute the probability of
at least one goal occurring in any specific time window.

Validated markets:
- Goal before first hydration break (~30 min)
- Goal after second hydration break (~75-90 min)
- Goal in first-half stoppage time (~45-48 min)
- Goal in second-half stoppage time (~90-95 min)
- Goal in first half after hydration break (~30-45 min)
- Goal between hydration breaks (~30-75 min)
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchContext:
    """
    Container for match-level inputs used across multiple models.

    Parameters
    ----------
    lambda_total : float
        Expected total goals for the match (from betting market over/under).
    home_win_prob : float
        Market-implied probability of home team winning in 90 minutes.
    away_win_prob : float
        Market-implied probability of away team winning in 90 minutes.
    draw_prob : float
        Market-implied probability of draw after 90 minutes.
    match_type : str
        One of: 'high_tempo', 'standard', 'cautious', 'low_block', 'third_place'
        Controls the early-window suppression correction magnitude.
    altitude_m : float
        Venue altitude in metres. Triggers late-window suppression above 1500m.
    """
    lambda_total: float
    home_win_prob: float
    away_win_prob: float
    draw_prob: float
    match_type: str = 'standard'
    altitude_m: float = 0.0

    def __post_init__(self):
        total = self.home_win_prob + self.away_win_prob + self.draw_prob
        if not (0.98 <= total <= 1.02):
            raise ValueError(
                f"Win/draw probabilities must sum to ~1.0, got {total:.3f}. "
                "Ensure vig has been removed before passing to MatchContext."
            )
        if self.match_type not in CORRECTION_BRACKETS:
            raise ValueError(
                f"match_type must be one of {list(CORRECTION_BRACKETS.keys())}"
            )


# Empirically validated correction brackets from tournament data.
# Values represent percentage points subtracted from naive Poisson estimate
# for the pre-hydration-break window. Positive = downward correction.
CORRECTION_BRACKETS = {
    'high_tempo':    (6, 8),    # Two elite attacking sides, minimal caution
    'standard':      (8, 10),   # Balanced knockout match
    'cautious':      (10, 12),  # Similar-quality teams, tactical match
    'low_block':     (8, 12),   # Elite attack vs defensive underdog
    'third_place':   (4, 6),    # Minimal tactical caution, nothing to lose
}

# Altitude suppression factor for late-window goal markets (post-75 min).
# Validated: England vs Mexico at Azteca (2200m) produced +33 RBP.
ALTITUDE_SUPPRESSION_THRESHOLD_M = 1500
ALTITUDE_LATE_WINDOW_FACTOR = 0.80  # 20% reduction in effective λ


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam)."""
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def p_at_least_one_goal(lam_window: float) -> float:
    """
    P(≥1 goal in a window) given expected goals in that window.

    Parameters
    ----------
    lam_window : float
        Expected goals specifically within the time window of interest.
        Compute as: lambda_total * (window_minutes / 90)

    Returns
    -------
    float
        Probability of at least one goal in the window.
    """
    return 1.0 - math.exp(-lam_window)


def window_lambda(lambda_total: float, start_min: float, end_min: float,
                  match_minutes: float = 90.0) -> float:
    """
    Expected goals in a specific time window, assuming uniform goal distribution.

    Parameters
    ----------
    lambda_total : float
        Expected total goals across the full match.
    start_min : float
        Start of window (inclusive), in minutes.
    end_min : float
        End of window (exclusive), in minutes.
    match_minutes : float
        Total match duration for normalisation (default 90).

    Returns
    -------
    float
        Expected goals in the specified window.
    """
    window_size = end_min - start_min
    return lambda_total * (window_size / match_minutes)


def goal_before_hydration_break(ctx: MatchContext,
                                 break_minute: float = 30.0) -> dict:
    """
    P(≥1 goal before the first hydration break).

    Applies empirically validated early-window suppression corrections
    based on match type. Crowds systematically overprice this market
    by anchoring on naive Poisson without accounting for tactical caution
    in the opening phase of knockout matches.

    Parameters
    ----------
    ctx : MatchContext
    break_minute : float
        Minute at which first hydration break occurs (default 30).

    Returns
    -------
    dict with keys:
        'naive_poisson': float — raw Poisson estimate before correction
        'correction_applied': tuple — (low, high) correction in pct points
        'adjusted_low': float — lower bound of calibrated range
        'adjusted_high': float — upper bound of calibrated range
        'point_estimate': float — midpoint of calibrated range
    """
    lam_window = window_lambda(ctx.lambda_total, 0, break_minute)
    naive = p_at_least_one_goal(lam_window)

    correction_low, correction_high = CORRECTION_BRACKETS[ctx.match_type]

    adjusted_low = max(0.30, naive - correction_high / 100)
    adjusted_high = max(0.35, naive - correction_low / 100)
    point_estimate = (adjusted_low + adjusted_high) / 2

    return {
        'naive_poisson': round(naive, 4),
        'correction_applied': (correction_low, correction_high),
        'adjusted_low': round(adjusted_low, 4),
        'adjusted_high': round(adjusted_high, 4),
        'point_estimate': round(point_estimate, 4),
    }


def goal_after_second_hydration_break(ctx: MatchContext,
                                       break_minute: float = 75.0,
                                       match_minutes: float = 90.0) -> dict:
    """
    P(≥1 goal after the second hydration break in regulation).

    Late-window goals are suppressed at altitude. Crowd typically prices
    this 5-10 points too high. Only match/exceed crowd when specific named
    late-goal threat evidence exists (e.g. Merino 88'+90' pattern).

    Parameters
    ----------
    ctx : MatchContext
    break_minute : float
        Minute at which second hydration break occurs (default 75).
    match_minutes : float
        End of regulation (default 90).

    Returns
    -------
    dict
    """
    lam_window = window_lambda(ctx.lambda_total, break_minute, match_minutes)

    # Altitude suppression for venues above threshold
    if ctx.altitude_m >= ALTITUDE_SUPPRESSION_THRESHOLD_M:
        lam_window *= ALTITUDE_LATE_WINDOW_FACTOR
        altitude_adjusted = True
    else:
        altitude_adjusted = False

    naive = p_at_least_one_goal(lam_window)

    # Default: submit 5-10pts below crowd. Without altitude or named
    # late-goal evidence, apply a modest downward correction.
    crowd_adjustment = -0.07  # -7 ppt as midpoint of 5-10pt range
    point_estimate = max(0.10, naive + crowd_adjustment)

    return {
        'naive_poisson': round(naive, 4),
        'altitude_adjusted': altitude_adjusted,
        'altitude_m': ctx.altitude_m,
        'point_estimate': round(point_estimate, 4),
        'note': (
            'Altitude suppression applied (λ reduced by 20%).'
            if altitude_adjusted else
            'No altitude adjustment. Default -7ppt correction applied.'
        )
    }


def goal_in_stoppage_time(ctx: MatchContext,
                           half: str = 'first',
                           stoppage_minutes: float = None,
                           team_has_st_goals_this_tournament: int = 0) -> dict:
    """
    P(≥1 goal during stoppage time in a given half).

    Crowds systematically overprice this market at ~18-22% (first half)
    and ~22-26% (second half). Base rate math consistently wins.

    Parameters
    ----------
    ctx : MatchContext
    half : str
        'first' or 'second'
    stoppage_minutes : float
        Length of stoppage time (default: 3 for first half, 5 for second).
    team_has_st_goals_this_tournament : int
        Number of matches in which a team has scored in stoppage time.
        If ≥2, applies 2x base rate multiplier (validated: Argentina pattern).

    Returns
    -------
    dict
    """
    if stoppage_minutes is None:
        stoppage_minutes = 3.0 if half == 'first' else 5.0

    match_minutes = 95.0  # effective denominator including stoppage
    lam_window = window_lambda(ctx.lambda_total, 0, stoppage_minutes, match_minutes)

    # Tournament-proven late-goal pattern adjustment
    if team_has_st_goals_this_tournament >= 2:
        lam_window *= 2.0
        pattern_note = f"2x multiplier applied: team scored in ST in {team_has_st_goals_this_tournament} matches."
    elif team_has_st_goals_this_tournament == 1:
        lam_window *= 1.4
        pattern_note = "1.4x multiplier applied: team scored in ST in 1 match."
    else:
        pattern_note = "No tournament ST pattern. Base rate only."

    p = p_at_least_one_goal(lam_window)

    return {
        'half': half,
        'stoppage_minutes': stoppage_minutes,
        'base_lambda_window': round(window_lambda(ctx.lambda_total, 0,
                                                   stoppage_minutes, match_minutes), 4),
        'adjusted_lambda_window': round(lam_window, 4),
        'point_estimate': round(p, 4),
        'pattern_note': pattern_note,
        'crowd_anchor': '18-22% (first half), 22-26% (second half)',
    }


def goal_between_hydration_breaks(ctx: MatchContext,
                                   first_break: float = 30.0,
                                   second_break: float = 75.0) -> dict:
    """
    P(≥1 goal in the window between the two hydration breaks).

    This is a ~45-minute window. For λ ≥ 2.5, this resolves YES ~73% of
    the time. Crowd consistently anchors 58-65%, underpricing this market.

    Parameters
    ----------
    ctx : MatchContext
    first_break : float
        End of first hydration break window (default 30).
    second_break : float
        Start of second hydration break window (default 75).

    Returns
    -------
    dict
    """
    lam_window = window_lambda(ctx.lambda_total, first_break, second_break)
    p = p_at_least_one_goal(lam_window)

    return {
        'window_minutes': second_break - first_break,
        'lambda_window': round(lam_window, 4),
        'point_estimate': round(p, 4),
        'crowd_anchor': '58-65% for λ≥2.5 matches',
        'note': 'Crowd underprices this market. Submit estimate directly.'
    }


def goal_in_first_half_after_break(ctx: MatchContext,
                                    break_minute: float = 30.0,
                                    halftime_minute: float = 47.0) -> dict:
    """
    P(≥1 goal in the window between first hydration break and halftime).

    Approximately 15-17 minutes. Not subject to early-window suppression
    (teams have settled). Crowd anchors ~40-46%, correct range is 32-37%.

    Parameters
    ----------
    ctx : MatchContext
    break_minute : float
        First hydration break minute (default 30).
    halftime_minute : float
        Effective end of first half including stoppage (default 47).

    Returns
    -------
    dict
    """
    lam_window = window_lambda(ctx.lambda_total, break_minute, halftime_minute)
    p = p_at_least_one_goal(lam_window)

    return {
        'window_minutes': halftime_minute - break_minute,
        'lambda_window': round(lam_window, 4),
        'point_estimate': round(p, 4),
        'crowd_anchor': '40-46%',
        'note': (
            'No early-window suppression needed. '
            'Crowd overprices by treating as "goal in first half" market.'
        )
    }
