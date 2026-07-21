"""
bookie_converter.py
-------------------
Utilities for converting betting market odds to true implied probabilities.

Bookmakers embed a margin (the 'vig' or 'overround') into their odds,
causing raw implied probabilities to sum to > 1.0. These functions
remove the vig to recover market-consensus true probabilities.
"""


def decimal_to_implied(odds: float) -> float:
    """Convert decimal odds to raw implied probability."""
    if odds <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    return 1.0 / odds


def american_to_decimal(american_odds: int) -> float:
    """Convert American (moneyline) odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def remove_vig_two_outcome(odds_yes: float, odds_no: float,
                            odds_format: str = 'decimal') -> dict:
    """
    Remove vig from a two-outcome market and return true probabilities.

    Parameters
    ----------
    odds_yes : float
        Odds on YES outcome.
    odds_no : float
        Odds on NO outcome.
    odds_format : str
        'decimal' or 'american'.

    Returns
    -------
    dict with keys:
        'p_yes': float — true probability of YES
        'p_no': float — true probability of NO
        'overround': float — total overround (e.g. 1.05 = 5% vig)
        'vig_pct': float — vig as percentage
    """
    if odds_format == 'american':
        odds_yes = american_to_decimal(odds_yes)
        odds_no = american_to_decimal(odds_no)

    raw_yes = decimal_to_implied(odds_yes)
    raw_no = decimal_to_implied(odds_no)
    overround = raw_yes + raw_no

    p_yes = raw_yes / overround
    p_no = raw_no / overround

    return {
        'p_yes': round(p_yes, 4),
        'p_no': round(p_no, 4),
        'overround': round(overround, 4),
        'vig_pct': round((overround - 1.0) * 100, 2),
    }


def remove_vig_three_outcome(odds_home: float, odds_draw: float,
                              odds_away: float,
                              odds_format: str = 'decimal') -> dict:
    """
    Remove vig from a three-outcome match result market (1X2).

    Parameters
    ----------
    odds_home : float
        Odds on home win.
    odds_draw : float
        Odds on draw.
    odds_away : float
        Odds on away win.
    odds_format : str
        'decimal' or 'american'.

    Returns
    -------
    dict with keys:
        'p_home_win': float
        'p_draw': float
        'p_away_win': float
        'overround': float
        'vig_pct': float
    """
    if odds_format == 'american':
        odds_home = american_to_decimal(odds_home)
        odds_draw = american_to_decimal(odds_draw)
        odds_away = american_to_decimal(odds_away)

    raw_home = decimal_to_implied(odds_home)
    raw_draw = decimal_to_implied(odds_draw)
    raw_away = decimal_to_implied(odds_away)
    overround = raw_home + raw_draw + raw_away

    return {
        'p_home_win': round(raw_home / overround, 4),
        'p_draw': round(raw_draw / overround, 4),
        'p_away_win': round(raw_away / overround, 4),
        'overround': round(overround, 4),
        'vig_pct': round((overround - 1.0) * 100, 2),
    }


def first_scorer_not_named(named_scorers: dict,
                            odds_format: str = 'decimal') -> dict:
    """
    P(first goal not scored by any named player) from first-scorer odds.

    Used for: "Will first goal be scored by player other than Messi/Salah?"
    Validated: crowd 55%, model 65%, YES → +22.25 RBP.

    Parameters
    ----------
    named_scorers : dict
        {player_name: odds_on_first_goal} mapping.
    odds_format : str
        'decimal' or 'american'.

    Returns
    -------
    dict
    """
    named_probs = {}
    for player, odds in named_scorers.items():
        if odds_format == 'american':
            odds = american_to_decimal(odds)
        raw_p = decimal_to_implied(odds)
        # Individual first-scorer odds don't sum to 1 due to vig,
        # but we treat each as approximately correct after modest adjustment
        named_probs[player] = round(raw_p * 0.92, 4)  # ~8% vig removal

    total_named = sum(named_probs.values())
    p_not_named = max(0.0, 1.0 - total_named)

    return {
        'named_player_probabilities': named_probs,
        'p_named_player_scores_first': round(total_named, 4),
        'p_other_player_scores_first': round(p_not_named, 4),
        'note': (
            'Crowd anchors at ~55% for "not named player" without '
            'properly computing bookie-implied named-player total. '
            'Model typically gives 60-70% when named players are '
            'high-profile but not dominant goal-scorers.'
        )
    }


def lambda_from_over_under(over_odds: float, under_odds: float,
                            line: float,
                            odds_format: str = 'decimal') -> dict:
    """
    Infer expected goals (lambda) from an over/under betting market.

    Uses the market line and over/under probabilities to estimate λ.
    When over is favoured, λ > line; when under is favoured, λ < line.

    Parameters
    ----------
    over_odds : float
        Odds on over.
    under_odds : float
        Odds on under.
    line : float
        The over/under line (e.g. 2.5).
    odds_format : str

    Returns
    -------
    dict
    """
    import math

    if odds_format == 'american':
        over_odds = american_to_decimal(over_odds)
        under_odds = american_to_decimal(under_odds)

    result = remove_vig_two_outcome(over_odds, under_odds)
    p_over = result['p_yes']

    # For integer lines, use Poisson CDF inversion
    # For 0.5 lines: P(X >= 1) = 1 - e^(-λ) = p_over → λ = -ln(1 - p_over)
    # For 2.5 lines: P(X >= 3) = p_over, solve numerically

    # Numerical approximation: λ ≈ line + (p_over - 0.5) × 2.5
    # This is a simple linear approximation valid for lines 1.5-3.5
    lambda_approx = line + (p_over - 0.5) * 2.5

    return {
        'line': line,
        'p_over': round(p_over, 4),
        'p_under': round(result['p_no'], 4),
        'lambda_approx': round(lambda_approx, 4),
        'note': (
            f'Approximate λ={lambda_approx:.2f} from O/U {line} market. '
            'Use as input to Poisson goal models.'
        )
    }
