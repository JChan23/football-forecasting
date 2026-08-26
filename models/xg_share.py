# xG-share decomposition models for underdog market pricing.

# Core insight: When a crowd is asked "will Team B (underdog) hold a lead at any point?", they anchor on Team B's win probability or recent narrative
# The correct model is: P(underdog leads at any point) ≈ P(underdog scores first) = xG_underdog / (xG_home + xG_away)
# The second term (underdog comes from behind to lead) is negligible and can be ignored for strong favourites.

def underdog_holds_lead(xg_underdog: float,
                         xg_favourite: float,
                         p_underdog_comeback_lead: float = 0.01) -> dict:

    # P(underdog holds a lead at any point in regulation, excl. shootout).
    # Decomposition: P(lead) = P(scores first) + P(equalises then goes ahead)
    # The second term is negligible (<2%) for typical favourite/underdog matchups and is treated as a fixed small constant.

    # xg_underdog: Expected goals for the underdog team (from model or betting market).
    # xg_favourite: Expected goals for the favourite team.
    # p_underdog_comeback_lead: Probability underdog scores twice before favourite equalises (default 0.01 — effectively negligible for strong favourites).

    if xg_underdog + xg_favourite <= 0:
        raise ValueError("Combined xG must be positive.")

    xg_share = xg_underdog / (xg_underdog + xg_favourite)
    p_scores_first = xg_share
    p_holds_lead = p_scores_first + p_underdog_comeback_lead

    return {
        'xg_underdog': round(xg_underdog, 4),
        'xg_favourite': round(xg_favourite, 4),
        'xg_share': round(xg_share, 4),
        'p_scores_first': round(p_scores_first, 4),
        'p_comeback_lead': round(p_underdog_comeback_lead, 4),
        'p_holds_lead': round(min(p_holds_lead, 1.0), 4),
        'crowd_typical_error': (
            'Crowds anchor on win probability or tournament narrative '
            '(e.g. "never trailed") rather than xG scoring-first probability. '
            'Typical crowd overestimate: +3 to +8ppt.'
        )
    }


def first_scorer_probability(xg_home: float, xg_away: float) -> dict:

    # Probability each team scores first, derived from xG shares.
    # Also returns P(0-0 in regulation) as the complement.

    # xg_home: Expected goals for home/first-named team.
    # xg_away: Expected goals for away/second-named team.

    import math

    total_xg = xg_home + xg_away
    if total_xg <= 0:
        raise ValueError("Combined xG must be positive.")

    # P(no goal in regulation) from Poisson
    p_no_goal = math.exp(-total_xg)

    # Given a goal is scored, probability it's by each team
    p_home_first_given_goal = xg_home / total_xg
    p_away_first_given_goal = xg_away / total_xg

    # Unconditional
    p_home_first = p_home_first_given_goal * (1 - p_no_goal)
    p_away_first = p_away_first_given_goal * (1 - p_no_goal)

    return {
        'xg_home': round(xg_home, 4),
        'xg_away': round(xg_away, 4),
        'p_home_scores_first': round(p_home_first, 4),
        'p_away_scores_first': round(p_away_first, 4),
        'p_no_goal_in_regulation': round(p_no_goal, 4),
    }
