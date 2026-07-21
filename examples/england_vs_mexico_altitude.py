"""
Worked Example 2: England vs Mexico (Round of 16, Estadio Azteca)
Market: "Will a goal be scored after the second hydration break?"

Result: YOU 31%, CROWD 49%, OUTCOME NO → RBP +33.04

This is the largest single-market win in the dataset. The altitude of
the Estadio Azteca (2,200m above sea level) inverts the normal late-game
logic: instead of fatigue opening up space and creating goals, high-altitude
fatigue suppresses attacking quality and reduces late-game scoring rates.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.poisson_goals import (
    MatchContext, goal_after_second_hydration_break,
    window_lambda, p_at_least_one_goal
)
from calibration.scoring import rbp_gap, expected_rbp_gap


def run():
    print("=" * 60)
    print("EXAMPLE 2: England vs Mexico — Altitude Late Window")
    print("=" * 60)

    # Step 1: Establish match parameters
    lambda_goals = 2.0  # Under 2.5 at -170; altitude environment
    azteca_altitude_m = 2240  # Estadio Azteca altitude

    print(f"\nStep 1: Match parameters")
    print(f"  λ_goals = {lambda_goals} (Under 2.5 at -170)")
    print(f"  Venue altitude = {azteca_altitude_m}m (Estadio Azteca)")
    print(f"  Altitude threshold = 1500m → suppression applies")

    # Step 2: Naive calculation (ignoring altitude)
    naive_lambda_window = window_lambda(lambda_goals, 75, 90)
    naive_p = p_at_least_one_goal(naive_lambda_window)
    print(f"\nStep 2: Naive Poisson (no altitude adjustment)")
    print(f"  λ_window (min 75-90) = {naive_lambda_window:.4f}")
    print(f"  P(≥1 goal) = {naive_p:.1%}")
    print(f"  → Crowd anchors here: ~49%")

    # Step 3: Apply altitude suppression
    ctx = MatchContext(
        lambda_total=lambda_goals,
        home_win_prob=0.24,   # Mexico (home) at +330
        away_win_prob=0.52,   # England at -125
        draw_prob=0.24,
        match_type='standard',
        altitude_m=azteca_altitude_m
    )

    result = goal_after_second_hydration_break(ctx)

    print(f"\nStep 3: Altitude-adjusted calculation")
    print(f"  Altitude factor applied: {result['altitude_adjusted']}")
    print(f"  Suppression factor: 0.80 (20% reduction in effective λ)")
    print(f"  Adjusted P(≥1 goal) = {result['point_estimate']:.1%}")

    # Step 4: Why altitude inverts the late-game logic
    print(f"\nStep 4: Why altitude inverts the normal late-window logic")
    print("  Normal logic: fatigue → defensive lapses → more late goals")
    print("  Altitude logic: fatigue → reduced attacking quality →")
    print("                  shorter passes, less pressing, lower goal rate")
    print("  At 2200m, second-half scoring rates drop ~15-20% vs sea level.")
    print("  England couldn't sustain their attacking patterns in thin air.")

    # Step 5: Evaluate forecast
    your_prob = 0.31
    crowd_prob = 0.49
    outcome = 0  # NO — no goal after second hydration break

    actual_rbp = rbp_gap(your_prob, crowd_prob, outcome, doubled=True)
    expected_value = expected_rbp_gap(your_prob, crowd_prob, result['point_estimate'], doubled=True)

    print(f"\nStep 5: Forecast evaluation")
    print(f"  Your submission: {your_prob:.0%}")
    print(f"  Crowd: {crowd_prob:.0%}")
    print(f"  Outcome: {'YES' if outcome else 'NO'}")
    print(f"  Actual RBP: +{actual_rbp:.2f}")
    print(f"  Expected RBP (pre-match): +{expected_value:.2f}")

    print("\nKey insight:")
    print("  The crowd anchored at ~49% — the standard late-window rate —")
    print("  without accounting for altitude. This produced an 18ppt gap.")
    print("  Always check venue altitude before pricing late-window markets.")
    print(f"  Venues above {1500}m: apply 15-20% downward λ adjustment.")


if __name__ == "__main__":
    run()
