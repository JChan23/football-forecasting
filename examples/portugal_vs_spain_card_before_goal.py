"""
Worked Example 1: Portugal vs Spain (Quarter-Final)
Market: "Will a card be shown before the first goal?"

Result: YOU 67%, CROWD 50%, OUTCOME YES → RBP +32.29

This example demonstrates the competing exponential processes model.
The crowd treats this as a coin flip (~50%); the rate-based model
gives a principled ~64-67% estimate.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.competing_processes import card_before_first_goal
from calibration.bookie_converter import remove_vig_three_outcome
from calibration.scoring import rbp_gap, expected_rbp_gap


def run():
    print("=" * 60)
    print("EXAMPLE 1: Portugal vs Spain — Card Before First Goal")
    print("=" * 60)

    # Step 1: Establish match lambda from market odds
    # Spain +130, Draw +195, Portugal (away) implied ~22% win
    match_probs = remove_vig_three_outcome(
        odds_home=2.30,   # Portugal
        odds_draw=2.95,
        odds_away=3.25,   # Spain
        odds_format='decimal'
    )
    print("\nStep 1: Market-implied match probabilities (vig-removed)")
    for k, v in match_probs.items():
        print(f"  {k}: {v:.1%}")

    # Step 2: Estimate match parameters
    lambda_goals = 1.8  # Under 2.5 favoured, 7 of last 10 H2H under 2.5
    lambda_cards = 3.2  # Tournament average; rivalry context → slight uplift

    print(f"\nStep 2: Match parameters")
    print(f"  λ_goals = {lambda_goals} (Under 2.5 favoured, H2H history)")
    print(f"  λ_cards = {lambda_cards} (tournament avg; rivalry context)")

    # Step 3: Apply competing processes model
    result = card_before_first_goal(
        lambda_goals=lambda_goals,
        lambda_cards=lambda_cards,
        card_context='rivalry'  # Iberian derby
    )
    print("\nStep 3: Competing exponential processes model")
    print(f"  Formula: λ_cards / (λ_cards + λ_goals)")
    print(f"  = {result['lambda_cards_adjusted']:.2f} / ({result['lambda_cards_adjusted']:.2f} + {result['lambda_goals']:.2f})")
    print(f"  P(card before first goal) = {result['p_card_before_goal']:.1%}")
    print(f"  Crowd anchor: {result['crowd_anchor']}")
    print(f"  Estimated edge vs crowd: +{result['estimated_edge_vs_crowd_ppt']:.1f}ppt")

    # Step 4: Evaluate the forecast
    your_prob = 0.67
    crowd_prob = 0.50
    outcome = 1  # YES — card was shown before first goal

    actual_rbp = rbp_gap(your_prob, crowd_prob, outcome, doubled=True)
    expected_value = expected_rbp_gap(your_prob, crowd_prob, result['p_card_before_goal'], doubled=True)

    print(f"\nStep 4: Forecast evaluation")
    print(f"  Your submission: {your_prob:.0%}")
    print(f"  Crowd: {crowd_prob:.0%}")
    print(f"  Outcome: {'YES' if outcome else 'NO'}")
    print(f"  Actual RBP: +{actual_rbp:.2f}")
    print(f"  Expected RBP (pre-match): +{expected_value:.2f}")

    print("\nKey insight:")
    print("  Crowds treat 'card before goal' as a coin flip, anchoring at ~50%.")
    print("  The rate model shows cards arrive ~3.2x per 90 min vs goals ~1.8x.")
    print(f"  λ_cards / (λ_cards + λ_goals) = {result['p_card_before_goal']:.0%} ≠ 50%.")
    print("  This edge of ~15-17ppt produces +30 RBP per instance consistently.")


if __name__ == "__main__":
    run()
