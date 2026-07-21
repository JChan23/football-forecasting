"""
Worked Example 3: Morocco vs Canada (Round of 16)
Market: "Will the match be decided by a penalty shootout?"

Result: YOU 12%, CROWD 22%, OUTCOME NO → RBP +11.33

This example demonstrates the two-stage penalty shootout decomposition.
The crowd anchors at ~22% using a lazy heuristic ("knockout matches often
go to penalties"). The correct model decomposes this into:

    P(shootout) = P(draw after 90 min) × P(draw after extra time | drew at 90)

This was validated in 4 consecutive tournament instances, all resolving NO.
Average RBP per instance: +12.6.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.competing_processes import penalty_shootout_probability, match_goes_to_et
from calibration.bookie_converter import (
    remove_vig_three_outcome, remove_vig_two_outcome
)
from calibration.scoring import rbp_gap, expected_rbp_gap


def run():
    print("=" * 60)
    print("EXAMPLE 3: Morocco vs Canada — Penalty Shootout")
    print("=" * 60)

    # Step 1: Convert match odds to vig-free probabilities
    # Morocco -135, Draw +230, Canada +460
    match_probs = remove_vig_three_outcome(
        odds_home=1.741,   # Morocco (-135 American → 1.741 decimal)
        odds_draw=3.30,    # +230 American → 3.30 decimal
        odds_away=5.60,    # +460 American → 5.60 decimal
        odds_format='decimal'
    )

    draw_prob = match_probs['p_draw']
    print(f"\nStep 1: Vig-removed match probabilities")
    print(f"  Morocco win: {match_probs['p_home_win']:.1%}")
    print(f"  Draw (→ ET): {draw_prob:.1%}")
    print(f"  Canada win: {match_probs['p_away_win']:.1%}")
    print(f"  Overround: {match_probs['overround']:.3f} ({match_probs['vig_pct']:.1f}% vig)")

    # Step 2: Check direct shootout odds (bookie cross-validation)
    # Bookie offered: Yes 6.5x, No 1.11x
    direct_shootout = remove_vig_two_outcome(6.5, 1.11)
    print(f"\nStep 2: Direct bookie shootout odds (6.5x / 1.11x)")
    print(f"  Raw implied P(shootout) = {1/6.5:.1%}")
    print(f"  Vig-removed P(shootout) = {direct_shootout['p_yes']:.1%}")

    # Step 3: Apply decomposition model
    result = penalty_shootout_probability(
        draw_prob_90=draw_prob,
        p_draw_after_et=0.42,       # Historical World Cup ET average
        high_scoring_match=False    # Under 2.5 favoured
    )

    print(f"\nStep 3: Two-stage decomposition")
    print(f"  P(draw after 90 min) = {result['draw_prob_90']:.1%}")
    print(f"  P(draw after ET | drew at 90) = {result['p_draw_after_et']:.1%}")
    print(f"  Formula: {result['formula']}")
    print(f"  P(shootout) = {result['draw_prob_90']:.1%} × {result['p_draw_after_et']:.1%} = {result['p_shootout']:.1%}")

    # Step 4: Reconcile with bookie direct odds
    model_p = result['p_shootout']
    bookie_p = direct_shootout['p_yes']
    midpoint = (model_p + bookie_p) / 2

    print(f"\nStep 4: Reconciling model vs bookie direct odds")
    print(f"  Model (decomposition): {model_p:.1%}")
    print(f"  Bookie direct: {bookie_p:.1%}")
    print(f"  Midpoint (submitted): {midpoint:.1%} → rounded to 12%")

    # Step 5: Why the crowd gets this wrong
    print(f"\nStep 5: Why the crowd anchors at ~22%")
    print(f"  Heuristic: 'About 1 in 5 knockout matches go to penalties'")
    print(f"  This ignores the conditional structure.")
    print(f"  Correct: Only matches that draw at 90 can have a shootout.")
    print(f"  Morocco had only {draw_prob:.0%} draw probability → limits upside.")
    print(f"  Crowd adds ~10ppt of noise above the model estimate.")

    # Step 6: Evaluate forecast
    your_prob = 0.12
    crowd_prob = 0.22
    outcome = 0  # NO — Morocco won in regulation

    actual_rbp = rbp_gap(your_prob, crowd_prob, outcome, doubled=True)

    print(f"\nStep 6: Forecast evaluation")
    print(f"  Your submission: {your_prob:.0%}")
    print(f"  Crowd: {crowd_prob:.0%}")
    print(f"  Outcome: {'YES' if outcome else 'NO'}")
    print(f"  Actual RBP: +{actual_rbp:.2f}")

    # Tournament record
    print(f"\nTournament record on shootout/ET decomposition:")
    results = [
        ("Morocco vs Canada",   0.12, 0.22, 0, True),
        ("Brazil vs Norway",    0.12, 0.22, 0, True),
        ("England vs Mexico",   0.13, 0.24, 0, True),
        ("Portugal vs Spain ET",0.26, 0.32, 0, True),
    ]
    total_rbp = 0
    for match, yp, cp, oc, dbl in results:
        rbp = rbp_gap(yp, cp, oc, dbl)
        total_rbp += rbp
        print(f"  {match}: {yp:.0%} vs crowd {cp:.0%} → {'+' if rbp > 0 else ''}{rbp:.2f} RBP")
    print(f"  Total: +{total_rbp:.2f} RBP | Average: +{total_rbp/len(results):.2f} per instance")


if __name__ == "__main__":
    run()
