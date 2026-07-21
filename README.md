# Football Match Forecasting Framework

A probabilistic forecasting framework for football match markets, built and validated across the 2026 FIFA World Cup.

**Competition result:** Jump Trading Probability Cup 2026
- **Ranked 55th out of 4,013 active forecasters** overall (top 1.4%)
- **Ranked 11th out of 1,272 active forecasters** in the knockout stage (top 0.87%)
- **+3.9 RBP gap vs crowd** across 996 settled forecasts (better than 88% of all participants)

---

## Overview

This framework implements a set of mathematically grounded models for pricing football match prop markets. Each model was iteratively validated against live results, with the framework updated through Bayesian reasoning as new data accumulated across the tournament.

The core principle is that **crowds systematically misprice certain market structures** by applying intuitive heuristics rather than rate-based probability calculations. The framework identifies and exploits these mispricing patterns.

---

## Mathematical Foundation

### 1. Poisson Goal Model

Goals in football approximate a Poisson process with rate λ (expected goals per 90 minutes, derived from the betting market over/under line).

For any time window [t₁, t₂]:

$$\lambda_{window} = \lambda_{total} \times \frac{t_2 - t_1}{90}$$

$$P(\geq 1 \text{ goal in window}) = 1 - e^{-\lambda_{window}}$$

**Key insight:** Crowds anchor on naive Poisson without accounting for tactical suppression in early windows (first 30 minutes) or altitude suppression in late windows.

### 2. Competing Exponential Processes

When two independent Poisson processes (e.g. cards and goals) race to fire first:

$$P(\text{cards before goals}) = \frac{\lambda_{cards}}{\lambda_{cards} + \lambda_{goals}}$$

This is exact for exponential inter-event times, which is the natural assumption for Poisson-distributed events.

**Crowd error:** Treat this as a ~50% coin flip. Model consistently gives 61–67% for card-before-goal markets given tournament-average rates (λ_cards ≈ 3.2, λ_goals ≈ 1.8–2.4).

**Validated:** 2 instances, both YES, average RBP +31.25 per instance.

### 3. Penalty Shootout Decomposition

$$P(\text{shootout}) = P(\text{draw at 90}) \times P(\text{draw after ET} \mid \text{drew at 90})$$

Where P(draw after ET | drew at 90) ≈ 0.42 historically (World Cup knockouts).

**Crowd error:** Anchor at ~22% using "knockout matches often go to penalties" heuristic, ignoring the conditional probability structure.

**Validated:** 4 instances, all NO, average RBP +12.6 per instance.

### 4. xG-Share Underdog Lead Decomposition

$$P(\text{underdog holds lead}) \approx P(\text{underdog scores first}) = \frac{xG_{underdog}}{xG_{home} + xG_{away}}$$

The second term (comeback lead) is negligible (<2%) for typical mismatches.

**Crowd error:** Anchor on win probability or narrative ("Switzerland never trailed this tournament") rather than xG scoring-first probability.

### 5. Player Threshold (Poisson CDF)

For a **starter** with per-90 rate r playing M minutes:

$$\lambda_{player} = r \times \frac{M}{90} \times d_{opp}$$

$$P(X \geq k) = 1 - \sum_{i=0}^{k-1} \frac{e^{-\lambda} \lambda^i}{i!}$$

For a **substitute** entering at minute m:

$$P(X \geq k) = P(\text{comes on}) \times P(X \geq k \mid M = 90 - m \text{ minutes})$$

**Crowd error:** Price "quality player = metric" without accounting for the minutes constraint on substitutes.

---

## Validated Market Edges

| Market | Model | Direction vs Crowd | Instances | Win Rate |
|--------|--------|-------------------|-----------|----------|
| Card before first goal | λ_c/(λ_c+λ_g) | +15–17ppt above crowd | 2 | 100% |
| Penalty shootout / ET | P(draw_90)×P(draw_ET) | 7–10ppt below crowd | 4 | 100% |
| First-half stoppage time goal | 10–13% base rate | 8–10ppt below crowd | 4+ | >85% |
| First goal in second half | P(0 FH goals)×P(SH goal) | 10–12ppt below crowd | 3 | 100% |
| Both halves same goals | Poisson enumeration | 6–8ppt below crowd | 3 | 100% |
| Card in each half | P(≥1 FH)×P(≥1 SH) | 8–12ppt below crowd | 1 | 100% |
| Named sub SoT | Minutes-constraint chain | Well below crowd | 2 | 100% |
| First goal by attribute | Scorer distribution | Model-driven | 2 | 100% |
| Goal between hydration breaks | 45-min window math | Above crowd | 1 | 100% |
| Goal after 2nd hydration break (altitude) | λ × 0.80 | 15–20ppt below crowd | 1 | 100% |

---

## Early-Window Suppression Corrections

The pre-hydration-break market (first 30 minutes) is systematically overpriced by crowds who apply naive Poisson without accounting for tactical caution in knockout match openings.

| Match Type | Correction (ppt) | Floor |
|-----------|-----------------|-------|
| High-tempo, both elite attacking | −6 to −8 | 40% |
| Standard balanced knockout | −8 to −10 | 38% |
| Cautious, similar-quality teams | −10 to −12 | 36% |
| Elite attack vs low-block underdog | −8 to −12 | 35% |
| Third-place playoff | −4 to −6 | 40% |

**Altitude venues (>1,500m):** Apply additional 15–20% reduction to λ for late-window (post-75 min) markets.

---

## Project Structure

```
football-forecasting/
├── models/
│   ├── poisson_goals.py        # Time-window goal probability
│   ├── competing_processes.py  # Race markets (card/goal, shootout)
│   ├── xg_share.py             # xG-based underdog decomposition
│   ├── player_threshold.py     # Named player SoT/goals threshold
│   └── derived_markets.py      # Compound markets
├── calibration/
│   ├── scoring.py              # Brier score, RBP gap, calibration
│   └── bookie_converter.py     # Odds → vig-free probabilities
├── examples/
│   ├── portugal_vs_spain_card_before_goal.py   # +32.29 RBP
│   ├── england_vs_mexico_altitude.py           # +33.04 RBP
│   └── morocco_vs_canada_shootout.py           # +11.33 RBP
├── notebooks/
│   └── forecasting_framework.ipynb             # Interactive walkthrough
├── tests/
│   └── test_models.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/JChan23/football-forecasting.git
cd football-forecasting
pip install -e .
```

---

## Quick Start

```python
from models import (
    MatchContext,
    goal_before_hydration_break,
    card_before_first_goal,
    penalty_shootout_probability,
)
from calibration import remove_vig_three_outcome

# 1. Convert match odds to vig-free probabilities
probs = remove_vig_three_outcome(
    odds_home=2.30,
    odds_draw=3.00,
    odds_away=3.20,
    odds_format='decimal'
)

# 2. Build match context
ctx = MatchContext(
    lambda_total=2.4,
    home_win_prob=probs['p_home_win'],
    away_win_prob=probs['p_away_win'],
    draw_prob=probs['p_draw'],
    match_type='standard',
    altitude_m=0
)

# 3. Price the "goal before hydration break" market
result = goal_before_hydration_break(ctx)
print(f"P(goal before break): {result['point_estimate']:.1%}")
print(f"  Naive Poisson was: {result['naive_poisson']:.1%}")
print(f"  Correction applied: -{result['correction_applied'][1]}ppt")

# 4. Price the "card before first goal" market
card_result = card_before_first_goal(
    lambda_goals=2.4,
    lambda_cards=3.2,
    card_context='standard'
)
print(f"\nP(card before first goal): {card_result['p_card_before_goal']:.1%}")
print(f"  Formula: λ_c/(λ_c+λ_g) = {card_result['lambda_cards_adjusted']:.1f}/({card_result['lambda_cards_adjusted']:.1f}+{card_result['lambda_goals']:.1f})")

# 5. Price the penalty shootout market
shootout = penalty_shootout_probability(
    draw_prob_90=probs['p_draw'],
    p_draw_after_et=0.42
)
print(f"\nP(penalty shootout): {shootout['p_shootout']:.1%}")
print(f"  Crowd typically anchors at: {shootout['crowd_anchor']}")
```

---

## Worked Examples

### Example 1: Card Before First Goal (Portugal vs Spain, QF)
**YOU: 67% | CROWD: 50% | OUTCOME: YES | RBP: +32.29**

```python
from models import card_before_first_goal

result = card_before_first_goal(
    lambda_goals=1.8,       # Under 2.5 favoured; H2H history
    lambda_cards=3.2,
    card_context='rivalry'  # Iberian derby → +15% card rate
)
# Output: P(card before goal) = 0.671
# Formula: 3.68 / (3.68 + 1.8) = 67.1%
# Crowd anchored at 50% treating this as a coin flip.
```

### Example 2: Late Window at Altitude (England vs Mexico, R16)
**YOU: 31% | CROWD: 49% | OUTCOME: NO | RBP: +33.04**

```python
from models import MatchContext, goal_after_second_hydration_break

ctx = MatchContext(
    lambda_total=2.0,
    home_win_prob=0.24,
    away_win_prob=0.52,
    draw_prob=0.24,
    altitude_m=2240   # Estadio Azteca — triggers 20% λ suppression
)
result = goal_after_second_hydration_break(ctx)
# Output: point_estimate = 0.308 (vs naive 0.385 at sea level)
# Altitude suppression reduces effective λ by 20%
```

### Example 3: Shootout Decomposition (Morocco vs Canada, R16)
**YOU: 12% | CROWD: 22% | OUTCOME: NO | RBP: +11.33**

```python
from models import penalty_shootout_probability

result = penalty_shootout_probability(
    draw_prob_90=0.275,   # Draw +230 → vig-removed 27.5%
    p_draw_after_et=0.42
)
# Output: P(shootout) = 0.275 × 0.42 = 11.6%
# Crowd anchored at 22% ignoring the conditional structure
```

---

## Calibration Results

The framework achieved **elite calibration** across 996 settled forecasts in the Jump Trading Probability Cup 2026, with stated confidence consistently matching actual outcome frequencies across all probability buckets.

```python
from calibration import calibration_by_bucket

# Example calibration check
forecasts = [(0.20, 0), (0.65, 1), (0.12, 0), (0.80, 1), ...]
result = calibration_by_bucket(forecasts)
# Platform verdict: "You're tracking the line — your confidence
#                   consistently matches reality. That's elite calibration."
```

**Key performance metrics:**
- RBP gap vs crowd: **+3.9** (better than 88% of participants)
- Contrarian win rate on total goals markets: **67%** (+8.3 avg RBP gap)
- Confidence bias: **+6% (optimal)**

---

## Key Lessons

1. **Proper scoring rules reward honesty.** The Brier score is strictly proper — the optimal submission is always your true belief. Gaming the score is mathematically impossible.

2. **Crowds systematically misprice time-window markets.** Any market scoped to a specific period (stoppage time, before hydration break) is underpriced relative to full-match intuitions.

3. **Rate ratios, not 50/50.** Race markets ("which happens first") have exact solutions from competing Poisson processes. The crowd's coin-flip anchor ignores this.

4. **Conditional structure matters.** Shootout markets require P(draw_90) × P(draw_ET), not a naive "knockout matches go to penalties" base rate.

5. **Minutes constraints kill substitute edge.** A player with 2.0 SoT/90 entering at the 65th minute has λ ≈ 0.56, not 2.0. Crowds ignore this.

6. **Altitude inverts late-game logic.** At 2,200m, fatigue suppresses rather than opens up late-match scoring. The standard "more goals after 75 min" assumption fails.

---

## Author

**Joshua Chan** | FMS Undergraduate, London School of Economics
GitHub: [@JChan23](https://github.com/JChan23)

*Built and validated across the 2026 FIFA World Cup, July 2026.*
