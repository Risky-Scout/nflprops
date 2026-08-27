# Phase 11 — Explanation and reporting

## Objective
Make every number defensible. If you cannot say why a projection moved, you cannot
trust it and you certainly cannot bet it.

## Spec sections
§69.

## Files to implement
```
src/nflprops/explain/attribution.py
src/nflprops/explain/report.py
src/nflprops/pipelines/weekly.py       # ties reporting into the weekly loop
```

## Requirements
1. Attribution is computed by **ablating each effect group through the simulator with
   common random numbers** — a causal decomposition within the model, not a post-hoc
   SHAP narrative over a black box.
2. Effect groups: baseline, team volume, player role, QB, opponent, game-market
   environment, injury redistribution.
3. Components sum to the total within a documented tolerance. Enforced by test.
4. Probability-space adjustments (calibration, market-aware) are reported separately
   in percentage points, after the projection decomposition.
5. Reports show distribution, not just a point: median, the percentile ladder, push
   probability, MC standard error, and availability-scenario entropy.

## Acceptance tests
```
tests/unit/test_explanation_additivity.py
tests/unit/test_attribution_uses_common_random_numbers.py
```

## Definition of done
- [ ] `nflprops report --season 2026 --week N` produces a per-player explanation
- [ ] Every published prop can be traced to its effect decomposition
