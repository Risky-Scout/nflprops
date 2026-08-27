# 2026 Prediction Readiness

The repository is not production-prediction ready merely because the simulator runs.
Publication is allowed only after all mandatory gates below pass.

## Gate 1 — provider/data contract
- Pin exact official BDL NFL OpenAPI bytes and commit `spec.lock.json`.
- `nflprops provider verify bdl --strict-fields` passes.
- Authorized response fixtures cover every production endpoint.
- Empirically inventory historical coverage available under the actual BDL plan.

## Gate 2 — market history collection
- Turn on live player-prop snapshots immediately.
- Preserve provider-updated time and collector-received time.
- Define deterministic OPEN/T-24h/T-1h/CLOSE buckets.
- Do not claim CLV for weeks where closing history was not collected.

## Gate 3 — point-in-time historical dataset
- Complete immutable raw storage and canonical Parquet/DuckDB writes.
- Zero leakage from final scores, future injuries, future roster states, future odds,
  future season aggregates, or calibration folds.
- Coverage report establishes usable training seasons and missingness by endpoint.

## Gate 4 — player/team states
- Fit role and efficiency shrinkage/decay with expanding-window validation.
- Role responds faster than skill.
- Injury/depth-chart redistribution is learned from historical substitutions.
- Rookie/no-history priors are explicitly validated.

## Gate 5 — structural football components
Fit and validate:
- team plays;
- dropback/run mix;
- sacks;
- directed-target rate;
- target allocation;
- rush allocation;
- completion/interception probabilities;
- receiving and rushing gain distributions;
- TD allocation;
- field-goal/XP components.

No component is promoted for in-sample fit alone.

## Gate 6 — coherent simulator
- All invariants pass across randomized games and edge cases.
- Deterministic byte-level reproduction passes.
- Simulation distributions match historical means, variances, zeros, and tails.
- Injury/QB-change scenarios behave plausibly.
- Tier-3/period markets remain disabled until PBP reconciliation is HIGH.

## Gate 7 — walk-forward market benchmark
Use expanding-window, same-timestamp comparisons only.
For every major prop family compare model vs no-vig market on:
- log loss;
- Brier score;
- calibration;
- CRPS/WIS where full outcome distributions are scored;
- stability by season, early/late season, position, favorite/underdog, injuries.

## Gate 8 — OOF calibration
- Generate true out-of-fold probabilities.
- Fit calibration only to prior folds.
- Prop-family/position/global fallback hierarchy is validated.
- PIT/reliability diagnostics show no material under/over-dispersion.

## Gate 9 — promotion
The model is allowed to publish production edges only when the locked promotion
criteria pass. Positive ROI alone is insufficient.

## Gate 10 — 2026 prospective operation
Before Week 1:
- freeze model/feature/provider-contract versions;
- start scheduled injury/roster/game-odds/prop collection;
- create an opening prediction snapshot before seeing closes;
- retain every published forecast immutably;
- settle and score every week;
- never retroactively alter a published probability.

The 2026 season is then the prospective proof period.
