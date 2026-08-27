# Phase 6 — Structural models

## Objective
Fit every component model the simulator calls. Each is a small, interpretable,
regularized model with an explicit interface, plus an empirical residual pool where
the simulator needs to draw noise.

## Spec sections
§32, §33, §34, §35, §36, §37, §38, §39, §41, §43, §45, §62.

## Files to implement
```
src/nflprops/models/base.py            # ComponentModel interface: fit / predict / sample / artifact
src/nflprops/models/registry.py
src/nflprops/models/plays.py           # team offensive plays (overdispersed count)
src/nflprops/models/pass_mix.py        # p_dropback | score state, quarter
src/nflprops/models/sacks.py           # p_sack | QB, team, opponent
src/nflprops/models/opportunities.py   # p_directed, target/carry share priors, kappa
src/nflprops/models/catch.py           # completion probability per target
src/nflprops/models/interceptions.py   # INT probability per attempt
src/nflprops/models/passing.py         # QB-side conditioning only (yards are DERIVED)
src/nflprops/models/rushing.py         # rush gain mean + residual pool
src/nflprops/models/receiving.py       # reception gain mean + residual pool
src/nflprops/models/touchdowns.py      # scoring opportunities, TD split, 2PT choice
src/nflprops/models/kicking.py         # FG make, XP make
```

## Requirements
1. **`passing.py` must NOT model passing yards.** QB passing yards, completions, and
   TDs are derived by aggregation in the simulator (spec §40). If you find yourself
   fitting a passing-yards regressor, you have violated the architecture.
2. Play counts are **overdispersed** relative to Poisson. Fit and validate dispersion;
   do not assume.
3. `pass_mix.py` must include current score differential and time remaining. This is
   the mechanism that creates game script. Fit it on realized in-game score states.
4. Residual pools (`rushing`, `receiving`) are **empirical**, stratified by
   `(position, aDOT bucket, catch-depth bucket)` for receiving and by
   `(position, box/context bucket)` for rushing. They must contain negative gains,
   zero gains, and explosive tails. They are frozen into the model artifact.
5. `opportunities.py` produces Dirichlet concentration `kappa` from posterior
   uncertainty: higher uncertainty -> lower kappa -> fatter share variance.
6. `touchdowns.py` includes the two-point-conversion choice model
   `P(go_for_two | differential, quarter, time)`. Do not approximate XP attempts as
   equal to TDs.
7. `kicking.py`: v1 uses a distance-marginal make rate with an opportunity-quality
   adjustment. Distance-aware FG modeling is a v2 item gated on PBP validation —
   record this in the model card, do not imply precision you do not have.
8. Optional residual-ML challenger: `structural_GLM + residual_ML`. It is registered
   as a challenger, never as the default, and is promoted only through Phase 10 gates.

## Acceptance tests
```
tests/unit/test_component_model_interface.py
tests/unit/test_no_passing_yards_regressor.py      # architectural guard
tests/unit/test_play_count_overdispersion.py
tests/unit/test_residual_pools_have_negative_tail.py
tests/unit/test_kappa_responds_to_uncertainty.py
tests/unit/test_two_point_model_present.py
```

## Definition of done
- [ ] Every model in `models/` fits on historical data and serializes to an artifact
- [ ] Each model reports out-of-sample fit statistics in the training report
- [ ] No model imports anything under `providers/`

## Explicitly out of scope
The simulator loop itself. Calibration. Market.
