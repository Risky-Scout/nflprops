# Phase 5 — Dynamic state system

## Objective
Fit and maintain empirical-Bayes posteriors for every player, team, and role metric,
including posterior variance — which the simulator needs for allocation concentration.

## Spec sections
§25, §26, §27, §28.

## Files to implement
```
src/nflprops/state/empirical_bayes.py
src/nflprops/state/player.py
src/nflprops/state/team.py
src/nflprops/state/role.py
src/nflprops/state/store.py
```

## Requirements
1. Update rule: `theta_t = w_t*x_t + (1-w_t)*theta_prior_t`, `w_t = n_t/(n_t+k)`.
   Transition: `theta_prior_{t+1} = lambda*theta_t + (1-lambda)*theta_population`.
2. **`k` and `lambda` are LEARNED per metric** by maximizing out-of-sample predictive
   likelihood on a walk-forward grid. They are not guessed and not global. Fitted
   values are stored in the artifact and printed in the training report.
3. Track posterior variance and effective sample size, not just the mean.
4. **Role state must move faster than skill state.** After fitting, assert
   `lambda_role < lambda_skill` per position group. If the fit does not produce this,
   report it as a finding — do not force it silently.
5. Priors are conditioned on `(position, depth, experience)` for rookies and
   early-season players.
6. State snapshots are written keyed by `(entity, metric, as_of)` and are PIT-filtered
   like any other table.

## Acceptance tests
```
tests/unit/test_eb_shrinkage_monotone.py       # more observations -> less shrinkage
tests/unit/test_role_faster_than_skill.py
tests/unit/test_state_variance_tracked.py
tests/leakage/test_state_timestamp_not_future.py
```

## Definition of done
- [ ] `nflprops state update --as-of <ts>` writes all three state tables
- [ ] Fitted `k` and `lambda` per metric written to the training report
- [ ] Walk-forward likelihood improvement over a fixed-k baseline is documented

## Explicitly out of scope
Structural models. Simulation.
