# Phase 7 — Coherent simulator

## Objective
The heart of the system. One simulation per game per `as_of`, quarter by quarter plus
overtime, from which **every** prop is derived. All invariants in
`contracts/invariants.yml` assert on every simulated game.

## Spec sections
§4, §29 through §52 inclusive. Read all of Part VI before writing code.

## Files to implement
```
src/nflprops/simulation/rng.py           # deterministic blake2b seeding — NEVER hash()
src/nflprops/simulation/environment.py   # shared pace/score shocks
src/nflprops/simulation/scenarios.py     # availability scenario mixture
src/nflprops/simulation/allocations.py   # Dirichlet-multinomial target/carry allocation
src/nflprops/simulation/redistribute.py  # learned injury redistribution
src/nflprops/simulation/quarter.py       # one quarter for one game
src/nflprops/simulation/overtime.py
src/nflprops/simulation/scoring.py       # opportunities -> TD/FG/XP/2PT events
src/nflprops/simulation/game.py          # full-game orchestration
src/nflprops/simulation/props.py         # prop derivation from draws
src/nflprops/simulation/invariants.py    # assert everything in contracts/invariants.yml
```

## Contracts consumed
`contracts/invariants.yml` (every rule), `contracts/prop_map.yml` (every derivation).

## Requirements
1. **Invariants abort the run.** Never warn-and-continue. `InvariantViolation` is
   raised with the offending game, draw index, and rule id.
2. Injury redistribution follows the learned sequence (depth replacement bonus ->
   same-position -> cross-position -> OTHER -> normalize). **Never proportionally
   redistribute a missing player's share across everyone.**
3. Availability is a **mixture over scenarios**, not a point estimate:
   `P(prop) = sum_s P(s) * P(prop|s)`.
4. Overtime runs when the simulated score is tied at end of regulation. Full-game
   props settle including OT.
5. TDs are **assigned to already-simulated events**, never generated independently.
6. `first_td` retains the `NONE` state. Do not renormalize it away.
7. Push probability is read directly off the integer-valued empirical distribution.
8. The simulator exports a **joint** draw matrix (subject to a configurable retention
   policy), not only marginals. Correlated props are never priced by multiplying
   marginals.
9. Adaptive stopping on Monte Carlo SE per spec §52, with a relative-SE floor for
   longshots.
10. Common random numbers across model versions and across lines on the same game.

## Acceptance tests
```
tests/invariants/test_all_invariants_on_random_games.py   # property-based, many seeds
tests/invariants/test_scenario_mixture.py
tests/invariants/test_inactive_zero_opportunity.py
tests/invariants/test_qb_derived_identities.py
tests/invariants/test_first_td_none_state.py
tests/invariants/test_period_sums.py
tests/determinism/test_same_seed_same_bytes.py
tests/determinism/test_no_builtin_hash.py                  # greps for hash( in simulation/
tests/unit/test_simulator_runs_without_optional_features.py
tests/unit/test_push_probability_from_integer_support.py
```

## Definition of done
- [ ] Every rule id in `contracts/invariants.yml` has a corresponding assertion
- [ ] Property-based invariant test passes over ≥200 randomized game configurations
- [ ] Every prop in `contracts/prop_map.yml` is derivable from a single simulation
- [ ] Simulator runs to completion with every optional feature absent
- [ ] Identical seed produces byte-identical output

## Explicitly out of scope
Calibration, devigging, EV, CLV.
