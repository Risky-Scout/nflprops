# Phase 10 — Walk-forward backtesting

## Objective
Prove — or disprove — that the model beats the market at the same information
timestamp, with zero leakage and honest sample accounting.

## Spec sections
§61, §64, §65, §66, §67, §74.

## Files to implement
```
src/nflprops/backtest/walkforward.py
src/nflprops/backtest/metrics.py
src/nflprops/backtest/leakage.py
src/nflprops/backtest/promotion.py
src/nflprops/backtest/dataset.py     # backtest row contract
```

## Backtest row contract (every row)
```
prediction_id, as_of, canonical_game_id, canonical_player_id, prop_type,
line, odds, vendor, market_type,
feature_snapshot_id, state_snapshot_id, model_version,
p_raw, p_calibrated, p_fundamental, p_final,
market_fair, devig_method, devig_confidence,
actual_value, outcome, closing_line, closing_odds, quoted_at_close
```

## Requirements
1. **Expanding window only.** Random splits are a hard failure.
2. Every leakage test in spec §66 runs in CI and fails the build.
3. Market benchmark comparison uses the devigged market probability **at the same
   `as_of`**, not the closing price. Comparing your T-24H model to the closing line
   and calling it a win is the classic self-deception.
4. Promotion gates (spec §65) are ALL required. ROI alone promotes nothing.
5. Weeks 1–4 and weeks 14+ are evaluated as separate stability slices.
6. Report effective sample size per prop family, including tier-3 props' reduction
   from the `pbp_quality == HIGH` gate.

## Acceptance tests
```
tests/leakage/test_no_random_split.py
tests/leakage/test_all_leakage_rules.py
tests/determinism/test_byte_reproducibility.py   # spec §67 full rebuild test
tests/unit/test_promotion_gate_requires_all.py
tests/unit/test_market_benchmark_same_asof.py
```

## Definition of done
- [ ] `nflprops backtest --seasons ...` produces a full report
- [ ] Model vs market log loss and Brier reported per prop family
- [ ] Every promotion gate evaluated and printed with pass/fail
- [ ] Reproducibility test green

## Explicitly out of scope
Explanations, reporting UI.
