# Phase 8 — Calibration and dispersion

## Objective
Turn raw simulated probabilities into calibrated ones, and verify the simulator is
not under-dispersed. This is where most structural simulators quietly fail.

## Spec sections
§53, §63, §64.

## Files to implement
```
src/nflprops/calibration/oof.py          # out-of-fold prediction generation
src/nflprops/calibration/calibrators.py  # logistic / beta / isotonic
src/nflprops/calibration/hierarchy.py    # prop family -> position -> global fallback
src/nflprops/calibration/dispersion.py   # PIT / rank histograms, variance inflation
```

## Requirements
1. **Out-of-fold predictions only.** Fitting a calibrator to in-sample fitted
   probabilities is a hard failure, tested for.
2. Fallback hierarchy `prop family -> position -> global`, with shrinkage toward the
   broader group for small samples. Method chosen per family by OOS log loss.
3. Dispersion diagnostic is **mandatory, not optional**: PIT histograms per prop
   family. A U-shape means under-dispersion.
4. Remedy order is enforced by review, not code: (a) find the missing structural
   variance source, (b) widen residual pools / lower kappa where justified, (c) only
   last, a fitted variance-inflation factor — which must be recorded in the model
   card as a known crutch.
5. Calibration folds must not overlap their training targets. Tested.

## Acceptance tests
```
tests/leakage/test_calibration_uses_oof_only.py
tests/leakage/test_calibration_fold_no_overlap.py
tests/unit/test_calibration_fallback_hierarchy.py
tests/unit/test_pit_histogram_computed.py
```

## Definition of done
- [ ] Reliability curves and PIT histograms produced per prop family
- [ ] Any applied variance inflation is explicitly listed in `model_card.md`
- [ ] Calibrated log loss beats raw log loss out of sample

## Explicitly out of scope
Market comparison and EV (Phase 9).
