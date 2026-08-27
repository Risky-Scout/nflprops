# Phase 9 — Market system

## Objective
Odds conversion, devigging (including the hard one-sided cases), cross-book
consensus, the snapshot collector, closing-line logic, EV, and CLV.

## START THE COLLECTOR EARLY
The collector piece of this phase should be deployed **as soon as Phase 1 lands**.
BDL retains no historical live prop data. Every week you do not collect is a week of
market history that can never be recovered. This is the most time-sensitive item in
the entire build.

## Spec sections
§54, §55, §56, §57, §58, §59, §60.

## Files to implement
```
src/nflprops/market/odds.py         # american <-> implied <-> decimal
src/nflprops/market/devig.py        # proportional baseline; power/shin as challengers
src/nflprops/market/consensus.py    # cross-book pooling
src/nflprops/market/snapshots.py    # collector + checkpoint retention
src/nflprops/market/closing.py      # deterministic closing-line rule
src/nflprops/market/ev.py
src/nflprops/market/clv.py
src/nflprops/market/rules/          # versioned settlement rules, separate from model
```

## Requirements
1. Proportional devig is the transparent baseline. Power/Shin are challengers,
   promoted only on held-out evidence.
2. **One-sided markets** (`first_td`, milestone `anytime_td`) get the explicit
   treatment in spec §56. Every prediction row carries `devig_method` and
   `devig_confidence`. Never treat a raw implied probability as fair.
3. Push probability contributes zero to EV; `EV_over = P_over*(d-1) - P_under`.
4. Collector retains `OPEN, T-48H, T-24H, T-12H, T-6H, T-3H, T-1H, T-30M, T-10M,
   CLOSE`. Not just "latest".
5. Closing line = latest valid quote at least `close_buffer_seconds` before kickoff,
   frozen in config. Missing-window games are marked `closing_line_missing` and
   excluded from CLV aggregates — **not filled**.
6. Settlement rules live in `market/rules/` and are versioned separately from model
   rules. The simulator exposes both `offensive_tds` and `all_tds`; the settlement
   rule picks which prices a given vendor's `anytime_td`.
7. **Prop availability bias:** record whether each prop was still quoted at close.
   Report edge conditional on availability.

## Acceptance tests
```
tests/unit/test_odds_conversion_roundtrip.py
tests/unit/test_proportional_devig.py
tests/unit/test_one_sided_devig_marks_confidence.py
tests/unit/test_first_td_devig_includes_none.py
tests/unit/test_push_zero_ev.py
tests/unit/test_closing_line_rule_deterministic.py
tests/unit/test_settlement_rules_versioned.py
```

## Definition of done
- [ ] Collector runs on a schedule and writes all checkpoints
- [ ] Devig produces fair probabilities with a stated method and confidence per row
- [ ] CLV report segments by prop family, vendor, and close-availability

## Explicitly out of scope
Backtest orchestration (Phase 10).
