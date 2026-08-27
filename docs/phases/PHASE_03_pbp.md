# Phase 3 — Play-by-play subsystem

## Objective
Reconstruct player attribution and period splits from BDL play text, and gate tier-3
prop labels on validated reconciliation. This phase decides whether half/quarter,
first-TD, and longest-pass props are trainable at all.

## Spec sections
§6 (what PBP does NOT contain), §20, §21, §22.

## Files to implement
```
src/nflprops/pbp/aliases.py
src/nflprops/pbp/classify.py
src/nflprops/pbp/parser.py
src/nflprops/pbp/reconcile.py
tests/provider_contract/test_bdl_yardline_semantics.py
```

## Requirements
1. **First, run the yard-line semantics test.** `start_yard_line`/`end_yard_line`
   orientation is undefined in the spec. Empirically determine the convention from
   known plays (e.g. kickoffs, scoring plays where `end_yard_line` should be the goal
   line, drives with known field position). Document the finding in
   `docs/BDL_YARDLINE_SEMANTICS.md`. **Until it passes, `is_red_zone_candidate` and
   `is_goal_to_go_candidate` emit null with `is_missing=1`.** Do not guess 0–100
   offense-oriented.
2. `aliases.py` builds alias sets from canonical rosters for both teams in the game:
   full name, "F.Last", suffix variants, hyphen/apostrophe normalization, punctuation
   stripping.
3. `classify.py` assigns `play_family` from `type_slug`/`type_abbreviation`/`type_text`
   BEFORE free-text parsing.
4. `parser.py` extracts `parsed_passer_id, parsed_rusher_id, parsed_receiver_id,
   parsed_kicker_id` plus `parser_confidence`. **An alias that matches two players on
   the same team is a parse failure, not a coin flip.**
5. `reconcile.py` compares reconstructed totals against `/nfl/v1/stats` for: pass
   attempts, completions, pass yards, rush attempts, rush yards, receptions,
   receiving yards, rushing TDs, receiving TDs, FG made. Assigns
   `pbp_quality in {HIGH, MEDIUM, LOW, FAIL}` using the config threshold.
6. Games below HIGH are excluded from tier-3 label sets. The exclusion, with counts,
   is written to the training manifest so effective sample size is always visible.

## Acceptance tests
```
tests/provider_contract/test_bdl_yardline_semantics.py
tests/unit/test_alias_collision_is_failure.py
tests/unit/test_play_family_classification.py
tests/unit/test_reconciliation_scoring.py
tests/unit/test_tier3_gate.py             # LOW-quality game cannot enter tier-3 labels
```

## Definition of done
- [ ] Reconciliation report produced for every ingested season
- [ ] `%` of games at HIGH reported per season and written to `docs/PBP_COVERAGE.md`
- [ ] If HIGH coverage is below a usable threshold, **say so explicitly** and mark the
      affected tier-3 props as not production-ready rather than shipping weak labels

## Explicitly out of scope
EPA computation. Features. Models.
