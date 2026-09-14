# `player_prop_pricing_artifacts` / `player_prop_prices` — the canonical player-prop pricing product

**Status:** PHASE 9 certified (9A read-only audit, 9B push-aware fair-pricing
math, 9C immutable SQL persistence + schema, 9D official-checkpoint
integration + explicit publication gate, 9E certification + product
contract). Best-executable-price selection, devigged multi-book consensus,
opportunity ranking, confidence scoring, Kelly sizing, generic count-style
sportsbook milestones, an API, a dashboard, and WizardOfOdds publishing are
**post-Phase-9** and are explicitly **deferred** — see
[§26 Deferred pricing products](#26-deferred-pricing-products).

This document is the normative product contract. It does not restate the
mathematics line-by-line; the executable sources of truth are
`src/nflprops/market/odds.py`, `src/nflprops/market/current_pricing.py`,
`src/nflprops/orchestration/pricing_store.py`,
`src/nflprops/orchestration/flows/checkpoints.py`, and
`migrations/versions/0006_player_prop_pricing.py`. See also
`docs/IMPLEMENTATION_SPEC.md` §54-§60 (the original market-system design
this module was built from) and `docs/THRESHOLD_MILESTONE_PROBABILITIES.md`
(the sibling sportsbook-independent model product).

---

## 1. Purpose

`player_prop_pricing_artifacts` + `player_prop_prices` is the canonical,
immutable record of **current sportsbook market pricing**: for every quote a
book actually offers, the model's raw and push-aware fair probabilities
against that exact line, the book's own offered price, and the resulting
expected value — derived from **the same coherent game simulation**
(`nflprops.simulation.game.simulate_game` → `GameSimulationResult`) that
`player_game_projections` (Phase 7) and `player_game_threshold_events`
(Phase 8) read in the same checkpoint. There is no second simulation and no
pricing-specific simulation.

The full contract, end to end:

```
shared GameSimulationResult
        |
raw p_win / p_push / p_loss                (§4)
        |
conditional non-push model fair probability (§5)
        |
model fair decimal / American odds          (§6)
        |
actual sportsbook offered price             (unchanged, from the quote)
        |
push-aware EV                               (§7)
        |
immutable canonical SQL pricing artifact    (§14)
```

Phase 9 certifies exactly the first five links in that chain. It carries
**no** best-executable price and **no** devigged multi-book consensus.

---

## 2. Sportsbook-independent model price vs. sportsbook offer

Every persisted row keeps two categorically different quantities strictly
separate (§8 elaborates the exact field-by-field terminology):

* **MODEL RAW PROBABILITY** (`p_model_raw`) and **MODEL FAIR PRICE**
  (`p_model_fair_nonpush`, `model_fair_decimal`, `model_fair_american`) are
  pure functions of the shared simulation draws and the quoted line. They do
  not depend on which book is asking, what odds it offers, or how many other
  books exist.
* **BOOK OFFER** (`american_odds`) and **BOOK DEVIGGED PROBABILITY**
  (`p_market_fair`, `devig_method`, `devig_confidence`) are properties of the
  quote itself.

Nothing in this module ever collapses these into one "edge" field, and
nothing here selects a best price or a consensus across books (deferred,
§26).

---

## 3. Player universe and per-quote grain

Unlike the Phase-7/8 canonical products (which are complete for every
Phase-7-eligible player regardless of whether a book quotes them), pricing
is **legitimately sparse**: one row exists **per currently-quoted
quote/side**, never for an unquoted player/prop. A player with zero posted
quotes contributes zero pricing rows — this is not a gap in the product, it
is the product (§20 zero-quote certification).

---

## 4. Raw settlement semantics (LOCKED)

For draw vector `X` (length `n_draws`) and threshold line `L`:

```
OVER at line L:
    p_win  = P(X > L)
    p_push = P(X = L)
    p_loss = P(X < L)

UNDER at line L:
    p_win  = P(X < L)
    p_push = P(X = L)
    p_loss = P(X > L)

p_win + p_push + p_loss = 1   (exact, by construction)
```

Implemented in `nflprops.simulation.props.summarize_prop`
(`p_over = mean(values > line)`, `p_under = mean(values < line)`,
`p_push = mean(values == line)`) and read unchanged by
`nflprops.market.current_pricing._price_quote`.

**`p_loss` is never persisted.** It is always derivable as
`p_loss = 1 - p_model_raw - p_push` — the same convention Phase 8 uses for
`p_miss = 1 - p_hit`.

Integer lines have real, nonzero push probability (received directly off
the empirical integer-valued draw distribution, never approximated from a
continuous density — SPEC §57). Half lines (`L` not integer-valued, e.g.
`100.5`) always have `p_push = 0` for an integer-valued stat, since no draw
can equal a non-integer line exactly.

---

## 5. Conditional non-push model fair probability (LOCKED)

```
p_model_fair_nonpush = p_win / (1 - p_push)
                      = p_win / (p_win + p_loss)
```

This is **P(win | the wager does not push)** — implemented in
`nflprops.market.odds.conditional_nonpush_fair_probability`. It is a MODEL
quantity: it must never be confused with `p_model_raw` (the unconditional
raw win probability) or with `p_market_fair` (the sportsbook's own devigged
implied probability — a book-side quantity computed from the book's own two
quoted prices, unrelated to this formula).

Validation is non-clipping: `0 <= p_win <= 1`, `0 <= p_push <= 1`,
`p_win + p_push <= 1` (with a `1e-9` floating-point slack, never a
scientific settlement epsilon) are enforced with a hard `ValueError` on
violation — an invalid input is never silently repaired.

**All-push boundary:** when `p_push == 1` (every draw pushes), there is no
executable non-push outcome to condition on. `p_model_fair_nonpush` is
`None` — never `NaN`, never a fabricated value.

---

## 6. Model fair odds (LOCKED)

```
model_fair_decimal   = (1 - p_push) / p_win   =  1 / p_model_fair_nonpush
model_fair_american   derived from p_model_fair_nonpush via the existing
                       American-odds conversion (nflprops.market.odds.
                       implied_to_american) — NEVER from p_model_raw.
```

Implemented in `nflprops.market.odds.fair_decimal_odds` /
`fair_american_odds`, both built strictly on top of
`conditional_nonpush_fair_probability` — there is exactly one place the
conditional formula is computed, and both odds representations derive from
it.

Boundary contract, never an infinity or a `NaN`:

| Case | `model_fair_decimal` | `model_fair_american` |
|---|---|---|
| `p_win = 0`, `p_nonpush > 0` | `None` | `None` |
| `p_win = 1`, `p_push = 0` | `1.0` | `None` |
| all-push (`p_push = 1`) | `None` | `None` |

---

## 7. Push-aware EV (LOCKED, UNCHANGED since Phase 6)

```
EV = p_win * (D - 1) - p_loss          (D = offered decimal odds)
```

Implemented in `nflprops.market.odds.expected_value`, using the **raw**
`p_win`/`p_loss` (never `p_model_fair_nonpush`) against the **actual
offered** sportsbook decimal price. Pushes contribute exactly `0` — EV is
deliberately **not** conditionalized on non-push outcomes; conditionalizing
it would double-count the push adjustment (`p_model_fair_nonpush` already
answers "what would be fair if this couldn't push"; EV answers "what do I
actually make at the book's real price, pushes included as free rolls").
`tests/unit/test_fair_pricing.py::test_ev_is_not_conditionalized_on_fair_probability`
certifies this distinction directly.

---

## 8. Terminology — never say "fair" unqualified

| Field | Exact meaning |
|---|---|
| `p_model_raw` | The model's raw (unconditional) probability of winning the offered wager, from the shared simulation draws. |
| `p_model_fair_nonpush` | The model's win probability **conditional on the wager not pushing**. A MODEL quantity. |
| `model_fair_decimal` / `model_fair_american` | Odds representations of `p_model_fair_nonpush`. MODEL quantities. |
| `p_market_fair` | The sportsbook's own implied probability after the active book-side devig method (`proportional_two_sided` today). A BOOK quantity — computed from the book's two quoted prices, never from the model. |
| `american_odds` | The actual, executable sportsbook offer. Never renamed to imply fairness. |

Every place in code and in this document that could otherwise say bare
"fair" says **model fair** or **market/book fair** instead.

---

## 9. Exact numerical certification — integer line

`X = [99, 100, 100, 101, 102]`, `line = 100`, certified directly against the
production helpers (`tests/unit/test_fair_pricing.py`,
`tests/orchestration/test_phase9e_pricing_certification.py`):

```
OVER:  p_win=0.4  p_push=0.4  p_loss=0.2
       p_model_fair_nonpush = 2/3
       model_fair_decimal   = 1.5
       model_fair_american  = -200

UNDER: p_win=0.2  p_push=0.4  p_loss=0.4
       p_model_fair_nonpush = 1/3
       model_fair_decimal   = 3.0
       model_fair_american  = +200

at offered decimal 2.00:
       OVER  EV = +0.20
       UNDER EV = -0.20
```

---

## 10. Half-line certification

Same vector, `line = 100.5`:

```
OVER: p_win=0.4  p_push=0  p_loss=0.6
      p_model_fair_nonpush = 0.4   (== p_model_raw, since p_push == 0)
      model_fair_decimal   = 2.5
      model_fair_american  = +150
```

**Identity certified:** `p_push == 0 ⇒ p_model_fair_nonpush == p_model_raw`
exactly (`conditional_nonpush_fair_probability(p, 0) == p` for any valid
`p`) — proven directly in `test_fair_pricing.py` and re-certified end to end
through a real official checkpoint in
`test_phase9d_checkpoint_pricing_integration.py::test_push_eligible_integer_line_persists_phase9b_math_exactly`.

---

## 11. Binary one-sided markets

The five supported binary products — `anytime_td`, `anytime_td_1q`,
`anytime_td_1h`, `anytime_td_2h`, `first_td` — have `p_push = 0` always
(there is no "push" state for a yes/no event), so
`p_model_fair_nonpush == p_model_raw` for every row, and
`model_fair_decimal`/`model_fair_american` derive normally wherever
`p_model_raw` is strictly inside `(0, 1)`.

`p_market_fair` is `None` and `devig_method` is `None` for these rows: a
one-sided BDL quote cannot be paired-devigged, and the one-sided
field/borrowed-overround devig methods (SPEC §56) remain uncertified stubs
(§26). `devig_confidence` is the type-safe
`DevigConfidence.ONE_SIDED_UNBENCHMARKED` (`"one_sided_unbenchmarked"`) —
never falsely reported as `FULL`, `PARTIAL`, or `BORROWED`.

---

## 12. Count-style sportsbook milestone boundary (LOCKED — not implemented)

A `MarketType.MILESTONE` quote for any prop **outside** the five binary
products above (e.g. a hypothetical "2+ field goals" sportsbook quote) has
no defined `AT_LEAST` hit-probability distribution in the certified
pricing pipeline. Since Phase 9B, this **fails explicitly** —
`nflprops.market.current_pricing.UnsupportedMilestoneMarketError`, naming
`prop_type`/`market_type`/`line` — instead of silently returning zero rows
(the pre-9B behavior) or silently treating it as binary. Phase 9E adds no
support for these; the Phase-8 model threshold ladder (`fg_made`,
`passing_tds`, ... at 1+/2+/3+) remains available as a fully independent,
already-certified product (`docs/THRESHOLD_MILESTONE_PROBABILITIES.md`) —
it is not wired into sportsbook pricing.

---

## 13. Unknown market types (LOCKED — fail closed)

`current_pricing.py` allowlists exactly `{"over_under", "milestone"}`
(`MarketType.OVER_UNDER.value`, `MarketType.MILESTONE.value`). Any other
`market_type` string raises `UnsupportedMarketTypeError` — never silently
falls through to the milestone branch (the pre-9B behavior).

---

## 14. Canonical persistence contract

Migration `0006_player_prop_pricing` (revises `0005_player_threshold_events`
— no other migration exists or is needed for Phase 9):

* **`player_prop_pricing_artifacts`** — one header row per `run_id`
  (PK, FK → `prediction_runs.run_id`, not `ON DELETE CASCADE`): `season,
  week, game_id, as_of, model_version, row_count, scientific_content_sha256,
  created_at`. `row_count >= 0` (CHECK) — `row_count == 0` is a **valid,
  complete** artifact, proving pricing ran for a zero-quote run; its
  *absence* is what means "pricing never ran," never its row count.
* **`player_prop_prices`** — one row per priced quote/side, PK
  `prediction_id` (the pre-existing, **unchanged** blake2b identity from
  `nflprops.market.current_pricing.prediction_id` — Phase 9C never invents
  a second competing identity), FK `run_id` → `prediction_runs.run_id`.
  CHECK constraints mirror the certified boundary semantics exactly:
  unit-interval probabilities, `p_model_raw + p_push <= 1.000000001` (the
  same `1e-9` float slack as the application layer, never a settlement
  epsilon), `model_fair_decimal IS NULL OR >= 1.0`, `n_draws > 0`,
  `american_odds != 0`, `market_type IN ('over_under','milestone')`,
  `side IN ('OVER','UNDER','HIT')`, and `line`/`market_type` nullability
  paired correctly. 8 indexes (`run_id`; `season,week,game_id`; `player_id`;
  `prop_type`; `vendor`; `side`; `run_id,player_id`; `run_id,vendor`).

`persist_player_prop_pricing` (`nflprops.orchestration.pricing_store`) is
the single persistence entry point for both tables. It loads the parent
`prediction_runs` row exactly once and validates
`season/week/game_id/model_version/n_draws` and `as_of` (against
`scheduled_as_of`) before writing anything; independently re-checks
`quote_available_at <= as_of` (PIT defense in depth, never trusting the
upstream pricing layer alone); and recomputes every `prediction_id` via the
certified `current_pricing.prediction_id` function, hard-erroring on any
mismatch.

**Local ⇄ PostgreSQL parity** is certified in
`tests/orchestration/test_pricing_store.py` (local Warehouse) and
`test_pricing_store_postgres.py` (ephemeral Docker PostgreSQL): migration
upgrade/incremental/downgrade/re-upgrade, FK rejection, every CHECK
constraint, idempotent/immutable persist, atomic rejection, and the
zero-quote round trip are certified on both backends.

---

## 15. Scientific content hash

`compute_scientific_content_hash` (`pricing_store.py`) is a deterministic,
order-independent SHA-256 over the complete row set's scientific fields
only: rows sorted by `prediction_id`, each serialized as its scientific
fields in fixed order (deterministic scalar serialization — `repr()` for
floats, ISO-8601 UTC for datetimes, a null sentinel distinct from any real
value), prefixed by the schema/version marker `"player_prop_prices/v1"` so a
future schema change can never collide ambiguously with today's hash.
`rows=[]` yields the fixed **canonical empty-artifact hash** — proven
identical across independently-constructed empty frames.

Row-set **order never affects the result**: persisting the identical
scientific frame in original/reversed/shuffled order produces the same
hash and is treated as an idempotent no-op, never a spurious conflict
(`test_pricing_store.py::test_row_order_never_affects_hash_or_causes_a_spurious_conflict`).

---

## 16. Scientific equality (LOCKED, corrected in Phase 9C-fix)

`SCIENTIFIC_FIELDS` (`pricing_store.py`) — participates in row conflict
detection and the artifact hash:

```
run_id, season, week, game_id, player_id, prop_type, market_type, vendor,
side, line, american_odds,
p_model_raw, p_push, p_model_fair_nonpush, model_fair_decimal,
model_fair_american, p_market_fair, devig_method, devig_confidence,
ev_per_unit, edge,
n_draws, model_version, as_of,
quote_available_at, quote_time_source, provider_updated_at, opened_at,
collector_received_at,
model_mean, model_median, p05, p10, p25, p50, p75, p90, p95,
p_model_calibrated
```

The distribution-summary columns (`model_mean`, `model_median`, the seven
percentiles) and `p_model_calibrated` are deliberately **included**: they
characterize the whole modeled distribution, not merely the probability at
one quoted line, and are **not** implied by `p_model_raw`/`p_push` alone —
two distributions can share a win/push probability at one line while
differing in mean, median, or any other percentile. (An earlier draft of
this module incorrectly excluded them as "derived"; that was a genuine
scientific-immutability gap, corrected before certification.)

Exactly two columns are **excluded** from scientific equality, both
**proven** pure deterministic functions of fields that ARE in the hash:

* `confidence_tier` = `prop_confidence_tier(prop_type)` — a static,
  versioned lookup keyed only by the already-scientific `prop_type`.
* `quote_age_seconds` = `(as_of - quote_available_at).total_seconds()` —
  both operands already-scientific, already-hash-protected columns.

Both are exposed as named, directly-callable functions
(`recompute_confidence_tier`, `recompute_quote_age_seconds`) and proven
against every persisted row in `test_pricing_store.py`.

`created_at` (both tables) is the **only** operational, non-scientific
field.

---

## 17. Checkpoint execution order (final, PHASE 9D)

```
claim prediction_run
      -> scheduled_as_of PIT manifest                          [PHASE 5]
      -> build football state
      -> ONE coherent game simulation (compute_game_prediction) [PHASE 6]
             |
             +-- 1. build + persist player_game_projections     [PHASE 7]
             |      (failure -> FAILED, no thresholds, no pricing)
             |
             +-- 2. build + persist player_game_threshold_events [PHASE 8]
             |      (failure -> PARTIAL, projections retained, no pricing)
             |
             +-- 3. price_current_markets(...) EXACTLY ONCE      [PHASE 6/9B]
             |      (failure -> PARTIAL, no canonical pricing artifact)
             |
             +-- 4. persist_player_prop_pricing(...)             [PHASE 9C]
             |      canonical SQL, from the SAME priced frame as step 3
             |      (failure -> PARTIAL, no canonical pricing artifact,
             |       legacy mirror never runs)
             |
             +-- 5. persist_current_pricing(...)                 [legacy]
             |      Warehouse/Parquet compatibility mirror, from the SAME
             |      priced frame -- no second pricing call
             |      (failure -> PARTIAL, ALL canonical artifacts retained)
             |
             +-- 6. explicit terminal eligibility check          [PHASE 9D]
                    (_pricing_artifact_invariant_violation -- explicit
                     runtime code, never a Python `assert`)
      -> 7. terminal prediction_run status
```

Steps 3-5 all consume the identical `GameSimulationResult` object
(`computation.simulation`, verified by object identity) and the identical
priced-quote list (`priced`) — there is never a second simulation and never
a second pricing calculation, certified for zero quotes, one quote, and
multiple books/vendors
(`test_phase9d_checkpoint_pricing_integration.py::test_exactly_one_pricing_call_and_one_simulation_*`).

---

## 18. Publication-eligibility runtime gate (LOCKED, PHASE 9D correction)

A run cannot reach `SUCCESS`/`MODEL_ONLY` or `SUCCESS`/`PUBLISHED` unless
**all three** canonical artifacts exist: the complete Phase-7 projection
artifact, the complete Phase-8 threshold artifact, and the complete
Phase-9 pricing artifact **header** (row count aside).

This is enforced by `_pricing_artifact_invariant_violation`
(`checkpoints.py`) — **explicit runtime code**, not a Python `assert`
(`assert` is compiled out entirely under `python -O` /
`PYTHONOPTIMIZE=1`, which would silently remove exactly this check). A
missing canonical pricing artifact fails the run closed
(`PARTIAL`/`NOT_PUBLISHED`, `failure_code="PRICING_ARTIFACT_INVARIANT_VIOLATION"`)
regardless of interpreter optimization flags. Certified structurally (an
AST parse proving zero `assert` nodes in the guarded functions) and
behaviorally (forcing the guarded field false immediately before each
terminal transition and requiring the run never reaches `SUCCESS`) in
`test_phase9d_checkpoint_pricing_integration.py`.

---

## 19. `MODEL_ONLY` final contract

> **`publication_status == MODEL_ONLY` ⇔** a successful modeled run that has
> a complete Phase-7 projection artifact, a complete Phase-8 threshold
> artifact, a complete Phase-9 pricing artifact **header**, **and**
> `player_prop_prices` row count `== 0` for that run.

A missing pricing artifact header is **never** `MODEL_ONLY` — it is a
pricing-failure `PARTIAL`. Bet365 absence, zero total quotes, or any single
book's absence do not change this: no sportsbook is required for
`MODEL_ONLY` (§22).

---

## 20. Failure matrix (fail closed)

| Situation | `status` | `publication_status` | projections | thresholds | canonical pricing | legacy mirror |
|---|---|---|---|---|---|---|
| no game modeled | `FAILED` | `NOT_PUBLISHED` (`GAME_NOT_MODELED`) | none | none | none | not run |
| projection failure | `FAILED` | `NOT_PUBLISHED` | none persisted | none | none | not run |
| threshold failure | `PARTIAL` | `NOT_PUBLISHED` (`THRESHOLD_ERROR`) | retained | none (atomic) | none | not run |
| pricing **calculation** failure | `PARTIAL` | `NOT_PUBLISHED` (`PREDICTION_ERROR` family) | retained | retained | none | not run |
| canonical pricing **persistence** failure | `PARTIAL` | `NOT_PUBLISHED` (`PRICING_PERSISTENCE_ERROR`) | retained | retained | none (Phase-9C atomic) | not run |
| legacy mirror failure | `PARTIAL` | `NOT_PUBLISHED` (`LEGACY_PRICING_MIRROR_ERROR`) | retained | retained | **retained** | not written |
| all succeed, zero priced rows | `SUCCESS` | `MODEL_ONLY` | complete | complete | complete, `row_count=0` | mirrored |
| all succeed, priced rows exist | `SUCCESS` | `PUBLISHED` | complete | complete | complete, `row_count>0` | mirrored |

Certified end to end in
`test_phase9d_checkpoint_pricing_integration.py` (calculation/persistence/
mirror failure injection, each with a clean recovering retry) and
`test_phase8d_checkpoint_threshold_integration.py` (threshold/no-game-modeled
cases, unchanged since Phase 8).

---

## 21. Zero-quote end-to-end certification

Through the real official checkpoint path (`game_checkpoint_flow`), a
zero-quote run produces: one simulation, one pricing calculation, a
complete `E*30` projection artifact, a complete `E*131` threshold artifact,
a `player_prop_pricing_artifacts` header with `row_count = 0` and the
canonical empty scientific hash, zero `player_prop_prices` rows, and
`SUCCESS`/`MODEL_ONLY`. A retry is an exact no-op: the same empty artifact,
the same hash, the original `created_at` retained.

---

## 22. Multi-book / provider independence

`latest_prop_quotes` selects the latest-known quote **per vendor**
(`(game_id, player_id, prop_type, vendor)`), so the same player/prop quoted
by several books produces one **independent** pricing row per book/side —
never a best-price collapse, never a consensus collapse, and no book (not
even Bet365) is required. Reordering quote rows, or a book being wholly
absent, does not change any other book's row, and never changes the
Phase-7/8 model artifacts (which are quote-independent by construction).

---

## 23. PIT, catch-up, and reschedule

Unchanged from Phase 5: `quote_available_at <= scheduled_as_of` is enforced
both by the upstream pricing layer (`current_pricing._price_quote`) and,
independently, by the persistence layer
(`PricingFutureQuoteError` in `pricing_store.py`) — persistence never
assumes it is the only PIT guard. A catch-up execution for the same
`scheduled_as_of` produces a scientifically identical pricing artifact
(same hash, same `prediction_id` set) regardless of real wall-clock catch-up
time. A kickoff reschedule produces a new parent run identity with its own,
disjoint pricing history; the original run's artifact and rows are never
migrated, relabeled, or mutated.

---

## 24. Retry / conflict certification

Exact scientific retry (any row order, any `created_at`) is an idempotent
no-op at both the row and artifact level: no duplicate rows, the same
`prediction_id` set, the same `scientific_content_sha256`, the original
`created_at` retained on both tables. Any genuine scientific difference —
at the individual-row level or the whole-artifact level — is a hard
conflict error (`PricingRowConflictError` / `PricingArtifactConflictError`)
raised before anything is written; the previously-stored artifact and rows
are left byte-for-byte unchanged.

---

## 25. Legacy mirror status

`predictions` (Warehouse/Parquet) remains a **compatibility output only**
after Phase 9 — it is not deleted, and it is not the canonical scientific
record. The canonical scientific checkpoint pricing record is
`player_prop_pricing_artifacts` + `player_prop_prices`. The legacy mirror's
own idempotency (natural-key append) is unchanged and is not the
immutability authority; Phase 9C/9D's guarantees are.

---

## 26. Deferred pricing products

Explicitly **not** part of certified Phase 9, regardless of whether stub
code or a `raise NotImplementedError` already exists for it:

* generic count-style sportsbook milestones (`2+`/`3+` on non-binary props)
* power devig, Shin devig, one-sided field devig (`first_td`), borrowed
  overround
* robust multi-book player-prop consensus, freshness weighting, outlier
  removal
* best executable price selection
* opportunity ranking, confidence scoring
* Kelly sizing / bankroll-fraction recommendation (the helper
  `nflprops.market.odds.kelly_fraction` exists and is push-aware, but is not
  wired into any pricing output and is not certified for use)

---

## 27. Production-automation boundary

Phase 9 also does not implement, and none of the above certification covers:
daily data-refresh scheduling, daily retraining, recalibration,
champion/challenger promotion, artifact deployment, a public API, a
dashboard, or WizardOfOdds publication. These are the immediate
post-Phase-9 production program, built on top of — never inside — this
certified pricing product.

---

## Certification test map

| Concern | Test module |
|---|---|
| fair-pricing math (helpers), boundary/degenerate matrix, EV lock | `tests/unit/test_fair_pricing.py`, `tests/unit/test_odds_conversion.py` |
| integration pricing frame, binary milestone, fail-closed unsupported/unknown market | `tests/simulation_pricing/`, `tests/orchestration/test_pricing_store.py` |
| immutable/idempotent persistence, parent provenance, PIT, scientific hash, conflict/atomicity | `tests/orchestration/test_pricing_store.py`, `test_pricing_store_postgres.py` |
| official-checkpoint integration, one-sim/one-price, failure matrix, retry, catch-up, reschedule, multi-book | `tests/orchestration/test_phase9d_checkpoint_pricing_integration.py` |
| explicit publication-eligibility runtime gate | `test_phase9d_checkpoint_pricing_integration.py` (forcing + structural AST tests) |
| final end-to-end certification | `tests/orchestration/test_phase9e_pricing_certification.py` |
