# `player_game_threshold_events` — the canonical raw threshold / milestone probability product

**Status:** PHASE 8 certified (8A read-only audit, 8B in-memory engine + versioned
catalog, 8C immutable persistence + schema, 8D official-checkpoint integration, 8E
certification + product contract). Fair odds / push-adjusted pricing / devig / EV /
consensus / ranking / Kelly / API / dashboard / WizardOfOdds publishing are
**Phase 9+** and are explicitly **deferred** — see [Production boundary](#20-production-boundary).

This document is the normative product contract. It does not restate the
mathematics line-by-line; the executable sources of truth are
`contracts/threshold_catalog.yml`, `src/nflprops/thresholds/`,
`src/nflprops/orchestration/threshold_event_store.py`, and
`migrations/versions/0005_player_threshold_events.py`.

---

## 1. Purpose

`player_game_threshold_events` is the **canonical, sportsbook-independent, raw
model** answer to "how likely is player *X* to reach *at least* *T* of stat *S*
in this game?" for a fixed, versioned catalog of round, model-motivated
thresholds and count milestones.

It is produced from **exactly one coherent game simulation**
(`nflprops.simulation.game.simulate_game` → `GameSimulationResult`) — the *same*
`GameSimulationResult` object that `player_game_projections` (Phase 7) and
current-market pricing (Phase 6) read in the same checkpoint. There is no second
simulation, no threshold-specific simulation, and no quote-dependent simulation.

It carries **only** `p_hit`. It has **no** American odds, no push probability, no
devig, no fair value, no EV, no vendor, no sportsbook line or price.

---

## 2. Player universe

The threshold product covers **every Phase-7 eligible real player for the game —
no more, no less, and with no position filtering.**

Eligibility is the certified Phase-7A rule, reused unchanged
(`nflprops.projections.eligible_player_states`); Phase 8 does not re-derive or
re-interpret it. A player is eligible iff, decided purely from pre-simulation
`PlayerState`:

```
player.active
AND (
    target_share > 0
    OR rush_share  > 0
    OR player is the simulator-selected starting QB
    OR player is the simulator-selected starting K
)
```

`E` in this document is the number of eligible players for the game/checkpoint.
Eligibility is pre-simulation and sportsbook-independent by construction, so the
threshold player universe is **identical** to the projection player universe for
the same run (certified end-to-end in
`tests/orchestration/test_phase8e_threshold_certification.py`).

---

## 3. `E × 131` completeness rule (LOCKED)

For every **successfully modeled** official game/checkpoint the persisted
threshold-row count is **exactly `E × 131`**: every eligible player crossed with
every one of the 131 canonical catalog events.

* No position-based omission. A kicker still receives the full `passing_yards`
  ladder; a receiver still receives the full `fg_made` ladder.
* A legitimate all-zero distribution is **not** dropped — it stays a row with
  `p_hit = 0.0`.
* The in-memory builder raises before returning if the frame is not exactly
  `E × 131` (`nflprops.thresholds.build.ThresholdEventError`), and the Phase-8C
  persistence layer independently re-checks completeness against the run's
  Phase-7 `player_game_projections` artifact before writing a single row
  (`ThresholdArtifactIncompleteError`). A partial canonical artifact is never
  stored.

Storage-scale consequences are in [§18](#18-storage-scale). They must never
motivate row omission.

---

## 4. The 23 catalog stat series

Source of truth: `contracts/threshold_catalog.yml`, mirrored byte-for-byte to
`src/nflprops/resources/contracts/threshold_catalog.yml`
(`python tools/sync_runtime_resources.py --check`).

**Canonical wording:**

> **22 Phase-7 registry-backed threshold stat series + 1 catalog-derived stat
> series (`offensive_tds`) = 23 total Phase-8 catalog stat series.**

Do not describe this as "22 catalog stats". The catalog covers 23 stat *series*;
22 of them are backed one-to-one by a Phase-7 registry stat, and 1
(`offensive_tds`) is a catalog-derived series computed from Phase-7 registry
columns.

### 4.1 Event counts

| | events |
|---|---:|
| STANDARD threshold events (`classification: STANDARD_THRESHOLD_ELIGIBLE`, 16 series) | **111** |
| MILESTONE threshold events (`classification: MILESTONE_ONLY`, 7 series) | **20** |
| **total canonical events** | **131** |

`expected_event_count: 131` in the catalog is enforced at load
(`ThresholdCatalogError` if the ladders don't sum to it).

### 4.2 Registry classification (of the 30 Phase-7 registry stats)

| group | count | notes |
|---|---:|---|
| STANDARD registry stats with a Phase-8 ladder | 16 | `passing_yards, passing_completions, passing_attempts, rushing_yards, rush_attempts, receiving_yards, receptions, targets, longest_reception, longest_rush, longest_pass, rushing_receiving_yards, kicking_points, passing_yards_1h, receiving_yards_1h, rushing_yards_1h` |
| MILESTONE registry stats with a Phase-8 ladder | 6 | `passing_tds, interceptions, fg_made, fg_attempts, passing_tds_1h, fg_made_1h` |
| binary Phase-7 stats **not** duplicated in Phase 8 | 5 | `anytime_td, anytime_td_1q, anytime_td_1h, anytime_td_2h, first_td` — see [§8](#8-binary-event-non-duplication) |
| registry stats with **no** independent Phase-8 ladder | 3 | `receiving_tds, rushing_tds` (components of `anytime_td` / `offensive_tds`), `xp_made` (component of `kicking_points`) |
| **total** | **30** | |

`16 + 6 = 22` registry-backed catalog series; `+ offensive_tds` (derived) `= 23`.

### 4.3 The ladders

The exact `values:` lists are in `contracts/threshold_catalog.yml` and must not
be edited outside a deliberate, versioned catalog revision. They are round,
support-spanning, model-motivated **positive integers** chosen from the realistic
NFL player range and public vernacular — never copied from a sportsbook alt-line
menu, a quote count, or a vendor.

---

## 5. `AT_LEAST` semantics and the exact `>=` definition

`event_type` is exactly `AT_LEAST` — the only Phase-8 event type
(`ck_player_game_threshold_events_event_type` enforces it at the database level).

For draw vector `v` (length `n_draws`) and integer threshold `T`:

```
hit(v_i)  =  v_i >= T          # AT_LEAST: value == threshold IS a hit
p_hit     =  count(v_i >= T) / n_draws
```

There is **no push**: an `AT_LEAST` event is a two-outcome model event
(reached / did not reach). `value == threshold` counts as a hit.

---

## 6. Very important semantic distinction: "milestone" here ≠ `MarketType.MILESTONE`

The Phase-8 term **milestone** (`classification: MILESTONE_ONLY`, and the
"MILESTONE threshold events" count) means only this: **a model event
`stat >= threshold` over a small natural count ladder (e.g. `fg_made` in
`{1, 2, 3, 4}`), carrying a raw binary probability `p_hit`.**

It is **not** the same concept as the existing sportsbook
`nflprops.domain.enums.MarketType.MILESTONE` settlement / pricing semantics.

A Phase-8 threshold/milestone event has:

* **no push**
* **no sportsbook price**
* **no devig / no vig removal**
* **no fair odds**
* **no expected value**

Those all belong to later pricing work ([§20](#20-production-boundary)). Phase 8
produces the raw model probability and nothing else.

---

## 7. `offensive_tds` semantics (LOCKED)

`offensive_tds` is the one **catalog-derived** stat series. It is **not** one of
the 30 Phase-7 registry stats; it is computed elementwise, draw-by-draw, on the
**same** `GameSimulationResult`, as the sum of two declared Phase-7 registry
columns:

```
offensive_tds(draw_i)  =  receiving_tds(draw_i) + rushing_tds(draw_i)
```

* **`passing_tds` is never included.** A quarterback throwing a touchdown is the
  *receiver's* offensive TD, not the passer's.
* No new distribution, no resampling, no RNG, no fitted approximation.
* Its ladder is **exactly `{2, 3}`** (`2+`, `3+`). `1+` is intentionally absent
  because the Phase-7 `anytime_td` binary distribution already represents
  `receiving_tds + rushing_tds >= 1`. Re-storing it here would duplicate an
  existing Phase-7 product.

Certified in `tests/thresholds/test_build.py`
(`test_offensive_tds_is_receiving_plus_rushing_only_excludes_passing_tds`,
`test_offensive_tds_matches_receiving_plus_rushing_registry_vectors`) and
end-to-end in `test_phase8e_threshold_certification.py`
(`test_offensive_tds_excludes_passing_tds_end_to_end`).

---

## 8. Binary-event non-duplication

Phase 8 does **not** persist threshold rows for these five binary Phase-7 events:

```
anytime_td      anytime_td_1q      anytime_td_1h      anytime_td_2h      first_td
```

Their authoritative model probability remains the **mean of their Phase-7 binary
draw vector**, stored as `player_game_projections.mean`. They are not in
`contracts/threshold_catalog.yml`, and `SELECT ... FROM
player_game_threshold_events WHERE stat_name IN (...)` returns zero rows for a
certified run (`test_binary_phase7_events_have_no_threshold_rows`).

---

## 9. Persistence grain, deterministic ID, columns

One row per **`(run_id, player_id, stat_name, event_type, threshold)`**.

```
threshold_event_id = SHA-256(
    run_id | player_id | stat_name | event_type | threshold
)
```

via the shared deterministic SHA helper
(`nflprops.collection.resource_availability.deterministic_id`, the same one
`compute_projection_id` / `compute_run_id` use) — **never** Python's built-in
`hash()`. Same pipe-joined scheme, `int(threshold)` normalized.

Stored columns: `threshold_event_id, run_id, season, week, game_id, player_id,
team_id, position_group, stat_name, event_type, threshold, p_hit, n_draws,
catalog_version, created_at`.

**Scientific identity / equality** = `run_id, season, week, game_id, player_id,
team_id, position_group, stat_name, event_type, threshold, p_hit, n_draws,
catalog_version`. `p_hit` is compared **exactly** — never tolerance-based.
`created_at` is operational metadata only and is **excluded** from scientific
equality.

`p_miss` is **derivable** as `1 - p_hit` and is **never stored**.

---

## 10. Scientific immutability & idempotency (Phase 8C — LOCKED)

`nflprops.orchestration.threshold_event_store.persist_player_game_threshold_events`
is the **single** persistence entry point.

* **Idempotent retry:** same `threshold_event_id` + identical scientific fields
  (any `created_at`) → no-op; the stored row, including its original
  `created_at`, is left exactly as it was.
* **Immutable conflict:** same `threshold_event_id` + any differing scientific
  field → hard `ThresholdEventConflictError`; nothing is written.
* **Atomic batch:** every validation (schema, values, catalog, parent
  provenance, completeness, conflict) runs **before** any insert. One bad row
  aborts the whole call — no partial write. Local backend uses
  `key=["threshold_event_id"], keep="first"`; PostgreSQL uses `INSERT ... ON
  CONFLICT DO NOTHING` as a backstop behind the application-layer conflict
  check.
* **Repeated retry row count stays `E × 131`** — never `2 × E × 131`
  (`test_official_retry_is_fully_idempotent_no_duplicate_threshold_rows`).

Boundary probabilities `p_hit = 0.0` and `p_hit = 1.0` are valid canonical
values and persist **exactly** — never clipped, never repaired
(`ck_player_game_threshold_events_p_hit_unit_interval` allows the closed
interval; `test_zero_and_one_boundary_probabilities_persist_exactly`).

---

## 11. Parent-run provenance (Phase 8C — strengthens Phase 7C)

Before any row is written, the one `prediction_runs` row for `run_id` is loaded
exactly once (`run_store.get_run`), and every incoming row's `season, week,
game_id, n_draws` must **equal** the parent's. A disagreement is a hard
`ThresholdEventProvenanceError` — the parent run is authoritative, the child
value is **never** silently reconciled, and nothing (not even the agreeing
subset) is written.

`run_id` must reference a real `prediction_runs` row that **also** already has a
complete Phase-7 `player_game_projections` artifact; that artifact is the
authoritative eligible-player set for the completeness check. With no projection
artifact, persistence fails closed.

`n_draws` on every persisted threshold row equals the parent run's `n_draws`, and
`p_hit` is computed over **all** `simulation.n_draws` draws — never a retained
sub-sample.

---

## 12. Checkpoint execution order (Phase 8D)

Within one official (or `MANUAL`) checkpoint execution
(`nflprops.orchestration.flows.checkpoints`):

```
claim prediction_run
      -> scheduled_as_of PIT manifest (data_manifest_sha256)   [Phase 5]
      -> build football state
      -> ONE coherent game simulation  (compute_game_prediction)  [Phase 6/7]
             |
             +-- 1. build player_game_projections     (from that GameSimulationResult)
             |   2. validate rows == E * 30, else FAIL before anything downstream
             |   3. persist player_game_projections   (Phase 7C, immutable/idempotent)
             |
             +-- 4. build player_game_threshold_events (SAME GameSimulationResult,
             |                                          SAME player_states, no quotes)
             |   5. persist player_game_threshold_events (Phase 8C, immutable/idempotent,
             |                                            E * 131 completeness enforced)
             |
             +-- 6. price current sportsbook markets   (SAME GameSimulationResult)
                 7. persist predictions + retained joint draws
      -> 8. terminal prediction_run status
```

* **Exactly one** football simulation per game/checkpoint.
  `build_player_game_projections`, `build_player_game_threshold_events`, and
  `price_current_markets` receive the **same** `GameSimulationResult` instance —
  object-identity certified
  (`test_exactly_one_simulation_feeds_projections_thresholds_and_pricing`).
* **Current-market pricing is never persisted before both complete canonical
  model artifacts exist.** Projections persist before thresholds; thresholds
  persist before any current price.
* `run_id / season / week` for persistence come from the official run context;
  `game_id / n_draws` from the simulation must agree with the parent (Phase 8C
  provenance gate).
* The Phase-5 `data_manifest_sha256` (hash of PIT *inputs*) and the Phase-6
  `simulation_input_sha256` (quote-independent) are **unchanged** by threshold
  persistence — threshold events are an *output*.

### Official checkpoints covered

`T48H`, `T24H`, `T6H`, `T90M`, `T30M`, `MANUAL`. Phase-5 semantics are preserved
unchanged:

* `scheduled_as_of = kickoff_at − checkpoint_offset`; the actual worker start
  time never replaces knowledge time.
* A catch-up run **before kickoff** uses the original scheduled cutoff and
  produces byte-identical canonical artifacts
  ([§16](#16-catch-up-consistency)).
* A checkpoint first discovered **at/after kickoff** retains the existing
  fail-closed behaviour (`CHECKPOINT_MISSED`).
* Schedule-revision identity behaviour is unchanged
  ([§17](#17-kickoff-reschedule)).

**Phase 8 introduces no use of the current wall clock into model knowledge.**

---

## 13. `MODEL_ONLY` semantics (Phase 8 certification invariant)

> **`publication_status == MODEL_ONLY`  ⇔  a successful modeled run that has
> BOTH a complete Phase-7 `E × 30` projection artifact AND a complete Phase-8
> `E × 131` threshold artifact, and zero executable current-market price rows.**

* **Zero sportsbook quotes** after a valid model → one simulation, complete
  `E × 30` projections persisted, complete `E × 131` threshold events persisted,
  zero priced rows, `SUCCESS / MODEL_ONLY`. Valid and preserved.
* **Bet365 absent** (or any single book absent, or many books present) does not
  change this — no sportsbook is required for `MODEL_ONLY`
  ([§15](#15-sportsbook-independence)).
* A **missing or incomplete threshold artifact must never produce
  `MODEL_ONLY`.** If the threshold build or persist step fails, the run is
  `PARTIAL / NOT_PUBLISHED` ([§14](#14-failure-semantics)).
* **No usable game model** (execution reached the model path but no coherent
  `GameSimulationResult`) → `FAILED / NOT_PUBLISHED / GAME_NOT_MODELED`, with
  **no** projection artifact, **no** threshold artifact, **no** pricing. Never
  `MODEL_ONLY`.

---

## 14. Failure semantics (Phase 8D — fail closed)

Reuses the existing `PredictionRunStatus` / `PublicationStatus` enums and the
free-text `failure_code` / `failure_detail` columns. **No new run-status field
and no schema migration were introduced for the threshold failure distinction**
(the threshold-persist failure re-uses `failure_code = "THRESHOLD_ERROR"`).

| Situation | `status` | `publication_status` | model artifacts | pricing |
|---|---|---|---|---|
| no game modeled | `FAILED` | `NOT_PUBLISHED` (`GAME_NOT_MODELED`) | none | none |
| simulation / input failure | `FAILED` | `NOT_PUBLISHED` | none | none |
| projection build / validation / provenance / persistence failure | `FAILED` | `NOT_PUBLISHED` | none persisted | **not run** |
| threshold **build** failure (after projections persisted) | `PARTIAL` | `NOT_PUBLISHED` (`THRESHOLD_ERROR`) | complete `E × 30` projections **retained** | **not run** |
| threshold **persistence** failure (after projections persisted) | `PARTIAL` | `NOT_PUBLISHED` (`THRESHOLD_ERROR`) | complete projections retained; **no partial threshold artifact** (atomic) | **not run** |
| pricing failure (after both canonical artifacts persisted) | `PARTIAL` | `NOT_PUBLISHED` | complete projections **and** complete `E × 131` threshold artifact both retained | attempted, not marked published |
| projections + thresholds persisted, pricing produced zero rows normally | `SUCCESS` | `MODEL_ONLY` | both complete | zero prices (valid) |
| projections + thresholds persisted, pricing produced rows | `SUCCESS` | `PUBLISHED` | both complete | priced |

A retry after a threshold or pricing failure completes cleanly and does **not**
duplicate the already-persisted projection rows
(`test_retry_after_threshold_failure_completes_cleanly_without_duplication`,
`test_pricing_failure_after_both_artifacts_is_partial_and_retainable`).

---

## 15. Publication-eligibility invariant (for later production)

Phase 8 itself does not publish anywhere. It does enforce the internal invariant
later pricing/publishing phases depend on:

> A run cannot be publication-eligible unless a **complete Phase-7 projection
> artifact** AND a **complete Phase-8 threshold artifact** both exist for it.

Current-market pricing may legitimately be absent only for a `MODEL_ONLY`
zero-quote run. A **partial** threshold artifact never qualifies — and by the
atomic-persist rule a partial threshold artifact is never stored in the first
place.

---

## 16. Sportsbook independence

Using the same PIT / model inputs, the threshold scientific artifact is
**identical** across: zero quotes, Bet365-only, non-Bet365-only, many books, and
reordered quote rows. Identical means: same eligible player universe, same row
count, same `p_hit` values, same `threshold_event_id`s, same scientific columns.

Only the downstream current-pricing artifact (`predictions`) may differ.

Certified: `tests/thresholds/test_build.py`
(`test_quote_independence_same_simulation_same_frame`,
`test_bet365_presence_or_absence_cannot_matter`,
`test_no_quote_or_vendor_parameter_and_no_market_import`) and
`tests/orchestration/test_phase8d_checkpoint_threshold_integration.py`
(`test_bet365_absence_and_book_order_do_not_change_threshold_artifact`,
`test_zero_books_still_full_threshold_artifact`).

---

## 17. Catch-up consistency

A scheduled checkpoint execution and a later catch-up execution **for the same
`scheduled_as_of`** select the same PIT inputs / manifest and produce the
**byte-identical** threshold scientific frame and the identical
`threshold_event_id` set. Phase 8 introduces no current-wall-clock read into
model knowledge (`test_catch_up_execution_produces_identical_threshold_artifact`).

---

## 18. Kickoff reschedule

Phase-5 schedule-revision behaviour is preserved. A revised kickoff produces a
new checkpoint identity and a new `run_id` under the existing revision rules →
new `threshold_event_id`s. The prior revision's `prediction_runs` row and its
threshold rows remain, immutable, attached to their **own** parent run; they are
**never** migrated or relabelled onto the new run. Both histories coexist in the
canonical table
(`test_kickoff_reschedule_gives_distinct_threshold_history`).

---

## 19. Numerical acceptance (Phase 8E)

For every persisted row, independently recomputed from the shared simulation
draws:

```
p_hit  ==  count(source_vector >= threshold) / n_draws        # exact, no tolerance
```

where `source_vector` is the Phase-7 registry extractor output for the stat, or —
for `offensive_tds` — the elementwise sum of the `receiving_tds` and
`rushing_tds` registry vectors on the same simulation.

Exact hand-recomputed proofs run for a representative event in every canonical
category — a passing-yards threshold, a rushing-yards threshold, a
receiving-yards threshold, a passing-TD milestone, a field-goal milestone, a
first-half (`passing_yards_1h`) threshold, and an `offensive_tds` milestone
(`test_persisted_p_hit_is_exact_empirical_frequency`).

**Monotonicity** over every complete ladder: for a fixed `(player_id,
stat_name)`, `T1 < T2  ⇒  p_hit(T1) >= p_hit(T2)`. Enforced in the builder
(`ThresholdEventError` on violation) and re-checked from persisted rows for all
`E × 23` ladders
(`test_every_persisted_ladder_is_monotone_non_increasing`).

**Boundary:** `p_hit == 0.0` (an honest all-zero distribution, e.g. a kicker's
`passing_yards` ladder) and `p_hit == 1.0` (a lead back clearing the low
`rush_attempts` rungs on every draw) both persist exactly; every stored `p_hit`
is an exact `k / n_draws` frequency that re-divides to the stored value
bit-for-bit.

---

## 20. Production boundary

Phase 8 produces canonical **raw model probabilities** only. It explicitly does
**not** do any of the following — they remain later phases:

* sportsbook fair odds
* push-adjusted / executable probabilities
* devigged consensus / best-price
* expected value
* opportunity ranking / confidence scores
* Kelly sizing
* public API
* dashboard
* WizardOfOdds / SportsOdds publishing
* daily model retraining / recalibration

`player_game_threshold_events` is the sportsbook-independent model artifact those
phases build on; it does not itself price anything.

---

## 21. Database contract (migration `0005_player_threshold_events` — certified, no Phase-8E migration)

`player_game_threshold_events`:

* **PK** `threshold_event_id` (Text).
* **FK** `fk_player_game_threshold_events_run_id`: `run_id → prediction_runs.run_id`
  — intentionally **not** `ON DELETE CASCADE` (threshold outputs outlive a run
  row's lifecycle bookkeeping, exactly like `player_game_projections`).
* **Scientific uniqueness** `uq_player_game_threshold_events_identity`
  `(run_id, player_id, stat_name, event_type, threshold)`.
* **CHECK** constraints:
  `ck_..._threshold_positive` (`threshold > 0`),
  `ck_..._n_draws_positive` (`n_draws > 0`),
  `ck_..._p_hit_unit_interval` (`p_hit >= 0 AND p_hit <= 1` — closed interval, so
  `0.0` and `1.0` are legal),
  `ck_..._event_type` (`event_type = 'AT_LEAST'`).
* **8 indexes:** `run_id`; `(run_id, player_id)`; `game_id`; `player_id`;
  `stat_name`; `(stat_name, threshold)`; `(season, week)`;
  `(season, week, game_id)`.
* `downgrade()` drops the 8 indexes then the table; `upgrade` / `downgrade`
  round-trip and local ⇄ PostgreSQL parity are certified in
  `tests/orchestration/test_threshold_event_persistence_postgres.py` and
  `tests/orchestration/test_threshold_event_store.py`.

The revision id is `0005_player_threshold_events` (28 chars) — kept ≤ 32 for
Alembic's default `alembic_version.version_num VARCHAR(32)`, the same constraint
noted on `0002`–`0004`.

---

## 22. Storage scale

Canonical rows per game per checkpoint = **`E × 131`**. This is a deliberate
product property; it must **not** be reduced by position-based row omission.

| `E` | rows / game / checkpoint | rows / game across 5 checkpoints | rows / 16-game slate (5 checkpoints) |
|---:|---:|---:|---:|
| 20 | `20 × 131` = **2,620** | `2,620 × 5` = **13,100** | `13,100 × 16` = **209,600** |
| 40 | `40 × 131` = **5,240** | `5,240 × 5` = **26,200** | `26,200 × 16` = **419,200** |

Roughly 0.2–0.42 M rows per slate at 5 official checkpoints. The immutable,
idempotent persistence layer stores each `(run_id, …)` grid once; retries and
catch-up runs for the same `run_id` add nothing.

---

## 23. Known type-debt note (non-blocking, not a Phase-8 scientific defect)

`src/nflprops/orchestration/flows/checkpoints.py` passes a `Warehouse` where the
persistence helpers annotate `StorageBackend`. `mypy --strict` reports a
structural method-signature incompatibility for this (the `Warehouse.append` /
`Warehouse.write` methods return `Path`, the `StorageBackend` protocol declares
`-> None`). This is a **pre-existing repository-wide** typing shape issue, not
introduced by Phase 8 and not a scientific defect. Phase 8E deliberately does
**not** undertake the structural typing refactor to resolve it. New Phase-8
modules (`nflprops.thresholds.*`,
`nflprops.orchestration.threshold_event_store`) are `--strict`-clean; the only
new instances of the pattern in `checkpoints.py` are the two added call sites
(`persist_player_game_threshold_events(warehouse, …)` and the extra
`update_run_status(warehouse, …)` in the `THRESHOLD_ERROR` branch), which reuse
the established shape. Tracked for a future dedicated `Warehouse` / `StorageBackend`
protocol-alignment pass.

---

## Certification test map

| Concern | Test module |
|---|---|
| in-memory engine, catalog, monotonicity, `offensive_tds`, binary non-dup, quote independence, one-sim | `tests/thresholds/test_build.py`, `tests/thresholds/test_catalog.py` |
| immutable / idempotent persistence, parent provenance, `E × 131` completeness, catalog validation | `tests/orchestration/test_threshold_event_store.py` |
| schema, CHECK / UNIQUE / FK, indexes, upgrade / downgrade, local ⇄ PostgreSQL parity | `tests/orchestration/test_threshold_event_persistence_postgres.py` |
| official-checkpoint integration, one simulation, failure matrix, retry, catch-up, reschedule, Bet365 independence, dispatcher e2e | `tests/orchestration/test_phase8d_checkpoint_threshold_integration.py` |
| end-to-end artifact grid / universe, exact numerical proofs (5 categories), monotonicity from persisted rows, `0.0` / `1.0` boundary, catalog shape, `MODEL_ONLY` | `tests/orchestration/test_phase8e_threshold_certification.py` |
