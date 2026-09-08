# `player_game_projections` — the canonical sportsbook-independent player forecast product

**Status:** PHASE 7 certified (7A eligibility, 7B in-memory engine, 7C immutable
persistence + parent provenance, 7D official-checkpoint integration, 7E
certification). Phase 8 (threshold ladders / milestone pricing / odds) is
explicitly **deferred** — see [Phase-8 deferral](#phase-8-deferral).

---

## 1. What this table is

`player_game_projections` is the **canonical, summarized, sportsbook-independent
player-distribution product** for one official game/checkpoint. One row per
`(run_id, player_id, stat_name)`; `projection_id = SHA-256(run_id | player_id |
stat_name)`.

It is produced from **exactly one coherent game simulation**
(`nflprops.simulation.game.simulate_game` → `GameSimulationResult`) — the *same*
`GameSimulationResult` object that current-market pricing reads. It carries **no**
sportsbook column (no line, price, vendor, devig, EV, consensus) and **no**
separate median column (`p50` *is* the median).

### `simulation_player_results` vs `player_game_projections`

| Table | Grain | Content | Role |
|---|---|---|---|
| `simulation_player_results` | one row per `(run_id, draw_id, player_id)` | retained **draw-level** coherent simulation output (the first `retained_joint_draws` joint draws) | raw reproducibility / joint-analysis artifact; a *bounded sample* of the draws |
| `player_game_projections` | one row per `(run_id, player_id, stat_name)` | **summarized** per-stat distribution (`mean` + 7 locked percentiles) over **all** `simulation.n_draws` | the canonical sportsbook-independent player forecast product |

`player_game_projections` is **never** summarized from the retained joint subset —
always from the full `simulation.n_draws` (see [Full-draw behavior](#7-full-draw-behavior)).

---

## 2. Player eligibility (Phase 7A — LOCKED)

A real player is eligible for the projection product for a game iff, decided
**purely from pre-simulation `PlayerState`** (never realized draws, never quotes,
vendors, popularity, or position heuristics beyond the existing QB/K allocation):

```
player.active
AND (
    target_share > 0
    OR rush_share  > 0
    OR player is the simulator-selected starting QB
    OR player is the simulator-selected starting K
)
```

Certified inclusions / exclusions (`tests/projections/test_eligibility.py`,
`tests/projections/test_starter_selection.py`):

| Case | Eligible? |
|---|---|
| simulator-selected starting QB (even with zero target/rush share) | **yes** |
| simulator-selected starting K (even with zero target/rush share) | **yes** |
| positive `target_share` player | **yes** |
| positive `rush_share` player | **yes** |
| active backup / rotational player with positive modeled opportunity | **yes** |
| roster-only player, active, zero modeled opportunity | **no** |
| synthetic `__OTHER__` / `__QB__` filler | **no** |
| inactive player | **no** |

`E` in this document is the number of eligible players for a game/checkpoint.
Eligibility is pre-simulation and sportsbook-independent by construction.

---

## 3. The 30-stat registry (Phase 7A/7B — FROZEN)

Exactly **20 native + 10 derived = 30** entries, registry order = natives then
deriveds. Source of truth: `nflprops.projections.stats.REGISTRY`
(`tests/projections/test_registry.py`).

* **native** — a real per-draw column of `GameSimulationResult.player_draws`,
  read straight through `nflprops.simulation.results.player_distribution`. No
  reinterpretation.
* **derived** — a distribution the current pricing path already computes from the
  same draws via `nflprops.simulation.props.prop_values`. There is exactly one
  definition of each `PropType` distribution, shared with current-market pricing.
  The `anytime_td*` family entries are the `>= 1` binary (identical to the
  `p_hit` threshold `props.summarize_prop` applies); `first_td` is already the
  per-player 0/1 indicator `props.prop_values` returns.

| # | stat_name | display_name | category | unit | source_kind | source column / formula | associated PropType(s) |
|---:|---|---|---|---|---|---|---|
| 1 | `targets` | Targets | receiving | count | native | `player_draws.targets` | — (modeled opportunity input) |
| 2 | `receptions` | Receptions | receiving | count | native | `player_draws.receptions` | `RECEPTIONS` |
| 3 | `receiving_yards` | Receiving Yards | receiving | yards | native | `player_draws.receiving_yards` | `RECEIVING_YARDS` |
| 4 | `receiving_tds` | Receiving TDs | receiving | count | native | `player_draws.receiving_tds` | — (component of `ANYTIME_TD`) |
| 5 | `longest_reception` | Longest Reception | receiving | yards | native | `player_draws.longest_reception` | `LONGEST_RECEPTION` |
| 6 | `rush_attempts` | Rush Attempts | rushing | count | native | `player_draws.rush_attempts` | `RUSHING_ATTEMPTS` |
| 7 | `rushing_yards` | Rushing Yards | rushing | yards | native | `player_draws.rushing_yards` | `RUSHING_YARDS` |
| 8 | `rushing_tds` | Rushing TDs | rushing | count | native | `player_draws.rushing_tds` | — (component of `ANYTIME_TD`) |
| 9 | `longest_rush` | Longest Rush | rushing | yards | native | `player_draws.longest_rush` | `LONGEST_RUSH` |
| 10 | `passing_attempts` | Passing Attempts | passing | count | native | `player_draws.passing_attempts` | `PASSING_ATTEMPTS` |
| 11 | `passing_completions` | Passing Completions | passing | count | native | `player_draws.passing_completions` | `PASSING_COMPLETIONS` |
| 12 | `passing_yards` | Passing Yards | passing | yards | native | `player_draws.passing_yards` | `PASSING_YARDS` |
| 13 | `passing_tds` | Passing TDs | passing | count | native | `player_draws.passing_tds` | `PASSING_TDS` |
| 14 | `interceptions` | Interceptions Thrown | passing | count | native | `player_draws.interceptions` | `INTERCEPTIONS` |
| 15 | `longest_pass` | Longest Completion | passing | yards | native | `player_draws.longest_pass` | `LONGEST_PASS` |
| 16 | `fg_attempts` | FG Attempts | kicking | count | native | `player_draws.fg_attempts` | — |
| 17 | `fg_made` | FG Made | kicking | count | native | `player_draws.fg_made` | `FG_MADE` |
| 18 | `xp_made` | XP Made | kicking | count | native | `player_draws.xp_made` | — (component of `KICKING_POINTS`) |
| 19 | `kicking_points` | Kicking Points | kicking | points | native | `player_draws.kicking_points` = `3*fg_made + xp_made` | `KICKING_POINTS` |
| 20 | `rushing_receiving_yards` | Rush + Rec Yards | combo | yards | native | `player_draws.rushing_receiving_yards` = `rushing_yards + receiving_yards` | `RUSHING_RECEIVING_YARDS` |
| 21 | `anytime_td` | Anytime TD | scoring | binary (0/1) | derived | `(prop_values(ANYTIME_TD) >= 1)` where `prop_values = receiving_tds + rushing_tds` | `ANYTIME_TD` |
| 22 | `passing_yards_1h` | Passing Yards (1H) | passing / half | yards | derived | `prop_values(PASSING_YARDS_1H)` = `q1_passing_yards + q2_passing_yards` | `PASSING_YARDS_1H` |
| 23 | `passing_tds_1h` | Passing TDs (1H) | passing / half | count | derived | `q1_passing_tds + q2_passing_tds` | `PASSING_TDS_1H` |
| 24 | `receiving_yards_1h` | Receiving Yards (1H) | receiving / half | yards | derived | `q1_receiving_yards + q2_receiving_yards` | `RECEIVING_YARDS_1H` |
| 25 | `rushing_yards_1h` | Rushing Yards (1H) | rushing / half | yards | derived | `q1_rushing_yards + q2_rushing_yards` | `RUSHING_YARDS_1H` |
| 26 | `fg_made_1h` | FG Made (1H) | kicking / half | count | derived | `q1_fg_made + q2_fg_made` | `FG_MADE_1H` |
| 27 | `anytime_td_1q` | Anytime TD (1Q) | scoring / period | binary (0/1) | derived | `((q1_receiving_tds + q1_rushing_tds) >= 1)` | `ANYTIME_TD_1Q` |
| 28 | `anytime_td_1h` | Anytime TD (1H) | scoring / half | binary (0/1) | derived | `((Σ q1..q2 receiving_tds+rushing_tds) >= 1)` | `ANYTIME_TD_1H` |
| 29 | `anytime_td_2h` | Anytime TD (2H) | scoring / half | binary (0/1) | derived | `((Σ q3..q4 (+ q5 OT) receiving_tds+rushing_tds) >= 1)` | `ANYTIME_TD_2H` |
| 30 | `first_td` | First TD Scorer | scoring | binary (0/1) | derived | `(GameSimulationResult.first_td_player == player_id)` | `FIRST_TD` |

`category` / `display_name` / `unit` above are the documented presentation
contract for a future public surface; they are **not** stored columns — the
persisted row carries only the scientific fields (§5).

### q1–q5 decomposition columns — INTERNAL, NON-PUBLIC

`GameSimulationResult.player_draws` also carries **35** per-period columns:

```
q{1,2,3,4,5}_receiving_yards   q{1,2,3,4,5}_rushing_yards
q{1,2,3,4,5}_receiving_tds     q{1,2,3,4,5}_rushing_tds
q{1,2,3,4,5}_passing_yards     q{1,2,3,4,5}_passing_tds
q{1,2,3,4,5}_fg_made
```

(7 stats × 5 periods, where `q5` is overtime.) They are **INTERNAL_NONPUBLIC**
draw-aligned inputs consumed only by `nflprops.simulation.props._sum_cols` to
build the half/period derived distributions (entries 22–29 above). They are
**never** emitted as `player_game_projections` rows and are not part of the
30-stat product.

---

## 4. PropType coverage (Phase 7E §6)

All **25** `nflprops.domain.enums.PropType` members are preserved: each resolves
to a coherent shared distribution and has a canonical entry in the 30-stat
registry. Executable proof: `tests/projections/test_phase7e_prop_coverage.py`
and `tests/simulation_pricing/test_projection_shared_distribution.py`.
Confidence-tier partition is unchanged: **9 / 6 / 10**.

| PropType | tier | pre-Phase-7 priced/representable | canonical Phase-7 distribution | native/derived | preserved |
|---|:--:|:--:|---|:--:|:--:|
| `PASSING_ATTEMPTS` | 1 | yes | `passing_attempts` | native | **YES** |
| `PASSING_COMPLETIONS` | 1 | yes | `passing_completions` | native | **YES** |
| `PASSING_YARDS` | 1 | yes | `passing_yards` | native | **YES** |
| `INTERCEPTIONS` | 1 | yes | `interceptions` | native | **YES** |
| `RUSHING_ATTEMPTS` | 1 | yes | `rush_attempts` | native | **YES** |
| `RUSHING_YARDS` | 1 | yes | `rushing_yards` | native | **YES** |
| `RECEPTIONS` | 1 | yes | `receptions` | native | **YES** |
| `RECEIVING_YARDS` | 1 | yes | `receiving_yards` | native | **YES** |
| `RUSHING_RECEIVING_YARDS` | 1 | yes | `rushing_receiving_yards` | native | **YES** |
| `PASSING_TDS` | 2 | yes | `passing_tds` | native | **YES** |
| `ANYTIME_TD` | 2 | yes | `anytime_td` (`>= 1` binary) | derived | **YES** |
| `KICKING_POINTS` | 2 | yes | `kicking_points` | native | **YES** |
| `FG_MADE` | 2 | yes | `fg_made` | native | **YES** |
| `LONGEST_RUSH` | 2 | yes | `longest_rush` | native | **YES** |
| `LONGEST_RECEPTION` | 2 | yes | `longest_reception` | native | **YES** |
| `PASSING_YARDS_1H` | 3 | yes | `passing_yards_1h` | derived | **YES** |
| `PASSING_TDS_1H` | 3 | yes | `passing_tds_1h` | derived | **YES** |
| `RECEIVING_YARDS_1H` | 3 | yes | `receiving_yards_1h` | derived | **YES** |
| `RUSHING_YARDS_1H` | 3 | yes | `rushing_yards_1h` | derived | **YES** |
| `FG_MADE_1H` | 3 | yes | `fg_made_1h` | derived | **YES** |
| `ANYTIME_TD_1H` | 3 | yes | `anytime_td_1h` | derived | **YES** |
| `ANYTIME_TD_2H` | 3 | yes | `anytime_td_2h` | derived | **YES** |
| `ANYTIME_TD_1Q` | 3 | yes | `anytime_td_1q` | derived | **YES** |
| `FIRST_TD` | 3 | yes | `first_td` | derived | **YES** |
| `LONGEST_PASS` | 3 | yes | `longest_pass` | native | **YES** |

---

## 5. Persisted scientific fields & percentiles

Stored columns: `projection_id, run_id, season, week, game_id, player_id,
team_id, position_group, stat_name, n_draws, mean, p05, p10, p25, p50, p75, p90,
p95, created_at`.

**Scientific identity** = `run_id, season, week, game_id, player_id, team_id,
position_group, stat_name, n_draws, mean, p05, p10, p25, p50, p75, p90, p95`.
`created_at` is operational metadata only — **excluded** from scientific
equality.

### Empirical percentiles (Phase 7B — LOCKED)

`p05 p10 p25 p50 p75 p90 p95` are empirical inverse-CDF **order statistics** of
the ascending draw vector — **no interpolation**, no library default method:

```
index = ceil(q * N) - 1     # clamped to [0, N-1]
value = sorted_values[index]
```

`p50` is the canonical median; there is deliberately no separate `median`
column. Locked fixture (`tests/projections/test_summary_math.py`):

```
[0,1,2,3,4,5,6,7,8,9]  ->  p05=0  p10=0  p25=2  p50=4  p75=7  p90=8  p95=9
```

`mean` is the arithmetic mean of the full vector — no trimming, winsorization,
calibration, rounding, or sportsbook adjustment. A coherent all-zero vector is
valid and summarizes to all zeros.

---

## 6. Immutable persistence & parent provenance (Phase 7C)

`nflprops.orchestration.projection_store.persist_player_game_projections` is the
single persistence entry point.

* **Deterministic id:** `projection_id = SHA-256(run_id | player_id |
  stat_name)` via the shared deterministic SHA helper — never Python `hash()`.
* **Parent run required:** `run_id` must reference an existing
  `prediction_runs(run_id)` row (FK `fk_player_game_projections_run_id`).
  Ad-hoc / backtest frames without one stay in memory and must not enter this
  table.
* **Parent provenance (LOCKED):** before any row is written, every incoming
  row's `season`, `week`, `game_id`, `n_draws` must equal the parent
  `prediction_runs` row's value. A disagreement is a hard
  `ProjectionProvenanceError` — the parent run is authoritative, the child value
  is **never** silently reconciled, and nothing (not even the agreeing subset)
  is written.
* **Idempotent retry:** same `projection_id` + identical scientific fields (any
  `created_at`) → no-op; the stored row, including its original `created_at`, is
  left exactly as it was.
* **Immutable conflict:** same `projection_id` + any differing scientific field →
  hard `ProjectionConflictError`; nothing is written.
* **Atomic batch:** one conflicting or provenance-mismatched row aborts the
  whole call — no partial insert.
* **DB constraints (migration `0004`, no new migration in 7D/7E):** PK
  `projection_id`; FK `run_id → prediction_runs.run_id` (not `ON DELETE
  CASCADE`); `UNIQUE (run_id, player_id, stat_name)`; indexes on `run_id`,
  `game_id`, `player_id`, `stat_name`, `(season, week)`, `(game_id, player_id)`.
  Certified on ephemeral PostgreSQL (`tests/orchestration/test_projection_persistence_postgres.py`).

---

## 7. Full-draw behavior (Phase 7B/7D — §23)

Every projection summary is computed from **all** `GameSimulationResult.n_draws`.
`retained_joint_draws` only bounds the separate `simulation_player_results`
table; no down-sampling is inserted between simulation and summarization.
Certified: with `n_draws = 256` and `retained_joint_draws = 8`, every persisted
projection row has `n_draws = 256`.

---

## 8. Official-checkpoint integration (Phase 7D)

Within one official (or MANUAL) checkpoint execution
(`nflprops.orchestration.flows.checkpoints`):

```
claim prediction_run
      -> scheduled_as_of PIT manifest (data_manifest_sha256)
      -> build football state
      -> ONE coherent game simulation  (compute_game_prediction)
             |
             +--> build player_game_projections   (from that GameSimulationResult)
             |    validate: rows == E * 30, else FAIL before pricing
             |    persist player_game_projections  (Phase 7C, immutable/idempotent)
             |
             +--> price current sportsbook markets (SAME GameSimulationResult)
                  persist predictions + retained joint draws
      -> terminal prediction_run status
```

* **Exactly one** football simulation per game/checkpoint. `build_player_game_projections`
  and `price_current_markets` receive the **same** `GameSimulationResult`
  instance (object-identity certified).
* Projection persistence happens **before** pricing and is **never** conditioned
  on pricing success.
* `run_id / season / week` for persistence come from the official run context;
  `game_id / n_draws` from the simulation must agree with the parent (Phase 7C
  gate).
* The Phase-5 `data_manifest_sha256` (a hash of PIT **inputs**) and the Phase-6
  `simulation_input_sha256` (quote-independent) are **unchanged** by projection
  persistence — projections are an *output*.

### Terminal status mapping

| Situation | `status` | `publication_status` | notes |
|---|---|---|---|
| projections built + persisted, pricing produced rows | `SUCCESS` | `PUBLISHED` | unchanged Phase-5/6 semantics |
| projections built + persisted, pricing produced **zero** rows normally | `SUCCESS` | `MODEL_ONLY` | valid — see [MODEL_ONLY](#9-model_only-semantics) |
| projections built + persisted, then a genuine **pricing exception** | `PARTIAL` | `NOT_PUBLISHED` | projection artifact preserved (never deleted) |
| projection build / validation / provenance / persistence fails | `FAILED` | `NOT_PUBLISHED` | pricing does **not** run; deterministic errors are non-retryable |
| no PIT-visible game at all | `FAILED` | `NOT_PUBLISHED` | `failure_code = GAME_NOT_FOUND` |
| game PIT-visible but **no usable game model** (no coherent simulation) | `FAILED` | `NOT_PUBLISHED` | `failure_code = GAME_NOT_MODELED` (Phase 7E correction — see below) |
| post-kickoff first discovery | `FAILED` | `NOT_PUBLISHED` | `failure_code = CHECKPOINT_MISSED` (unchanged) |

There is no Phase-5 `DATA_HOLD` data-gate machinery in this flow, so the
certified fallback for "no model / no artifact" is `FAILED / NOT_PUBLISHED`, not
`DATA_HOLD`. No new run status was introduced.

---

## 9. `MODEL_ONLY` semantics (Phase 7E certification invariant)

> **`publication_status == MODEL_ONLY`  ⇔  a complete, valid `E × 30`
> `player_game_projections` artifact exists for the run.**

* **Zero sportsbook quotes** after a valid model → one simulation, complete
  `E × 30` projections **persisted**, zero priced rows, `SUCCESS / MODEL_ONLY`.
  This path is valid and preserved.
* **No usable game model** (execution reached the model path, but no coherent
  `GameSimulationResult` could be produced, so **no** projection artifact
  exists) → `FAILED / NOT_PUBLISHED / GAME_NOT_MODELED`. It must **not** be
  `SUCCESS / MODEL_ONLY`.

**Phase 7E correction:** the pre-7E `not game_modeled` branch of
`game_checkpoint_flow` returned `SUCCESS / MODEL_ONLY` with no artifact. That
violated the invariant above and was corrected in Phase 7E to
`FAILED / NOT_PUBLISHED` with the narrowly scoped `GAME_NOT_MODELED` code.

---

## 10. `E × 30` completeness (Phase 7E §3)

For every **successfully modeled** official game/checkpoint the persisted
projection-row count is **exactly `E × 30`** (`E` eligible players × 30 approved
stats). A build that is not exactly `E × 30` raises before pricing — no
incomplete "successful" projection artifact is ever published.

---

## 11. Quote independence (Phase 7E §11)

The projection scientific output — eligible player set, stat rows, `mean`, all
seven percentiles, and `projection_id`s — is **identical** across: zero quotes,
Bet365-only, Bet365 absent, many books, changed lines, changed prices, and
reordered quotes. Only the `predictions` (priced) rows differ. Certified in
`tests/simulation_pricing/test_projection_quote_independence.py` and the
Bet365-absence checkpoint test in
`tests/orchestration/test_phase7d_checkpoint_projection_integration.py`.

---

## 12. Catch-up / PIT & kickoff reschedule (Phase 7E §18/§19)

* **Catch-up:** a checkpoint with `scheduled_as_of = 18:30` executed late at
  `19:05` uses only data with `available_at <= 18:30` and produces the **same**
  `run_id`, simulation, projections, and `projection_id`s as an on-time run.
* **Kickoff reschedule:** a revised kickoff → a new `run_id` → new
  `projection_id`s. The prior revision's `prediction_runs` row and its
  projection rows remain, immutable, and do **not** satisfy the new schedule
  revision. Both histories coexist in the canonical table.

---

## 13. Historical semantics (unchanged)

Phase 7 does **not** backfill projections or alter historical modeling. Existing
historical behavior stands: for seasons where the PIT injury feed was
unavailable (2022–2025 per the live BDL injury-history audit), persisted
prediction provenance records `injury_data_available = false` rather than
silently agreeing with a "no designation" reading.

---

## Phase-8 deferral

The following are **Phase 8** and are intentionally **not** implemented here:
threshold ladders, standard milestones, arbitrary threshold probabilities,
American odds for thresholds, `player_threshold_prices`, consensus / best-price,
confidence surfaces, Opportunities, Kelly, final-forecast policy, retraining /
recalibration, API, dashboard, and WizardOfOdds / SportsOdds publishing.

`player_game_projections` is the sportsbook-independent model artifact those
phases build on; it does not itself price thresholds.
