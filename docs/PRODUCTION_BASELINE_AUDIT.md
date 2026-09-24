# Production Baseline Audit

Date: 2026-09-04
Purpose: Phase 0 baseline freeze required by the 2026–2027 Production Automation
blueprint, before any production-automation code is written.

## 1. Repository identity

- Branch: `main`
- HEAD: `eb64b41135de35836620a41ca4034d73b914200d` ("Implement CLV and close
  availability reporting")
- `origin/main`: same SHA — local `main` is not ahead/behind origin.
- Working tree at audit time (uncommitted, not created by this audit):
  - Modified: `src/nflprops/backtest/dataset.py`, `src/nflprops/backtest/historical.py`,
    `src/nflprops/pipelines/pregame.py`, `tests/backtest/test_historical_fold_adapter.py`,
    `tests/unit/test_pregame_provenance_wiring.py`
  - Untracked: `src/nflprops/market/timing.py`, `tests/unit/test_quote_time_semantics.py`
  - **Content**: this is a complete, self-consistent leakage fix — it introduces
    `quote_knowledge_time()` / `quote_time_source()` in the new `market/timing.py`
    module to distinguish, per market mode, the actual *knowledge* timestamp of a
    quote (`collector_received_at` for live markets vs. the reconstructed
    `available_at` for historical opening markets) instead of trusting a
    provider-echoed timestamp. It is wired into `pregame.py`'s `_price_quote` /
    provenance audit and into the backtest row contract (`quote_available_at`,
    `quote_time_source`, `quote_age_seconds`). Full test suite (below) passes with
    this diff in place, including the new `test_quote_time_semantics.py`. This is
    in-flight work from a prior session, not something this audit touched or
    committed — left as-is pending user direction.

## 2. Quality gates (run against the working tree, including the uncommitted diff above)

```
python -m ruff check src tests      -> All checks passed!
python -m compileall -q src tests   -> clean (no output)
python -m pytest -q                 -> exit code 0, no failures/errors
```

30 tests are skipped, all via explicit `pytest.mark.skip(reason="PHASE N not yet
implemented")`, e.g.:

```
tests/invariants/test_all_invariants_on_random_games.py   PHASE 7
tests/invariants/test_scenario_mixture.py                 PHASE 7
tests/leakage/test_feature_registry_complete.py           PHASE 4
tests/provider_contract/test_bdl_yardline_semantics.py    PHASE 3
tests/unit/test_alias_collision_is_failure.py             PHASE 3
tests/unit/test_attribution_uses_common_random_numbers.py PHASE 11
tests/unit/test_component_model_interface.py              PHASE 6
tests/unit/test_eb_shrinkage_monotone.py                  PHASE 5
tests/unit/test_entity_resolution_stable.py                PHASE 2
tests/unit/test_explanation_additivity.py                 PHASE 11
tests/unit/test_first_td_devig_includes_none.py            PHASE 9
tests/unit/test_is_missing_columns_present.py               PHASE 4
tests/unit/test_kappa_responds_to_uncertainty.py            PHASE 6
tests/unit/test_manifest_roundtrip.py                       PHASE 0
tests/unit/test_no_passing_yards_regressor.py                PHASE 6
tests/unit/test_one_sided_devig_marks_confidence.py           PHASE 9
tests/unit/test_opponent_from_reversed_rows.py                PHASE 4
tests/unit/test_pit_histogram_computed.py                      PHASE 8
tests/unit/test_play_count_overdispersion.py                    PHASE 6
tests/unit/test_push_probability_from_integer_support.py         PHASE 7
tests/unit/test_quality_gates.py                                  PHASE 2
tests/unit/test_raw_store_immutable.py                            PHASE 2
tests/unit/test_reconciliation_scoring.py                          PHASE 3
tests/unit/test_residual_pools_have_negative_tail.py                 PHASE 6
tests/unit/test_role_faster_than_skill.py                             PHASE 5
tests/unit/test_simulator_runs_without_optional_features.py            PHASE 7
tests/unit/test_snapshots_append_only.py                                 PHASE 2
tests/unit/test_state_variance_tracked.py                                 PHASE 5
tests/unit/test_tier3_gate.py                                              PHASE 3
tests/unit/test_two_point_model_present.py                                  PHASE 6
```

**Important:** these "PHASE N" labels belong to the repo's *own* pre-existing spec
(`docs/IMPLEMENTATION_SPEC.md`), a different, older phase numbering than the
Cursor blueprint's Phase 0–27. They should not be confused. None of them
correspond 1:1 to blueprint phases. Every skip here is a genuine, currently
unimplemented feature (challenger devig methods, empirical-Bayes shrinkage
extensions, entity resolution edge cases, etc.) — not stale labeling on top of
working code. This satisfies the blueprint's "trust the executable implementation"
instruction: I verified the underlying functions still raise `NotImplementedError`
rather than silently having been completed.

**Stop-condition check (§5.3):** the full suite is green. No pre-existing
regression is being hidden. Phase 0 stop condition does not block starting work.

## 3. Currently implemented production commands (`src/nflprops/cli.py`)

Implemented and working:
- `predict SEASON WEEK AS_OF [--draws]` — builds PIT states, simulates each game
  once, prices current player-prop quotes from that one simulation.
- `run SEASON WEEK [--as-of] [--draws]` — refreshes BDL current-week inputs then
  calls `predict`.
- `settle SEASON WEEK` — settles structured full-game props from BDL player-game
  stats.
- `report`, `reproduce`, `coverage` — implemented (not read in full detail here).

Explicitly unimplemented (`raise NotImplementedError("PHASE N")`, repo's own
numbering):
- Provider spec/ingest group: two commands under `PHASE 2`.
- Snapshot group: two commands under `PHASE 9`.
- PBP group: two commands under `PHASE 3`.
- Features group: one command under `PHASE 4`.
- State group: one command under `PHASE 5`.
- `train` (`PHASE 6`), `backtest` (`PHASE 10`).

Other known `NotImplementedError` sites:
- `src/nflprops/simulation/props.py:174` — intentional guard for unsupported prop
  types (not a gap, a fail-closed check).
- `src/nflprops/state/empirical_bayes.py:151` — `PHASE 5` shrinkage extension.
- `src/nflprops/market/devig.py:61,66,86,103` — `PHASE 9` challenger devig methods
  (Shin/power); current default devig path (proportional, two-sided) is
  implemented and in production use.

## 4. Current data backend

- Parquet files queried through DuckDB, no database server
  (`src/nflprops/resources/contracts/warehouse_tables.yml` header: *"Storage:
  Parquet files queried through DuckDB. No database server."*).
- `Warehouse` class (`src/nflprops/data/warehouse.py`) already exposes the
  backend-agnostic interface the blueprint's Phase 1 asks for:
  `read(table)`, `write(table, frame, sort_by=...)`, `append(table, frame, ...)`,
  `append_records(...)`, `tables()`, `exists(table)`, plus `query(sql)` and
  `register_views()`.
- No PostgreSQL, no S3/object-store client, no Alembic migrations, no
  `storage/base.py` / `storage/postgres.py` / `storage/object_store.py` modules
  exist yet. `NFLPROPS_DATA_ROOT` (local disk, defaults to `./data`) is the only
  configured storage location today (`.env.example`).

## 5. Existing tables / contracts

`src/nflprops/resources/contracts/`:
- `warehouse_tables.yml` — canonical table inventory (layers: raw → bronze →
  silver → features/states → predictions). Documents that snapshot tables
  (rosters, injuries, market/`player_prop_snapshots`) are point-in-time,
  append-only, and that `player_prop_snapshots` specifically is irreplaceable
  live history that cannot be reconstructed later.
- `feature_registry.yml`, `bdl_endpoints.yml`, `invariants.yml`, `prop_map.yml`.

No `simulation_artifacts`, `collector_runs`, `prediction_runs`,
`player_game_projections`, `player_threshold_prices`, `prop_market_consensus`,
`opportunities`, or `workflow_events` tables/contracts exist yet — these are all
net-new per the blueprint.

## 6. Existing market / closing / CLV functionality

Already implemented and covered by passing tests:
- `market/odds.py` — probability↔American-odds conversion, EV.
- `market/devig.py` — proportional two-sided devig (default); Shin/power devig
  are stubs (`PHASE 9`).
- `market/consensus.py` — `game_market_consensus` (median across vendors for
  **game-level** spread/total), `latest_prop_quotes`, `fair_over_probability`.
  This is game-market consensus only — there is **no multi-book player-prop
  consensus module yet** (blueprint Phase 11's `prop_market_consensus` /
  weighted-median-with-outlier-detection is a genuine gap, not an oversight).
- `market/closing.py` — deterministic closing-quote selection: latest valid live
  quote received no later than `kickoff_at - close_buffer_seconds`
  (`close_buffer_seconds = 60` in `configs/base.toml`'s `[market]` section),
  fails closed on same-timestamp conflicts rather than picking a favorable price.
- `market/clv.py` — CLV/close-availability reporting: probability CLV and cents
  CLV computed only for same-threshold quotes; line moves recorded separately as
  line-unit CLV; missing closes are never imputed and never enter CLV means. This
  matches the blueprint's Phase 20/25 CLV rules essentially verbatim already.
- `market/timing.py` (new, uncommitted — see §1) — the quote-knowledge-time
  leakage fix.
- `market/snapshots.py`, `market/ev.py`, `market/rules/` also exist.

No push-aware *exact* sportsbook pricing module separate from the simulation
distribution summarization was independently verified line-by-line in this
audit; `predict_week`/`_price_quote` in `pipelines/pregame.py` currently do
simulate-once-then-price-every-quote already (see §7), which is the direction
the blueprint's Phase 6 decomposition wants, just not yet split into the four
named functions (`simulate_game_for_prediction`, `build_player_game_projections`,
`build_threshold_prices`, `price_current_markets`).

## 7. Simulation / pricing flow today

`predict_week()` (`pipelines/pregame.py`) already: loads PIT inputs → builds
team/player states → constructs `GameSimulationInput` → simulates each game
**once** → loops over that game's available sportsbook quotes and prices each
from the single simulation's draws (`_price_quote`) → attaches PIT provenance
audit per prediction row. It does not currently persist a standalone
`player_game_projections` artifact (full expected box score per real player,
independent of any sportsbook quote existing) or a `player_threshold_prices`
artifact (standard milestone grid) — both are net-new per blueprint Phases 7–8.
Joint-draw retention (`retain_joint_draws`) is already wired into `predict_week`
per the CLI signature.

## 8. Currently skipped tests corresponding to now-implementable production
features

None of the 30 skipped tests map to blueprint-described production-automation
features (storage backend, collector, Prefect, confidence, opportunities,
FastAPI, Streamlit, etc.) — those tests don't exist yet because that code
doesn't exist yet. The 30 skips are all pre-existing repo-native "PHASE N"
placeholders unrelated to this blueprint's phase numbering (see §2/§3). None
should be unskipped as part of blueprint work; new tests should be added
alongside new blueprint-phase code instead, per the blueprint's own instruction
not to unskip unrelated future-phase tests to manufacture completeness.

## 9. Stale `STATUS: SKELETON` headers vs. genuinely incomplete code

A large number of files carry a `STATUS: SKELETON` header comment (roughly 50
files across `explain/`, `calibration/`, `providers/`, `features/`,
`simulation/`, `pipelines/`, `models/`, `state/`, `pbp/`, `backtest/`,
`market/`, `data/`, plus two stray `domain/models.py.before_*` backup files).
Per the blueprint's own instruction (§0), the header is stale where the file's
functions are actually invoked by passing tests and by the working `predict`/
`run`/`settle` commands — which is the overwhelming majority of these files
(e.g. `simulation/scoring.py`, `simulation/quarter.py`, `models/passing.py`,
`market/snapshots.py`, `pbp/parser.py` are all exercised by the green test
suite and by live prediction). The header is **not** stale, and genuinely marks
incomplete code, specifically for the functions that raise `NotImplementedError`
enumerated in §3 (empirical-Bayes shrinkage extension, challenger devig
methods) and the CLI command stubs. No file-level line-by-line audit of all ~50
headers was performed beyond confirming this pattern; a targeted re-check is
cheap if a specific file's status becomes load-bearing for a later phase.

The two `domain/models.py.before_vendor_fix` / `domain/models.py.before_open_prop_type`
files are backup copies left in the source tree (not `.py` files imported by
anything, matched only because they contain the string `.py` and the skeleton
marker) — worth flagging to the user as tree clutter, not touched by this audit.

## 10. Scope gap vs. the 2026–2027 production blueprint

The repository today is a complete **single-machine research/backtesting
system**: DuckDB+Parquet storage, a Typer CLI run by hand, one BDL provider,
no scheduler, no API, no dashboard, no continuous collector, no confidence
scoring, no opportunities ranking, no publication/DATA_HOLD gate, no deployment
target. The blueprint's Phases 1–27 describe building an entire second system
around it: managed PostgreSQL, S3-compatible object storage, a continuous
multi-cadence live collector service, Prefect 3 orchestration on a systemd-
supervised Linux VPS worker, a confidence-scoring engine, an opportunities
board, a publication/DATA_HOLD engine, FastAPI, Streamlit, and automated
CSV/Excel/JSON exports — followed by a full Week 1 production certification.

This is real, multi-week infrastructure work, and several of its Phase-0-blocking
items (P0) require **user-held external accounts/credentials this session has no
access to**: provisioning a managed Postgres instance, an S3-compatible bucket
and its access keys, a Linux VPS, and (optionally) Prefect Cloud. Code-level
scaffolding for all of these (storage interface, Alembic setup, provider
interfaces, collector logic, Prefect flow definitions, FastAPI/Streamlit apps,
confidence/opportunities/publication modules) can be built and tested locally
against the existing DuckDB backend without those credentials; actually wiring
and deploying against real Postgres/S3/a VPS cannot.

## 11. Conflicts found

None yet at the code level. The one thing worth flagging explicitly, per the
blueprint's own "document a CONFLICT" instruction, is not a code conflict but a
sequencing one:

```
CONFLICT
EXACT FILE: n/a (process-level)
EXACT CURRENT BEHAVIOR: repository has zero production infrastructure
  (no Postgres, no object storage, no VPS, no Prefect deployment).
EXACT BLUEPRINT REQUIREMENT: Phase 1 (P0) requires a managed PostgreSQL
  instance and S3-compatible object storage to exist and be reachable before
  backend-parity tests can pass and before production writes may move off
  DuckDB; Phase 4/5 require an always-on Linux VPS and Prefect worker.
WHY BOTH CANNOT COEXIST: these are real external services requiring the
  user's own accounts, credentials, and money; they cannot be provisioned by
  this agent autonomously.
MINIMUM CHANGE REQUIRED: user provides (a) a Postgres connection string,
  (b) S3-compatible endpoint/bucket/keys, (c) a target Linux host, or
  explicitly defers Phases 1/4/5's live-infra portions while the code-level
  scaffolding for them is built and tested against local DuckDB/local disk
  first.
```

## 12. Recommendation (not a blueprint requirement, offered per Phase 0's
"report any conflict" instruction)

Proceed phase-by-phase per §35 of the blueprint, committing one phase at a
time as instructed, but split each infrastructure-dependent phase into a
"build the interface/code, test against local backend" step (doable now) and
a "wire to real managed service" step (needs user-provided credentials/access).
Football-output phases (6–13: split simulation from pricing, player game
projections, threshold pricing, push-aware odds, prop consensus, best price)
have no external dependency and can proceed immediately.
