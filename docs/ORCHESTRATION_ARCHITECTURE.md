# Orchestration Architecture (Phase 5)

Phase 5 converts the Phase-4 collection/prediction components into a
deterministic production orchestration layer using Prefect 3, while
preserving every existing point-in-time (PIT) guarantee. It adds
orchestration — it does not change football prediction mathematics,
collection mechanics, or PIT selector logic.

**Prefect execution time never defines model knowledge time.
`scheduled_as_of` defines model knowledge time.**

## What Prefect does and does not do here

Prefect is a *scheduler and audit trail*, not a data pipeline. It decides
*when* to call pre-existing, unmodified pipeline code:

- `nflprops.collection.service.collect_once` (Phase 4) — resource fetching,
  retry/backoff, status classification, `collector_runs`/
  `collector_resource_runs` bookkeeping. Prefect never reimplements any of
  this; it only decides when to call it.
- `nflprops.pipelines.pregame.predict_game` (a thin Phase-5 wrapper over the
  pre-existing `predict_week`) — state construction, simulation, market
  pricing, PIT enforcement, invariant checks. Prefect never reimplements
  any of this either.

Core model modules (`simulation`, `state`, `backtest`, `market`,
`collection`) do not import `prefect`. Only
`nflprops.orchestration.tasks`, `nflprops.orchestration.flows.*`, and
`nflprops.orchestration.deployments` do.

`nflprops.orchestration.checkpoints` and `nflprops.orchestration.run_store`
are deliberately Prefect-free pure Python: checkpoint timing math and run
identity/claiming are unit-testable at full speed with no Prefect runtime.

## Package layout

```
src/nflprops/orchestration/
    checkpoints.py    # CheckpointName, offsets, scheduled_as_of, due/catch-up/missed
    run_store.py      # PredictionRunRecord, deterministic run_id, atomic claim, status transitions
    tasks.py           # retry classification (transient vs deterministic); no Prefect execution logic
    flows/
        collection.py  # collection_once_flow, collection_dispatch_flow
        checkpoints.py # game_checkpoint_flow, checkpoint_dispatch_flow
    deployments.py     # deployment construction + `python -m nflprops.orchestration.deployments register`
```

## Official checkpoints

Five official, game-relative pregame checkpoints, plus `MANUAL` for ad-hoc
diagnostic runs (never produced by the dispatcher):

| Checkpoint | Offset before kickoff |
|---|---|
| T48H | 172800s (48h) |
| T24H | 86400s (24h) |
| T6H  | 21600s (6h) |
| T90M | 5400s (90m) |
| T30M | 1800s (30m) |

Configured in `[checkpoints.offset_seconds]` (`configs/base.toml`, mirrored
byte-identical in `src/nflprops/resources/configs/base.toml`). Validated at
startup (`CheckpointOffsets.validate`): all offsets must be `> 0`, all
names unique, and `T48H > T24H > T6H > T90M > T30M`. Invalid production
configuration fails closed — it never silently corrects itself.

**Distinct from `[market.collector].checkpoints`**: that config key
(`"OPEN","T-48H",...,"CLOSE"`) is a set of market-snapshot labels for the
still-unimplemented Phase-9 market snapshot collector
(`nflprops/market/snapshots.py`, a skeleton) — not official model
checkpoint identities. No code reads `[market.collector]` today, and no
Phase-5 official-checkpoint code path (`orchestration.checkpoints`,
`orchestration.run_store`, `orchestration.flows.*`, the checkpoint CLI
commands) reads it either. Official model checkpoint orchestration reads
only `[checkpoints]`/`[checkpoints.offset_seconds]`.

### The scheduled_as_of rule (non-negotiable)

For kickoff `K` and offset `O`:

```
scheduled_as_of = K - O
as_of           = scheduled_as_of
```

`scheduled_as_of` is **never** the wall-clock time a worker happens to
start, a task happens to run, or a database row happens to be written. If
a worker is late but still runs before kickoff, it still uses the
*original* `scheduled_as_of` as the model's `as_of` — this is what keeps a
late run scientifically valid: the warehouse is point-in-time, so
re-deriving the exact same cutoff later reproduces the exact same
knowledge state, never a leaked later one.

### Due / catch-up / missed (`evaluate_checkpoint`)

- `now < scheduled_as_of` → **not due**.
- `scheduled_as_of <= now < kickoff_at`, discovered within one
  `dispatcher_tick_seconds` of `scheduled_as_of` → **run** (the ordinary,
  on-time case).
- Same window, discovered later (a worker recovering from an outage) →
  **run** if `catch_up_before_kickoff` (default `true`), using the
  *original* `scheduled_as_of` — never `now`. If `catch_up_before_kickoff`
  is `false`, this case is **missed** instead.
- `now >= kickoff_at` → **missed**, unconditionally. An official pregame
  forecast must never execute after kickoff, no matter how it was
  discovered.

A checkpoint that is never claimed/run before kickoff is not a new status
value — it is `status=FAILED, failure_code=CHECKPOINT_MISSED,
publication_status=NOT_PUBLISHED`, with no simulation ever invoked.

## Run identity (`prediction_runs`)

`prediction_runs` is *operational/audit lifecycle metadata* for official
checkpoint executions — never a store of prediction outputs. Prediction
outputs stay in the pre-existing `predictions` / `simulation_player_results`
tables, immutable regardless of a run's status.

### Deterministic run ID

```
run_id = SHA256(
    game_id + "|" + checkpoint_name + "|" + scheduled_as_of + "|" + kickoff_at
    + "|" + model_version + "|" + config_sha256 + "|" + source_sha256
)
```

(`nflprops.orchestration.run_store.compute_run_id`, backed by
`nflprops.collection.resource_availability.deterministic_id` — the same
pipe-joined-SHA-256 convention Phase 4 already established. Never Python's
built-in `hash()`.) Including `kickoff_at` is intentional: a materially
rescheduled game is a different checkpoint-schedule revision and must
produce a different `run_id`, even for the same `checkpoint_name`.

### Schema

See `migrations/versions/0003_prediction_runs.py` for the authoritative
PostgreSQL DDL (indexes: `game_id`; `season, week`; `checkpoint_name`;
`scheduled_as_of`; `status`; `game_id, kickoff_at`) and
`contracts/warehouse_tables.yml`'s `orchestration.prediction_runs` entry
for the mirrored contract description. Local storage (`Warehouse`) stores
the same columns as a Parquet-backed table; column set is identical across
backends.

Status vocabulary (`PredictionRunStatus`): `SCHEDULED, RUNNING, SUCCESS,
PARTIAL, DATA_HOLD, FAILED`. Publication vocabulary (`PublicationStatus`,
Phase-5 scope only — the full publication-gate engine is a later phase):
`PUBLISHED, MODEL_ONLY, DATA_HOLD, NOT_PUBLISHED`.

`game_checkpoint_flow`'s outcome mapping:

| Outcome | status | publication_status | failure_code |
|---|---|---|---|
| Simulated, non-empty predictions | SUCCESS | PUBLISHED | — |
| Simulated, zero priceable quotes | SUCCESS | MODEL_ONLY | — |
| No PIT-visible game for `game_id`/`scheduled_as_of` | FAILED | NOT_PUBLISHED | GAME_NOT_FOUND |
| PIT leakage detected | FAILED | NOT_PUBLISHED | LEAKAGE_VIOLATION |
| Simulation invariant violated | FAILED | NOT_PUBLISHED | INVARIANT_VIOLATION |
| Any other exception | FAILED | NOT_PUBLISHED | PREDICTION_ERROR |
| Never claimed/run before kickoff | FAILED | NOT_PUBLISHED | CHECKPOINT_MISSED |

`DATA_HOLD` is reserved for a future phase's full data-quality gate engine
— Phase 5 never assigns it, since no existing hard gate currently produces
that signal from inside this pipeline.

### Allowed status transitions

`SCHEDULED -> RUNNING`, `RUNNING -> {SUCCESS, PARTIAL, DATA_HOLD, FAILED}`,
and `SCHEDULED -> FAILED` (CHECKPOINT_MISSED / pre-execution validation
failure only). All four of SUCCESS/PARTIAL/DATA_HOLD/FAILED are terminal —
`update_run_status` raises `RunStatusTransitionError` on any other
transition, including re-entering a terminal state. Identity fields
(`scheduled_as_of`, `kickoff_at`, `model_version`, `config_sha256`,
`source_sha256`, `data_manifest_sha256`) are fixed forever at claim time —
`update_run_status` never touches them.

### Idempotent claiming

`claim_checkpoint(backend, record)` is the sole mechanism that decides
whether a checkpoint may be executed:

- **PostgreSQL**: a single `INSERT ... ON CONFLICT DO NOTHING`, atomic
  under concurrent transactions by construction. No conflict target is
  named: `prediction_runs` carries two unique constraints that are
  logically equivalent for an identical record (the `run_id` primary key,
  and `uq_prediction_runs_identity` — `run_id` is a deterministic hash of
  exactly that identity tuple). An earlier revision named only
  `(run_id)` as the arbiter, which left a genuine race under true
  concurrent inserts of the identical row: one transaction could raise a
  real `IntegrityError` on `uq_prediction_runs_identity` instead of being
  silently absorbed. A real concurrency test (`ThreadPoolExecutor` +
  `threading.Barrier`, separate connections, 2 and 8 competing workers,
  against ephemeral PostgreSQL — see
  `tests/orchestration/test_migration_postgres.py`) reproduced this and
  confirmed the fix: a bare `ON CONFLICT DO NOTHING` suppresses a
  violation of *any* unique/exclusion constraint on the table. Still a
  pure PostgreSQL-native primitive — no application-level locking.
- **Local (`Warehouse`)**: a deterministic check-then-insert, sufficient
  for single-process tests — not claimed to be production-safe under real
  multi-process concurrency (only PostgreSQL is required to be).

A Prefect retry of `game_checkpoint_flow`'s task reuses the exact same
`ctx.run_id` — it can never create a second official identity for the same
checkpoint. Repeated persistence of `predictions` rows on retry is
idempotent via the pre-existing `prediction_id` natural key (`Warehouse`/
`PostgresStorageBackend` `.append(..., key=[...])` semantics).

### Kickoff revisions

A rescheduled kickoff changes `scheduled_as_of` for every checkpoint, and
therefore changes every `run_id` for that game going forward. Completed
runs tied to the old kickoff are never deleted or rewritten — they remain
permanent historical evidence. `checkpoint_satisfied(..., kickoff_at=...)`
only considers a checkpoint satisfied by a run tied to the *current*
canonical `kickoff_at`; an old revision's row never satisfies the new
schedule, even for the same `checkpoint_name`. Final-forecast designation
(considering only the current kickoff revision) is deferred — Phase 5
always stores `is_final_forecast=false`.

## Data manifest fingerprint

`data_manifest_sha256` (`nflprops.orchestration.manifest`) is a real,
content-level fingerprint of the actual selected PIT input dataset for one
`(game_id, scheduled_as_of)` checkpoint — not row counts/max-timestamps
alone, and not a re-hash of `StateProvenanceContext.state_snapshot_id`
(an earlier Phase-5 revision did exactly that; a read-only audit
established it didn't cover market data or injury-feed-availability
provenance, and this module replaced it).

`build_checkpoint_manifest(warehouse, game_id=..., scheduled_as_of=...)`
selects, for records eligible at `available_at <= scheduled_as_of` (or
`collector_received_at <= scheduled_as_of` for `collector_resource_runs`,
which has no `available_at` column) using the exact same
`nflprops.features.asof.filter_pit(frame, as_of, strict=False)` primitive
`predict_week`/`build_team_states`/`build_player_states` already use
internally:

- the target game's own selected schedule/state row
- historical player-game and team-game stats, scoped to the two teams
  actually playing in `game_id` (not every team in the warehouse)
- roster rows for those two teams
- injury rows for the players those scoped roster/stat rows reference
- `collector_resource_runs` rows for the INJURIES resource type, plus the
  derived `injury_feed_available` boolean itself (via the existing
  `injury_feed_available_at`) — both are part of the hashed payload, not
  just the boolean's downstream row count
- game-odds and player-prop-quote rows for `game_id`
- reference `players` rows for the players referenced above

Each component records `row_count`, a `content_sha256` over every selected
row's full column content (sorted by stable canonical keys first — row
order in the source table, which depends on append history rather than
content, never affects the hash), and `min`/`max` timestamps where
applicable. The top-level `data_manifest_sha256` is
`nflprops.domain.hashing.hash_payload` (sorted-key JSON, SHA-256, never
Python's built-in `hash()`) over the full canonicalized component dict.

This is a pure function of `(warehouse content, game_id, scheduled_as_of)`
with no `now`/wall-clock dependency at all, so it is computed once before
a checkpoint is claimed and is catch-up-safe by construction: an on-time
run and a late-recovering worker computing the identical
`(game_id, scheduled_as_of)` against unchanged warehouse content always
produce the identical manifest. `config_sha256` (resolved config) and
`source_sha256` (code fingerprint) remain separate, independent
`prediction_runs` columns — this module represents data only.

## Retry policy: transient vs. deterministic (`orchestration.tasks`)

Only genuinely transient orchestration/infrastructure failures may be
retried: temporary database/object-storage connectivity, a Prefect worker
interruption, or a transient provider/collection-infra error where the
underlying call is idempotent. Default policy: `retries=2,
retry_delay_seconds=[15, 60]`.

Never retried — retrying reproduces the identical failure and only delays
surfacing it: `LeakageError` (either variant in this codebase),
`AssertionError` (simulation invariant violations), `ValueError`,
`KeyError`, `TypeError` (invalid config / unknown canonical entity / any
other deterministic input-validation failure).

`nflprops.orchestration.tasks.is_transient_failure` is a plain function
(no Prefect dependency) so this classification is unit-testable without
exercising Prefect's retry/sleep machinery. `retry_condition_fn` wraps it
for Prefect's `@task(retry_condition_fn=...)` hook. The pre-existing
provider HTTP client retry loop is untouched and is not wrapped in a
second retry loop here.

## Collection dispatch

`collection_dispatch_flow` wraps `collect_once` via `collection_once_flow`
(a thin `@flow`) and adds only a due/not-due decision
(`collection_due`): due immediately if no prior cycle exists for
`(provider, season, week)`; otherwise due iff `now - latest_started_at >=`
the cadence for the nearest unstarted game in scope, recomputed fresh on
every call so entering a tighter cadence band applies immediately. Uses
the latest *attempted* cycle's `started_at` regardless of
SUCCESS/PARTIAL/FAILED, so a failing provider cannot create a tight
one-minute retry storm.

Phase 4's foreground infinite loop (`nflprops.collection.loop`) is
untouched and remains available outside Prefect. Prefect never launches
that loop as one long-running task — a Prefect deployment schedules
`collection_dispatch_flow` to run every `dispatcher_tick_seconds`
(default 60s) and it returns quickly, doing real work only when due.

## Checkpoint dispatch and per-game isolation

`checkpoint_dispatch_flow` enumerates every currently-scheduled game in
`(season, week)`, and for each of the five official checkpoints: skips it
if already satisfied for the current kickoff; otherwise evaluates
due/catch-up/missed; computes the deterministic `run_id` and data
manifest; atomically claims it; and, if claimed for execution, runs one
isolated `game_checkpoint_flow`.

`game_checkpoint_flow` never raises — every outcome (success, empty,
leakage, invariant violation, unexpected error, game not found) becomes a
terminal `prediction_runs` row. The dispatcher additionally wraps each
`game_checkpoint_flow` call in a defense-in-depth `try/except` so that even
an unexpected bug in status bookkeeping cannot abort the loop: one game's
failure never prevents other due games in the same dispatch cycle from
running to completion.

## Prefect flow parameter validation

`collection_once_flow`, `collection_dispatch_flow`, `game_checkpoint_flow`,
and `checkpoint_dispatch_flow` are all declared with
`validate_parameters=False`: they take live, in-process objects
(`Warehouse`, `Config`, `FullProvider`, `SimulationConfig`,
`CheckpointRunContext`) that Prefect's default Pydantic-based parameter
validation cannot (and should not attempt to) build a JSON schema for.
The two deployment-facing adapter flows in `deployments.py`
(`collection_dispatch_deployment_flow`, `checkpoint_dispatch_deployment_flow`)
are the ones Prefect actually schedules and validates parameters for —
they take only JSON-serializable values (`provider_name: str`,
`season: int`, `week: int`) and construct the live objects in-process
before delegating to the flows above, exactly as the existing
`nflprops collect once` CLI command already does.

## Deployments and registration

`nflprops.orchestration.deployments.build_deployments` constructs (never
registers) the two required deployments — `collection-dispatch` and
`checkpoint-dispatch` — against the configured work pool
(`[orchestration].prefect_work_pool`, default `nflprops-production`) and
tick interval (`[orchestration].dispatcher_tick_seconds`, default 60s).
Pure object construction: no network call, no resource creation.

```
python -m nflprops.orchestration.deployments register
```

registers/upserts them against whatever Prefect API `PREFECT_API_URL` (and
`PREFECT_API_KEY`, for Prefect Cloud) points to — a self-hosted Prefect
server and Prefect Cloud are configured identically. Missing
`PREFECT_API_URL` raises `DeploymentRegistrationError` explicitly; this
command never silently falls back to an ephemeral local server and never
creates a cloud account or work pool on your behalf.

## Local testing without Prefect Cloud

Prefect 3's synchronous execution model auto-starts a temporary ephemeral
local server when no `PREFECT_API_URL` is configured — this is what every
Phase-5 test relies on (directly or via `prefect.testing.utilities.
prefect_test_harness` where a shared server across multiple flow calls is
useful). No automated test makes a network call to Prefect Cloud.

## Manual checkpoints

`nflprops checkpoint run --game-id <id> --as-of <ts> --season <s> --week
<w>` runs a one-off diagnostic prediction claimed under
`checkpoint_name=MANUAL` — excluded from `OFFICIAL_CHECKPOINTS`, never
produced by the dispatcher, and never satisfies an official checkpoint's
due/claim logic. `nflprops checkpoint due <season> <week> --at <ts>`
dry-inspects due-ness (default) or, with `--execute`, actually claims and
runs due checkpoints — the default never executes a prediction.

## Systemd worker (deployment artifact only)

`deploy/systemd/nflprops-prefect-worker.service` is a template for a
Linux (Ubuntu 24.04 LTS) production host — never installed on a developer
machine. It runs `prefect worker start --pool nflprops-production --type
process` as a dedicated unprivileged `nflprops` user, reads secrets from
an external `EnvironmentFile` (`/etc/nflprops/nflprops.env`, never inlined
into the unit itself), and restarts on failure (`Restart=always`).
`tests/orchestration/test_systemd_unit.py` statically validates these
properties (no `systemctl` calls).

## Deferred to later phases

- The full publication-gate engine (confidence-based gating beyond the
  Phase-5 `PublicationStatus` vocabulary).
- Final-forecast designation across checkpoint fallback
  (T30M → T90M → T6H → T24H → T48H); Phase 5 always stores
  `is_final_forecast=false`.
- `DATA_HOLD` as an actively-assigned status — reserved for a future
  data-quality gate.
- The Phase-6 simulation/pricing architectural decomposition
  (`simulate_game_for_prediction` / `build_player_game_projections` /
  `build_threshold_prices` / `price_current_markets`). Phase 5's only
  prediction-pipeline change is the minimal `predict_game` single-game
  filter wrapper needed to run the existing model for exactly one game.
- Player-game-projections artifact, threshold/milestone pricing,
  multi-book consensus, best-price logic, Kelly sizing, Opportunities,
  API/Streamlit/Excel surfaces, automated settlement/CLV changes,
  production cloud provisioning (no Prefect Cloud account, work pool, or
  systemd installation was created by this phase).
