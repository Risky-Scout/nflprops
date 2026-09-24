# Collection Architecture (Phase 4)

Generalized, provider-neutral, point-in-time continuous collection. Builds
directly on Phase 3's provider abstraction (`nflprops.domain.protocols`,
`nflprops.providers.registry`) — the collector drives any `FullProvider`,
never a concrete provider class.

**`collector_resource_runs` is authoritative for feed availability.**
**`injury_snapshot_runs` is legacy/deprecated after this phase.**

## The collection cycle

`nflprops.collection.service.collect_once(*, provider, season, week,
warehouse, config, now)` runs exactly one cycle:

1. Fetch the week's schedule (`GAMES`) — foundational; everything else's
   scope (team IDs, game IDs) depends on it. If GAMES doesn't come back
   checked, the cycle is `FAILED` immediately; nothing else is attempted.
2. Attempt each other resource the provider supports
   (`ROSTERS`, `INJURIES`, `GAME_ODDS`, `PLAYER_PROPS`) **in isolation** — one
   resource's exception never blocks or rolls back another's success.
3. Append every canonical row returned to its existing warehouse table
   (`games`, `roster_snapshots`, `injury_snapshots`, `game_odds_snapshots`,
   `player_prop_snapshots`) using that table's established natural-key
   append-only semantics — a collection cycle never overwrites a previously
   stored snapshot, even a byte-identical one collected at a different time.
4. Record one `collector_resource_runs` row per attempted resource, and one
   `collector_runs` row for the cycle as a whole.

`collect_once` is deterministic given a frozen `now` — no sleeping, no
threading, no Prefect dependency. `nflprops.collection.loop.run_collection_loop`
wraps it in a simple foreground `while` loop (injectable clock/sleeper,
graceful `SIGINT`/`SIGTERM` handling); Phase 5 orchestrates/wraps *that*, it
does not replace it.

## Two tables, not one

A single aggregate `collector_runs.status` is never sufficient evidence of
resource-level availability — injuries can succeed with zero rows while odds
fails, in the very same cycle.

- **`collector_runs`** — one row per overall cycle: provider, season/week,
  timing, cadence used, rollup counts, overall `status`
  (`RUNNING`/`SUCCESS`/`PARTIAL`/`FAILED`), `source_sha256`/`config_sha256`.
- **`collector_resource_runs`** — one row per resource fetch attempted within
  a cycle: `resource_type`, `scope_type`/`scope_json` (exactly what was
  requested — never claiming more precision than the provider call actually
  had), `collection_status`, `row_count`, `retry_count`, and — only when the
  resource was successfully checked — `collector_received_at`. **This table,
  not `collector_runs.status`, answers "was resource X available at time
  T."**

## Resource statuses

| Status | Meaning |
|---|---|
| `SUCCESS` | Provider call succeeded, valid response. `row_count` may legitimately be 0 (e.g. a healthy-slate injury check). |
| `EMPTY_RESPONSE` | Structurally successful but suspiciously empty for a resource where emptiness isn't a valid complete reading (games, rosters). |
| `MARKET_NOT_POSTED` | Market-only. Request succeeded; nothing is currently posted. Proves the feed was *checked* — never that a quote exists. |
| `PARTIAL_RESPONSE` | Usable data for only part of the explicitly requested scope. |
| `RATE_LIMITED` | Final failure after the provider client's own retry budget was exhausted, caused by rate limiting. |
| `PROVIDER_ERROR` | Any other final provider/transport failure. |
| `UNSUPPORTED` | The configured provider doesn't implement this capability at all (checked via the Phase 3 Protocol capability system) — not a data-success state, and the resource fetch is never even attempted. |

Only `SUCCESS` and `MARKET_NOT_POSTED` establish availability
(`nflprops.collection.models.FEED_CHECKED_STATUSES`).

### `market_feed_checked` vs. `market_quote_available`

`resource_feed_available_at()` answers "was the market endpoint
successfully queried" — never "does this specific quote exist." Whether a
particular quote exists is a completely separate question, answered by
presence of a row in `game_odds_snapshots`/`player_prop_snapshots`
themselves. Do not conflate the two.

## Feed availability rule

`nflprops.collection.resource_availability.resource_feed_available_at(
resource_runs, *, resource_type, as_of, provider=None, scope_type=None)`:
True only when a resource-run record exists with `collector_received_at <=
as_of` and `collection_status` in `FEED_CHECKED_STATUSES`.
`provider=None` (the default) means "any provider's successful collection
counts" — what provider-agnostic pipeline code should use, since it has no
reason to know which provider is configured.

## Scope model

`scope_type` is `LEAGUE` / `WEEK` / `GAME` / `TEAM`, matching what the
provider call actually covered — never claiming precision the call didn't
have:

- `GAMES`, `GAME_ODDS` — `WEEK` (one call covers the whole week).
- `ROSTERS` — `WEEK` (one resource-run aggregates all teams playing that
  week; BDL's per-team calls are looped internally).
- `INJURIES` — `LEAGUE` (BDL's injuries endpoint has no team/week filter).
- `PLAYER_PROPS` — `GAME`, one resource-run per game (BDL's real call shape
  is per-game).

## Cadence

`nflprops.collection.cadence.cadence_seconds(time_to_kickoff)` — pure
function of the nearest **unstarted** game's time-to-kickoff, config-driven
via `[collection.cadence]` (defaults match the blueprint's production
table: 1800/1200/600/300/120/60s at the 48h/24h/6h/90m/30m boundaries). No
unstarted game (bye week, all games started) → `no_future_game_poll_seconds`
(1800s default) — post-slate schedule-discovery mode. A started or
negative-time game is treated identically to "no game" — it never selects
pregame minute-level polling.

## Retry/backoff

Reuses `BDLClient`'s existing single retry loop (`[provider.bdl.retry]`:
transient statuses `408/429/500/502/503/504`, exponential backoff with
jitter, `Retry-After` honored) — **the collector never wraps this in a
second, independent retry loop.** `BDLClient.pop_retry_count()` /
`BDLProvider.pop_retry_count()` expose, read-only, how many retries the
*existing* loop performed since the last call; `collect_once` reads this
exactly once per resource fetch. A provider that doesn't expose this
(e.g. the in-memory fake test provider) reports `retry_count=0` — "not
observable," not a false claim of zero retries.

`[collection.retry]` was deliberately **not** added as a config section —
there is nothing to configure that `[provider.bdl.retry]` doesn't already
own.

## Timestamp semantics

- **provider/effective timestamp** — when the provider says the record
  itself changed (e.g. `provider_updated_at`, `opened_at`).
- **`collector_received_at`** — when *this system* received the response.
  Set by the provider's own mapper at the moment of the HTTP response (e.g.
  `MappingContext.ingested_at` for BDL); `collect_once` never overrides it.
- **prediction `as_of`** — the point-in-time cutoff a prediction is built
  against; unrelated to either of the above except that both must be
  `<= as_of` for a leakage-free read.

## `injury_snapshot_runs` → `collector_resource_runs`

Phase 2 built `injury_snapshot_runs` as an injury-specific collection-attempt
log — the same "successful zero-row fetch is still available" insight this
phase generalizes to every resource. Phase 4 reconciles them:

- `nflprops.collection.migrate_injury_runs.migrate_legacy_injury_runs(warehouse)`
  translates every existing legacy row into a `collector_resource_runs` row
  (`resource_type=INJURIES`, `scope_type=LEAGUE`), using a deterministic id
  derived from the legacy row's own natural key (`provider`,
  `available_at`) — reruns never duplicate.
- `nflprops.data.injury_availability.injury_feed_available_at()` is now a
  thin wrapper over `resource_feed_available_at(..., resource_type=INJURIES)`
  — it reads `collector_resource_runs`, not `injury_snapshot_runs`.
- `LeanIngestor.ingest_week()` (the pre-existing one-shot ingestion path) no
  longer writes `injury_snapshot_runs` at all — that call site was removed
  once `collector_resource_runs` became authoritative, so old installations
  stop growing a table nothing reads for availability. `collect_once` writes
  `collector_resource_runs` directly and never touches `injury_snapshot_runs`.
- `injury_snapshot_runs` is kept as a legacy/deprecated, **read-only**
  artifact rather than deleted, to avoid unnecessary migration risk (per the
  warehouse contract's `deprecated: true` marker) — its only remaining
  purpose is `migrate_legacy_injury_runs()` reading whatever rows an old
  installation already has. `record_injury_collection_run()` (the writer)
  is retained only for legacy-data test/migration simulation, not called by
  any production code path.

**Future integration note (Phase 5+):** if a future phase builds a more
general orchestration-level `collector_runs` concept of its own (e.g. inside
Prefect), reconcile it with *this* `collector_runs`/`collector_resource_runs`
pair rather than creating a third competing system.

## What Phase 4 deliberately does not do

Prefect orchestration, official model checkpoints, player projection
outputs, prop consensus, best-price logic, confidence scoring,
Opportunities, API/dashboard, settlement/CLV changes, cloud provisioning.
