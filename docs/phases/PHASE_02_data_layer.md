# Phase 2 — Immutable data layer

## Objective
Turn canonical objects into a queryable, point-in-time-correct, reproducible
warehouse. Raw immutability, bronze/silver parquet, DuckDB access, entity resolution
with crosswalks, append-only snapshots, data-quality gates, and an empirical coverage
report.

## Spec sections
§2 (point-in-time), §9 (canonical IDs), §13, §16, §17, §18, §19.

## Files to implement
```
src/nflprops/data/raw_store.py
src/nflprops/data/warehouse.py
src/nflprops/data/entity_resolution.py
src/nflprops/data/snapshots.py
src/nflprops/data/quality.py
src/nflprops/pipelines/coverage_report.py
src/nflprops/pipelines/bootstrap.py
```

## Contracts consumed
`contracts/warehouse_tables.yml` — table names, keys, PIT requirements, layer rules.

## Requirements
1. Raw store is append-only and content-addressed. Rewriting an existing raw file with
   different bytes is an error.
2. Every time-varying silver table carries `event_time, available_at, ingested_at,
   provider, provider_record_id`.
3. Where `available_at` must be reconstructed for backfilled history, set
   `available_at_is_estimated = true`. Estimated rows are excluded from
   strict-leakage training sets.
4. Entity resolution mints canonical IDs once and writes crosswalk rows. It NEVER
   rewrites an existing canonical ID. Name collisions on the same team are surfaced,
   not auto-resolved.
5. Snapshots (`roster_snapshots`, `injury_snapshots`) are append-only. Deduplicate on
   `raw_record_hash` but retain first and last `available_at` per distinct state.
6. Quality gates emit `INFO`/`WARN`/`BLOCK`. `BLOCK` halts the pipeline and raises
   `DataQualityError`. Nothing is auto-corrected silently.
7. Required quality gates (spec §19), at minimum: final games with zero stat rows;
   stat rows whose team is not in that game; team stat rows without a paired opponent
   row; **sum of player targets exceeding team pass attempts**; impossible negative
   counts; duplicate (game, player) rows; advanced-stat weeks with no matching game;
   injury rows for players absent from every roster snapshot.
8. `coverage_report.py` empirically discovers coverage. It hardcodes no counts.

## Acceptance tests
```
tests/unit/test_raw_store_immutable.py
tests/unit/test_pit_columns_present.py
tests/unit/test_entity_resolution_stable.py
tests/unit/test_snapshots_append_only.py     # Saturday must not overwrite Monday
tests/unit/test_quality_gates.py
tests/leakage/test_available_at_filter.py
```

## Definition of done
- [ ] `nflprops ingest bootstrap --provider bdl` completes on fixture data
- [ ] `nflprops coverage` prints a real, discovered coverage table
- [ ] Re-running ingestion is idempotent — no duplicate silver rows
- [ ] Every table in `contracts/warehouse_tables.yml` exists with the declared keys

## Explicitly out of scope
Features, states, models, PBP parsing.
