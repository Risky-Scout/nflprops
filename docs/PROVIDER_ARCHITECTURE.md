# Provider Architecture (Phase 3)

This document describes the provider-agnostic architecture that decouples
the NFL-props pipeline from any specific data/odds provider.

**BALLDONTLIE (BDL) is an implementation, not the domain model.**
**Bet365 is a reference sportsbook, not a required sportsbook.**

## Layering

```
upstream provider payload
       |
provider-specific client        (nflprops.providers.bdl.client.BDLClient)
       |
provider-specific mapper/quirks (nflprops.providers.bdl.mapper / .quirks)
       |
canonical domain model          (nflprops.domain.models)
       |
pipelines / warehouse / model   (nflprops.pipelines.*, features/, state/, simulation/)
```

Nothing below the canonical domain layer needs to understand BDL field
names, BDL IDs, BDL status abbreviations, BDL pagination shape, or BDL
sportsbook-naming quirks. A provider's own client/mapper/quirks modules are
the *only* place provider-specific parsing may live.

## Provider interface

`nflprops.domain.protocols` (pre-existing, extended by this phase) defines
the generic contract as five composable, `runtime_checkable` `Protocol`
classes, plus a convenience union:

- `ReferenceDataProvider` — `teams()`, `players()`, `active_players()`, `roster()`
- `ScheduleProvider` — `games()`
- `StatisticsProvider` — `player_game_stats()`, `player_season_stats()`,
  `team_game_stats()`, `team_season_stats()`, `advanced_passing/rushing/receiving()`,
  `plays()`, `standings()`
- `AvailabilityProvider` — `injuries()`
- `MarketProvider` — `game_odds()`, `opening_game_odds()`, `player_props()`,
  `opening_player_props()`
- `FullProvider` — the union of all five, plus `name`, `spec_sha256`,
  `spec_captured_at`

Every method returns canonical `nflprops.domain.models` objects — never a
provider-native dict. A provider need not implement every capability; the
registry reports which ones it satisfies so the pipeline can fail early.

## Capability model

A provider's capabilities are reported by
`nflprops.providers.registry.capabilities(name)`, which instantiates the
registered provider and checks `isinstance(provider, <Protocol>)` for each
of the five capability groups above — never `hasattr()` probing. This is
what makes a partial provider (e.g. reference data only) explicit rather
than silently missing methods failing at call time deep in a pipeline.

## Provider adapter (BDL)

`nflprops.providers.bdl.provider.BDLProvider` structurally satisfies
`FullProvider` today — every protocol method is implemented, backed by
`BDLClient` (transport: auth, pagination, retry/backoff) and
`providers.bdl.mapper` (raw payload → canonical model, with
`providers.bdl.quirks` absorbing BDL's specific oddities). None of BDL's
working pagination, normalization, injury mapping, roster behavior, or
retry/error behavior changed in this phase.

## Provider factory / registry

`nflprops.providers.registry` is the single provider-construction
mechanism:

```python
from nflprops.providers.registry import get_provider
provider, warehouse = get_provider("bdl", cfg)
```

`nflprops.pipelines.lean` registers `"bdl"` and `"balldontlie"` (an alias)
against `build_bdl_provider` at import time — the one place in the codebase
that legitimately knows the concrete `BDLProvider`/`BDLClient` classes.
`LeanIngestor` (the ingestion orchestrator) is type-hinted against
`FullProvider`, not `BDLProvider` — it only ever calls Protocol-defined
methods. An unregistered provider name fails explicitly (`KeyError`); there
is no silent fallback to BDL.

## Sportsbook / vendor normalization

`nflprops.market.vendors` is the canonical, provider-agnostic sportsbook
identity system — deliberately separate from BDL's own
`player_prop_vendors` enum (`nflprops.domain.enums.Vendor`), which is
provider-native documentation of what BDL happens to support today (and
does not include Bet365).

- `canonical_vendor(raw)` — deterministic, formatting-only normalization
  (`strip().lower()`). `"bet365"`, `"Bet365"`, `"BET365"` all resolve to
  `"bet365"`. This is a mechanical rule, not a hand-maintained alias table —
  it applies identically to every vendor, not a Bet365 special case.
- `GameOdds.vendor` / `PlayerProp.vendor` now hold the canonicalized value;
  the original string is preserved separately in `vendor_raw`.
- `SportsbookConfig` (loaded from `[market.sportsbooks]`) carries
  `allow_all_supported`, `blocked`, and `reference_book`.
- `is_vendor_allowed(vendor, cfg)` / `is_reference_book(vendor, cfg)` are the
  only vendor-policy checks this phase implements. No reliability
  weighting, staleness filtering, or outlier removal — that is later-phase
  consensus logic.

### `[market.sportsbooks]` config

```toml
[market.sportsbooks]
allow_all_supported = true
blocked = []
reference_book = "bet365"
```

`reference_book` is display/comparison-only. Its presence or absence never
changes whether ingestion or prediction succeeds — no code path may require
Bet365 to exist.

## Provenance

Every time-varying canonical record already carries `provider` and
`provider_record_id` (via `PITMixin`). Market rows (`GameOdds`, `PlayerProp`)
additionally carry `vendor` / `vendor_raw` and `raw_record_hash` — a
deterministic SHA-256 over the raw payload
(`nflprops.domain.hashing.hash_payload`: sorted-key JSON, stable UTF-8,
never Python's built-in `hash()`). Canonical `game_id`/`team_id`/`player_id`
remain the only IDs pipeline/model code should ever use; provider-native IDs
stay inside provider adapters and provenance fields — canonical IDs are
never replaced by them.

## What Phase 3 deliberately does not do

- Implement a second real (live) provider — `tests/provider_contract/fake_provider.py`
  is a small in-memory proof, used only in tests, that imports nothing from
  `nflprops.providers.bdl`.
- Continuous collection, Prefect orchestration, prop consensus, best-price
  selection, confidence scoring, APIs/dashboards, or any simulator/model
  behavior change.
- Reconcile `injury_snapshot_runs` (Phase 2) with a generalized
  `collector_runs` table. **Done in Phase 4** — see
  `docs/COLLECTION_ARCHITECTURE.md`: `collector_resource_runs` is now the
  sole authoritative feed-availability source, and no production code path
  writes new `injury_snapshot_runs` rows anymore.
