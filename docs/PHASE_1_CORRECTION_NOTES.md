# Phase 0–1 Correction Notes — BDL Provider Foundation

**Correction date:** 2026-08-20
**Official source of truth:** https://www.balldontlie.io/openapi/nfl.yml

This correction pass preserves the original provider-neutral blueprint and makes the BDL boundary substantially executable. It does **not** claim that the full predictive model is implemented.

## Verified official BDL contract facts

The official NFL OpenAPI document reviewed on 2026-08-20 declares:

- OpenAPI `3.1.0`
- API title `BALLDONTLIE - NFL API`
- API version `1.0.0`
- production server `https://api.balldontlie.io`
- API-key authentication in the `Authorization` request header

Provider-contract details corrected in this repository include:

- the standard cursor is an **integer**, not a string;
- advanced component names are `NFLAdvancedPassingStats`, `NFLAdvancedRushingStats`, and `NFLAdvancedReceivingStats`;
- most shared array query parameters use literal bracketed wire names (for example `player_ids[]`);
- `team_stats` uses endpoint-local unbracketed form/explode arrays: `team_ids`, `seasons`, `game_ids`;
- live/opening player-prop `vendors` is an unbracketed form/explode array;
- live player props are documented as live/non-historical and as returning all results without ordinary pagination, even though the example contains metadata resembling a cursor;
- opening player props do not expose cursor/per-page query parameters;
- `NFLOpeningPlayerProp` currently lists `updated_at` as required while defining `opened_at`; the provider adapter tolerates either and emits only canonical `opened_at`;
- DFS list/draftable filters use bracketed wire names such as `slate_ids[]`, `providers[]`, `game_ids[]`, `positions[]`;
- DFS cursors are integers;
- the DFS draftable schema component is `DfsDraftable`.

## Implemented in this pass

### Provider transport

`src/nflprops/providers/bdl/client.py`

- Authorization header
- timeout
- retry/no-retry status policy
- bounded exponential backoff + jitter
- Retry-After handling
- standard integer cursor pagination
- loop detection
- endpoint-specific array parameter encoding
- raw-response hook
- JSON validation

### Permissive BDL raw schemas

`src/nflprops/providers/bdl/raw_models.py`

The provider boundary is deliberately permissive (`extra="allow"`). Canonical domain models remain strict.

Implemented raw models cover:

- teams
- players
- roster/depth chart
- games
- player game stats
- player season stats
- team game/season stats
- injuries
- standings
- advanced passing/rushing/receiving
- play-by-play
- current/opening game odds
- current/opening player props
- DFS slates/events/roster slots/detail
- DFS draftables
- pagination/meta structures

### Provider quirks and normalization

`src/nflprops/providers/bdl/quirks.py`

- opening-prop timestamp inconsistency
- Decimal line parsing
- possession time
- height/weight
- experience
- position group
- injury status normalization

### Raw → canonical mapper

`src/nflprops/providers/bdl/mapper.py`

Provider IDs are translated to deterministic canonical IDs. Provider-specific quirks do not leak into downstream modeling modules.

### BDL provider facade

`src/nflprops/providers/bdl/provider.py`

Implemented methods cover reference data, schedules, statistics, availability, markets, play-by-play, and optional DFS resources.

### Spec pin/drift workflow

`src/nflprops/providers/bdl/spec.py`

The exact remote OpenAPI bytes must be pinned atomically and hashed. Schema drift is compared without silently upgrading production.

## Important deliberate blocker: exact OpenAPI bytes

The repository still contains an obvious placeholder at:

`specs/providers/bdl/nfl.yml`

This sandbox could inspect the live official document through its browsing subsystem, but its filesystem/runtime downloader could not resolve the external host. I therefore did **not** manufacture or partially reconstruct a supposed authoritative copy.

On an internet-connected development machine run:

```bash
nflprops provider pin bdl --url https://www.balldontlie.io/openapi/nfl.yml
nflprops provider verify bdl --strict-fields
```

The provider defaults to fail-closed if the real pinned spec is absent.

After pinning, commit:

- `specs/providers/bdl/nfl.yml`
- the generated spec lock/hash metadata

Then run:

```bash
make verify-phase-01
```

## Second reproducibility blocker: dependency lock

The sandbox also could not reach package indexes, so `uv.lock` was not generated. Do not treat an environment-specific `pip freeze` as a substitute.

On a networked development machine:

```bash
uv lock
```

Commit `uv.lock` before a model artifact is called fully reproducible.

## Real API fixtures

No real BDL response was fabricated. Before Phase 1 is production-verified, capture authorized, redacted fixtures for representative responses from every modeled endpoint and use them for offline contract tests.

## Current test status

After this correction pass:

- source compilation: PASS
- internal contract validation: PASS
- provider-boundary containment guard: PASS
- pytest: **49 passed, 46 skipped, 0 failed**

The remaining skips are intentional work for later phases plus:

1. exact pinned-spec machine verification;
2. empirical BDL yard-line semantics validation.

## Release rule

A final production release must not be approved merely because skipped tests are green. All phase acceptance tests relevant to the declared release scope must be active and passing.
