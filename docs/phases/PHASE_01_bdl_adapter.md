# Phase 1 — BDL provider adapter

## Objective
Build the complete, provider-independent boundary: auth, HTTP client, retry,
pagination, raw Pydantic schemas for every endpoint, canonical mappers, the provider
facade implementing the capability protocols, and contract tests against saved
fixtures. After this phase, canonical domain objects can be produced for every
endpoint in the contract, and **nothing downstream knows BDL exists**.

## BLOCKER — resolve first
`specs/providers/bdl/nfl.yml` is a placeholder. Before writing mappers:

```bash
nflprops provider pin bdl --url <bdl openapi url>
nflprops provider verify bdl
```

`verify` diffs `contracts/bdl_endpoints.yml` against the pinned spec and fails on any
missing/extra endpoint, parameter, field, type, or enum value. **If it fails, stop and
report the diff.** Do not edit the contract to silence a failure without understanding
why the two disagree.

## Spec sections
§6, §7, §8, §9, §10, §11, §12, §13, §14, §15.

## Files to implement
```
src/nflprops/domain/models.py        (canonical domain objects — strict)
src/nflprops/domain/protocols.py     (capability protocols)
src/nflprops/providers/registry.py
src/nflprops/providers/bdl/endpoints.py
src/nflprops/providers/bdl/client.py
src/nflprops/providers/bdl/raw_models.py
src/nflprops/providers/bdl/quirks.py
src/nflprops/providers/bdl/mapper.py
src/nflprops/providers/bdl/provider.py
tools/verify_spec_coverage.py        (full implementation)
```

## Contracts consumed
`contracts/bdl_endpoints.yml` — this is the source of truth for every endpoint,
parameter name, array-bracket convention, field list, tier, and quirk.

## Requirements
1. `endpoints.py` is the ONLY file containing endpoint path strings. Enforced by
   `tests/unit/test_import_boundaries.py`.
2. `client.py` does auth, requests, retries, backoff, timeouts, param encoding,
   decoding, logging, and the raw-persistence hook. It computes **no football
   features**. Interface: `get(path, params)` and `paginated_get(path, params)`.
3. Retry on `408, 429, 500, 502, 503, 504` and transport errors, exponential backoff
   with bounded jitter, honor `Retry-After` on 429. Never retry `400, 401, 403, 404`.
4. Param encoding is driven by the `array_param` flag in the contract. `/nfl/v1/team_stats`
   uses unbracketed names; `/nfl/v1/games` takes `season_type` as an array while
   `/nfl/v1/stats` takes it as a scalar. These are data, not special cases in code.
5. `/nfl/v1/odds/player_props` does NOT use the cursor loop — handle explicitly.
6. `raw_models.py` is permissive: extra fields allowed, documented inconsistencies
   tolerated. `domain/models.py` is strict, typed, normalized.
7. `quirks.py` holds every provider oddity, including the opening-prop
   `opened_at`/`updated_at` inconsistency. Nothing downstream sees a quirk.
8. Type normalization: line values and spreads -> `Decimal`; `possession_time` -> int
   seconds; `height`/`weight` -> inches/pounds with raw string retained.
9. `provider.py` implements `ReferenceDataProvider`, `ScheduleProvider`,
   `StatisticsProvider`, `AvailabilityProvider`, `MarketProvider`.
10. Every response is persisted raw with its metadata sidecar before mapping, with the
    `Authorization` header redacted.

## Acceptance tests
```
tests/provider_contract/test_spec_coverage.py          # contract == pinned spec
tests/provider_contract/test_param_encoding.py         # bracketed vs unbracketed
tests/provider_contract/test_season_type_shape.py      # array on games, scalar on stats
tests/provider_contract/test_pagination.py             # cursor loop + props exception
tests/provider_contract/test_opening_prop_quirk.py     # opened_at OR updated_at
tests/provider_contract/test_decimal_line_parsing.py   # no binary float at ingest
tests/provider_contract/test_nullable_fields.py
tests/provider_contract/test_unknown_new_field_tolerated.py
tests/provider_contract/test_incompatible_change_fails_loudly.py
tests/provider_contract/test_retry_policy.py
tests/unit/test_import_boundaries.py
tests/unit/test_no_secret_leakage.py
```
All tests run against saved fixtures in `tests/fixtures/bdl/`. **No test hits the
live network.**

## Definition of done
- [ ] `nflprops provider verify bdl` exits 0 against the real pinned spec
- [ ] Every endpoint in the contract has a client method, raw model, and mapper
- [ ] Every field in the contract appears in the raw model
- [ ] `nflprops provider verify bdl --strict-fields` reports zero unmapped fields
      OR each unmapped field is listed in `providers/bdl/UNMAPPED.md` with a reason
- [ ] `grep -rn "nfl/v1" src/ --include=*.py | grep -v endpoints.py` returns nothing
- [ ] `grep -rn "providers.bdl" src/nflprops/{models,features,simulation,state}` returns nothing

## Explicitly out of scope
Parquet, DuckDB, entity resolution, features, models. Mapping produces canonical
objects in memory; persisting them is Phase 2.
