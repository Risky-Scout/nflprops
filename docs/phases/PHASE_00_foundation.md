# Phase 0 — Foundation

## Objective
Establish the package, configuration system, structured logging, run manifests, and
canonical ID minting. Nothing in this phase touches the network or any model. When
this phase is done, `nflprops --help` runs, config loads and validates, and a run
manifest can be written and re-read.

## Spec sections
§0, §3 (determinism), §7 (spec pinning skeleton), §9 (canonical IDs), §71 (config).

## Files to implement
```
pyproject.toml                       (already present — verify, add deps only if listed in spec §62)
src/nflprops/__init__.py
src/nflprops/version.py
src/nflprops/errors.py
src/nflprops/logging_.py
src/nflprops/config.py
src/nflprops/domain/ids.py
src/nflprops/domain/enums.py
src/nflprops/data/manifests.py
src/nflprops/cli.py                  (command tree only; subcommands raise NotImplementedError)
tools/verify_spec_coverage.py        (scaffold; full logic lands in Phase 1)
```

## Contracts consumed
- `configs/base.toml`, `configs/providers/bdl.toml`, `configs/models/2026_v1.toml`
- `contracts/warehouse_tables.yml` (for path constants only)

## Requirements
1. `config.py` loads TOML with layered override: `base.toml` -> provider toml ->
   model toml -> environment variables -> CLI flags. Validation via Pydantic. Unknown
   keys are an error, not a warning.
2. Config exposes `config_sha256()` over the fully-resolved config, for manifests.
3. `ids.py` mints canonical IDs as UUIDv5 over
   `(entity_kind, first_seen_provider, first_seen_provider_id)` with a fixed
   namespace constant. Minting is pure and deterministic. There is no "regenerate".
4. `logging_.py` emits structured JSON logs with `run_id` bound. It **redacts** any
   value matching the configured API-key env var anywhere in a log record.
5. `manifests.py` writes and reads run manifests containing git commit, python
   version, lockfile sha, config sha, and spec sha placeholders.
6. `errors.py` defines the exception hierarchy: `NflpropsError` ->
   `ConfigError`, `ProviderError` (-> `ProviderAuthError`, `ProviderRateLimitError`,
   `ProviderSchemaError`), `DataQualityError`, `LeakageError`, `InvariantViolation`,
   `ContractViolation`.

## Acceptance tests
```
tests/unit/test_config_layering.py
tests/unit/test_canonical_ids_stable.py     # same inputs -> same UUID, always
tests/unit/test_no_secret_leakage.py        # API key never appears in a log record
tests/unit/test_manifest_roundtrip.py
```

## Definition of done
- [ ] `nflprops --help` lists every command from spec §73
- [ ] `pytest tests/unit -q` green
- [ ] `make lint` and `make typecheck` green
- [ ] No network call exists anywhere in the codebase yet

## Explicitly out of scope
HTTP, BDL, parquet, models, simulation. Subcommands beyond `--help` may raise
`NotImplementedError`.
