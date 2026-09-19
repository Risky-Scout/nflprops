# Platform automation

GitHub-controlled remote compute for NFL simulation, historical replay,
retraining, and recalibration, so none of that runs on Joseph's Mac or on
the WizardOfOdds serving host. This document describes the platform layer
only -- it contains and changes no model science. See
`docs/IMPLEMENTATION_SPEC.md` and `docs/phases/` for the science build.

## Scope boundary

Platform consumes Science through an explicit git SHA
(`science_ref`), never by copying or rewriting Science code, and never by
assuming "whatever is latest." Platform and Science currently advance on
independent branches; until they're integrated onto the same commit,
`science_ref` checkouts will not contain `nflprops.platform`, and Platform
checkouts will not contain a Science entry point. Both directions fail
closed -- see `nflprops.platform.remote_training.resolve_science_entrypoint`
-- rather than silently succeeding with nothing done.

Untouched by this layer: `src/nflprops/calibration/`,
`src/nflprops/distributions/`, `src/nflprops/simulation/`,
`src/nflprops/projections/`, `src/nflprops/thresholds/`,
`src/nflprops/market/current_pricing.py`, simulation/calibration/pricing
math, promotion thresholds, `PropType` definitions, scientific
compatibility rules, and the public schema. No Alembic migration is part of
this change -- see `.github/workflows/ci.yml`'s `platform-scope-guard` job,
which fails any PR that touches either boundary.

## Components

| Component | Where |
|---|---|
| Remote training workflow | `.github/workflows/remote-training.yml` |
| Remote training harness (inputs, SHA verification, entry-point resolution, run report) | `src/nflprops/platform/remote_training.py` |
| Training-data snapshot manifest + verified download | `src/nflprops/platform/data_snapshot.py` |
| Operational health reporting | `src/nflprops/platform/health.py`, `nflprops platform health` |
| WizardOfOdds deployment foundation (not yet triggered) | `.github/workflows/deploy-wizard.yml` |
| Ordinary CI (now includes byte-compile, migration-head validation, scope guard) | `.github/workflows/ci.yml` |

## Remote training workflow contract

Dispatched manually (`workflow_dispatch`) with required inputs
`science_ref` (explicit 40-character git SHA), `data_manifest_sha256`
(explicit SHA-256 of the pinned training-data manifest),
`manifest_object_key` (where that manifest lives in object storage), and
`mode` (`production` or `smoke`).

1. Rejects a non-SHA `science_ref` before doing anything else.
2. Checks out this repository at exactly `science_ref` and verifies
   `git rev-parse HEAD` equals it byte-for-byte.
3. Downloads and SHA-256-verifies the pinned training-data snapshot into an
   ephemeral runner workspace (`nflprops platform remote-training
   prepare-data`) -- get-only against the object store; the source
   snapshot is never mutated.
4. Invokes the Science entry point named by `entry_point` (default
   `nflprops.calibration.phase10c3a_runner`) via `nflprops.platform.
   remote_training.resolve_science_entrypoint`, passing the locked
   `n_draws`.
5. Writes a machine-readable run report (repository, science SHA, workflow
   SHA, data manifest SHA, model/config versions where available, n_draws,
   start/end timestamps, runner identity, exit status, challenger
   payload/hash metadata, validation result, promotion eligibility result)
   and uploads it as a workflow artifact, `if: always()`.
6. Always cleans the ephemeral training-data workspace, `if: always()`.

`mode=production` always forces `n_draws=20000`
(`remote_training.PRODUCTION_N_DRAWS`) -- no input can override it.
`mode=smoke` is capped at `remote_training.MAX_SMOKE_N_DRAWS` (2,000, far
below production) and its run report always reports
`promotion_evidence_eligible=false`, independent of whatever the Science
entry point itself returns.

Production runs serialize against each other via a fixed concurrency group
(`remote-training-production`, `cancel-in-progress: false`); smoke runs
never block or get blocked by production.

This module never imports or calls
`nflprops.calibration.registry.promote_calibration_champion` (or any other
promotion function) -- champion promotion stays a separate, explicit,
human-gated action outside this workflow. See
`tests/platform/test_remote_training.py::test_module_never_references_promotion`.

Any failure -- checkout, SHA verification, data download, manifest
verification, missing Science entry point, or a Science-side exception --
fails the GitHub job. Nothing here silently continues with default data,
silently reduces `n_draws`, or auto-promotes a challenger. The last-known-good
champion is unaffected by anything in this workflow.

## Ordinary CI

`.github/workflows/ci.yml`'s `test` job already runs contract validation,
the runtime-resource sync check, `ruff`, `pytest -q` (which already
includes the Docker-gated PostgreSQL/MinIO integration tests -- see the
`docker` pytest marker in `pyproject.toml` -- since GitHub-hosted
`ubuntu-latest` runners ship Docker), and now a byte-compile pass
(`python -m compileall src`). `migration-head` validates the Alembic
migration graph resolves to exactly one head. `platform-scope-guard` fails
any pull request that touches a science-scoped path or adds/modifies an
Alembic migration. None of this runs in `remote-training.yml`, which stays
a separate, manually-dispatched, potentially multi-hour workflow.

## WizardOfOdds deployment foundation

`.github/workflows/deploy-wizard.yml` is a foundation only: it is not
triggered by any push and nothing in this change invokes it. It enforces
`github.ref == refs/heads/main`, uses the `wizardofodds.com` GitHub
environment, authenticates with the existing `WIZARD_SSH_*` secrets over
key-based SSH with `StrictHostKeyChecking=yes` against the pinned
`WIZARD_SSH_KNOWN_HOSTS`, and performs an atomic release (upload to a
SHA-named release directory, extract, flip a `current` symlink, prune old
releases, rollback the symlink on a failed post-deploy health check).
Everything it touches is namespaced under `$RELEASE_ROOT
(/opt/wizardofodds/nflprops-releases)` -- it never writes to the existing
WNBA deployment, the existing separate NFL game-model deployment, nginx, or
DNS.

## EXTERNAL_PROVISIONING_STILL_REQUIRED

- A self-hosted GitHub Actions runner registered with labels
  `[self-hosted, nflprops-training]`, on a dedicated remote Linux machine
  capable of multi-hour CPU workloads -- not Joseph's Mac, not the
  WizardOfOdds 1-vCPU/~2GB host. `remote-training.yml`'s job simply queues
  (never silently falls back elsewhere) until this exists.
- Object storage (S3-compatible; MinIO or AWS S3) provisioned for
  production, with `OBJECT_STORE_ENDPOINT`/`OBJECT_STORE_REGION`/
  `OBJECT_STORE_BUCKET`/`OBJECT_STORE_ACCESS_KEY`/`OBJECT_STORE_SECRET_KEY`
  added as repository (or environment) secrets consumed by
  `remote-training.yml`.
- Production PostgreSQL, with `DATABASE_URL` added as a secret consumed by
  `remote-training.yml`.
- The GitHub `wizardofodds.com` environment, with `WIZARD_SSH_HOST`,
  `WIZARD_SSH_KNOWN_HOSTS`, `WIZARD_SSH_PORT`, `WIZARD_SSH_PRIVATE_KEY`,
  `WIZARD_SSH_USER` (already-documented secret names, per the platform
  brief) actually populated, and `$RELEASE_ROOT` on that host created with
  the deploy user's write permission.
- Once Platform and Science are integrated onto the same commit: the real
  `nflprops.calibration.phase10c3a_runner` (or whatever the finalized
  entry point is named) must accept the keyword arguments
  `remote_training.execute_remote_training` calls it with
  (`science_ref`, `data_manifest_sha256`, `data_dir`, `n_draws`, `mode`,
  `promotion_evidence_eligible`) and return a `dict` with the documented
  keys (`model_version`, `config_version`, `challenger_payload_hash`,
  `validation_result`, `promotion_eligibility_result`).
