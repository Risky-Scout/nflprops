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
| Remote training workflow (manual dispatch; GitHub-hosted `ubuntu-24.04`) | `.github/workflows/remote-training.yml` |
| Remote training smoke workflow (push-triggered; GitHub-hosted `ubuntu-24.04`; synthetic fixture data only) | `.github/workflows/remote-training-smoke.yml` |
| Synthetic fixture-warehouse generator for the smoke workflow | `tools/generate_smoke_warehouse.py` |
| Remote training harness (inputs, SHA verification, entry-point resolution, run report) | `src/nflprops/platform/remote_training.py` |
| Training-data snapshot manifest + verified download | `src/nflprops/platform/data_snapshot.py` |
| Operational health reporting | `src/nflprops/platform/health.py`, `nflprops platform health` |
| WizardOfOdds deployment foundation (not yet triggered) | `.github/workflows/deploy-wizard.yml` |
| Ordinary CI (now includes byte-compile, migration-head validation, scope guard) | `.github/workflows/ci.yml` |
| BLOCK 2B: OS-level single-writer lock for the live warehouse | `src/nflprops/platform/writer_lock.py` |
| BLOCK 2B: immutable manifest/atomic-publish primitive (snapshots + result bundles) | `src/nflprops/platform/immutable_bundle.py` |
| BLOCK 2B: canonical warehouse snapshot lifecycle (create/list/verify/restore) | `src/nflprops/platform/warehouse_snapshot.py` |
| BLOCK 2B: Wizard-host runtime CLI (`python -m nflprops.platform.wizard_runtime`) | `src/nflprops/platform/wizard_runtime.py` |
| BLOCK 2B: GitHub <-> Wizard snapshot download / result-bundle upload (foundation, not yet triggered) | `.github/workflows/wizard-snapshot-transfer.yml` |
| BLOCK 2B: read-only Wizard resource/collision probe | `.github/workflows/wizard-probe.yml` |
| BLOCK 2B: `nflprops-runtime` systemd unit foundation | `deploy/systemd/nflprops-runtime.service` |

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

Runs on the GitHub-hosted `ubuntu-24.04` runner -- no self-hosted machine is
required. Real production/smoke dispatches still require the object-store
and `DATABASE_URL` secrets listed under EXTERNAL_PROVISIONING_STILL_REQUIRED
below.

## GitHub-hosted execution proof (`remote-training-smoke.yml`)

Because `workflow_dispatch` for `remote-training.yml` is only reachable from
the GitHub UI/API once the workflow file exists on the default branch
(`main`), `.github/workflows/remote-training-smoke.yml` proves the same
GitHub-hosted Platform -> Science execution path on every push to
`dev/nflprops-production`, ahead of that merge:

1. Checks out at `github.sha` and re-verifies `git rev-parse HEAD` matches.
2. Generates a small synthetic 2-season warehouse in-repo
   (`tools/generate_smoke_warehouse.py`) -- never the real historical
   snapshot, never an object-store download. No `OBJECT_STORE_*` or
   `DATABASE_URL` secret is read by this workflow.
3. Invokes `nflprops.platform.remote_training run --mode smoke` against that
   fixture directly via `--data-dir`, exercising the identical Platform ->
   Science handoff `remote-training.yml` uses.
4. Asserts the resulting report has `mode=smoke`,
   `promotion_evidence_eligible=false`, and `n_draws` within
   `MAX_SMOKE_N_DRAWS` before declaring success.
5. Records runner OS/CPU/RAM, disk before/after, elapsed time, max RSS
   (via `/usr/bin/time -v` where available), and exit status alongside the
   run report as an uploaded artifact, `if: always()`.

`--mode` is hardcoded to `smoke` in this workflow file -- there is no input
that could escalate it to `production` from a push -- and
`nflprops.platform.remote_training` independently forces
`promotion_evidence_eligible=False` for any non-production mode regardless
of what the Science entry point itself reports.

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

## BLOCK 2B: zero-cost production state + Wizard runtime foundation

**Architecture lock** (do not reopen without a measured production
resource/concurrency failure): GitHub Actions remains heavy compute/CI/
training/validation/deployment; the existing WizardOfOdds server is the
persistent nflprops runtime (canonical state, collector, checkpoint
scheduler-worker, a future API); DuckDB + versioned immutable files is
durable production state. No PostgreSQL service, no Oracle, no
DigitalOcean compute/database, no Cloudflare R2, no new subscription --
`StorageSettings` (`nflprops.data.storage.settings`) now accepts
`NFLPROPS_ENV=production` with `NFLPROPS_STORAGE_BACKEND=duckdb` (an
explicit absolute `NFLPROPS_DATA_ROOT` is still required; the relative dev
default is rejected) alongside the pre-existing postgres path, which
remains supported but is no longer required.

**Single-writer guarantee.** `nflprops.platform.writer_lock.WriterLock` is
a bounded, OS-level (`fcntl.flock`) interprocess lock at
`<state_root>/locks/writer.lock` (`default_lock_path`) -- the ONE
coordination point every process that mutates the live warehouse must go
through. Acquisition is non-blocking-polled with a bounded deadline and
fails closed (`WriterLockTimeoutError`) rather than hanging forever;
staleness needs no separate check because the kernel releases a held
`flock` the instant its owning process's file descriptors close, including
on a crash.

**Immutable snapshot contract.** `nflprops.platform.warehouse_snapshot.
create_snapshot` is the one function allowed to read the live warehouse for
durability: under the writer lock, it `CHECKPOINT`s the ancillary DuckDB
query-layer file if one exists, copies every warehouse file (+ the
checkpointed `.duckdb` copy) into a staging directory, builds a
deterministic manifest (snapshot_id, created_at, source identity incl.
hostname + Alembic migration head, per-file SHA-256/byte-count, top-level
manifest SHA-256 -- `nflprops.platform.immutable_bundle`), verifies it, and
atomically renames it into `<state_root>/snapshots/<snapshot_id>/`. Never
overwrites an existing snapshot with different content
(`BundleConflictError`). `list_snapshots`/`verify_snapshot`/
`restore_snapshot` round out the lifecycle; `restore_snapshot` always
targets a fresh path and structurally refuses to target the live
warehouse root.

**GitHub <-> Wizard transfer** (`.github/workflows/wizard-snapshot-transfer.yml`,
foundation only, `workflow_dispatch`, not yet triggered by anything):
`download-snapshot` takes an explicit `snapshot_id` + explicit
`expected_manifest_sha256`, SCPs the one named snapshot directory into
`RUNNER_TEMP` over strict-known-hosts SSH, and verifies it
(`nflprops.platform.wizard_runtime bundle-verify`) before trusting
anything -- it never opens the live DuckDB file and never writes back to
the Wizard host. `upload-result-bundle` builds+verifies a result bundle
locally (run/report identity, science SHA, workflow SHA, data snapshot id,
and whatever payload files apply) and hands it to the WIZARD RUNTIME
OWNER's own already-deployed CLI over SSH to atomically publish into a
staging "incoming" area under `/var/lib/nflprops/publications` --
installing a published bundle into live state is a separate, later,
explicit step this block does not implement (no final public prediction
publication yet).

**Wizard read-only probe** (`.github/workflows/wizard-probe.yml`,
`workflow_dispatch`): uses the same `WIZARD_SSH_*`/`wizardofodds.com`
mechanism to report CPU/RAM/swap/disk, candidate nflprops path
availability, write permission, existing systemd service names, and
listening ports -- strictly read-only (no install, no service management,
no nginx, no writes). See the BLOCK 2B final report for what it found on
the real host.

**`nflprops-runtime` systemd foundation**
(`deploy/systemd/nflprops-runtime.service`, not installed by anything in
this change): one runtime-owner process (simplifies the single-writer
guarantee -- every write already happens from inside this one unit), runs
as the existing `wizard-deploy` SSH user, absolute paths throughout,
`EnvironmentFile=/etc/nflprops/nflprops-runtime.env` (outside git;
template at `deploy/systemd/nflprops-runtime.env.example`),
`Restart=on-failure` with a bounded `RestartSec`/`StartLimitBurst`. Its
placeholder `ExecStart` (`Type=oneshot`) only runs `nflprops platform
health` -- continuous collection/checkpointing is a Block 3 decision, not
started here. `deploy-wizard.yml` additively installs/updates this unit
(via non-interactive `sudo -n`, skipping with a warning rather than
hanging if passwordless sudo isn't configured) and extends its post-deploy
health check/rollback to cover both `nflprops-wizard-web` and
`nflprops-runtime` -- still main-only, still `workflow_dispatch`-only,
still not invoked by anything in this change.

**Extended health CLI** (`nflprops platform health`): now also reports
runtime version/SHA, warehouse path, warehouse readable/writable, writer
lock status, latest snapshot id/time/hash + verification status, disk
free, memory available, current Alembic migration head, and explicit
collector/checkpoint placeholders (honestly "not yet activated -- Block
3", never a fabricated status). Every new check is read-only and never
creates a directory or contends for the writer lock as a side effect of
merely being asked.

## EXTERNAL_PROVISIONING_STILL_REQUIRED

- Object storage (S3-compatible; MinIO or AWS S3) -- OPTIONAL as of
  BLOCK 2B (no longer required for production). Only needed if
  `remote-training.yml` production/smoke dispatches still want the
  object-store-backed training-data snapshot path
  (`nflprops.platform.data_snapshot`); provision with
  `OBJECT_STORE_ENDPOINT`/`OBJECT_STORE_REGION`/`OBJECT_STORE_BUCKET`/
  `OBJECT_STORE_ACCESS_KEY`/`OBJECT_STORE_SECRET_KEY` if/when needed.
- Production PostgreSQL -- OPTIONAL as of BLOCK 2B (no longer required).
  The locked zero-cost architecture is DuckDB + versioned immutable
  snapshots on the Wizard host; only provision `DATABASE_URL` if a future
  decision reopens the PostgreSQL path.
- ~~The GitHub `wizardofodds.com` environment, with `WIZARD_SSH_*` secrets
  populated~~ -- DONE: confirmed populated (`WIZARD_SSH_HOST`,
  `WIZARD_SSH_KNOWN_HOSTS`, `WIZARD_SSH_PORT`, `WIZARD_SSH_PRIVATE_KEY`,
  `WIZARD_SSH_USER`) as of BLOCK 2B's read-only probe. `$RELEASE_ROOT`
  (`/opt/wizardofodds/nflprops-releases`) and the BLOCK 2B state root
  (`/var/lib/nflprops`, `/var/log/nflprops`, `/etc/nflprops`) on that host
  still need the deploy user's write permission confirmed/created before
  `deploy-wizard.yml` or `wizard-snapshot-transfer.yml` are actually
  dispatched -- see the BLOCK 2B final report's probe results.
- Once Platform and Science are integrated onto the same commit: the real
  `nflprops.calibration.phase10c3a_runner` (or whatever the finalized
  entry point is named) must accept the keyword arguments
  `remote_training.execute_remote_training` calls it with
  (`science_ref`, `data_manifest_sha256`, `data_dir`, `n_draws`, `mode`,
  `promotion_evidence_eligible`) and return a `dict` with the documented
  keys (`model_version`, `config_version`, `challenger_payload_hash`,
  `validation_result`, `promotion_eligibility_result`).
