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
(/home/wizard-deploy/nflprops)` -- the isolated runtime root approved by
the BLOCK 2B probe (see "BLOCK 2B final report" below) -- it never writes
to the existing
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
staging "incoming" area under `/home/wizard-deploy/nflprops/publications` --
installing a published bundle into live state is a separate, later,
explicit step this block does not implement (no final public prediction
publication yet).

**Wizard read-only probe** (`.github/workflows/wizard-probe.yml`,
`workflow_dispatch`): uses the same `WIZARD_SSH_*`/`wizardofodds.com`
mechanism to report CPU/RAM/swap/disk, candidate nflprops path
availability, write permission, existing systemd service names, and
listening ports -- strictly read-only (no install, no service management,
no nginx, no writes). See "BLOCK 2B final report" below for what it found
on the real host.

**`nflprops-runtime` systemd foundation**
(`deploy/systemd/nflprops-runtime.service`, not installed by anything in
this change): one runtime-owner process (simplifies the single-writer
guarantee -- every write already happens from inside this one unit), runs
as the existing `wizard-deploy` SSH user, absolute paths throughout,
`EnvironmentFile=/home/wizard-deploy/nflprops/nflprops-runtime.env`
(outside git, mode 600; template at
`deploy/systemd/nflprops-runtime.env.example`), `Restart=on-failure` with
a bounded `RestartSec`/`StartLimitBurst`, and hard lightweight-runtime
caps (`MemoryHigh=384M`, `MemoryMax=512M`, `MemorySwapMax=0`,
`TasksMax=16`, `CPUQuota=50%`). Its placeholder `ExecStart`
(`Type=oneshot`) is
`/home/wizard-deploy/nflprops/current/.venv/bin/python -m nflprops.cli
platform health --deploy-gate`, the read-only health gate and nothing
else. Continuous collection/checkpointing is a Block 3 decision and is
not started here. The unit is installed once by root (see "One-time root
setup" below); `deploy-wizard.yml` never writes `/etc` and never uses
root beyond the scoped `sudo -n systemctl restart
nflprops-runtime.service`.

**Release contract** (`deploy-wizard.yml` + `deploy/wizard/*.sh`,
streamed over SSH, main-only, `workflow_dispatch`-only, not invoked by
anything yet):

1. Upload the archive to `releases/<release-id>.tar.gz`, where
   `<release-id>` is `<sha>-<run_id>-<run_attempt>`, unique per deploy.
2. `prepare_release.sh` runs entirely **before** activation:
   - Preflight: Python ≥ 3.11 with `venv`, the installed unit matches
     this release's unit file byte for byte (SHA-256), the scoped sudo
     rule exists, and ≥ 3 GiB is free.
   - Extract into the immutable `releases/<release-id>/` and write
     `RELEASE_SHA`.
   - Build that release's **own** `.venv` and install
     `-e .[orchestration,runtime]`. The core dependencies plus Prefect and
     alembic only; no `challenger` (lightgbm/shap), no database driver,
     no object-store client.
   - Run the pre-activation checks from the candidate venv: `import
     nflprops`, `platform health --help`, `wizard_runtime --help`,
     `wizard_runtime snapshot --help`, a read-only `snapshot list`, and
     `platform health --deploy-gate --expect-version <sha>`.
   - Mark the release `.prepared`. Any failure deletes the partial
     release, and `current` is never touched.
3. `activate_release.sh` refuses a release that is not `.prepared`. It
   then switches `current` atomically (a new symlink plus `rename(2)`)
   and runs the **health gate**:
   - `sudo -n systemctl restart nflprops-runtime.service`
   - `systemctl is-active nflprops-runtime.service`
   - `current/.venv/bin/python -m nflprops.cli platform health
     --deploy-gate --expect-version <sha>`

   All three must pass. On failure it restores the previous `current`,
   restarts the service on it, and runs the same gate against the
   previous SHA. The exit status is always propagated:
   - 0: deployed
   - 1: rolled back, with rollback health verified (the workflow fails)
   - 2: **rollback failed** (hard fail)
   - 3: first deploy failed with no previous release, and `current` was
     removed (hard fail)
   - 4: refused before switching

   On success, only the active release and its rollback target are kept.
   Nothing in the deploy path uses `|| true`, and nothing references
   `nflprops-wizard-web` (no such service exists).

**Deploy gate.** `platform health --deploy-gate` bases its exit status on
every check except `latest_snapshot`. No snapshot exists until Block 3's
checkpointing starts, but the check is still reported.
`--expect-version` adds a critical `release_version` check. The running
SHA comes from `<release>/RELEASE_SHA` unless
`NFLPROPS_RUNTIME_VERSION_SHA` overrides it, so leave that variable unset
in the env file.

**Measured locally (not on Wizard):** I ran a real `prepare_release.sh`
followed by `activate_release.sh` against a scratch runtime root, with a
real venv, a real `pip install`, the real health gate and only systemctl
faked. Both succeeded. The venv took **~894 MiB** and about 50 s to
install. With two releases kept, venvs use ~1.8 GiB of the ~11 GB free.
This must be re-measured on the Wizard host during the first real
deploy.

**Extended health CLI** (`nflprops platform health`): now also reports
runtime version/SHA, warehouse path, warehouse readable/writable, writer
lock status, latest snapshot id/time/hash + verification status, disk
free, storage growth (live DuckDB/Parquet warehouse size, snapshot
count/size against bounded retention, publication size, and measured
snapshot-to-snapshot growth -- never an extrapolated capacity claim),
memory pressure (RAM available, swap used, Linux PSI `full avg60`),
current Alembic migration head, and explicit
collector/checkpoint placeholders (honestly "not yet activated -- Block
3", never a fabricated status). Every new check is read-only and never
creates a directory or contends for the writer lock as a side effect of
merely being asked.

## BLOCK 2B final report: measured Wizard probe

The measurements below were supplied by the operator from a read-only
probe of the real host. The GitHub-hosted attempt through a temporary
push trigger (`eb0de38`, reverted in `dc9db13`; run 35765851043) was
**rejected** by the `wizardofodds.com` environment's branch protection
before it connected, so these figures do not come from that run.

**Measured resources**

| Resource | Measured |
|---|---|
| CPU | 1 core |
| RAM | 1.9 GiB total, ~1.4 GiB available |
| Swap | 496 MiB, effectively fully used |
| Root filesystem | 49 GB total, 37 GB used, ~11 GB available (79%) |
| Existing nflprops production paths | none |
| `wizard-deploy` can write | `/home/wizard-deploy` |
| `wizard-deploy` cannot write | `/var/lib`, `/var/log`, `/opt/wizardofodds`, `/etc` |
| nflprops systemd service-name collision | none |
| Occupied ports | 21, 22, 80, 443, 3306, 33060, 8000, 8080, 8461 |

**Decision: `WIZARD_RESOURCE_PROBE_SAFE = YES`, with strict
lightweight-runtime constraints.**

The Wizard host is approved for canonical DuckDB state, the collector,
checkpoint scheduling/orchestration, immutable snapshot creation, health
monitoring, and later a lightweight API. It is **not** approved for the
20k-draw production simulation, historical replay, model training,
calibration optimization, or any other heavy compute. GitHub Actions
remains the heavy-compute platform. This does not reopen the
infrastructure architecture.

**Production layout** (`nflprops.platform.runtime_layout`). Every path
is writable by `wizard-deploy` without root and isolated from every
other workload:

```
/home/wizard-deploy/nflprops/
    current/        -> releases/<sha> (atomic symlink flip)
    releases/       <sha>-<run>-<attempt>/ with its own .venv; active + previous kept
    state/          NFLPROPS_DATA_ROOT=state/warehouse (+ state/nflprops.duckdb)
    snapshots/      immutable warehouse snapshots, bounded retention
    publications/   immutable GitHub result bundles
    backups/
    logs/
    locks/          writer.lock (single-writer guarantee)
    nflprops-runtime.env   (mode 600, outside git, never overwritten)
```

`NFLPROPS_RUNTIME_ROOT` selects the root. If it is unset, the runtime
falls back to the warehouse root's parent, so dev checkouts behave as
before. `resolve_runtime_layout` refuses any root or warehouse path that
overlaps `/home/wizard-deploy/nfl-production-2026` or
`/var/www/sportsodds`. Nothing touches the WNBA or game-model state.

**Constraints applied because swap is saturated**
- `memory_pressure` health check: RAM, swap, and PSI are always
  reported. Saturated swap alone is a WARNING in the detail. The check
  fails when MemAvailable < 256 MiB, when swap is ≥ 90% used and
  MemAvailable < 512 MiB, or when PSI `full avg60` > 10. When memory is
  tight, the runtime alerts; it never falls back to heavy local compute.
- One runtime-owner process under the unit's hard caps (`MemoryMax=512M`,
  `MemorySwapMax=0`, `TasksMax=16`, `CPUQuota=50%`, `Nice=10`).
- No memory-heavy caching was added.
- The future API runs as a single worker, bound to localhost only.

**Constraints applied because only ~11 GB is free**
- Bounded snapshot retention: `NFLPROPS_SNAPSHOT_RETENTION`, default 7,
  must be ≥ 1. `wizard_runtime snapshot create` prunes to it under the
  writer lock, and `snapshot prune` enforces it on demand.
- If the warehouse files and migration head have not changed since the
  latest snapshot, no duplicate snapshot is written.
- Release retention drops from 5 to 2: the active release and its
  rollback target. Each release carries its own ~894 MiB venv.
- The `storage_growth` health check reports live DuckDB/Parquet size,
  snapshot count and size against retention, publication size, and
  measured growth between the oldest and newest retained snapshot. It
  fails if retention is exceeded or if free space drops below 2 GiB
  (`disk_free` uses the same 2 GiB floor).
- **Season-long capacity is not claimed.** It can only be judged once
  real production growth has been measured.

**Ports.** 8000 and 8080 (and every other occupied port above) are
refused by `runtime_layout.validate_api_port`. The future API port is
chosen at deployment time, localhost only, after re-running `ss -tln`.
Passing the validator is necessary but not sufficient.

**Health under the zero-cost architecture.** When the backend is
`duckdb` and no `DATABASE_URL` is set, `database` reports "not required".
An unconfigured object store reports "optional". Neither one makes the
report unhealthy anymore. Before this change, `nflprops platform health`
could never pass on the locked architecture.

**Not done in BLOCK 2B (by design)**
- No deploy was dispatched, no production collection was started, and
  nothing was installed on the host.
- The one-time root setup (below) has not been run.
- Block 3 has not started. `main` was not merged.

**Previously known deploy blockers, now fixed in the closeout**
- Each release now builds its own `.venv` before activation (step 2
  above).
- The deploy health check no longer targets the nonexistent
  `nflprops-wizard-web` or ends in `|| true`. The gate is the real
  `nflprops-runtime.service` plus the versioned deploy-gate health, with
  verified rollback.

## One-time root setup (Block 3; NOT performed in BLOCK 2B)

GitHub's SSH user (`wizard-deploy`) has no root, and the deploy workflow
never asks for it. Until an operator with root runs the steps below,
`prepare_release.sh`'s preflight fails closed ("not installed" or
"sudoers"). All production files and state stay under
`/home/wizard-deploy/nflprops/`. Root owns only the unit file and one
sudoers line.

The installed unit must match the unit file of the release being deployed
**byte for byte**, because prepare checks its SHA-256. Copy it from the
exact commit you will deploy:

```bash
# 1. From a checkout of the SHA to be deployed, as wizard-deploy (no root):
scp -P "$WIZARD_SSH_PORT" deploy/systemd/nflprops-runtime.service \
  wizard-deploy@"$WIZARD_SSH_HOST":/home/wizard-deploy/nflprops-runtime.service.pending

# 2. On the Wizard host, as an administrator with root:
sudo install -m 644 -o root -g root \
  /home/wizard-deploy/nflprops-runtime.service.pending \
  /etc/systemd/system/nflprops-runtime.service
sudo systemctl daemon-reload

# Scoped rule: wizard-deploy may restart ONLY this unit, nothing else.
echo 'wizard-deploy ALL=(root) NOPASSWD: /usr/bin/systemctl restart nflprops-runtime.service' \
  | sudo tee /etc/sudoers.d/nflprops-runtime >/dev/null
sudo chmod 440 /etc/sudoers.d/nflprops-runtime
sudo visudo -cf /etc/sudoers.d/nflprops-runtime

# 3. Verify, as wizard-deploy (must list the command, no password prompt):
sudo -n -l /usr/bin/systemctl restart nflprops-runtime.service
rm /home/wizard-deploy/nflprops-runtime.service.pending

# 4. Only AFTER the first successful deploy-wizard.yml run (so `current`
#    exists), enable start-at-boot:
sudo systemctl enable nflprops-runtime.service
```

Any later change to `deploy/systemd/nflprops-runtime.service` needs step
2's `install` and `daemon-reload` again. Until then, deploys fail closed
with "installed nflprops-runtime.service differs". The sudoers rule
never needs to change.

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
  `WIZARD_SSH_USER`), and the read-only probe ran against the real host.
- ~~Deploy-user write permission for the state root~~ -- RESOLVED by the
  probe. `wizard-deploy` cannot write `/var/lib`, `/var/log`,
  `/opt/wizardofodds`, or `/etc`, so every nflprops path moved to
  `/home/wizard-deploy/nflprops/`, which it can write without root.
- One-time root setup on the Wizard host (Block 3, not yet done). See
  "One-time root setup" above for the exact commands: the unit file plus
  one scoped sudoers line.
- Once Platform and Science are integrated onto the same commit: the real
  `nflprops.calibration.phase10c3a_runner` (or whatever the finalized
  entry point is named) must accept the keyword arguments
  `remote_training.execute_remote_training` calls it with
  (`science_ref`, `data_manifest_sha256`, `data_dir`, `n_draws`, `mode`,
  `promotion_evidence_eligible`) and return a `dict` with the documented
  keys (`model_version`, `config_version`, `challenger_payload_hash`,
  `validation_result`, `promotion_eligibility_result`).
