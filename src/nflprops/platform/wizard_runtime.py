"""The Wizard-host runtime CLI (BLOCK 2B + BLOCK 3), invoked via
`python -m nflprops.platform.wizard_runtime <command>` (also mounted as
`nflprops wizard-runtime`) by systemd and the main-only GitHub workflows.

BLOCK 3:
  run                 the long-running runtime (systemd ExecStart)
  once                ONE explicit MANUAL collection cycle (certification)
  status              read-only runtime status / next due work
  report              read-only warehouse certification report
  checkpoint manual   claim + prepare ONE explicit MANUAL checkpoint
  checkpoint pending  read-only list of checkpoints pending GitHub execution
BLOCK 4:
  ingest-stats        append completed-game outcome history (final games only)
  execute-checkpoint  GITHUB ONLY: verify + run one prepared checkpoint (20k)
  result-ingest       validate + install a published GitHub result bundle
BLOCK 2B:
  snapshot create|list|verify|restore|prune, lock-status, result-bundle
  build, bundle-verify|publish|stage-path

Paths come from the collector's own config (`run.data_root`, i.e.
NFLPROPS_DATA_ROOT; the warehouse is `<data_root>/canonical`) and the
probe-approved runtime layout (`NFLPROPS_RUNTIME_ROOT`) -- see
`nflprops.platform.runtime_layout`.

This module never trains, never recalibrates, never simulates: the only
science-adjacent work on the Wizard host is PIT collection and checkpoint
PREPARATION (claim + manifest + snapshot); heavy execution is GitHub's.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import typer

from nflprops.data.storage.settings import StorageSettings
from nflprops.platform.immutable_bundle import (
    BundleError,
    build_manifest,
    publish_atomically,
    read_manifest,
    stage_bundle_dir,
    verify_directory_against_manifest,
    write_manifest,
)
from nflprops.platform.runtime_layout import (
    RuntimeLayout,
    resolve_layout_from_config,
    snapshot_retention,
)
from nflprops.platform.warehouse_snapshot import (
    WarehouseSnapshotError,
    create_snapshot,
    list_snapshots,
    prune_snapshots,
    restore_snapshot,
    verify_snapshot,
)
from nflprops.platform.writer_lock import WriterLockError

app = typer.Typer(
    name="wizard-runtime",
    help="Wizard-host lightweight runtime: collection, checkpoint preparation, snapshots.",
    no_args_is_help=True,
)

snapshot_app = typer.Typer(help="Immutable warehouse snapshot lifecycle.")
result_bundle_app = typer.Typer(help="GitHub result-bundle transport/verification.")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(result_bundle_app, name="result-bundle")


def _layout() -> RuntimeLayout:
    # StorageSettings still enforces the production guard (absolute
    # NFLPROPS_DATA_ROOT); the warehouse path itself is the collector's.
    StorageSettings.from_env()
    return resolve_layout_from_config()


def _protected_snapshots(layout: RuntimeLayout) -> frozenset[str]:
    """Snapshot ids a pending checkpoint request references (never
    pruned). Read-only; never creates the warehouse directory."""
    from nflprops.platform.checkpoint_prepare import protected_snapshot_ids_at

    return protected_snapshot_ids_at(layout.warehouse_root)


def _warehouse_and_snapshot_roots() -> tuple[Path, Path]:
    layout = _layout()
    return layout.warehouse_root, layout.snapshots


@snapshot_app.command("create")
def snapshot_create(
    migration_head: str = typer.Option(
        ..., help="The Alembic migration head at the time of this snapshot."
    ),
    lock_timeout_seconds: float = typer.Option(
        60.0, help="Bounded wait for the writer lock before failing closed."
    ),
    keep: int = typer.Option(
        0,
        help="Snapshots to retain after creating this one (0 = "
        "NFLPROPS_SNAPSHOT_RETENTION, default 7). Retention is never unlimited.",
    ),
) -> None:
    """Checkpoint + copy the live warehouse into a new immutable snapshot,
    then enforce bounded retention. The ONE command that reads the live
    warehouse for durability purposes."""
    layout = _layout()
    retain = keep or snapshot_retention()
    try:
        info = create_snapshot(
            warehouse_root=layout.warehouse_root,
            snapshot_root=layout.snapshots,
            lock_path=layout.writer_lock,
            lock_timeout_seconds=lock_timeout_seconds,
            migration_head=migration_head,
            hostname=socket.gethostname(),
        )
        pruned = prune_snapshots(
            layout.snapshots,
            keep=retain,
            protected=_protected_snapshots(layout),
            lock_path=layout.writer_lock,
            lock_timeout_seconds=lock_timeout_seconds,
        )
    except (WarehouseSnapshotError, WriterLockError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        f"SUCCEEDED: snapshot_id={info.snapshot_id} files={info.file_count} "
        f"bytes={info.total_bytes} manifest_sha256={info.manifest_sha256} "
        f"retained={retain} pruned={len(pruned)}"
    )


@snapshot_app.command("prune")
def snapshot_prune(
    keep: int = typer.Option(
        0, help="Snapshots to retain (0 = NFLPROPS_SNAPSHOT_RETENTION, default 7)."
    ),
) -> None:
    """Delete all but the newest `keep` snapshots (bounded retention)."""
    layout = _layout()
    try:
        pruned = prune_snapshots(
            layout.snapshots,
            keep=keep or snapshot_retention(),
            protected=_protected_snapshots(layout),
            lock_path=layout.writer_lock,
        )
    except (WarehouseSnapshotError, WriterLockError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    for snapshot_id in pruned:
        typer.echo(f"PRUNED: {snapshot_id}")
    typer.echo(f"SUCCEEDED: pruned={len(pruned)}")


@snapshot_app.command("list")
def snapshot_list() -> None:
    _warehouse_root, snapshot_root = _warehouse_and_snapshot_roots()
    for info in list_snapshots(snapshot_root):
        typer.echo(
            f"{info.snapshot_id}\tcreated_at={info.created_at}\t"
            f"files={info.file_count}\tbytes={info.total_bytes}\t"
            f"sha256={info.manifest_sha256}"
        )


@snapshot_app.command("verify")
def snapshot_verify(snapshot_id: str = typer.Argument(...)) -> None:
    _warehouse_root, snapshot_root = _warehouse_and_snapshot_roots()
    try:
        info = verify_snapshot(snapshot_root, snapshot_id)
    except WarehouseSnapshotError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"VERIFIED: {info.snapshot_id} sha256={info.manifest_sha256}")


@snapshot_app.command("restore")
def snapshot_restore(
    snapshot_id: str = typer.Argument(...),
    dest_path: str = typer.Argument(
        ..., help="A NEW, empty path -- never the live warehouse."
    ),
) -> None:
    """Restore a verified snapshot into `dest_path`, a fresh recovery
    location. Never overwrites the live warehouse -- see
    `nflprops.platform.warehouse_snapshot.restore_snapshot`."""
    warehouse_root, snapshot_root = _warehouse_and_snapshot_roots()
    try:
        info = restore_snapshot(
            snapshot_root, snapshot_id, Path(dest_path), warehouse_root=warehouse_root
        )
    except WarehouseSnapshotError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"RESTORED: {info.snapshot_id} -> {dest_path}")


@app.command("lock-status")
def lock_status() -> None:
    """Non-blocking report of whether the writer lock is currently held."""
    from nflprops.platform.writer_lock import WriterLock

    lock_path = _layout().writer_lock
    held = WriterLock(lock_path).is_locked_by_other()
    typer.echo(f"lock_path={lock_path} held_by_other={held}")


# ---------------------------------------------------------- result bundles


@result_bundle_app.command("build")
def result_bundle_build(
    bundle_id: str = typer.Option(..., help="Explicit, unique result-bundle id."),
    source_dir: str = typer.Option(
        ..., help="Directory of already-produced files to package (run report, "
        "validation report, calibration payload, etc.)."
    ),
    staging_dir: str = typer.Option(
        ..., help="Empty directory to stage the bundle + manifest into before transfer."
    ),
    science_sha: str = typer.Option(..., help="Science git SHA that produced this bundle."),
    workflow_sha: str = typer.Option(..., help="Workflow git SHA that ran the job."),
    data_snapshot_id: str = typer.Option(
        "", help="The Wizard snapshot_id this run consumed, if any."
    ),
) -> None:
    """Copy `source_dir`'s files into `staging_dir` and write a deterministic
    manifest -- the GitHub-side half of the result-bundle contract (task
    §5). This never installs anything into live state; a later, separate,
    explicit step (the Wizard runtime owner) decides whether/how to consume
    a published bundle."""
    import shutil

    src = Path(source_dir)
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        target = staging / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    source_identity = {
        "science_sha": science_sha,
        "workflow_sha": workflow_sha,
        "data_snapshot_id": data_snapshot_id or None,
    }
    manifest = build_manifest(
        bundle_id=bundle_id,
        source_identity=source_identity,
        schema_version="nflprops.platform.wizard_runtime.result_bundle/v1",
        root_dir=staging,
    )
    write_manifest(manifest, staging)
    verify_directory_against_manifest(staging, manifest)
    typer.echo(f"SUCCEEDED: staged bundle_id={bundle_id} manifest_sha256={manifest.manifest_sha256}")


# --------------------------------------------------------------------
# Generic immutable-bundle commands -- shared by BOTH halves of the
# transfer contract: a downloaded warehouse snapshot (task §4) and a
# result bundle either half of the round trip (task §5), since both are
# just an `nflprops.platform.immutable_bundle` directory + manifest.json.


@app.command("bundle-verify")
def bundle_verify(
    bundle_dir: str = typer.Option(..., help="A downloaded (or local) bundle directory."),
    expected_manifest_sha256: str = typer.Option(
        ..., help="The manifest SHA-256 the caller independently expects."
    ),
) -> None:
    """Verify a bundle directory (downloaded snapshot OR result bundle)
    against its own manifest AND an independently-known expected SHA-256.
    Fails closed on any mismatch -- shared by both the snapshot-download
    and result-bundle-upload halves of the transfer contract."""
    root = Path(bundle_dir)
    try:
        manifest = read_manifest(root)
        verify_directory_against_manifest(
            root, manifest, expected_manifest_sha256=expected_manifest_sha256
        )
    except BundleError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"VERIFIED: bundle_id={manifest.bundle_id} sha256={manifest.manifest_sha256}")


@app.command("bundle-publish")
def bundle_publish(
    staging_dir: str = typer.Option(..., help="A verified staged bundle directory."),
    final_dir: str = typer.Option(..., help="The immutable destination directory."),
) -> None:
    """Atomically rename a verified staged bundle into its final immutable
    location (same-volume rename). Never overwrites different content at
    `final_dir` -- see `nflprops.platform.immutable_bundle.publish_atomically`."""
    try:
        published = publish_atomically(Path(staging_dir), Path(final_dir))
    except BundleError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc

    status = "PUBLISHED" if published else "ALREADY_PUBLISHED_IDENTICAL"
    typer.echo(f"{status}: {final_dir}")


@app.command("bundle-stage-path")
def bundle_stage_path(
    final_dir: str = typer.Option(...),
    bundle_id: str = typer.Option(...),
) -> None:
    """Print a fresh same-volume staging directory path for `final_dir`
    (a thin CLI wrapper so a shell workflow step can create one without
    importing Python directly)."""
    staging = stage_bundle_dir(Path(final_dir), bundle_id=bundle_id)
    typer.echo(str(staging))


# ------------------------------------------------------------ BLOCK 3 runtime

checkpoint_app = typer.Typer(help="Checkpoint scheduling/preparation (never execution).")
app.add_typer(checkpoint_app, name="checkpoint")


def _runtime_components(
    *, with_provider: bool = True
) -> tuple[RuntimeLayout, Any, Any, Any, str | None]:
    """(layout, config, warehouse, provider|None, release_sha). The
    provider is the certified registry-built one (BDL by default); a
    missing API key fails closed here, before any write."""
    import os

    from nflprops.config import load
    from nflprops.platform.runtime_layout import running_release_sha

    layout = _layout()
    cfg = load()
    provider = None
    if with_provider:
        from nflprops.pipelines.lean import (
            LeanIngestor,  # noqa: F401 -- registers "bdl"
        )
        from nflprops.providers.registry import get_provider

        provider, warehouse = get_provider(os.environ.get("NFLPROPS_PROVIDER", "bdl"), cfg)
    else:
        from nflprops.data.warehouse import Warehouse

        warehouse = Warehouse(layout.warehouse_root, layout.data_root / "nflprops.duckdb")
    if warehouse.root.resolve() != layout.warehouse_root.resolve():
        raise RuntimeError(
            f"provider warehouse {warehouse.root} != runtime warehouse {layout.warehouse_root}"
        )
    return layout, cfg, warehouse, provider, running_release_sha()


def _build_loop(
    layout: RuntimeLayout, cfg: Any, warehouse: Any, provider: Any, release_sha: str | None
) -> Any:
    import os

    from nflprops.pipelines.lean import LeanIngestor
    from nflprops.platform.checkpoint_retry import (
        DEFAULT_SLOT_RETRY_BASE_SECONDS,
        DEFAULT_SLOT_RETRY_MAX_SECONDS,
    )
    from nflprops.platform.checkpoint_worker import DEFAULT_WORKER_TIMEOUT_SECONDS
    from nflprops.platform.runtime_layout import current_migration_head
    from nflprops.platform.runtime_loop import (
        DEFAULT_CHECKPOINT_BATCH_LIMIT,
        DEFAULT_CHECKPOINT_COOLDOWN_SECONDS,
        DEFAULT_CHECKPOINT_STARTUP_GRACE_SECONDS,
        DEFAULT_OUTCOME_INGEST_INTERVAL_SECONDS,
        DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
        DEFAULT_TICK_SECONDS,
        RuntimeLoop,
    )

    def _bootstrap() -> None:
        LeanIngestor(
            provider, warehouse, goat=bool(cfg.get_path("provider.bdl.tier.goat", False))
        ).bootstrap()

    from nflprops.platform.storage_guard import resolve_raw_retention_days

    retention_days, retention_warning = resolve_raw_retention_days(
        os.environ.get("NFLPROPS_RAW_RETENTION_DAYS")
    )
    if retention_warning:
        typer.echo(f"WARNING: {retention_warning}", err=True)
    return RuntimeLoop(
        layout=layout,
        warehouse=warehouse,
        config=cfg,
        provider=provider,
        migration_head=current_migration_head(),
        release_sha=release_sha,
        tick_seconds=float(os.environ.get("NFLPROPS_RUNTIME_TICK_SECONDS", DEFAULT_TICK_SECONDS)),
        snapshot_interval_seconds=float(
            os.environ.get("NFLPROPS_SNAPSHOT_INTERVAL_SECONDS", DEFAULT_SNAPSHOT_INTERVAL_SECONDS)
        ),
        snapshot_retention=snapshot_retention(),
        reference_bootstrap=_bootstrap,
        checkpoint_timeout_seconds=float(
            os.environ.get(
                "NFLPROPS_CHECKPOINT_PREPARE_TIMEOUT_SECONDS", DEFAULT_WORKER_TIMEOUT_SECONDS
            )
        ),
        checkpoint_batch_limit=max(
            1,
            int(
                os.environ.get(
                    "NFLPROPS_CHECKPOINT_PREPARE_BATCH_LIMIT", DEFAULT_CHECKPOINT_BATCH_LIMIT
                )
            ),
        ),
        checkpoint_cooldown_seconds=max(
            0.0,
            float(
                os.environ.get(
                    "NFLPROPS_CHECKPOINT_PREPARE_COOLDOWN_SECONDS",
                    DEFAULT_CHECKPOINT_COOLDOWN_SECONDS,
                )
            ),
        ),
        checkpoint_startup_grace_seconds=max(
            0.0,
            float(
                os.environ.get(
                    "NFLPROPS_CHECKPOINT_PREPARE_STARTUP_GRACE_SECONDS",
                    DEFAULT_CHECKPOINT_STARTUP_GRACE_SECONDS,
                )
            ),
        ),
        checkpoint_slot_retry_base_seconds=max(
            1.0,
            float(
                os.environ.get(
                    "NFLPROPS_CHECKPOINT_SLOT_RETRY_BASE_SECONDS",
                    DEFAULT_SLOT_RETRY_BASE_SECONDS,
                )
            ),
        ),
        checkpoint_slot_retry_max_seconds=max(
            1.0,
            float(
                os.environ.get(
                    "NFLPROPS_CHECKPOINT_SLOT_RETRY_MAX_SECONDS",
                    DEFAULT_SLOT_RETRY_MAX_SECONDS,
                )
            ),
        ),
        raw_retention_days=retention_days,
        outcome_ingest_interval_seconds=float(
            os.environ.get(
                "NFLPROPS_OUTCOME_INGEST_INTERVAL_SECONDS", DEFAULT_OUTCOME_INGEST_INTERVAL_SECONDS
            )
        ),
    )


@app.command("run")
def runtime_run() -> None:
    """The long-running lightweight runtime (systemd `Type=simple`):
    locked-cadence PIT collection + official checkpoint scheduling/
    preparation + bounded snapshots. Stops cleanly on SIGTERM/SIGINT.
    Never runs checkpoint science, training, replay, or calibration."""
    import os

    from nflprops.platform.runtime_loop import configure_json_logging

    configure_json_logging(os.environ.get("NFLPROPS_LOG_LEVEL", "INFO"))
    layout, cfg, warehouse, provider, release_sha = _runtime_components()
    loop = _build_loop(layout, cfg, warehouse, provider, release_sha)
    raise typer.Exit(loop.run())


@app.command("once")
def runtime_once() -> None:
    """ONE explicit collection cycle for the current target week, recorded
    as trigger=MANUAL (certification) -- distinct from cadence-SCHEDULED
    cycles. Does not prepare checkpoints."""
    import json
    from datetime import UTC, datetime

    layout, cfg, warehouse, provider, release_sha = _runtime_components()
    loop = _build_loop(layout, cfg, warehouse, provider, release_sha)
    from nflprops.platform.runtime_loop import TRIGGER_MANUAL

    try:
        loop._ensure_reference_data()
        now = datetime.now(UTC)
        assert loop.resolver is not None
        target = loop.resolver.resolve(warehouse, now)
        if target is None:
            typer.echo("FAILED: no upcoming game found for the season", err=True)
            raise typer.Exit(1)
        result = loop.collect(target, now, trigger=TRIGGER_MANUAL)
    finally:
        client = getattr(provider, "client", None)
        if client is not None:
            client.close()
    typer.echo(json.dumps({"trigger": TRIGGER_MANUAL, **loop._last["collection"],
                           "season": target.season, "week": target.week}, sort_keys=True))
    if result.status.value == "FAILED":
        raise typer.Exit(1)


@app.command("status")
def runtime_status() -> None:
    """Read-only: the runtime's last heartbeat/status plus lock state."""
    import json

    from nflprops.platform.writer_lock import WriterLock

    layout = _layout()
    status: dict[str, object] = {"status_file": str(layout.runtime_status)}
    if layout.runtime_status.is_file():
        status["runtime"] = json.loads(layout.runtime_status.read_text())
    else:
        status["runtime"] = None
    status["writer_lock_held_by_other"] = WriterLock(layout.writer_lock).is_locked_by_other()
    status["snapshots"] = [s.snapshot_id for s in list_snapshots(layout.snapshots)]
    typer.echo(json.dumps(status, indent=2, sort_keys=True, default=str))


@app.command("report")
def runtime_report() -> None:
    """Read-only certification report over the live warehouse: collection
    cycles/triggers, resource statuses, PIT timestamp sanity, natural-key
    duplicates, injury statuses, checkpoint runs/requests, upcoming games."""
    import json

    from nflprops.platform.runtime_report import build_report

    layout = _layout()
    typer.echo(json.dumps(build_report(layout.warehouse_root), indent=2, sort_keys=True, default=str))


@checkpoint_app.command("manual")
def checkpoint_manual(
    game_id: str = typer.Option(..., "--game-id"),
    as_of: str = typer.Option(..., "--as-of", help="Explicit timezone-aware ISO cutoff (<= now)."),
    season: int = typer.Option(..., help="Season the game belongs to."),
    week: int = typer.Option(..., help="Week the game belongs to."),
) -> None:
    """Claim + PREPARE one explicit MANUAL checkpoint: PIT cutoff ->
    immutable snapshot -> request bundle -> PENDING_REMOTE_EXECUTION.
    Never an official T48H/T24H/T6H/T90M/T30M; never simulates, promotes,
    or publishes."""
    import json
    from datetime import UTC, datetime

    from nflprops.platform.checkpoint_prepare import (
        CheckpointPrepareError,
        prepare_manual_checkpoint,
    )
    from nflprops.platform.runtime_layout import current_migration_head

    cutoff = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        raise typer.BadParameter("--as-of must include an explicit timezone")
    layout, cfg, warehouse, _provider, release_sha = _runtime_components(with_provider=False)
    try:
        prepared = prepare_manual_checkpoint(
            layout=layout,
            warehouse=warehouse,
            config=cfg,
            season=season,
            week=week,
            game_id=game_id,
            as_of=cutoff,
            now=datetime.now(UTC),
            migration_head=current_migration_head(),
            hostname=socket.gethostname(),
            release_sha=release_sha,
        )
    except (CheckpointPrepareError, WarehouseSnapshotError, WriterLockError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            {
                "checkpoint_name": "MANUAL",
                "run_id": prepared.run_id,
                "game_id": prepared.game_id,
                "scheduled_as_of": prepared.scheduled_as_of.isoformat(),
                "snapshot_id": prepared.snapshot_id,
                "snapshot_manifest_sha256": prepared.snapshot_manifest_sha256,
                "request_bundle_sha256": prepared.request_bundle_sha256,
                "request_bundle_dir": str(prepared.request_bundle_dir),
                "state": "PENDING_REMOTE_EXECUTION",
                "science_executed_on_wizard": False,
            },
            sort_keys=True,
        )
    )


@checkpoint_app.command("pending")
def checkpoint_pending() -> None:
    """Read-only: checkpoint requests awaiting GitHub execution."""
    import json

    from nflprops.platform.checkpoint_prepare import pending_requests

    layout = _layout()
    if not layout.warehouse_root.is_dir():
        typer.echo("[]")
        return
    from nflprops.data.warehouse import Warehouse

    rows = pending_requests(Warehouse(layout.warehouse_root)).to_dicts()
    typer.echo(json.dumps(rows, indent=2, sort_keys=True, default=str))


@checkpoint_app.command("executable")
def checkpoint_executable() -> None:
    """Read-only, for the GitHub executor (checkpoint-execute.yml): the
    PENDING_REMOTE_EXECUTION requests that still pass the fail-closed
    execution gate, oldest cutoff first, each with the identity GitHub
    must verify and -- if a result bundle for it is already published
    (an earlier executor died before ingest) -- that bundle's manifest
    SHA-256, so the executor resumes ingest instead of re-executing."""
    import json

    from nflprops.platform.immutable_bundle import read_manifest
    from nflprops.platform.result_ingest import result_bundle_id

    layout = _layout()
    if not layout.warehouse_root.is_dir():
        typer.echo("[]")
        return
    from nflprops.data.warehouse import Warehouse
    from nflprops.platform.checkpoint_prepare import executable_requests

    rows = []
    frame = executable_requests(Warehouse(layout.warehouse_root))
    for row in frame.sort("scheduled_as_of").iter_rows(named=True):
        published = layout.publications / result_bundle_id(row["run_id"])
        published_sha = None
        if published.is_dir():
            try:
                published_sha = read_manifest(published).manifest_sha256
            except BundleError:
                published_sha = None  # not a valid bundle: never resumed from
        rows.append({
            field: row[field]
            for field in (
                "run_id", "checkpoint_name", "game_id", "season", "week",
                "scheduled_as_of", "kickoff_at", "request_bundle_sha256",
                "snapshot_id", "snapshot_manifest_sha256", "data_manifest_sha256",
            )
        } | {"published_result_manifest_sha256": published_sha})
    typer.echo(json.dumps(rows, indent=2, sort_keys=True, default=str))


@checkpoint_app.command("refuse")
def checkpoint_refuse(
    run_id: str = typer.Option(..., help="The pending request's run_id (64 hex)."),
    workflow_run: str = typer.Option(..., help="URL of the GitHub run that refused it."),
    refusal_code: str = typer.Option(
        ..., help="The executor's SCIENTIFIC refusal code (remote_checkpoint)."
    ),
    evidence_b64: str = typer.Option(
        "", help="Base64 of the executor's refusal evidence JSON (refusal.json 'evidence')."
    ),
) -> None:
    """WIZARD SIDE: record that the GitHub executor's verification
    SCIENTIFICALLY refused one pending request (-> NOT_EXECUTABLE, run
    FAILED, failure_detail "<refusal_code>: ..."). An operational refusal
    code is refused (the request stays pending). Evidence, when given (and
    always for MISSING_REQUIRED_GAME_METADATA), is validated and appended in
    full to state/checkpoint_refusal_evidence.jsonl first. Only narrows
    publication; idempotent; never touches a completed request."""
    import base64
    import binascii
    import json
    import re

    from nflprops.data.warehouse import Warehouse
    from nflprops.platform.result_ingest import (
        MAX_REFUSAL_EVIDENCE_BYTES,
        REFUSAL_EVIDENCE_FILE,
        ResultIngestError,
        refuse_request,
    )

    if not re.fullmatch(r"[0-9a-f]{64}", run_id):
        raise typer.BadParameter("run_id must be 64 hex")
    if not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+/actions/runs/\d+", workflow_run):
        raise typer.BadParameter("workflow_run must be a GitHub Actions run URL")
    evidence = None
    if evidence_b64:
        if len(evidence_b64) > 2 * MAX_REFUSAL_EVIDENCE_BYTES:
            raise typer.BadParameter("evidence is too large")
        try:
            evidence = json.loads(base64.b64decode(evidence_b64, validate=True))
        except (binascii.Error, ValueError) as exc:
            raise typer.BadParameter(f"evidence is not base64 JSON: {exc}") from exc
    layout = _layout()
    try:
        status = refuse_request(
            Warehouse(layout.warehouse_root),
            run_id,
            refusal_code=refusal_code,
            detail=f"GitHub executor verification refused the request: {workflow_run}",
            lock_path=layout.writer_lock,
            evidence=evidence,
            evidence_log=layout.state / REFUSAL_EVIDENCE_FILE,
        )
    except (ResultIngestError, WriterLockError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{status}: {run_id}")


@checkpoint_app.command("operational-failure")
def checkpoint_operational_failure(
    run_id: str = typer.Option(..., help="The request's run_id (64 hex)."),
    workflow_run: str = typer.Option(..., help="URL of the GitHub run that failed."),
    failure_code: str = typer.Option(..., help="Operational failure code (checkpoint_failures)."),
) -> None:
    """WIZARD SIDE: append one OPERATIONAL execution failure to the
    operational failure log. Never changes the request or its run: the
    request stays scientifically pending."""
    import json

    from nflprops.data.warehouse import Warehouse
    from nflprops.platform.checkpoint_failures import (
        OperationalFailureError,
        record_operational_failure,
    )

    layout = _layout()
    try:
        record = record_operational_failure(
            layout, Warehouse(layout.warehouse_root),
            run_id=run_id, workflow_run=workflow_run, failure_code=failure_code,
        )
    except OperationalFailureError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(record, sort_keys=True))


@checkpoint_app.command("repair-false-refusal")
def checkpoint_repair_false_refusal(
    run_id: str = typer.Option(..., help="The pinned falsely refused run_id (64 hex)."),
    incident_id: str = typer.Option(..., help="The pinned incident id (refusal_incidents)."),
) -> None:
    """WIZARD SIDE: audited remediation of ONE pinned false NOT_EXECUTABLE
    refusal (`refusal_repair`): re-proves the incident, appends an
    immutable remediation record, then restores the request to
    PENDING_REMOTE_EXECUTION. Refuses any other request."""
    import json
    from datetime import UTC, datetime

    from nflprops.config import load
    from nflprops.data.warehouse import Warehouse
    from nflprops.platform.refusal_repair import (
        RefusalRepairError,
        repair_false_refusal,
    )
    from nflprops.platform.runtime_layout import running_release_sha

    layout = _layout()
    try:
        record = repair_false_refusal(
            layout, Warehouse(layout.warehouse_root), load(),
            run_id=run_id, incident_id=incident_id,
            repair_release_sha=running_release_sha(), now=datetime.now(UTC),
        )
    except (RefusalRepairError, WriterLockError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(record, sort_keys=True, default=str))


# ------------------------------------------------------------ BLOCK 4


@app.command("outcome-report")
def outcome_report_cmd(
    as_of: list[str] = typer.Option(  # noqa: B008
        [], help="ISO-8601 UTC cutoff(s): how many outcomes a checkpoint then could see."
    ),
    snapshot_id: str = typer.Option(
        "", help="Certify this immutable snapshot's outcome tables instead of the live ones."
    ),
) -> None:
    """Read-only: certify the versioned outcome tables (rows, versions,
    receipt times, never-visible-before-first-seen, per-week coverage). With
    --snapshot-id the snapshot's manifest is verified first (fail closed)
    and the report covers exactly that snapshot's outcome tables."""
    import json
    import re
    from datetime import datetime

    from nflprops.data.warehouse import Warehouse
    from nflprops.platform.stats_backfill import outcome_report
    from nflprops.platform.warehouse_snapshot import verify_snapshot

    probes = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in as_of]
    if any(p.tzinfo is None for p in probes):
        raise typer.BadParameter("--as-of must carry a UTC offset")
    layout = _layout()
    if not snapshot_id:
        report = outcome_report(Warehouse(layout.warehouse_root), as_of_probes=probes)
        typer.echo(json.dumps(report, indent=2, sort_keys=True, default=str))
        return
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{12}", snapshot_id):
        raise typer.BadParameter("snapshot_id format")
    try:
        info = verify_snapshot(layout.snapshots, snapshot_id)
    except (BundleError, WarehouseSnapshotError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    # The verified snapshot directory is a warehouse root; outcome_report
    # only reads it (Warehouse() creates nothing in an existing directory).
    report = outcome_report(Warehouse(layout.snapshots / snapshot_id), as_of_probes=probes)
    report["snapshot"] = {"snapshot_id": info.snapshot_id, "manifest_sha256": info.manifest_sha256}
    typer.echo(json.dumps(report, indent=2, sort_keys=True, default=str))


@app.command("games-backfill")
def games_backfill(
    season: int = typer.Option(..., help="Season, e.g. 2026."),
    weeks: str = typer.Option(..., help="Comma-separated weeks, e.g. 1,2."),
    expected_ids: str = typer.Option(
        ..., help="Comma-separated EXACT canonical game ids the receipts must restore."
    ),
    apply: bool = typer.Option(
        False, "--apply", help="Write (writer lock). Default: dry-run, read-only."
    ),
) -> None:
    """PR #21: restore missing `games` rows from GENUINE stored BDL
    BDL games-endpoint receipts (`platform.games_backfill`): receipt-time
    availability only, exact expected-ID guard, idempotent, never replaces
    a row, never touches an immutable snapshot. DRY-RUN unless --apply."""
    import json
    from datetime import UTC, datetime

    from nflprops.data.warehouse import Warehouse
    from nflprops.platform.games_backfill import (
        GamesBackfillError,
        apply_games_backfill,
        plan_games_backfill,
    )

    ids = tuple(part.strip() for part in expected_ids.split(",") if part.strip())
    week_tuple = tuple(int(part) for part in weeks.split(",") if part.strip())
    layout = _layout()
    warehouse = Warehouse(layout.warehouse_root)
    try:
        if apply:
            summary = apply_games_backfill(
                warehouse, layout.raw_root, season=season, weeks=week_tuple,
                expected_game_ids=ids, lock_path=layout.writer_lock, now=datetime.now(UTC),
            )
        else:
            plan = plan_games_backfill(
                warehouse, layout.raw_root, season=season, weeks=week_tuple,
                expected_game_ids=ids,
            )
            summary = {"applied": False, "dry_run": True, **plan.summary()}
    except (GamesBackfillError, WriterLockError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("ingest-stats")
def ingest_stats(
    seasons: str = typer.Option(..., help="Comma-separated seasons, e.g. 2024,2025,2026."),
    weeks: str = typer.Option("", help="Optional comma-separated weeks, e.g. 1,2,3,4."),
    recent_days: int = typer.Option(
        0, help="Only final games played in the last N days (recurring runtime ingest); 0 = all."
    ),
) -> None:
    """Append completed-game outcome history (`player_game_stats`,
    `team_game_stats`) for final games only as immutable versions: a
    provider correction is a new version, a stored version is never
    replaced, availability is the genuine receipt time. Never touches
    `games` or any other table."""
    import json
    from dataclasses import asdict
    from datetime import UTC, datetime, timedelta

    from nflprops.data.outcome_versions import OutcomeVersionError
    from nflprops.platform.stats_backfill import backfill_outcome_history

    layout, _cfg, warehouse, provider, _release_sha = _runtime_components()
    now = datetime.now(UTC)
    try:
        summaries = backfill_outcome_history(
            provider,
            warehouse,
            seasons=[int(part) for part in seasons.split(",") if part.strip()],
            lock_path=layout.writer_lock,
            now=now,
            weeks=[int(part) for part in weeks.split(",") if part.strip()] or None,
            since=now - timedelta(days=recent_days) if recent_days > 0 else None,
        )
    except (WriterLockError, OutcomeVersionError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        client = getattr(provider, "client", None)
        if client is not None:
            client.close()
    typer.echo(json.dumps([asdict(s) for s in summaries], sort_keys=True))


@app.command("execute-checkpoint")
def execute_checkpoint_cmd(
    request_dir: str = typer.Option(..., help="Downloaded request bundle directory."),
    expected_request_sha256: str = typer.Option(..., help="Request bundle manifest SHA-256."),
    snapshot_root: str = typer.Option(..., help="Directory holding the downloaded snapshot."),
    work_dir: str = typer.Option(..., help="Empty scratch directory for the restored warehouse."),
    out_dir: str = typer.Option(..., help="Empty directory to write the result files into."),
    science_sha: str = typer.Option(..., help="Git SHA of the executing checkout."),
    workflow_run: str = typer.Option(..., help="GitHub run id/url, for provenance."),
    verify_only: bool = typer.Option(
        False, help="Verify the request against its snapshot, then stop (no simulation)."
    ),
    refusal_file: str = typer.Option(
        "", help="Where to write the classified verification refusal (JSON), if any."
    ),
) -> None:
    """GITHUB SIDE ONLY: verify one prepared checkpoint against its
    snapshot, run the certified 20,000-draw checkpoint flow on a scratch
    restore, gate calibration, decide PUBLIC_READY, and write the result
    files. Refuses to run on the Wizard host (NFLPROPS_RUNTIME_ROOT set).

    Exit 3 = a SCIENTIFIC, deterministic verification refusal (the only
    case the workflow may record NOT_EXECUTABLE) -- including the
    pre-simulation science readiness gate (MISSING_REQUIRED_GAME_METADATA);
    exit 4 = an OPERATIONAL verification refusal (executor/environment/
    config-runtime mismatch: the request stays pending); exit 5 = model code
    failed unexpectedly inside the flow (MODEL_EXECUTION_FAILED: no result
    files, never COMPLETED, the request stays pending for a reviewed
    re-execution); exit 1 = any other (possibly transient) failure. Exits
    3, 4 and 5 write `--refusal-file` (class, code, message[, evidence])."""
    import json
    import os

    from nflprops.config import load
    from nflprops.data.warehouse import Warehouse
    from nflprops.platform.remote_checkpoint import (
        ModelExecutionError,
        RemoteExecutionError,
        execute_checkpoint,
        load_verified_request,
        verify_request_against_snapshot,
    )

    def _write_refusal(payload: dict[str, object]) -> None:
        if refusal_file:
            Path(refusal_file).write_text(json.dumps(payload, sort_keys=True) + "\n")

    if os.environ.get("NFLPROPS_RUNTIME_ROOT") or os.environ.get("GITHUB_ACTIONS") != "true":
        typer.echo("FAILED: execute-checkpoint runs only on GitHub Actions", err=True)
        raise typer.Exit(1)
    try:
        request, request_sha = load_verified_request(
            Path(request_dir), expected_manifest_sha256=expected_request_sha256
        )
        scratch = Path(work_dir) / "warehouse"
        info = restore_snapshot(Path(snapshot_root), request["snapshot_id"], scratch)
        warehouse = Warehouse(scratch, Path(work_dir) / "scratch.duckdb")
        cfg = load()
        try:
            run = verify_request_against_snapshot(
                request,
                warehouse,
                cfg,
                snapshot_id=info.snapshot_id,
                snapshot_manifest_sha256=info.manifest_sha256,
            )
            if verify_only:
                typer.echo(
                    f"VERIFIED: run {run.run_id} is executable against snapshot "
                    f"{info.snapshot_id}"
                )
                return
            executed = execute_checkpoint(
                request,
                run,
                warehouse,
                cfg,
                out_dir=Path(out_dir),
                request_bundle_sha256=request_sha,
                science_sha=science_sha,
                workflow_run=workflow_run,
            )
        except RemoteExecutionError as exc:
            _write_refusal(exc.as_dict())
            typer.echo(f"REFUSED [{exc.refusal_class}/{exc.refusal_code}]: {exc}", err=True)
            # Only a SCIENTIFIC refusal can never execute as claimed; an
            # OPERATIONAL one must leave the request pending.
            raise typer.Exit(3 if exc.scientific else 4) from exc
        except ModelExecutionError as exc:
            _write_refusal(exc.as_dict())
            typer.echo(f"MODEL_EXECUTION_FAILED: {exc}", err=True)
            raise typer.Exit(5) from exc
    except (RemoteExecutionError, BundleError, WarehouseSnapshotError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    result = executed.result
    typer.echo(json.dumps({k: result[k] for k in (
        "run_id", "run", "row_counts", "calibration_gate", "decision", "decision_reasons",
    )}, indent=2, sort_keys=True))


@app.command("result-ingest")
def result_ingest_cmd(
    bundle_id: str = typer.Option(..., help="Published result bundle id."),
    expected_manifest_sha256: str = typer.Option(..., help="Manifest SHA the executor reported."),
) -> None:
    """WIZARD SIDE: validate one published result bundle against its live
    pending request and install it under the writer lock (lightweight:
    reads and appends a few small files; never runs science)."""
    import json
    from datetime import UTC, datetime

    from nflprops.data.warehouse import Warehouse
    from nflprops.platform.result_ingest import ResultIngestError, ingest_result_bundle

    if "/" in bundle_id or bundle_id.startswith("."):
        raise typer.BadParameter("bundle_id must be a plain identifier")
    layout = _layout()
    try:
        summary = ingest_result_bundle(
            Warehouse(layout.warehouse_root),
            layout.publications / bundle_id,
            expected_manifest_sha256=expected_manifest_sha256,
            lock_path=layout.writer_lock,
            now=datetime.now(UTC),
        )
    except (ResultIngestError, BundleError, WriterLockError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
