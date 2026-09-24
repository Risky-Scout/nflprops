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
    if not layout.warehouse_root.is_dir():
        return frozenset()
    from nflprops.data.warehouse import Warehouse
    from nflprops.platform.checkpoint_prepare import protected_snapshot_ids

    return protected_snapshot_ids(Warehouse(layout.warehouse_root))


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
    from nflprops.platform.runtime_layout import current_migration_head
    from nflprops.platform.runtime_loop import (
        DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
        DEFAULT_TICK_SECONDS,
        RuntimeLoop,
    )

    def _bootstrap() -> None:
        LeanIngestor(
            provider, warehouse, goat=bool(cfg.get_path("provider.bdl.tier.goat", False))
        ).bootstrap()

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


if __name__ == "__main__":
    app()
