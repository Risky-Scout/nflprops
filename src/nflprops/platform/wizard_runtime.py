"""BLOCK 2B: the Wizard-host runtime CLI -- snapshot lifecycle + result-bundle
transfer commands, invoked via `python -m nflprops.platform.wizard_runtime
<command>` by systemd, cron, or a GitHub Actions SSH step (mirroring
`nflprops.platform.remote_training`'s `python -m` invocation convention).

Reads warehouse/snapshot/lock paths from `NFLPROPS_DATA_ROOT` (via
`StorageSettings.from_env`) and the BLOCK 2B convention that the warehouse
root's PARENT is the runtime state root -- see
`nflprops.platform.warehouse_snapshot` / `nflprops.platform.writer_lock`.

This module never imports Science code, never trains, never recalibrates,
and never runs on GitHub-hosted runners for anything other than the
snapshot-download/result-bundle-verify halves of the transfer contract
(task §4/§5) -- creating a warehouse snapshot only ever happens ON the
Wizard host, against the live warehouse.
"""

from __future__ import annotations

import socket
from pathlib import Path

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
from nflprops.platform.warehouse_snapshot import (
    WarehouseSnapshotError,
    create_snapshot,
    list_snapshots,
    restore_snapshot,
    verify_snapshot,
)
from nflprops.platform.writer_lock import default_lock_path

app = typer.Typer(
    name="wizard-runtime",
    help="Wizard-host warehouse snapshot lifecycle and result-bundle transfer (BLOCK 2B).",
    no_args_is_help=True,
)

snapshot_app = typer.Typer(help="Immutable warehouse snapshot lifecycle.")
result_bundle_app = typer.Typer(help="GitHub result-bundle transport/verification.")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(result_bundle_app, name="result-bundle")


def _warehouse_and_snapshot_roots() -> tuple[Path, Path]:
    settings = StorageSettings.from_env()
    warehouse_root = Path(settings.local_warehouse_root)
    snapshot_root = warehouse_root.parent / "snapshots"
    return warehouse_root, snapshot_root


@snapshot_app.command("create")
def snapshot_create(
    migration_head: str = typer.Option(
        ..., help="The Alembic migration head at the time of this snapshot."
    ),
    lock_timeout_seconds: float = typer.Option(
        60.0, help="Bounded wait for the writer lock before failing closed."
    ),
) -> None:
    """Checkpoint + copy the live warehouse into a new immutable snapshot.
    The ONE command that reads the live warehouse for durability purposes."""
    warehouse_root, snapshot_root = _warehouse_and_snapshot_roots()
    try:
        info = create_snapshot(
            warehouse_root=warehouse_root,
            snapshot_root=snapshot_root,
            lock_timeout_seconds=lock_timeout_seconds,
            migration_head=migration_head,
            hostname=socket.gethostname(),
        )
    except WarehouseSnapshotError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        f"SUCCEEDED: snapshot_id={info.snapshot_id} files={info.file_count} "
        f"bytes={info.total_bytes} manifest_sha256={info.manifest_sha256}"
    )


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

    _warehouse_root, snapshot_root = _warehouse_and_snapshot_roots()
    lock_path = default_lock_path(snapshot_root.parent)
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


if __name__ == "__main__":
    app()
