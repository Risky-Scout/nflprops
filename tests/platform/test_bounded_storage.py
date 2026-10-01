"""Bounded Wizard storage: zstd raw store + verified migration, orphan
sweeper, finite raw retention (30 d default, 45 d failed-cycle cap), disk guardrails, terminal-request snapshot
release, and hard-link deduplicated snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nflprops.data.raw_store import (
    COMPRESSED_SUFFIX,
    RawStore,
    compress_legacy_payloads,
    decompress_bytes,
)
from nflprops.data.warehouse import Warehouse
from nflprops.errors import DataQualityError
from nflprops.platform.checkpoint_prepare import (
    REMOTE_REQUESTS_TABLE,
    STATE_NOT_EXECUTABLE,
    STATE_PENDING_REMOTE_EXECUTION,
    protected_snapshot_ids,
)
from nflprops.platform.runtime_layout import resolve_runtime_layout
from nflprops.platform.storage_guard import (
    LEVEL_CRITICAL,
    LEVEL_FAIL_CLOSED,
    LEVEL_OK,
    LEVEL_WARNING,
    disk_level,
    prune_raw_payloads,
    sweep_orphans,
)
from nflprops.platform.warehouse_snapshot import (
    create_snapshot,
    restore_snapshot,
    verify_snapshot,
)

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
PAYLOAD = {"data": [{"player": i, "line": 50.5 + i, "vendor": "x" * 20} for i in range(200)]}


def _write(store: RawStore, payload=PAYLOAD, *, received: datetime = NOW):
    return store.write_json(
        provider="bdl", endpoint="/nfl/v1/odds/player_props", request_params={"api_key": "k"},
        payload=payload, requested_at=received, received_at=received, http_status=200,
        spec_sha256=None,
    )


def _age(path: Path, delta: timedelta) -> None:
    ts = (NOW - delta).timestamp()
    os.utime(path, (ts, ts))


# ------------------------------------------------------------------ raw store


def test_raw_payload_is_stored_compressed_and_round_trips_exactly(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    ref = _write(store)
    path = Path(ref.payload_path)
    assert path.name.endswith(COMPRESSED_SUFFIX)
    body = store._canonical_json_bytes(PAYLOAD)
    assert decompress_bytes(path.read_bytes()) == body
    assert hashlib.sha256(store.read_bytes(ref)).hexdigest() == ref.response_sha256
    assert store.read_json(ref) == json.loads(body)
    assert path.stat().st_size * 5 < len(body)
    meta = json.loads(Path(ref.metadata_path).read_text())
    assert meta["request_params"]["api_key"] == "<REDACTED>"
    # idempotent re-fetch: nothing rewritten
    mtime = path.stat().st_mtime_ns
    _write(store)
    assert path.stat().st_mtime_ns == mtime


def test_compressed_object_collision_is_refused(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    ref = _write(store)
    Path(ref.payload_path).write_bytes(store_bytes := b"\x28\xb5\x2f\xfd" + b"garbage")
    del store_bytes
    with pytest.raises(Exception):  # noqa: B017 -- corrupt frame or collision
        _write(store)


def _legacy(store: RawStore, payload=PAYLOAD) -> tuple[Path, bytes]:
    body = store._canonical_json_bytes(payload)
    digest = hashlib.sha256(body).hexdigest()
    target = store.root / "bdl" / "nfl__v1__odds__player_props"
    target.mkdir(parents=True, exist_ok=True)
    legacy = target / f"{digest}.json"
    legacy.write_bytes(body)
    return legacy, body


def test_legacy_payload_is_still_honored_and_never_duplicated(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    legacy, body = _legacy(store)
    ref = _write(store)
    assert Path(ref.payload_path) == legacy  # existing legacy object reused
    assert not legacy.with_name(legacy.name + ".zst").exists()
    assert store.read_bytes(ref) == body


def test_legacy_migration_verifies_then_removes_and_is_resumable(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    legacies = [_legacy(store, {"n": i, "pad": "y" * 500}) for i in range(5)]
    bad_dir = store.root / "bdl" / "x"
    bad_dir.mkdir(parents=True)
    bad = bad_dir / ("0" * 64 + ".json")
    bad.write_bytes(b"{}\n")  # bytes do not hash to their content address

    first = compress_legacy_payloads(store.root, max_files=2)
    assert first.converted == 2
    second = compress_legacy_payloads(store.root)
    assert second.converted == 3
    assert second.skipped_mismatched_address == [str(bad)]
    assert bad.exists()  # never touched
    for legacy, body in legacies:
        assert not legacy.exists()
        compressed = legacy.with_name(legacy.name + ".zst")
        assert decompress_bytes(compressed.read_bytes()) == body
        assert store.read_bytes(legacy) == body  # old path still resolves
    assert second.bytes_after < second.bytes_before

    # crash after writing .zst but before unlinking: re-verified, then removed
    legacy, body = _legacy(store, {"n": 99, "pad": "z" * 500})
    compressed = legacy.with_name(legacy.name + ".zst")
    from nflprops.data.raw_store import compress_bytes

    compressed.write_bytes(compress_bytes(body))
    assert compress_legacy_payloads(store.root).converted == 1
    assert not legacy.exists()

    # a .zst that does NOT match its legacy bytes is refused, legacy kept
    legacy, body = _legacy(store, {"n": 100})
    legacy.with_name(legacy.name + ".zst").write_bytes(compress_bytes(b"other"))
    with pytest.raises(DataQualityError):
        compress_legacy_payloads(store.root)
    assert legacy.exists()


# ------------------------------------------------------------------ guardrails


def test_disk_levels() -> None:
    assert disk_level(9.0) == LEVEL_OK
    assert disk_level(7.9) == LEVEL_WARNING
    assert disk_level(4.5) == LEVEL_CRITICAL
    assert disk_level(2.9) == LEVEL_FAIL_CLOSED


def _layout(tmp_path: Path):
    root = tmp_path / "nflprops"
    warehouse = Warehouse(root / "state" / "canonical")
    warehouse.root.mkdir(parents=True, exist_ok=True)
    return resolve_runtime_layout(warehouse.root, {"NFLPROPS_RUNTIME_ROOT": str(root)}), warehouse


def test_sweeper_removes_only_old_abandoned_artifacts(tmp_path: Path) -> None:
    layout, warehouse = _layout(tmp_path)
    pl.DataFrame({"x": [1]}).write_parquet(warehouse.root / "games.parquet")
    parts = warehouse.root / "player_prop_snapshots.parts"
    parts.mkdir()
    old_tmp = parts / "part-000000004-000000004.parquet.tmp"
    old_tmp.write_bytes(b"x")
    young_tmp = warehouse.root / "games.parquet.tmp"
    young_tmp.write_bytes(b"y")
    replaced = warehouse.root / "roster_snapshots.parts.replaced"
    replaced.mkdir()
    (replaced / "part-000000001-000000001.parquet").write_bytes(b"z")
    layout.raw_root.mkdir(parents=True)
    raw_tmp = layout.raw_root / "a.json.zst.tmp"
    raw_tmp.write_bytes(b"r")
    pending = layout.snapshots / "_pending" / ".pending.tmp-abc"
    pending.mkdir(parents=True)
    staging = layout.checkpoint_requests / ".run1.tmp-xyz"
    staging.mkdir(parents=True)
    incoming_old = layout.publications / "_incoming" / "bundle-old"
    incoming_old.mkdir(parents=True)
    incoming_new = layout.publications / "_incoming" / "bundle-new"
    incoming_new.mkdir(parents=True)
    published = layout.checkpoint_requests / "run1"
    published.mkdir()
    (published / "request.json").write_text("{}")
    _age(published / "request.json", timedelta(days=30))
    for path in (old_tmp, replaced, raw_tmp, pending, staging):
        _age(path, timedelta(hours=2))
    _age(young_tmp, timedelta(minutes=5))
    _age(incoming_old, timedelta(hours=30))
    _age(incoming_new, timedelta(hours=2))

    result = sweep_orphans(layout, now=NOW)
    removed = set(result.removed)
    assert {str(old_tmp), str(replaced), str(raw_tmp), str(pending), str(staging),
            str(incoming_old)} == removed
    assert young_tmp.exists() and incoming_new.exists()
    assert (warehouse.root / "games.parquet").exists() and published.exists()


def test_raw_retention_is_bounded_by_default_and_override_may_only_shorten() -> None:
    from nflprops.platform.runtime_loop import RuntimeLoop
    from nflprops.platform.storage_guard import (
        RAW_FAILED_HARD_CAP_DAYS,
        RAW_RETENTION_DAYS,
        resolve_raw_retention_days,
    )

    assert (RAW_RETENTION_DAYS, RAW_FAILED_HARD_CAP_DAYS) == (30, 45)
    assert resolve_raw_retention_days(None) == (30, None)
    assert resolve_raw_retention_days("") == (30, None)
    assert resolve_raw_retention_days("14") == (14, None)
    for longer_or_disabled in ("31", "365", "0", "-1", "inf", "never"):
        days, warning = resolve_raw_retention_days(longer_or_disabled)
        assert days == 30 and warning, longer_or_disabled
    default = RuntimeLoop.__dataclass_fields__["raw_retention_days"].default
    assert default == 30


def _aged(ref, days: float) -> None:
    for path in (Path(ref.payload_path), Path(ref.metadata_path)):
        _age(path, timedelta(days=days))


def test_raw_retention_covers_every_payload_class_and_nothing_else(tmp_path: Path) -> None:
    layout, warehouse = _layout(tmp_path)
    store = RawStore(layout.raw_root)
    d = timedelta(days=1)

    def at(days: float) -> datetime:
        return NOW - days * d

    warehouse.write("collector_runs", pl.DataFrame({
        "status": ["SUCCESS", "FAILED", "FAILED", "RUNNING"],
        "started_at": [at(31) - timedelta(minutes=1), at(31.5) - timedelta(minutes=1),
                       at(46) - timedelta(minutes=1), at(40) - timedelta(minutes=1)],
        "completed_at": [at(31) + timedelta(minutes=1), at(31.5) + timedelta(minutes=1),
                         at(46) + timedelta(minutes=1), None],
    }))
    canon_old = _write(store, {"a": "canon"}, received=at(31))
    discovery_old = _write(store, {"a": "discovery"}, received=at(32))
    failed_31 = _write(store, {"a": "failed31"}, received=at(31.5))
    failed_46 = _write(store, {"a": "failed46"}, received=at(46))
    unfinished_40 = _write(store, {"a": "unfinished"}, received=at(40))
    recent = _write(store, {"a": "recent"}, received=at(2))
    for ref, days in ((canon_old, 31), (discovery_old, 32), (failed_31, 31.5),
                      (failed_46, 46), (unfinished_40, 40), (recent, 2)):
        _aged(ref, days)
    # payload whose metadata is missing: aged by file mtime, 45-day cap
    orphan_young = layout.raw_root / "bdl" / "x" / ("a" * 64 + ".json.zst")
    orphan_old = layout.raw_root / "bdl" / "x" / ("b" * 64 + ".json.zst")
    orphan_young.parent.mkdir(parents=True)
    for path, days in ((orphan_young, 40), (orphan_old, 46)):
        path.write_bytes(b"x")
        _age(path, timedelta(days=days))
    # never touched: non-raw names under raw_root, canonical tables
    stray = layout.raw_root / "bdl" / "x" / "notes.txt"
    stray.write_text("keep")
    _age(stray, timedelta(days=400))
    pl.DataFrame({"x": [1]}).write_parquet(warehouse.root / "games.parquet")
    _age(warehouse.root / "games.parquet", timedelta(days=400))

    result = prune_raw_payloads(layout, retention_days=30, now=NOW)

    gone = (canon_old, discovery_old, failed_46)
    kept = (failed_31, unfinished_40, recent)
    for ref in gone:
        assert not Path(ref.payload_path).exists() and not Path(ref.metadata_path).exists()
    for ref in kept:
        assert Path(ref.payload_path).exists() and Path(ref.metadata_path).exists()
    assert not orphan_old.exists() and orphan_young.exists()
    assert stray.exists() and (warehouse.root / "games.parquet").exists()
    assert (result.removed_canonicalized, result.removed_discovery,
            result.removed_failed_cycle, result.removed_unattributed) == (1, 1, 1, 1)

    # a shortened override applies to canonicalized/discovery payloads only
    result = prune_raw_payloads(layout, retention_days=1, now=NOW)
    assert not Path(recent.payload_path).exists()
    assert Path(failed_31.payload_path).exists()  # failed-cycle cap stays 45
    for bad in (0, 31):
        with pytest.raises(ValueError):
            prune_raw_payloads(layout, retention_days=bad, now=NOW)


# ------------------------------------------------------------------ snapshots


def test_terminal_requests_release_their_snapshot(tmp_path: Path) -> None:
    _layout_unused, warehouse = _layout(tmp_path)
    warehouse.write(REMOTE_REQUESTS_TABLE, pl.DataFrame({
        "request_id": ["a", "b", "c"],
        "state": [STATE_PENDING_REMOTE_EXECUTION, STATE_NOT_EXECUTABLE, "COMPLETED"],
        "snapshot_id": ["s-pending", "s-not-exec", "s-done"],
    }))
    assert protected_snapshot_ids(warehouse) == frozenset({"s-pending"})


def test_snapshots_hard_link_unchanged_files_and_still_verify(tmp_path: Path) -> None:
    layout, warehouse = _layout(tmp_path)
    big = warehouse.root / "player_prop_snapshots.parquet"
    pl.DataFrame({"x": list(range(50_000))}).write_parquet(big)
    pl.DataFrame({"y": [1]}).write_parquet(warehouse.root / "games.parquet")
    (warehouse.root / "games.parquet.tmp").write_bytes(b"")  # never snapshotted
    kwargs = dict(warehouse_root=warehouse.root, snapshot_root=layout.snapshots,
                  lock_path=layout.writer_lock, migration_head="h", hostname="t")
    first = create_snapshot(**kwargs, created_at=NOW)
    pl.DataFrame({"y": [1, 2]}).write_parquet(warehouse.root / "games.parquet")
    second = create_snapshot(**kwargs, created_at=NOW + timedelta(hours=6))

    a = layout.snapshots / first.snapshot_id / big.name
    b = layout.snapshots / second.snapshot_id / big.name
    assert a.stat().st_ino == b.stat().st_ino  # unchanged file stored once
    ga = layout.snapshots / first.snapshot_id / "games.parquet"
    gb = layout.snapshots / second.snapshot_id / "games.parquet"
    assert ga.stat().st_ino != gb.stat().st_ino  # changed file copied
    assert not (layout.snapshots / second.snapshot_id / "games.parquet.tmp").exists()
    verify_snapshot(layout.snapshots, first.snapshot_id)
    verify_snapshot(layout.snapshots, second.snapshot_id)

    # pruning the older snapshot leaves the newer one intact and verifiable
    import shutil

    shutil.rmtree(layout.snapshots / first.snapshot_id)
    verify_snapshot(layout.snapshots, second.snapshot_id)
    restored = restore_snapshot(layout.snapshots, second.snapshot_id, tmp_path / "restore")
    assert restored.snapshot_id == second.snapshot_id
    assert (tmp_path / "restore" / big.name).stat().st_ino != b.stat().st_ino  # copied out


# ------------------------------------------------------------------ runtime


def test_runtime_fails_closed_on_snapshots_and_checkpoints_below_three_gib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nflprops.platform.runtime_loop as loop_module
    from nflprops.config import load
    from nflprops.platform.runtime_loop import RuntimeLoop

    layout, warehouse = _layout(tmp_path)
    calls: list[str] = []
    loop = RuntimeLoop(layout=layout, warehouse=warehouse, config=load(), provider=object(),
                       migration_head="h", release_sha=None, clock=lambda: NOW,
                       resolver=type("R", (), {"resolve": lambda self, w, n: None})())
    monkeypatch.setattr(loop, "_periodic_snapshot", lambda now: calls.append("snapshot"))
    monkeypatch.setattr(loop_module, "free_gib", lambda path: 2.5)
    loop.tick()
    assert calls == []
    assert loop._last["storage"]["level"] == LEVEL_FAIL_CLOSED
    status = json.loads(layout.runtime_status.read_text())
    assert status["last"]["storage"]["level"] == LEVEL_FAIL_CLOSED

    monkeypatch.setattr(loop_module, "free_gib", lambda path: 4.5)
    loop._housekeeping_at = None
    loop.tick()
    assert calls == ["snapshot"]
    assert loop._last["storage"]["level"] == LEVEL_CRITICAL


def _two_snapshots(layout, warehouse, *, corrupt_prior: bool = False):
    big = warehouse.root / "player_prop_snapshots.parquet"
    pl.DataFrame({"x": list(range(50_000))}).write_parquet(big)
    pl.DataFrame({"y": [1]}).write_parquet(warehouse.root / "games.parquet")
    kwargs = dict(warehouse_root=warehouse.root, snapshot_root=layout.snapshots,
                  lock_path=layout.writer_lock, migration_head="h", hostname="t")
    first = create_snapshot(**kwargs, created_at=NOW)
    prior_file = layout.snapshots / first.snapshot_id / big.name
    if corrupt_prior:
        data = bytearray(prior_file.read_bytes())
        data[len(data) // 2] ^= 0xFF  # same size, different bytes
        prior_file.write_bytes(bytes(data))
    pl.DataFrame({"y": [1, 2]}).write_parquet(warehouse.root / "games.parquet")
    second = create_snapshot(**kwargs, created_at=NOW + timedelta(hours=6))
    return big, prior_file, layout.snapshots / second.snapshot_id / big.name, second


def test_valid_earlier_snapshot_file_is_rehashed_then_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nflprops.platform.warehouse_snapshot as ws

    hashed: list[str] = []
    real = ws._sha256_file
    monkeypatch.setattr(ws, "_sha256_file", lambda p: hashed.append(str(p)) or real(p))
    layout, warehouse = _layout(tmp_path)
    big, prior_file, new_file, second = _two_snapshots(layout, warehouse)
    assert new_file.stat().st_ino == prior_file.stat().st_ino
    assert str(prior_file) in hashed  # the earlier file itself was re-hashed
    verify_snapshot(layout.snapshots, second.snapshot_id)
    assert new_file.read_bytes() == big.read_bytes()


def test_corrupted_earlier_snapshot_file_is_never_hard_linked(tmp_path: Path) -> None:
    layout, warehouse = _layout(tmp_path)
    big, prior_file, new_file, second = _two_snapshots(layout, warehouse, corrupt_prior=True)
    assert new_file.stat().st_ino != prior_file.stat().st_ino
    assert new_file.read_bytes() == big.read_bytes() != prior_file.read_bytes()
    verify_snapshot(layout.snapshots, second.snapshot_id)  # byte-for-byte vs its manifest


def test_unverifiable_copy_fails_closed_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    import nflprops.platform.warehouse_snapshot as ws

    layout, warehouse = _layout(tmp_path)
    pl.DataFrame({"y": [1]}).write_parquet(warehouse.root / "games.parquet")

    def bad_copy(src, dst, *a, **k):
        shutil.copyfile(src, dst)
        with open(dst, "ab") as handle:
            handle.write(b"!")

    monkeypatch.setattr(ws.shutil, "copy2", bad_copy)
    with pytest.raises(ws.WarehouseSnapshotError):
        create_snapshot(warehouse_root=warehouse.root, snapshot_root=layout.snapshots,
                        lock_path=layout.writer_lock, migration_head="h", hostname="t",
                        created_at=NOW)
    published = [p for p in layout.snapshots.iterdir() if not p.name.startswith("_")]
    assert published == []
