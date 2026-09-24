"""BLOCK 2B: `nflprops.platform.wizard_runtime`'s Typer CLI -- the
snapshot-lifecycle and result-bundle-transfer commands invoked by systemd/
cron on the Wizard host and by the GitHub snapshot-transfer workflow.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from nflprops.platform.wizard_runtime import app

runner = CliRunner()


@pytest.fixture()
def warehouse_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # NFLPROPS_DATA_ROOT is the collector's run.data_root; the warehouse
    # tables live in <data_root>/canonical (nflprops.pipelines.lean).
    data_root = tmp_path / "data"
    warehouse_root = data_root / "canonical"
    warehouse_root.mkdir(parents=True)
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(warehouse_root / "t.parquet")
    monkeypatch.setenv("NFLPROPS_DATA_ROOT", str(data_root))
    monkeypatch.delenv("NFLPROPS_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("NFLPROPS_ENV", "development")
    monkeypatch.setenv("NFLPROPS_STORAGE_BACKEND", "duckdb")
    return warehouse_root


def test_snapshot_create_then_list_then_verify(warehouse_env: Path) -> None:
    create = runner.invoke(
        app, ["snapshot", "create", "--migration-head", "0009_compact_pmf_payload"]
    )
    assert create.exit_code == 0, create.output
    assert "SUCCEEDED" in create.output

    listing = runner.invoke(app, ["snapshot", "list"])
    assert listing.exit_code == 0
    snapshot_id = listing.output.split("\t")[0].strip()
    assert snapshot_id

    verify = runner.invoke(app, ["snapshot", "verify", snapshot_id])
    assert verify.exit_code == 0
    assert "VERIFIED" in verify.output


def test_snapshot_verify_unknown_id_fails_closed(warehouse_env: Path) -> None:
    result = runner.invoke(app, ["snapshot", "verify", "no-such-snapshot"])
    assert result.exit_code == 1
    assert "FAILED" in result.output


def test_snapshot_restore_to_fresh_path(warehouse_env: Path, tmp_path: Path) -> None:
    create = runner.invoke(
        app, ["snapshot", "create", "--migration-head", "0009_compact_pmf_payload"]
    )
    snapshot_id = create.output.split("snapshot_id=")[1].split(" ")[0]

    dest = tmp_path / "recovery"
    restore = runner.invoke(app, ["snapshot", "restore", snapshot_id, str(dest)])
    assert restore.exit_code == 0, restore.output
    assert (dest / "t.parquet").exists()


def test_snapshot_restore_rejects_the_live_warehouse(warehouse_env: Path) -> None:
    create = runner.invoke(
        app, ["snapshot", "create", "--migration-head", "0009_compact_pmf_payload"]
    )
    snapshot_id = create.output.split("snapshot_id=")[1].split(" ")[0]

    restore = runner.invoke(
        app, ["snapshot", "restore", snapshot_id, str(warehouse_env)]
    )
    assert restore.exit_code == 1
    assert "FAILED" in restore.output


def test_lock_status_reports_free(warehouse_env: Path) -> None:
    result = runner.invoke(app, ["lock-status"])
    assert result.exit_code == 0
    assert "held_by_other=False" in result.output


def test_result_bundle_build_verify_publish_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    final = tmp_path / "final" / "run-1"
    source.mkdir()
    (source / "report.json").write_text('{"ok": true}')

    build = runner.invoke(
        app,
        [
            "result-bundle", "build",
            "--bundle-id", "run-1",
            "--source-dir", str(source),
            "--staging-dir", str(staging),
            "--science-sha", "a" * 40,
            "--workflow-sha", "b" * 40,
        ],
    )
    assert build.exit_code == 0, build.output
    manifest_sha = build.output.split("manifest_sha256=")[1].strip()

    verify = runner.invoke(
        app,
        [
            "bundle-verify",
            "--bundle-dir", str(staging),
            "--expected-manifest-sha256", manifest_sha,
        ],
    )
    assert verify.exit_code == 0, verify.output

    publish = runner.invoke(
        app,
        ["bundle-publish", "--staging-dir", str(staging), "--final-dir", str(final)],
    )
    assert publish.exit_code == 0, publish.output
    assert "PUBLISHED" in publish.output
    assert (final / "report.json").exists()


def test_result_bundle_verify_fails_closed_on_wrong_expected_sha(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "report.json").write_text("{}")
    runner.invoke(
        app,
        [
            "result-bundle", "build",
            "--bundle-id", "run-2",
            "--source-dir", str(source),
            "--staging-dir", str(staging),
            "--science-sha", "a" * 40,
            "--workflow-sha", "b" * 40,
        ],
    )

    verify = runner.invoke(
        app,
        [
            "bundle-verify",
            "--bundle-dir", str(staging),
            "--expected-manifest-sha256", "0" * 64,
        ],
    )
    assert verify.exit_code == 1
    assert "FAILED" in verify.output


def test_result_bundle_publish_never_overwrites_different_content(tmp_path: Path) -> None:
    final = tmp_path / "final" / "run-3"

    def _build_and_publish(content: str) -> object:
        source = tmp_path / f"source-{content}"
        staging = tmp_path / f"staging-{content}"
        source.mkdir()
        (source / "report.json").write_text(content)
        runner.invoke(
            app,
            [
                "result-bundle", "build",
                "--bundle-id", "run-3",
                "--source-dir", str(source),
                "--staging-dir", str(staging),
                "--science-sha", "a" * 40,
                "--workflow-sha", "b" * 40,
            ],
        )
        return runner.invoke(
            app,
            ["bundle-publish", "--staging-dir", str(staging), "--final-dir", str(final)],
        )

    first = _build_and_publish('{"v": 1}')
    assert first.exit_code == 0
    second = _build_and_publish('{"v": 2}')
    assert second.exit_code == 1
    assert "FAILED" in second.output
    assert (final / "report.json").read_text() == '{"v": 1}'


# ------------------------------- probe-approved layout + bounded retention


def test_runtime_root_env_places_snapshots_and_lock_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "nflprops"
    warehouse_root = runtime_root / "state" / "canonical"
    warehouse_root.mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(warehouse_root / "t.parquet")
    monkeypatch.setenv("NFLPROPS_DATA_ROOT", str(runtime_root / "state"))
    monkeypatch.setenv("NFLPROPS_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("NFLPROPS_ENV", "development")
    monkeypatch.setenv("NFLPROPS_STORAGE_BACKEND", "duckdb")

    create = runner.invoke(
        app, ["snapshot", "create", "--migration-head", "0009_compact_pmf_payload"]
    )
    assert create.exit_code == 0, create.output
    snapshot_id = create.output.split("snapshot_id=")[1].split(" ")[0]
    assert (runtime_root / "snapshots" / snapshot_id / "manifest.json").exists()
    assert (runtime_root / "locks" / "writer.lock").exists()
    assert not (runtime_root / "state" / "snapshots").exists()

    status = runner.invoke(app, ["lock-status"])
    assert str(runtime_root / "locks" / "writer.lock") in status.output


def test_snapshot_create_enforces_bounded_retention(
    warehouse_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFLPROPS_SNAPSHOT_RETENTION", "2")
    for value in range(4):
        pl.DataFrame({"a": [value]}).write_parquet(warehouse_env / "t.parquet")
        result = runner.invoke(
            app, ["snapshot", "create", "--migration-head", "0009_compact_pmf_payload"]
        )
        assert result.exit_code == 0, result.output
        assert "retained=2" in result.output
    listing = runner.invoke(app, ["snapshot", "list"])
    assert len([line for line in listing.output.splitlines() if line.strip()]) == 2


def test_snapshot_prune_command(warehouse_env: Path) -> None:
    for value in range(3):
        pl.DataFrame({"a": [value]}).write_parquet(warehouse_env / "t.parquet")
        runner.invoke(
            app,
            ["snapshot", "create", "--migration-head", "0009_compact_pmf_payload",
             "--keep", "10"],
        )
    result = runner.invoke(app, ["snapshot", "prune", "--keep", "1"])
    assert result.exit_code == 0, result.output
    assert "pruned=2" in result.output
