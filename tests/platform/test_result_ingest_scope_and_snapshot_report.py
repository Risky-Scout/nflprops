"""PR #20: the Wizard-side result ingest is bounded by the run it installs
(it never reads the large snapshot tables), and the snapshot outcome report
verifies the snapshot then reads it without writing anything."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_block4_checkpoint_roundtrip import (  # noqa: F401  (pytest fixtures)
    _github_execute,
    _publish,
    draws,
    wizard,
)

from nflprops.data.warehouse import Warehouse
from nflprops.platform.remote_checkpoint import RESULT_TABLES
from nflprops.platform.result_ingest import RESULTS_TABLE, ingest_result_bundle

#: Every table the ingest may read: the run's own result tables, its run
#: and request rows, and the results ledger. Never a snapshot/feed table.
RUN_SCOPED = {name for name, _key in RESULT_TABLES} | {
    "prediction_runs", "remote_checkpoint_requests", RESULTS_TABLE,
}


def test_result_ingest_reads_only_run_scoped_tables(
    wizard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    executed = _github_execute(wizard, tmp_path)
    final, sha = _publish(wizard, executed["out"], wizard["prepared"].run_id)
    reads: list[str] = []
    real_read = Warehouse.read

    def tracking(self: Warehouse, table: str, **kw: object):  # type: ignore[no-untyped-def]
        reads.append(table)
        return real_read(self, table, **kw)

    monkeypatch.setattr(Warehouse, "read", tracking)
    summary = ingest_result_bundle(wizard["warehouse"], final, expected_manifest_sha256=sha,
                                   lock_path=wizard["layout"].writer_lock, now=datetime.now(UTC))
    assert summary["status"] == "INGESTED"
    assert reads and set(reads) <= RUN_SCOPED, sorted(set(reads) - RUN_SCOPED)
    assert "player_prop_snapshots" not in reads


def _tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(str(path.relative_to(root)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_snapshot_outcome_report_verifies_and_never_writes(
    wizard: dict, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    from typer.testing import CliRunner

    from nflprops.platform import wizard_runtime

    layout = wizard["layout"]
    snapshot_id = wizard["prepared"].snapshot_id
    monkeypatch.setattr(wizard_runtime, "_layout", lambda: layout)
    before = _tree(layout.snapshots)
    result = CliRunner().invoke(wizard_runtime.app, [
        "outcome-report", "--snapshot-id", snapshot_id, "--as-of", "2025-09-15T16:30:00+00:00"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["snapshot"]["snapshot_id"] == snapshot_id
    assert report["player_game_stats"]["rows"] > 0
    assert report["team_game_stats"]["rows"] > 0
    assert report["player_game_stats"]["estimated_rows"] == 0
    assert _tree(layout.snapshots) == before  # nothing created or changed
    # A tampered snapshot is refused before anything is reported.
    victim = next(p for p in (layout.snapshots / snapshot_id).rglob("*.parquet"))
    victim.write_bytes(victim.read_bytes() + b"x")
    refused = CliRunner().invoke(wizard_runtime.app, ["outcome-report",
                                                      "--snapshot-id", snapshot_id])
    assert refused.exit_code == 1
    assert "FAILED" in refused.output
