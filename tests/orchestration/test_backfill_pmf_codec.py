"""BLOCK 2A: `tools/backfill_pmf_codec.py` -- safe, additive, dry-run-capable
conversion of legacy `player_prop_distribution_outcomes` rows into compact
`pmf_payload` columns on `player_prop_distributions`.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nflprops.data.warehouse import Warehouse
from nflprops.distributions.pmf_codec import decode_pmf
from nflprops.orchestration.distribution_store import (
    PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE,
    PLAYER_PROP_DISTRIBUTIONS_TABLE,
    compute_distribution_id,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "projections"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from _projection_fixtures import GAME_ID
from backfill_pmf_codec import run_backfill

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)
RUN_ID = "RUN-BACKFILL"


def _backend(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "wh")


def _write_legacy_row(
    backend: Warehouse,
    *,
    player_id: str,
    prop_type: str,
    outcomes: list[int],
    probabilities: list[float],
) -> tuple[str, int]:
    distribution_id = compute_distribution_id(
        run_id=RUN_ID, player_id=player_id, prop_type=prop_type
    )
    distribution_key = int(
        hashlib.sha256(distribution_id.encode("utf-8")).hexdigest()[:15], 16
    )
    dist_row = {
        "distribution_key": distribution_key,
        "distribution_id": distribution_id,
        "run_id": RUN_ID,
        "game_id": GAME_ID,
        "player_id": player_id,
        "team_id": "T",
        "position_group": "WR",
        "prop_type": prop_type,
        "support_min": min(outcomes),
        "support_max": max(outcomes),
        "n_draws": 1000,
        "outcome_count": len(outcomes),
        "raw_content_sha256": "legacy-fixture-hash",
        "created_at": NOW,
    }
    backend.append(
        PLAYER_PROP_DISTRIBUTIONS_TABLE, pl.DataFrame([dist_row]), key=["distribution_key"]
    )
    outcome_rows = [
        {"distribution_key": distribution_key, "outcome": o, "p_raw": p}
        for o, p in zip(outcomes, probabilities, strict=True)
    ]
    backend.append(
        PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE,
        pl.DataFrame(outcome_rows),
        key=["distribution_key", "outcome"],
    )
    return distribution_id, distribution_key


def test_dry_run_reports_would_backfill_without_writing(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _write_legacy_row(
        backend, player_id="p1", prop_type="receptions",
        outcomes=[0, 1, 2], probabilities=[0.5, 0.3, 0.2],
    )
    results = run_backfill(backend, run_id=None, apply=False)
    assert len(results) == 1
    assert results[0].status == "would_backfill"

    stored = backend.read(PLAYER_PROP_DISTRIBUTIONS_TABLE)
    assert "pmf_payload" not in stored.columns or stored["pmf_payload"].null_count() == 1


def test_apply_writes_valid_payload_and_keeps_legacy_rows(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    distribution_id, _key = _write_legacy_row(
        backend, player_id="p2", prop_type="receiving_yards",
        outcomes=[-5, 0, 10, 200], probabilities=[0.1, 0.4, 0.3, 0.2],
    )
    results = run_backfill(backend, run_id=None, apply=True)
    assert len(results) == 1
    assert results[0].status == "backfilled"

    stored = backend.read(PLAYER_PROP_DISTRIBUTIONS_TABLE)
    row = stored.filter(pl.col("distribution_id") == distribution_id).row(0, named=True)
    assert row["pmf_payload"] is not None
    decoded = decode_pmf(bytes(row["pmf_payload"]))
    assert decoded.outcomes == (-5, 0, 10, 200)
    assert decoded.probabilities == (0.1, 0.4, 0.3, 0.2)

    # legacy rows are never deleted
    legacy = backend.read(PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE)
    assert legacy.height == 4

    # re-running is a no-op: the row is no longer a candidate
    again = run_backfill(backend, run_id=None, apply=True)
    assert again == []


def test_mismatched_legacy_normalization_is_reported_and_not_written(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    distribution_id, _key = _write_legacy_row(
        backend, player_id="p3", prop_type="rushing_yards",
        outcomes=[0, 1], probabilities=[0.5, 0.4],  # sums to 0.9, not 1.0
    )
    results = run_backfill(backend, run_id=None, apply=True)
    assert len(results) == 1
    assert results[0].status == "mismatch"

    stored = backend.read(PLAYER_PROP_DISTRIBUTIONS_TABLE)
    row = stored.filter(pl.col("distribution_id") == distribution_id).row(0, named=True)
    assert row.get("pmf_payload") is None


def test_no_legacy_rows_is_skipped(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    distribution_id = compute_distribution_id(
        run_id=RUN_ID, player_id="p4", prop_type="fg_made"
    )
    distribution_key = int(
        hashlib.sha256(distribution_id.encode("utf-8")).hexdigest()[:15], 16
    )
    dist_row = {
        "distribution_key": distribution_key,
        "distribution_id": distribution_id,
        "run_id": RUN_ID,
        "game_id": GAME_ID,
        "player_id": "p4",
        "team_id": "T",
        "position_group": "K",
        "prop_type": "fg_made",
        "support_min": 0,
        "support_max": 0,
        "n_draws": 1000,
        "outcome_count": 1,
        "raw_content_sha256": "legacy-fixture-hash",
        "created_at": NOW,
    }
    backend.append(
        PLAYER_PROP_DISTRIBUTIONS_TABLE, pl.DataFrame([dist_row]), key=["distribution_key"]
    )
    results = run_backfill(backend, run_id=None, apply=True)
    assert len(results) == 1
    assert results[0].status == "skipped_no_legacy_rows"


def test_run_id_filters_candidates(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _write_legacy_row(
        backend, player_id="p5", prop_type="receptions",
        outcomes=[0, 1], probabilities=[0.6, 0.4],
    )
    results = run_backfill(backend, run_id="no-such-run", apply=False)
    assert results == []
