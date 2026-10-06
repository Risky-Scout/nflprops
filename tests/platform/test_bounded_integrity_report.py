"""PR #20: the read-only integrity report in bounded memory.

The PIT and natural-key duplicate checks must give EXACTLY the answers of
the PR #19 report (`_reference_runtime_report`, frozen) -- including its
diagonal-concat semantics (columns missing from some files are null, types
relaxed to the supertype) and `n_unique` struct semantics (null keys) --
while reading one file slice at a time. The production-shaped memory
measurement itself is `tools/report_memory_probe.py` (constrained-worker.yml).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _reference_runtime_report as reference

from nflprops.data.warehouse import _part_name, _parts_dir
from nflprops.platform import runtime_report
from nflprops.platform.runtime_report import ReportReadError, build_report

NOW = datetime(2026, 10, 6, 3, 0, tzinfo=UTC)
T0 = NOW - timedelta(days=3)
TS = pl.Datetime(time_unit="us", time_zone="UTC")


def _ts(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _write(root: Path, table: str, frames: list[pl.DataFrame]) -> None:
    """First frame = legacy single file, the rest = live parts (production
    storage shape)."""
    root.mkdir(parents=True, exist_ok=True)
    frames[0].write_parquet(root / f"{table}.parquet")
    if len(frames) > 1:
        parts = _parts_dir(root, table)
        parts.mkdir(exist_ok=True)
        for n, frame in enumerate(frames[1:], start=1):
            frame.write_parquet(parts / _part_name(n * 10, n * 10))


def _props(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"canonical_game_id": pl.String, "canonical_player_id": pl.String,
                "prop_type": pl.String, "vendor": pl.String,
                "collector_received_at": TS, "available_at": TS, "line_value": pl.Float64},
        orient="row",
    )


@pytest.fixture()
def adversarial(tmp_path: Path) -> Path:
    root = tmp_path / "canonical"
    # player_prop_snapshots: legacy + 3 parts; exact duplicate within a
    # file, a duplicate ACROSS files, keys with nulls (equal null keys are
    # one n_unique value), a future available_at, available_at after receipt.
    p = [("g1", f"p{i}", "rec_yds", "dk", _ts(i), _ts(i), 50.5) for i in range(40)]
    _write(root, "player_prop_snapshots", [
        _props([*p[:20], p[3]]),                                       # in-file dup
        _props([*p[20:30], p[25]]),                                    # in-file dup
        _props(p[30:] + [p[0]]                                         # cross-file dup
               + [("g1", None, "rec_yds", "dk", _ts(1), _ts(1), 1.0)] * 2
               + [("g1", "pz", None, None, None, _ts(2), 1.0)] * 3),
        _props([("g2", "p1", "rec_yds", "fd", _ts(9), NOW + timedelta(hours=1), 1.5),
                ("g2", "p2", "rec_yds", "fd", _ts(9), _ts(10), 1.5)]),  # future; after receipt
    ])
    # roster_snapshots: a key column absent from one file (diagonal concat
    # null-fills it), plus a duplicate.
    roster = pl.DataFrame({"canonical_team_id": ["t1", "t1", "t2", "t1"],
                           "canonical_player_id": ["a", "b", "c", "a"],
                           "available_at": [_ts(1), _ts(2), _ts(3), _ts(1)]})
    _write(root, "roster_snapshots", [roster, roster.drop("canonical_team_id").head(2),
                                      roster.drop("canonical_team_id").head(2)])
    # injury_snapshots: statuses incl. null; duplicate (player, ts, hash).
    injuries = pl.DataFrame({"canonical_player_id": ["a", "a", "b", "c", "c"],
                             "available_at": [_ts(1), _ts(1), _ts(2), _ts(3), _ts(4)],
                             "raw_record_hash": ["h1", "h1", "h2", "h3", "h4"],
                             "status": ["out", "out", None, "questionable", "out"]})
    _write(root, "injury_snapshots", [injuries.head(3), injuries.tail(2)])
    # game_odds_snapshots with a microsecond-vs-millisecond unit mix
    # (the concat relaxes to a supertype).
    odds = pl.DataFrame({"canonical_game_id": ["g1", "g1"], "vendor": ["dk", "dk"],
                         "collector_received_at": [_ts(5), _ts(5)],
                         "available_at": [_ts(5), _ts(6)]})
    _write(root, "game_odds_snapshots", [
        odds, odds.with_columns(pl.col("collector_received_at").cast(
            pl.Datetime(time_unit="ms", time_zone="UTC")))])
    pl.DataFrame({"canonical_game_id": ["g1", "g1", "g2"],
                  "available_at": [_ts(1), _ts(1), _ts(2)],
                  # distinct dates: upcoming_games (unchanged code) orders date ties
                  # nondeterministically in both implementations
                  "date": [NOW + timedelta(days=1), NOW + timedelta(days=1),
                           NOW + timedelta(days=2)], "season": [2026] * 3,
                  "week": [5] * 3, "status_state": ["pre"] * 3}).write_parquet(
        root / "games.parquet")
    pl.DataFrame({"collector_run_id": ["r1", "r1", "r2"],
                  "available_at": [_ts(1), _ts(1), _ts(2)],
                  "started_at": [_ts(1), _ts(1), _ts(2)],
                  "status": ["SUCCESS", "SUCCESS", "FAILED"]}).write_parquet(
        root / "collector_runs.parquet")
    # An empty table: skipped exactly as before.
    pl.DataFrame({"run_id": [], "status": []},
                 schema={"run_id": pl.String, "status": pl.String}).write_parquet(
        root / "prediction_runs.parquet")
    return root


def _checks(report: dict) -> dict:
    return {k: report.get(k) for k in ("pit", "natural_key_duplicates", "injury_status_counts")}


def test_bounded_report_equals_the_reference_on_adversarial_tables(adversarial: Path) -> None:
    old = reference.build_report(adversarial, now=NOW)
    new = build_report(adversarial, now=NOW)
    assert _checks(new) == _checks(old)
    # Not vacuous: the fixture really holds violations and duplicates.
    assert old["natural_key_duplicates"]["player_prop_snapshots"] == 3 + 1 + 2
    assert old["pit"]["player_prop_snapshots"]["future_available_at_rows"] == 1
    # the future row is also after its receipt; the null-receipt row never counts
    assert old["pit"]["player_prop_snapshots"]["available_after_received_rows"] == 2
    assert old["natural_key_duplicates"]["roster_snapshots"] > 0
    assert old["natural_key_duplicates"]["injury_snapshots"] == 1
    assert None in old["injury_status_counts"]
    # Everything outside the PIT/duplicate checks is unchanged too.
    assert {k: v for k, v in new.items() if k != "generated_at"} == {
        k: v for k, v in old.items() if k != "generated_at"}


def test_tiny_slices_and_candidate_batches_give_the_same_answers(
    adversarial: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_report, "_ROWS_PER_SLICE", 3)
    monkeypatch.setattr(runtime_report, "_CANDIDATE_BATCH_HASHES", 1)
    assert _checks(build_report(adversarial, now=NOW)) == _checks(
        reference.build_report(adversarial, now=NOW))


def test_hash_collisions_are_never_counted_as_duplicates(
    adversarial: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even if EVERY key hashed equally, pass 2 compares real key values."""
    monkeypatch.setattr(runtime_report, "_key_hash",
                        lambda key: (pl.struct(key).hash() * 0 + 7).alias("__key_hash"))
    assert _checks(build_report(adversarial, now=NOW)) == _checks(
        reference.build_report(adversarial, now=NOW))


def test_production_shape_scaled_down_equals_the_reference(tmp_path: Path) -> None:
    """Many files, legacy + parts, interleaved games, a few planted duplicates."""
    root = tmp_path / "canonical"
    frames = []
    for part in range(12):
        rows = [(f"g{(part * 7 + i) % 9}", f"p{i % 50}", ["a", "b", "c"][i % 3],
                 ["dk", "fd"][i % 2], _ts(part * 1000 + i), _ts(part * 1000 + i), float(i))
                for i in range(900)]
        frames.append(_props(rows))
    frames.append(_props([tuple(frames[0].row(5)), tuple(frames[3].row(7))]))
    _write(tmp_path / "canonical", "player_prop_snapshots", frames)
    new, old = build_report(root, now=NOW), reference.build_report(root, now=NOW)
    assert _checks(new) == _checks(old)
    assert old["natural_key_duplicates"]["player_prop_snapshots"] == 2


def test_only_the_needed_columns_are_read(adversarial: Path) -> None:
    files = runtime_report.table_files(adversarial, "player_prop_snapshots")
    schema = runtime_report._unified_schema(files)
    key = runtime_report._NATURAL_KEYS["player_prop_snapshots"]
    for path in files:
        for part in runtime_report._slices(path, key, schema):
            assert part.collect_schema().names() == key  # never line_value etc.


def test_fails_closed_on_an_unreadable_file(adversarial: Path) -> None:
    part = next(_parts_dir(adversarial, "player_prop_snapshots").glob("*.parquet"))
    part.write_bytes(b"not parquet")
    with pytest.raises((ReportReadError, pl.exceptions.PolarsError, OSError)):
        build_report(adversarial, now=NOW)


def test_report_never_writes(adversarial: Path) -> None:
    def tree(root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(p for p in root.rglob("*")):
            digest.update(str(path.relative_to(root)).encode())
            if path.is_file():
                digest.update(path.read_bytes())
        return digest.hexdigest()

    before = tree(adversarial)
    build_report(adversarial, now=NOW)
    assert tree(adversarial) == before


def test_probe_gate_thresholds_are_not_loosened() -> None:
    probe = (Path(__file__).resolve().parents[2] / "tools/report_memory_probe.py").read_text()
    assert "NEW_REPORT_MAX_PEAK_MIB = 320.0" in probe
    assert "INGEST_MAX_PEAK_MIB = 320.0" in probe
    ops = (Path(__file__).resolve().parents[2] / "deploy/wizard/ops.sh").read_text()
    assert 'POLARS_MAX_THREADS="$REPORT_POLARS_THREADS"' in ops
    assert json.loads(json.dumps({"ok": True}))["ok"]
