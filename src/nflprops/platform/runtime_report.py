"""BLOCK 3: read-only certification report over the live Wizard warehouse.

Opens Parquet files directly (never constructs a `Warehouse`, which would
create directories) and never takes the writer lock. Used by
`python -m nflprops.platform.wizard_runtime report` (wizard-ops.yml), which
runs OUTSIDE the runtime's cgroup on a small host: the PIT and duplicate
checks read one file at a time and never materialize a large table
(PR #20). Any table that cannot be read or aligned exactly fails the report
closed (`ReportReadError`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.data.warehouse import read_table, table_files

#: Canonical snapshot tables and the natural keys `collect_once` appends
#: them with (nflprops.collection.service). Duplicates on these keys would
#: mean a restart manufactured a second scientific observation.
_NATURAL_KEYS: dict[str, list[str]] = {
    "games": ["canonical_game_id", "available_at"],
    "roster_snapshots": ["canonical_team_id", "canonical_player_id", "available_at"],
    "injury_snapshots": ["canonical_player_id", "available_at", "raw_record_hash"],
    "game_odds_snapshots": ["canonical_game_id", "vendor", "collector_received_at"],
    "player_prop_snapshots": [
        "canonical_game_id",
        "canonical_player_id",
        "prop_type",
        "vendor",
        "collector_received_at",
    ],
    "collector_runs": ["collector_run_id"],
    "collector_resource_runs": ["resource_run_id"],
    "prediction_runs": ["run_id"],
    "remote_checkpoint_requests": ["request_id"],
    "runtime_collection_triggers": ["collector_run_id"],
}


#: Hash seeds for the duplicate check's partitioning hash (any fixed
#: values: equal keys hash equally within one process, which is all the
#: exact count relies on).
_HASH_SEEDS = (0x6E666C70, 0x726F7073, 0x52455054, 0x31393937)
#: Candidate rows (keys sharing a hash) re-read per batch: bounds the exact
#: comparison even if a table genuinely held millions of duplicates.
_CANDIDATE_BATCH_HASHES = 200_000
#: Rows decoded at once from any one file. A single legacy file holds up to
#: ~1.8M prop rows; polars does not stream a struct hash, so whole-file
#: reads decoded it all at once (~540 MiB measured). Slices are pushed down
#: to the Parquet reader, so only the row groups a slice covers are decoded.
_ROWS_PER_SLICE = 100_000


class ReportReadError(RuntimeError):
    """A table could not be read or aligned exactly; the report fails
    closed rather than certifying a table it did not fully check."""


def _read(root: Path, table: str) -> pl.DataFrame:
    return read_table(root, table)


def _unified_schema(files: list[Path]) -> dict[str, pl.DataType]:
    """The schema `pl.concat(scans, how="diagonal_relaxed")` gives the
    logical table: every column of any file, each at its supertype."""
    scans = [pl.scan_parquet(path) for path in files]
    lazy = scans[0] if len(scans) == 1 else pl.concat(scans, how="diagonal_relaxed")
    try:
        return dict(lazy.collect_schema())
    except Exception as exc:  # polars raises several types for bad files
        raise ReportReadError(f"cannot read schema of {files[0].parent}: {exc}") from exc


def _aligned(path: Path, columns: list[str], schema: dict[str, pl.DataType]) -> pl.LazyFrame:
    """One file's `columns` exactly as the diagonal concat presents them:
    present columns cast to the logical supertype, absent ones null."""
    names = set(pl.read_parquet_schema(path))
    return pl.scan_parquet(path).select(
        (pl.col(c) if c in names else pl.lit(None)).cast(schema[c]).alias(c) for c in columns
    )


def _collect(lazy: pl.LazyFrame, path: Path) -> pl.DataFrame:
    try:
        return lazy.collect(engine="streaming")
    except Exception as exc:
        raise ReportReadError(f"cannot read {path}: {exc}") from exc


def _row_count(path: Path) -> int:
    return int(_collect(pl.scan_parquet(path).select(pl.len()), path).item())


def _slices(
    path: Path, columns: list[str], schema: dict[str, pl.DataType]
) -> list[pl.LazyFrame]:
    """`_aligned(path, columns, schema)` as consecutive row slices of at most
    `_ROWS_PER_SLICE` rows (together: every row, in file order, once)."""
    n = _row_count(path)
    aligned = _aligned(path, columns, schema)
    return [aligned.slice(offset, _ROWS_PER_SLICE) for offset in range(0, n, _ROWS_PER_SLICE)]


def _pit_stats(
    files: list[Path], schema: dict[str, pl.DataType], current: datetime
) -> dict[str, Any]:
    """rows / max(available_at) / future_available_at_rows /
    available_after_received_rows over the logical table, one file at a
    time (only the two timestamp columns are ever read)."""
    has_available = "available_at" in schema
    has_received = has_available and "collector_received_at" in schema
    columns: list[str] = [c for c in ("available_at", "collector_received_at") if c in schema
               and (c == "available_at" or has_received)]
    totals: dict[str, Any] = {"rows": 0, "max_available_at": None,
                              "future_available_at_rows": 0,
                              "available_after_received_rows": 0}
    stats = [pl.len().alias("rows")]
    if has_available:
        stats.append(pl.col("available_at").max().alias("max_available_at"))
        stats.append((pl.col("available_at") > current).sum()
                     .alias("future_available_at_rows"))
    if has_received:
        stats.append((pl.col("available_at") > pl.col("collector_received_at")).sum()
                     .alias("available_after_received_rows"))
    for path in files:
        if not columns:
            totals["rows"] += _row_count(path)
            continue
        for part in _slices(path, columns, schema):
            _accumulate(totals, _collect(part.select(stats), path).row(0, named=True))
    return totals


def _accumulate(totals: dict[str, Any], row: dict[str, Any]) -> None:
    totals["rows"] += row["rows"]
    for name in ("future_available_at_rows", "available_after_received_rows"):
        totals[name] += row.get(name) or 0
    top = row.get("max_available_at")
    if top is not None and (totals["max_available_at"] is None or top > totals["max_available_at"]):
        totals["max_available_at"] = top


def _key_hash(key: list[str]) -> pl.Expr:
    return pl.struct(key).hash(*_HASH_SEEDS).alias("__key_hash")


def _duplicate_rows(files: list[Path], key: list[str], schema: dict[str, pl.DataType]) -> int:
    """Exactly `rows - n_unique(struct(key))` over the logical table, in
    bounded memory. Pass 1 keeps only a 64-bit hash of each row's (aligned)
    key; rows whose hash is unique are certainly unique keys (equal keys
    hash equally). Pass 2 re-reads only the rows sharing a hash and counts
    exact duplicates among their real key values, so a hash collision can
    never be reported as a duplicate."""
    hashes = pl.concat(
        [_collect(part.select(_key_hash(key)), path)["__key_hash"]
         for path in files for part in _slices(path, key, schema)]
    ).sort()
    repeated = hashes.filter(hashes == hashes.shift(1)).unique().sort()
    del hashes
    if repeated.is_empty():
        return 0
    duplicates = 0
    for start in range(0, repeated.len(), _CANDIDATE_BATCH_HASHES):
        batch = repeated.slice(start, _CANDIDATE_BATCH_HASHES).implode()
        rows = pl.concat(
            [_collect(part.filter(_key_hash(key).is_in(batch)), path)
             for path in files for part in _slices(path, key, schema)],
            how="vertical",
        )
        duplicates += rows.height - rows.select(pl.struct(key).n_unique()).item()
    return duplicates


def _status_counts(files: list[Path], schema: dict[str, pl.DataType]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for path in files:
        for piece in _slices(path, ["status"], schema):
            part = _collect(piece.group_by("status").len(), path)
            for status, n in part.iter_rows():
                counts[status] = counts.get(status, 0) + n
    return counts


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def build_report(warehouse_root: Path, *, now: datetime | None = None) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    report: dict[str, Any] = {"generated_at": current.isoformat(), "warehouse": str(warehouse_root)}

    runs = _read(warehouse_root, "collector_runs")
    triggers = _read(warehouse_root, "runtime_collection_triggers")
    if not runs.is_empty():
        joined = runs
        if not triggers.is_empty():
            joined = runs.join(
                triggers.select("collector_run_id", "trigger"), on="collector_run_id", how="left"
            )
        ordered = joined.sort("started_at")
        report["collection"] = {
            "cycles": ordered.height,
            "first_started_at": _iso(ordered["started_at"][0]),
            "last_started_at": _iso(ordered["started_at"][-1]),
            "by_status": dict(ordered.group_by("status").len().iter_rows()),
            "by_trigger": dict(ordered.group_by("trigger").len().iter_rows())
            if "trigger" in ordered.columns
            else {},
            "recent": [
                {k: _iso(v) for k, v in row.items()}
                for row in ordered.tail(8)
                .select(
                    [
                        c
                        for c in (
                            "collector_run_id",
                            "status",
                            "trigger",
                            "season",
                            "week",
                            "started_at",
                            "completed_at",
                            "cadence_seconds",
                        )
                        if c in ordered.columns
                    ]
                )
                .iter_rows(named=True)
            ],
        }
    else:
        report["collection"] = {"cycles": 0}

    resources = _read(warehouse_root, "collector_resource_runs")
    if not resources.is_empty():
        group = [c for c in ("resource_type", "collection_status") if c in resources.columns]
        report["resource_status_counts"] = [
            {k: _iso(v) for k, v in row.items()}
            for row in resources.group_by(group).len().sort(group).iter_rows(named=True)
        ]

    # PIT sanity and natural-key duplicates, in bounded memory: each table
    # is read one file at a time and only the columns a check needs (PR #20;
    # the former whole-table `n_unique` peaked at ~2.4 GiB on Wizard's
    # 4.8M-row prop table and was OOM-killed). Same answers, same keys.
    pit: dict[str, Any] = {}
    duplicates: dict[str, int] = {}
    for table, key in _NATURAL_KEYS.items():
        files = table_files(warehouse_root, table)
        if not files:
            continue
        schema = _unified_schema(files)
        present = [c for c in key if c in schema]
        stats = _pit_stats(files, schema, current)
        if stats["rows"] == 0:
            continue
        if present:
            duplicates[table] = _duplicate_rows(files, present, schema)
        entry: dict[str, Any] = {"rows": stats["rows"]}
        if "available_at" in schema:
            entry["max_available_at"] = _iso(stats["max_available_at"])
            entry["future_available_at_rows"] = stats["future_available_at_rows"]
        if {"available_at", "collector_received_at"} <= set(schema):
            entry["available_after_received_rows"] = stats["available_after_received_rows"]
        pit[table] = entry
    report["pit"] = pit
    report["natural_key_duplicates"] = duplicates

    injury_files = table_files(warehouse_root, "injury_snapshots")
    if injury_files:
        injury_schema = _unified_schema(injury_files)
        if "status" in injury_schema:
            totals = _status_counts(injury_files, injury_schema)
            counts = pl.DataFrame(
                {"status": list(totals), "len": list(totals.values())},
                schema={"status": injury_schema["status"], "len": pl.UInt32},
            ).sort("status")
            if counts.height:
                report["injury_status_counts"] = dict(counts.iter_rows())

    predictions = _read(warehouse_root, "prediction_runs")
    if not predictions.is_empty():
        report["prediction_runs"] = [
            {k: _iso(v) for k, v in row.items()}
            for row in predictions.group_by(["checkpoint_name", "status"])
            .len()
            .sort(["checkpoint_name", "status"])
            .iter_rows(named=True)
        ]
    requests = _read(warehouse_root, "remote_checkpoint_requests")
    if not requests.is_empty():
        report["checkpoint_requests"] = [
            {k: _iso(v) for k, v in row.items()}
            for row in requests.select(
                "run_id",
                "checkpoint_name",
                "game_id",
                "scheduled_as_of",
                "state",
                "snapshot_id",
                "snapshot_manifest_sha256",
                "execution_target",
            ).iter_rows(named=True)
        ]

    games = _read(warehouse_root, "games")
    if not games.is_empty():
        latest = games.sort("available_at").group_by("canonical_game_id").tail(1)
        upcoming = latest.filter(pl.col("date") > current).sort("date")
        report["upcoming_games"] = [
            {k: _iso(v) for k, v in row.items()}
            for row in upcoming.select(
                "canonical_game_id", "season", "week", "date", "status_state"
            )
            .head(20)
            .iter_rows(named=True)
        ]
    return report
