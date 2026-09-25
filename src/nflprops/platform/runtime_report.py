"""BLOCK 3: read-only certification report over the live Wizard warehouse.

Opens Parquet files directly (never constructs a `Warehouse`, which would
create directories) and never takes the writer lock. Used by
`python -m nflprops.platform.wizard_runtime report` (wizard-ops.yml).
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


def _read(root: Path, table: str) -> pl.DataFrame:
    return read_table(root, table)


def _scan(root: Path, table: str) -> pl.LazyFrame | None:
    """The logical table as a lazy scan over every file holding it, so the
    large snapshot tables are aggregated without being materialized."""
    files = table_files(root, table)
    if not files:
        return None
    scans = [pl.scan_parquet(path) for path in files]
    return scans[0] if len(scans) == 1 else pl.concat(scans, how="diagonal_relaxed")


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

    pit: dict[str, Any] = {}
    duplicates: dict[str, int] = {}
    for table, key in _NATURAL_KEYS.items():
        lazy = _scan(warehouse_root, table)
        if lazy is None:
            continue
        columns = set(lazy.collect_schema().names())
        present = [c for c in key if c in columns]
        stats = [pl.len().alias("rows")]
        if present:
            stats.append(pl.struct(present).n_unique().alias("unique_keys"))
        if "available_at" in columns:
            stats.append(pl.col("available_at").max().alias("max_available_at"))
            stats.append(
                (pl.col("available_at") > current).sum().alias("future_available_at_rows")
            )
        if {"available_at", "collector_received_at"} <= columns:
            stats.append(
                (pl.col("available_at") > pl.col("collector_received_at"))
                .sum()
                .alias("available_after_received_rows")
            )
        row = lazy.select(stats).collect(engine="streaming").row(0, named=True)
        if row["rows"] == 0:
            continue
        if present:
            duplicates[table] = row["rows"] - row["unique_keys"]
        entry: dict[str, Any] = {"rows": row["rows"]}
        if "available_at" in columns:
            entry["max_available_at"] = _iso(row["max_available_at"])
            entry["future_available_at_rows"] = row["future_available_at_rows"]
        if {"available_at", "collector_received_at"} <= columns:
            entry["available_after_received_rows"] = row["available_after_received_rows"]
        pit[table] = entry
    report["pit"] = pit
    report["natural_key_duplicates"] = duplicates

    injuries = _scan(warehouse_root, "injury_snapshots")
    if injuries is not None and "status" in injuries.collect_schema().names():
        counts = injuries.group_by("status").len().sort("status").collect(engine="streaming")
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
