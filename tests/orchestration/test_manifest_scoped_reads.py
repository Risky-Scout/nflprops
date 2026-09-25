"""BLOCK 3 hardening: the checkpoint manifest reads the growing live
snapshot tables pre-filtered (game / team / player scope + the PIT cutoff)
instead of loading them whole. The selected rows -- and therefore
`data_manifest_sha256`, a run identity input -- must be byte-for-byte
unchanged, on the legacy single-file layout and on the partitioned one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nflprops.data.warehouse import PARTITIONED_TABLES, Warehouse
from nflprops.orchestration import manifest as manifest_module
from nflprops.orchestration.manifest import compute_data_manifest_sha256

T0 = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
GAMES = [("g0", "t0", "t1"), ("g1", "t2", "t3"), ("g2", "t4", "t5")]
CYCLES = 6


def _ts(cycle: int) -> datetime:
    return T0 + timedelta(hours=cycle)


def _batches() -> list[tuple[str, pl.DataFrame, dict]]:
    """(table, frame, append kwargs) in live-collection order."""
    out: list[tuple[str, pl.DataFrame, dict]] = []
    games = pl.DataFrame(
        [
            {
                "canonical_game_id": g,
                "available_at": T0 - timedelta(days=1),
                "date": T0 + timedelta(days=3),
                "season": 2026,
                "week": 3,
                "home_canonical_team_id": home,
                "visitor_canonical_team_id": away,
            }
            for g, home, away in GAMES
        ]
    )
    out.append(("games", games, {"key": ["canonical_game_id", "available_at"]}))
    for cycle in range(CYCLES):
        at = _ts(cycle)
        teams = [t for _, home, away in GAMES for t in (home, away)]
        roster = pl.DataFrame(
            [
                {
                    "canonical_team_id": t,
                    "canonical_player_id": f"{t}-p{p}",
                    "available_at": at,
                    "position": "WR",
                    "depth": p,
                }
                for t in teams
                for p in range(3)
            ]
        )
        injuries = pl.DataFrame(
            [  # two records per (player, available_at): ties in the manifest sort
                {
                    "canonical_player_id": f"{t}-p{p}",
                    "available_at": at,
                    "raw_record_hash": f"{cycle}-{t}-{p}-{k}",
                    "status": ["questionable", "out"][k],
                }
                for t in teams
                for p in range(2)
                for k in range(2)
            ]
        )
        odds = pl.DataFrame(
            [
                {
                    "canonical_game_id": g,
                    "vendor": v,
                    "collector_received_at": at,
                    "available_at": at,
                    "spread_home_value": -3.5 + cycle,
                }
                for g, _, _ in GAMES
                for v in ("dk", "fd")
            ]
        )
        props = pl.DataFrame(
            [
                {
                    "canonical_game_id": g,
                    "canonical_player_id": f"{home}-p{p}",
                    "prop_type": "receiving_yards",
                    "vendor": v,
                    "collector_received_at": at,
                    "available_at": at,
                    "line_value": 40.5 + p + cycle,
                }
                for g, home, _ in GAMES
                for p in range(3)
                for v in ("dk", "fd")
            ]
        )
        runs = pl.DataFrame(
            [
                {
                    "resource_run_id": f"{cycle}-{r}",
                    "resource_type": r,
                    "collector_received_at": at,
                    "collection_status": "SUCCESS",
                }
                for r in ("GAMES", "ROSTERS", "INJURIES", "GAME_ODDS", "PLAYER_PROPS")
            ]
        )
        for table, frame in (
            ("roster_snapshots", roster),
            ("injury_snapshots", injuries),
            ("game_odds_snapshots", odds),
        ):
            spec = PARTITIONED_TABLES[table]
            out.append((table, frame, {"key": list(spec.key), "sort_by": list(spec.sort_by)}))
        spec = PARTITIONED_TABLES["player_prop_snapshots"]
        for g, _, _ in GAMES:  # per game, like collect_once
            out.append(
                (
                    "player_prop_snapshots",
                    props.filter(pl.col("canonical_game_id") == g),
                    {"key": list(spec.key), "sort_by": list(spec.sort_by)},
                )
            )
        out.append(("collector_resource_runs", runs, {"key": ["resource_run_id"]}))
    return out


def _legacy_single_file(root: Path) -> Warehouse:
    """Every table as ONE file, exactly as the pre-hardening append left it."""
    wh = Warehouse(root)
    frames: dict[str, list[pl.DataFrame]] = {}
    kwargs: dict[str, dict] = {}
    for table, frame, kw in _batches():
        frames.setdefault(table, []).append(frame)
        kwargs[table] = kw
    for table, parts in frames.items():
        out = pl.concat(parts, how="diagonal_relaxed").unique(
            subset=kwargs[table]["key"], keep="last", maintain_order=True
        )
        sort_by = kwargs[table].get("sort_by")
        if sort_by:
            out = out.sort(sort_by)
        out.write_parquet(root / f"{table}.parquet")
    return wh


def _partitioned(root: Path) -> Warehouse:
    wh = Warehouse(root)
    for table, frame, kw in _batches():
        wh.append(table, frame, **kw)
    return wh


@pytest.fixture(scope="module")
def warehouses(tmp_path_factory: pytest.TempPathFactory) -> tuple[Warehouse, Warehouse]:
    base = tmp_path_factory.mktemp("manifest")
    legacy = _legacy_single_file(base / "legacy")
    parted = _partitioned(base / "parted")
    assert (parted.root / "player_prop_snapshots.parts").is_dir()
    return legacy, parted


CUTOFFS = [T0 - timedelta(hours=1), _ts(0), _ts(2) + timedelta(minutes=30), _ts(CYCLES + 1)]


@pytest.mark.parametrize("game_id", [g for g, _, _ in GAMES])
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_scoped_reads_leave_the_manifest_hash_unchanged(
    warehouses: tuple[Warehouse, Warehouse],
    monkeypatch: pytest.MonkeyPatch,
    game_id: str,
    cutoff: datetime,
) -> None:
    legacy, parted = warehouses

    def sha(wh: Warehouse) -> str:
        return compute_data_manifest_sha256(wh, game_id=game_id, scheduled_as_of=cutoff)

    scoped_legacy = sha(legacy)
    scoped_parted = sha(parted)
    with monkeypatch.context() as unscoped:
        # The pre-hardening behavior: load every snapshot table whole.
        unscoped.setattr(
            manifest_module, "_read_scoped", lambda wh, table, where: wh.read(table)
        )
        unscoped_legacy = sha(legacy)
    assert scoped_legacy == unscoped_legacy == scoped_parted


def test_scoped_manifest_never_loads_other_games_or_future_rows(
    warehouses: tuple[Warehouse, Warehouse], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, parted = warehouses
    cutoff = _ts(2) + timedelta(minutes=30)
    loaded: dict[str, int] = {}
    original = Warehouse.read

    def spy(self: Warehouse, table: str, *, where: pl.Expr | None = None) -> pl.DataFrame:
        frame = original(self, table, where=where)
        loaded[table] = frame.height
        return frame

    monkeypatch.setattr(Warehouse, "read", spy)
    compute_data_manifest_sha256(parted, game_id="g1", scheduled_as_of=cutoff)
    full_props = original(parted, "player_prop_snapshots").height
    # g1's props at cycles 0..2 only: 3 players x 2 vendors x 3 cycles.
    assert loaded["player_prop_snapshots"] == 18 < full_props
    assert loaded["roster_snapshots"] == 2 * 3 * 3  # g1's two teams, 3 cycles
