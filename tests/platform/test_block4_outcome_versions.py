"""BLOCK 4: immutable versioned outcome history.

Covers the storage contract (`nflprops.data.outcome_versions`), every
reader class that must select a specific version (PIT state/manifest,
settlement, training/recalibration labels), and the Weeks 1-4 backfill
semantics (`nflprops.platform.stats_backfill`): genuine receipt-time
availability only, corrections appended, nothing ever fabricated.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestration"))

from _fixtures import (
    HOME_PLAYER_ID,
    TARGET_GAME_ID,
    _player_game_row,
    _team_game_row,
    build_pit_fixture_warehouse,
)

from nflprops.calibration.historical_runner import final_outcome_rows_for_game
from nflprops.data.outcome_versions import (
    CONTENT_SHA,
    FIRST_SEEN_AT,
    SOURCE_STATUS,
    VERSION_ID,
    OutcomeVersionError,
    append_outcome_versions,
    as_known_at,
    first_known,
    latest_final,
    migrate_legacy_rows,
)
from nflprops.data.quality import validate_core
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.manifest import build_checkpoint_manifest
from nflprops.pipelines.settle import reconcile_settlement_stats, settle_predictions
from nflprops.platform.stats_backfill import backfill_outcome_history
from nflprops.state.player import build_player_states
from nflprops.state.team import build_team_states

T0 = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)  # first final observation
T1 = T0 + timedelta(days=1)  # first correction
T2 = T0 + timedelta(days=3)  # second correction


def _player(at: datetime, receiving_yards: int, *, game: str = "g1", player: str = "p1") -> dict:
    return {
        "canonical_game_id": game, "canonical_player_id": player, "canonical_team_id": "h",
        "receiving_yards": receiving_yards, "receptions": 6, "receiving_targets": 8,
        "rushing_attempts": 0, "rushing_yards": 0, "passing_attempts": 0,
        "passing_completions": 0, "passing_touchdowns": 0, "passing_interceptions": 0,
        "available_at": at, "ingested_at": at, "available_at_is_estimated": False,
        "provider": "bdl", "provider_record_id": f"{game}:{player}:h",
    }


def _team(at: datetime, points: int, *, team: str = "h") -> dict:
    return {
        "canonical_game_id": "g1", "canonical_team_id": team, "total_points": points,
        "passing_attempts": 30, "rushing_attempts": 25, "passing_completions": 20, "sacks": 1,
        "available_at": at, "ingested_at": at, "available_at_is_estimated": False,
        "provider": "bdl", "provider_record_id": f"g1:{team}",
    }


def _append(wh: Warehouse, table: str, rows: list[dict], run: str = "r"):
    return append_outcome_versions(
        wh, table, pl.DataFrame(rows), ingest_run_id=run,
        source_status=["Final"] * len(rows),
    )


@pytest.fixture()
def versioned(tmp_path: Path) -> Warehouse:
    """g1/p1 receiving_yards 50 (T0) -> 57 (T1) -> 61 (T2); team h 20 -> 23."""
    wh = Warehouse(tmp_path / "canonical")
    _append(wh, "player_game_stats", [_player(T0, 50)])
    _append(wh, "player_game_stats", [_player(T1, 57)])
    _append(wh, "player_game_stats", [_player(T2, 61)])
    _append(wh, "team_game_stats", [_team(T0, 20), _team(T0, 17, team="v")])
    _append(wh, "team_game_stats", [_team(T1, 23), _team(T1, 17, team="v")])
    return wh


# ------------------------------------------------------------ storage


def test_multiple_corrected_player_versions_are_all_retained(versioned: Warehouse) -> None:
    stored = versioned.read("player_game_stats").sort(FIRST_SEEN_AT)
    assert stored["receiving_yards"].to_list() == [50, 57, 61]
    assert stored[FIRST_SEEN_AT].to_list() == [T0, T1, T2]
    assert stored["available_at"].to_list() == [T0, T1, T2]
    assert stored[VERSION_ID].n_unique() == 3
    assert stored[CONTENT_SHA].n_unique() == 3
    assert stored[SOURCE_STATUS].to_list() == ["Final"] * 3


def test_multiple_corrected_team_versions_only_append_changed_rows(
    versioned: Warehouse,
) -> None:
    stored = versioned.read("team_game_stats")
    # team v never changed: one version; team h: two.
    assert stored.filter(pl.col("canonical_team_id") == "v").height == 1
    assert stored.filter(pl.col("canonical_team_id") == "h")["total_points"].to_list() == [20, 23]


def test_old_versions_are_immutable_and_identical_refetch_adds_nothing(
    versioned: Warehouse,
) -> None:
    before = versioned.read("player_game_stats").sort(VERSION_ID)
    result = _append(versioned, "player_game_stats", [_player(T2 + timedelta(days=1), 61)])
    assert (result.new_keys, result.corrections, result.unchanged) == (0, 0, 1)
    assert versioned.read("player_game_stats").sort(VERSION_ID).equals(before)

    result = _append(versioned, "player_game_stats", [_player(T2 + timedelta(days=2), 64)])
    assert result.corrections == 1
    after = versioned.read("player_game_stats")
    kept = after.filter(pl.col(VERSION_ID).is_in(before[VERSION_ID].implode()))
    assert kept.select(before.columns).sort(VERSION_ID).equals(before)
    assert after.height == before.height + 1


def test_correction_observed_out_of_order_is_refused(versioned: Warehouse) -> None:
    before = versioned.read("player_game_stats")
    with pytest.raises(OutcomeVersionError, match="no later than"):
        _append(versioned, "player_game_stats", [_player(T1, 99)])
    assert versioned.read("player_game_stats").equals(before)


def test_conflicting_contents_in_one_batch_are_refused(tmp_path: Path) -> None:
    wh = Warehouse(tmp_path / "c")
    with pytest.raises(OutcomeVersionError, match="different contents"):
        _append(wh, "player_game_stats", [_player(T0, 50), _player(T0, 51)])
    assert not wh.exists("player_game_stats")


def test_correction_is_never_visible_before_it_was_first_seen(tmp_path: Path) -> None:
    """Even a batch carrying an (allowed) estimated availability cannot
    back-date a correction: it is visible only from first_seen_at."""
    wh = Warehouse(tmp_path / "c")
    estimated = {**_player(T0, 50), "available_at": T0 - timedelta(hours=6),
                 "available_at_is_estimated": True}
    append_outcome_versions(wh, "player_game_stats", pl.DataFrame([estimated]),
                            ingest_run_id="r", allow_estimated=True)
    correction = {**_player(T1, 57), "available_at": T0 - timedelta(hours=6),
                  "available_at_is_estimated": True}
    append_outcome_versions(wh, "player_game_stats", pl.DataFrame([correction]),
                            ingest_run_id="r", allow_estimated=True)
    stored = wh.read("player_game_stats").sort(FIRST_SEEN_AT)
    assert stored["available_at"].to_list() == [T0 - timedelta(hours=6), T1]
    assert stored["available_at_is_estimated"].to_list() == [True, False]


def test_estimated_availability_refused_unless_explicitly_allowed(tmp_path: Path) -> None:
    wh = Warehouse(tmp_path / "c")
    row = {**_player(T0, 50), "available_at_is_estimated": True}
    with pytest.raises(OutcomeVersionError, match="estimated"):
        append_outcome_versions(wh, "player_game_stats", pl.DataFrame([row]), ingest_run_id="r")


def test_legacy_rows_migrate_additively(tmp_path: Path) -> None:
    wh = Warehouse(tmp_path / "c")
    legacy = pl.DataFrame([_player(T0, 50)]).drop("ingested_at")
    wh.write("player_game_stats", legacy)
    _append(wh, "player_game_stats", [_player(T1, 57)])
    stored = wh.read("player_game_stats").sort(FIRST_SEEN_AT)
    assert stored["receiving_yards"].to_list() == [50, 57]
    migrated = stored.row(0, named=True)
    assert migrated[FIRST_SEEN_AT] == T0  # its own available_at, nothing estimated
    assert migrated["ingest_run_id"] == "legacy-migration"
    assert stored.select(legacy.columns).head(1).equals(legacy)


# ------------------------------------------------------------ PIT selection


def test_pit_lookup_before_and_after_each_correction(versioned: Warehouse) -> None:
    ps = versioned.read("player_game_stats")
    assert as_known_at(ps, "player_game_stats", T0 - timedelta(seconds=1)).is_empty()
    for cutoff, expected in (
        (T0, 50), (T1 - timedelta(seconds=1), 50), (T1, 57),
        (T2 - timedelta(seconds=1), 57), (T2, 61), (T2 + timedelta(days=30), 61),
    ):
        known = as_known_at(ps, "player_game_stats", cutoff)
        assert known["receiving_yards"].to_list() == [expected], cutoff
    ts = versioned.read("team_game_stats")
    before = as_known_at(ts, "team_game_stats", T1 - timedelta(seconds=1))
    after = as_known_at(ts, "team_game_stats", T1)
    assert before.sort("canonical_team_id")["total_points"].to_list() == [20, 17]
    assert after.sort("canonical_team_id")["total_points"].to_list() == [23, 17]


def test_first_known_vs_latest_final(versioned: Warehouse) -> None:
    ps = versioned.read("player_game_stats")
    assert first_known(ps, "player_game_stats")["receiving_yards"].to_list() == [50]
    assert latest_final(ps, "player_game_stats")["receiving_yards"].to_list() == [61]


def test_single_version_frames_are_returned_unchanged() -> None:
    frame = pl.DataFrame([_player(T0, 50, player="b"), _player(T0, 40, player="a")])
    assert as_known_at(frame, "player_game_stats", T1).equals(frame)
    assert latest_final(frame, "player_game_stats").equals(frame)


def test_state_builders_count_each_game_once_at_the_pit_version(tmp_path: Path) -> None:
    def _stamp(row: dict, at: datetime) -> dict:
        return {**row, "available_at": at, "ingested_at": at}

    wh = Warehouse(tmp_path / "c")
    p0 = _player_game_row(game_id="g1", team_id="h", player_id="p1", available_at=T0)
    teams0 = [_team_game_row(game_id="g1", team_id=t, available_at=T0) for t in ("h", "v")]
    append_outcome_versions(wh, "player_game_stats", pl.DataFrame([_stamp(p0, T0)]),
                            ingest_run_id="r")
    append_outcome_versions(wh, "team_game_stats",
                            pl.DataFrame([_stamp(t, T0) for t in teams0]), ingest_run_id="r")
    p1 = _stamp({**p0, "receiving_yards": p0["receiving_yards"] + 30}, T1)
    t1 = _stamp({**teams0[0], "passing_yards": 999}, T1)
    append_outcome_versions(wh, "player_game_stats", pl.DataFrame([p1]), ingest_run_id="r")
    append_outcome_versions(wh, "team_game_stats", pl.DataFrame([t1]), ingest_run_id="r")

    ps = wh.read("player_game_stats")
    ts = wh.read("team_game_stats")
    assert ps.height == 2 and ts.height == 3
    players = pl.DataFrame([{"canonical_player_id": "p1", "position_group": "WR"}])
    for cutoff in (T0, T1):
        one_ps = as_known_at(ps, "player_game_stats", cutoff)
        one_ts = as_known_at(ts, "team_game_stats", cutoff)
        assert one_ps.height == 1 and one_ts.height == 2
        # All versions present == only the PIT version present.
        assert build_player_states(ps, ts, players, as_of=cutoff, strict=False) == (
            build_player_states(one_ps, one_ts, players, as_of=cutoff, strict=False)
        )
        assert build_team_states(ts, ps, as_of=cutoff, strict=False) == build_team_states(
            one_ts, one_ps, as_of=cutoff, strict=False
        )
    # And the correction is actually reflected after it was seen.
    assert build_player_states(ps, ts, players, as_of=T0, strict=False) != build_player_states(
        ps, ts, players, as_of=T1, strict=False
    )


# ------------------------------------------------------------ final truth


def _prediction() -> pl.DataFrame:
    return pl.DataFrame({
        "prediction_id": ["pred-1"], "game_id": ["g1"], "player_id": ["p1"],
        "prop_type": ["receiving_yards"], "vendor": ["book"], "side": ["OVER"],
        "line": [55.5], "american_odds": [-110], "model_version": ["m"],
    })


def test_settlement_uses_latest_final_version(versioned: Warehouse) -> None:
    ps = versioned.read("player_game_stats")
    # Stored order puts the final version first; selection must not depend on it.
    shuffled = ps.sort(FIRST_SEEN_AT, descending=True)
    settled = settle_predictions(_prediction(), shuffled)
    assert settled.height == 1
    assert settled["actual_value"][0] == 61.0


def test_recalibration_settlement_stats_use_latest_final(versioned: Warehouse) -> None:
    reconciled = reconcile_settlement_stats(
        versioned.read("player_game_stats"), versioned.read("team_game_stats")
    )
    assert reconciled.height == 1
    assert reconciled["receiving_yards"].to_list() == [61]


def test_training_labels_use_latest_final(versioned: Warehouse) -> None:
    rows = final_outcome_rows_for_game(versioned.read("player_game_stats"), "g1")
    assert rows.height == 1
    assert rows["receiving_yards"].to_list() == [61]


def test_quality_accepts_versions_but_blocks_duplicate_versions(versioned: Warehouse) -> None:
    ps = versioned.read("player_game_stats")
    ts = versioned.read("team_game_stats")
    codes = {i.code for i in validate_core(player_stats=ps, team_stats=ts)}
    assert not codes & {"DUPLICATE_PLAYER_GAME", "DUPLICATE_TEAM_GAME", "UNPAIRED_TEAM_GAME"}
    doubled = pl.concat([ps, ps.head(1)])
    codes = {i.code for i in validate_core(player_stats=doubled, team_stats=ts)}
    assert "DUPLICATE_PLAYER_GAME_VERSION" in codes


# ------------------------------------------------------------ manifest


def test_manifest_is_deterministic_and_ignores_later_corrections(tmp_path: Path) -> None:
    kickoff = datetime(2025, 9, 8, 17, 0, tzinfo=UTC)
    as_of = kickoff - timedelta(hours=6)
    late = as_of + timedelta(hours=1)

    def _fixture(name: str) -> Warehouse:
        wh = build_pit_fixture_warehouse(
            tmp_path / name, kickoff_at=kickoff,
            quote_visible_at=as_of - timedelta(minutes=1),
            quote_hidden_at=as_of + timedelta(minutes=1),
        )
        for table in ("player_game_stats", "team_game_stats"):
            rows = wh.read(table).with_columns(pl.col("available_at").alias("ingested_at"))
            wh.write(table, migrate_legacy_rows(rows, table))
        return wh

    uncorrected = _fixture("uncorrected")
    corrected = _fixture("corrected")
    hist = corrected.read("player_game_stats").filter(
        pl.col("canonical_player_id") == HOME_PLAYER_ID
    ).row(0, named=True)
    correction = {**hist, "receiving_yards": hist["receiving_yards"] + 9,
                  "available_at": late, "ingested_at": late}
    append_outcome_versions(corrected, "player_game_stats", pl.DataFrame([correction]),
                            ingest_run_id="r")
    assert corrected.read("player_game_stats").height == 3

    def _sha(wh: Warehouse, cutoff: datetime) -> str:
        return build_checkpoint_manifest(
            wh, game_id=TARGET_GAME_ID, scheduled_as_of=cutoff
        ).data_manifest_sha256

    # Deterministic, and a correction first seen after the cutoff cannot
    # change what that checkpoint knew.
    assert _sha(corrected, as_of) == _sha(corrected, as_of) == _sha(uncorrected, as_of)
    # From the moment it was seen, the correction is the known version.
    assert _sha(corrected, late) != _sha(uncorrected, late)


# ------------------------------------------------------------ Weeks 1-4 backfill


class _WeeksProvider:
    """2026 Weeks 1-5; week 5 is in progress. Records carry the provider
    boundary's genuine receipt time, like `BDLProvider` (`MappingContext`)."""

    def __init__(self, received_at: datetime, yards: dict[str, int] | None = None) -> None:
        self.received_at = received_at
        self.yards = yards or {}

    def games(self, seasons=None, season_types=None, **_kw):
        rows = []
        for week in range(1, 6):
            rows.append({
                "canonical_game_id": f"w{week}", "season": 2026, "season_type": 2,
                "week": week, "status": "Final" if week < 5 else "2nd Quarter",
                "date": datetime(2026, 9, 6, 17, tzinfo=UTC) + timedelta(weeks=week - 1),
            })
        return rows

    def _pit(self) -> dict:
        return {"available_at": self.received_at, "ingested_at": self.received_at,
                "available_at_is_estimated": False, "provider": "bdl"}

    def player_game_stats(self, seasons=None, season_type=None, **_kw):
        if season_type != 2:
            return []
        return [{"canonical_game_id": f"w{w}", "canonical_player_id": "p1",
                 "canonical_team_id": "h", "receiving_yards": self.yards.get(f"w{w}", 40 + w),
                 "provider_record_id": f"w{w}:p1", **self._pit()} for w in range(1, 6)]

    def team_game_stats(self, seasons=None, season_type=None, **_kw):
        if season_type != 2:
            return []
        return [{"canonical_game_id": f"w{w}", "canonical_team_id": t, "total_points": 20,
                 "provider_record_id": f"w{w}:{t}", **self._pit()}
                for w in range(1, 6) for t in ("h", "v")]


def test_weeks_1_4_backfill_is_final_only_and_uses_genuine_receipt_time(tmp_path: Path) -> None:
    received = datetime(2026, 10, 2, 15, 0, tzinfo=UTC)
    wh = Warehouse(tmp_path / "canonical")
    results = backfill_outcome_history(
        _WeeksProvider(received), wh, seasons=[2026], weeks=[1, 2, 3, 4],
        lock_path=tmp_path / "writer.lock", now=received + timedelta(seconds=5),
    )
    assert results[0].season.final_games == 4
    assert results[0].player.new_keys == 4 and results[0].team.new_keys == 8
    ps = wh.read("player_game_stats")
    assert sorted(ps["canonical_game_id"].to_list()) == ["w1", "w2", "w3", "w4"]
    # No fabricated PIT: availability is the real receipt time, never the
    # game date (+lag), and is never flagged as an estimate.
    for frame in (ps, wh.read("team_game_stats")):
        assert set(frame["available_at"].to_list()) == {received}
        assert set(frame[FIRST_SEEN_AT].to_list()) == {received}
        assert frame["available_at_is_estimated"].to_list() == [False] * frame.height
        assert frame["provider_observed_at"].null_count() == frame.height
    # A checkpoint before the backfill genuinely did not know these outcomes.
    assert as_known_at(ps, "player_game_stats", received - timedelta(seconds=1)).is_empty()
    # Settlement/training see the final truth regardless of receipt time.
    assert latest_final(ps, "player_game_stats").height == 4


def test_weeks_1_4_rerun_appends_only_provider_corrections(tmp_path: Path) -> None:
    first = datetime(2026, 10, 2, 15, 0, tzinfo=UTC)
    later = first + timedelta(days=2)
    wh = Warehouse(tmp_path / "canonical")
    lock = tmp_path / "writer.lock"
    backfill_outcome_history(_WeeksProvider(first), wh, seasons=[2026], weeks=[1, 2, 3, 4],
                             lock_path=lock, now=first)
    before = wh.read("player_game_stats").sort(VERSION_ID)
    results = backfill_outcome_history(
        _WeeksProvider(later, yards={"w2": 99}), wh, seasons=[2026], weeks=[1, 2, 3, 4],
        lock_path=lock, now=later,
    )
    assert results[0].player.corrections == 1 and results[0].player.unchanged == 3
    assert results[0].team.corrections == 0
    after = wh.read("player_game_stats")
    assert after.filter(pl.col(VERSION_ID).is_in(before[VERSION_ID].implode())).select(
        before.columns
    ).sort(VERSION_ID).equals(before)
    w2 = after.filter(pl.col("canonical_game_id") == "w2").sort(FIRST_SEEN_AT)
    assert w2["receiving_yards"].to_list() == [42, 99]
    assert w2["available_at"].to_list() == [first, later]
    assert as_known_at(after, "player_game_stats", later - timedelta(seconds=1)).filter(
        pl.col("canonical_game_id") == "w2")["receiving_yards"].to_list() == [42]
    assert latest_final(after, "player_game_stats").filter(
        pl.col("canonical_game_id") == "w2")["receiving_yards"].to_list() == [99]


def test_backfill_refuses_records_without_genuine_receipt_time(tmp_path: Path) -> None:
    class _NoReceipt(_WeeksProvider):
        def _pit(self) -> dict:
            return {"provider": "bdl"}

    wh = Warehouse(tmp_path / "canonical")
    with pytest.raises(OutcomeVersionError, match="genuine"):
        backfill_outcome_history(_NoReceipt(T0), wh, seasons=[2026], weeks=[1],
                                 lock_path=tmp_path / "l", now=T0)
    assert not wh.exists("player_game_stats")


def test_backfill_refuses_estimated_provider_availability(tmp_path: Path) -> None:
    class _Estimated(_WeeksProvider):
        def _pit(self) -> dict:
            return {**super()._pit(), "available_at_is_estimated": True}

    wh = Warehouse(tmp_path / "canonical")
    with pytest.raises(OutcomeVersionError, match="estimated"):
        backfill_outcome_history(_Estimated(T0), wh, seasons=[2026], weeks=[1],
                                 lock_path=tmp_path / "l", now=T0)
    assert not wh.exists("player_game_stats")


def test_backfill_never_writes_games_or_checkpoint_tables(tmp_path: Path) -> None:
    received = datetime(2026, 10, 2, 15, 0, tzinfo=UTC)
    wh = Warehouse(tmp_path / "canonical")
    backfill_outcome_history(_WeeksProvider(received), wh, seasons=[2026], weeks=[1, 2, 3, 4],
                             lock_path=tmp_path / "l", now=received)
    assert set(wh.tables()) <= {"player_game_stats", "team_game_stats"}
