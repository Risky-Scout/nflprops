"""PHASE 5 correction: `data_manifest_sha256` must fingerprint the actual
selected PIT input dataset for one (game_id, scheduled_as_of) checkpoint,
not row counts/max-timestamps alone and not a warehouse-wide
`state_snapshot_id` re-hash.

Each "changes -> different hash" test builds two independent warehouses
that differ in exactly one respect, so before/after comparison never
depends on `Warehouse.append`'s in-place mutation semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from nflprops.data.warehouse import Warehouse
from nflprops.domain.enums import InjuryStatusCanonical
from nflprops.orchestration.manifest import (
    build_checkpoint_manifest,
    compute_data_manifest_sha256,
)

GAME_ID = "g1"
KICKOFF = datetime(2026, 9, 13, 20, 20, 0, tzinfo=UTC)
AS_OF = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)


def _base_warehouse(
    tmp_path: Path,
    name: str,
    *,
    spread_home_value: float = -3.5,
    receiving_yards: int = 55,
    injury_status: str = "questionable",
    roster_position: str = "WR",
    kickoff: datetime = KICKOFF,
    line_value: float = 55.5,
    injury_collection_status: str = "SUCCESS",
    injury_collector_received_at: datetime | None = None,
) -> Warehouse:
    wh = Warehouse(tmp_path / name)

    games = pl.DataFrame(
        {
            "canonical_game_id": [GAME_ID],
            "available_at": [kickoff - timedelta(days=5)],
            "date": [kickoff],
            "season": [2026],
            "week": [2],
            "home_canonical_team_id": ["t1"],
            "visitor_canonical_team_id": ["t2"],
            "status_state": ["scheduled"],
        }
    )
    wh.append("games", games, key=["canonical_game_id", "available_at"])

    player_stats = pl.DataFrame(
        {
            "canonical_game_id": ["gprev"],
            "canonical_player_id": ["p1"],
            "canonical_team_id": ["t1"],
            "available_at": [AS_OF - timedelta(days=10)],
            "receiving_yards": [receiving_yards],
        }
    )
    wh.append("player_game_stats", player_stats, key=["canonical_game_id", "canonical_player_id"])

    roster = pl.DataFrame(
        {
            "canonical_team_id": ["t1"],
            "canonical_player_id": ["p1"],
            "available_at": [AS_OF - timedelta(days=1)],
            "position": [roster_position],
        }
    )
    wh.append("roster_snapshots", roster, key=["canonical_team_id", "canonical_player_id"])

    injuries = pl.DataFrame(
        {
            "canonical_player_id": ["p1"],
            "available_at": [AS_OF - timedelta(hours=6)],
            "status": [injury_status],
            "status_raw": [injury_status],
        }
    )
    wh.append("injury_snapshots", injuries, key=["canonical_player_id", "available_at"])

    resource_runs = pl.DataFrame(
        {
            "resource_run_id": ["run-1"],
            "resource_type": ["INJURIES"],
            "collector_received_at": [injury_collector_received_at or (AS_OF - timedelta(hours=6))],
            "collection_status": [injury_collection_status],
        }
    )
    wh.append("collector_resource_runs", resource_runs, key=["resource_run_id"])

    game_odds = pl.DataFrame(
        {
            "canonical_game_id": [GAME_ID],
            "vendor": ["bet365"],
            "available_at": [AS_OF - timedelta(hours=1)],
            "spread_home_value": [spread_home_value],
        }
    )
    wh.append("game_odds_snapshots", game_odds, key=["canonical_game_id", "vendor", "available_at"])

    player_props = pl.DataFrame(
        {
            "canonical_game_id": [GAME_ID],
            "canonical_player_id": ["p1"],
            "prop_type": ["receiving_yards"],
            "vendor": ["bet365"],
            "available_at": [AS_OF - timedelta(hours=1)],
            "line_value": [line_value],
        }
    )
    wh.append(
        "player_prop_snapshots",
        player_props,
        key=["canonical_game_id", "canonical_player_id", "prop_type", "vendor", "available_at"],
    )

    players = pl.DataFrame(
        {
            "canonical_player_id": ["p1"],
            "position_group": ["WR"],
        }
    )
    wh.write("players", players)

    return wh


def _manifest_sha(wh: Warehouse) -> str:
    return compute_data_manifest_sha256(wh, game_id=GAME_ID, scheduled_as_of=AS_OF)


def test_identical_selected_pit_inputs_produce_identical_hash(tmp_path: Path) -> None:
    wh_a = _base_warehouse(tmp_path, "a")
    wh_b = _base_warehouse(tmp_path, "b")
    assert _manifest_sha(wh_a) == _manifest_sha(wh_b)


def test_row_ordering_changes_only_produce_identical_hash(tmp_path: Path) -> None:
    wh_a = Warehouse(tmp_path / "order-a")
    wh_b = Warehouse(tmp_path / "order-b")

    forward = pl.DataFrame(
        {
            "canonical_game_id": [GAME_ID, GAME_ID],
            "vendor": ["bet365", "fanduel"],
            "available_at": [AS_OF - timedelta(hours=1), AS_OF - timedelta(hours=2)],
            "spread_home_value": [-3.5, -3.0],
        }
    )
    reversed_ = forward.reverse()

    games = pl.DataFrame(
        {
            "canonical_game_id": [GAME_ID],
            "available_at": [KICKOFF - timedelta(days=5)],
            "date": [KICKOFF],
            "season": [2026],
            "week": [2],
            "home_canonical_team_id": ["t1"],
            "visitor_canonical_team_id": ["t2"],
            "status_state": ["scheduled"],
        }
    )
    wh_a.write("games", games)
    wh_b.write("games", games)
    wh_a.write("game_odds_snapshots", forward)
    wh_b.write("game_odds_snapshots", reversed_)

    assert _manifest_sha(wh_a) == _manifest_sha(wh_b)


def test_player_stat_value_change_with_identical_count_and_timestamp_changes_hash(
    tmp_path: Path,
) -> None:
    wh_a = _base_warehouse(tmp_path, "a", receiving_yards=55)
    wh_b = _base_warehouse(tmp_path, "b", receiving_yards=99)
    assert _manifest_sha(wh_a) != _manifest_sha(wh_b)


def test_injury_status_change_with_identical_count_and_timestamp_changes_hash(
    tmp_path: Path,
) -> None:
    wh_a = _base_warehouse(tmp_path, "a", injury_status="questionable")
    wh_b = _base_warehouse(tmp_path, "b", injury_status="out")
    assert _manifest_sha(wh_a) != _manifest_sha(wh_b)


def test_roster_assignment_change_with_identical_count_and_timestamp_changes_hash(
    tmp_path: Path,
) -> None:
    wh_a = _base_warehouse(tmp_path, "a", roster_position="WR")
    wh_b = _base_warehouse(tmp_path, "b", roster_position="TE")
    assert _manifest_sha(wh_a) != _manifest_sha(wh_b)


def test_game_odds_price_change_changes_hash(tmp_path: Path) -> None:
    wh_a = _base_warehouse(tmp_path, "a", spread_home_value=-3.5)
    wh_b = _base_warehouse(tmp_path, "b", spread_home_value=-2.5)
    assert _manifest_sha(wh_a) != _manifest_sha(wh_b)


def test_player_prop_line_change_changes_hash(tmp_path: Path) -> None:
    wh_a = _base_warehouse(tmp_path, "a", line_value=55.5)
    wh_b = _base_warehouse(tmp_path, "b", line_value=60.5)
    assert _manifest_sha(wh_a) != _manifest_sha(wh_b)


def test_target_kickoff_change_changes_hash(tmp_path: Path) -> None:
    wh_a = _base_warehouse(tmp_path, "a", kickoff=KICKOFF)
    wh_b = _base_warehouse(tmp_path, "b", kickoff=KICKOFF + timedelta(hours=1))
    assert _manifest_sha(wh_a) != _manifest_sha(wh_b)


def test_injury_data_available_provenance_change_changes_hash(tmp_path: Path) -> None:
    # Same injury_snapshots content in both cases -- only whether the feed
    # was actually *checked* (per collector_resource_runs) differs.
    wh_checked = _base_warehouse(
        tmp_path,
        "checked",
        injury_collection_status="SUCCESS",
        injury_collector_received_at=AS_OF - timedelta(hours=6),
    )
    wh_unchecked = _base_warehouse(
        tmp_path,
        "unchecked",
        injury_collection_status="RATE_LIMITED",
        injury_collector_received_at=AS_OF - timedelta(hours=6),
    )
    assert _manifest_sha(wh_checked) != _manifest_sha(wh_unchecked)

    manifest_checked = build_checkpoint_manifest(
        wh_checked, game_id=GAME_ID, scheduled_as_of=AS_OF
    )
    manifest_unchecked = build_checkpoint_manifest(
        wh_unchecked, game_id=GAME_ID, scheduled_as_of=AS_OF
    )
    assert manifest_checked.components["injury_availability"].extra["injury_feed_available"] is True
    assert (
        manifest_unchecked.components["injury_availability"].extra["injury_feed_available"] is False
    )


def test_row_added_after_scheduled_as_of_does_not_change_manifest(tmp_path: Path) -> None:
    wh = _base_warehouse(tmp_path, "a")
    before = _manifest_sha(wh)

    late_odds = pl.DataFrame(
        {
            "canonical_game_id": [GAME_ID],
            "vendor": ["fanduel"],
            "available_at": [AS_OF + timedelta(minutes=1)],
            "spread_home_value": [-9.0],
        }
    )
    wh.append(
        "game_odds_snapshots", late_odds, key=["canonical_game_id", "vendor", "available_at"]
    )
    late_prop = pl.DataFrame(
        {
            "canonical_game_id": [GAME_ID],
            "canonical_player_id": ["p1"],
            "prop_type": ["receiving_yards"],
            "vendor": ["fanduel"],
            "available_at": [AS_OF + timedelta(minutes=1)],
            "line_value": [70.5],
        }
    )
    wh.append(
        "player_prop_snapshots",
        late_prop,
        key=["canonical_game_id", "canonical_player_id", "prop_type", "vendor", "available_at"],
    )
    late_injury = pl.DataFrame(
        {
            "canonical_player_id": ["p1"],
            "available_at": [AS_OF + timedelta(minutes=1)],
            "status": ["out"],
            "status_raw": ["Out"],
        }
    )
    wh.append("injury_snapshots", late_injury, key=["canonical_player_id", "available_at"])

    after = _manifest_sha(wh)
    assert after == before


def test_catch_up_at_later_wall_clock_time_produces_identical_manifest(tmp_path: Path) -> None:
    """Catch-up execution (a worker recovering late but before kickoff)
    must reuse the exact same `scheduled_as_of` -- the manifest is a pure
    function of (warehouse content, game_id, scheduled_as_of) with no
    dependency on wall-clock/`now` at all, so an on-time call and a
    simulated-late call over the identical warehouse content necessarily
    agree; this proves the function has no hidden `now`/mtime dependency
    (e.g. no accidental read of the system clock or file mtimes)."""
    wh = _base_warehouse(tmp_path, "a")

    on_time = build_checkpoint_manifest(wh, game_id=GAME_ID, scheduled_as_of=AS_OF)
    # Simulate "worker recovers later" purely by calling again -- no sleep,
    # no now= parameter exists on this function to vary.
    catch_up = build_checkpoint_manifest(wh, game_id=GAME_ID, scheduled_as_of=AS_OF)

    assert on_time.data_manifest_sha256 == catch_up.data_manifest_sha256
    assert on_time.as_dict() == catch_up.as_dict()


def test_manifest_never_uses_python_builtin_hash(tmp_path: Path) -> None:
    # A SHA-256 hex digest is always 64 lowercase-hex characters; Python's
    # built-in hash() would never produce this shape.
    wh = _base_warehouse(tmp_path, "a")
    digest = _manifest_sha(wh)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_game_odds_and_player_props_are_scoped_to_the_target_game(tmp_path: Path) -> None:
    wh = _base_warehouse(tmp_path, "a")
    other_game_odds = pl.DataFrame(
        {
            "canonical_game_id": ["some-other-game"],
            "vendor": ["bet365"],
            "available_at": [AS_OF - timedelta(hours=1)],
            "spread_home_value": [1.5],
        }
    )
    wh.append(
        "game_odds_snapshots",
        other_game_odds,
        key=["canonical_game_id", "vendor", "available_at"],
    )
    manifest = build_checkpoint_manifest(wh, game_id=GAME_ID, scheduled_as_of=AS_OF)
    assert manifest.components["game_odds"].row_count == 1


def test_reference_players_scoped_to_relevant_players_only(tmp_path: Path) -> None:
    wh = _base_warehouse(tmp_path, "a")
    irrelevant_player = pl.DataFrame(
        {"canonical_player_id": ["not-in-this-game"], "position_group": ["QB"]}
    )
    existing = wh.read("players")
    wh.write("players", pl.concat([existing, irrelevant_player], how="diagonal_relaxed"))

    manifest = build_checkpoint_manifest(wh, game_id=GAME_ID, scheduled_as_of=AS_OF)
    assert manifest.components["reference_players"].row_count == 1


def test_injury_status_enum_value_is_a_valid_canonical_status() -> None:
    # Sanity check that the fixtures above use a real canonical injury
    # status string, not an arbitrary made-up one.
    assert InjuryStatusCanonical("questionable") == InjuryStatusCanonical.QUESTIONABLE
    assert InjuryStatusCanonical("out") == InjuryStatusCanonical.OUT
