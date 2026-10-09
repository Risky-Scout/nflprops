"""Historical player positions from nflverse weekly rosters (POS1-POS15).

The 2026 BDL `players` dimension marks departed players Unknown (group
OTHER). Historical replay must instead read the week-versioned nflverse
roster position, at or before the target week only. Synthetic data only.
"""

from __future__ import annotations

import dataclasses
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gate1_model_profiles import (
    AWAY,
    HOME,
    HOME_RB,
    HOME_WR,
    PLAYERS,
    TARGET_SLATE,
    TRADED,
    historical_world,
    live_world,
    prepared,
    simulator_input,
)

from nflprops.data.evidence_policy import EvidenceClass, classify_model_evidence
from nflprops.domain.model_profile import ModelProfile, ModelProfileError
from nflprops.features import historical_positions as hp
from nflprops.features.historical_evidence import EvidenceMode
from nflprops.features.historical_positions import (
    HistoricalPositionError,
    MatchClass,
    build_historical_positions,
    build_player_crosswalk,
    nflverse_week,
    resolve_position_groups,
)
from nflprops.pipelines.model_inputs import build_model_inputs
from nflprops.simulation.game import TeamSimulationInput, _ensure_players

# ----------------------------------------------------------- synthetic world

T_A, T_B = "t:a", "t:b"
TEAMS = pl.DataFrame({"canonical_team_id": [T_A, T_B], "abbreviation": ["AAA", "BBB"]})
KICK = datetime(2022, 9, 11, 17, 0, tzinfo=UTC)


def _games(n_weeks: int = 8) -> pl.DataFrame:
    rows = [
        {"canonical_game_id": f"g{w}", "season": 2022, "week": w, "postseason": False,
         "status_state": "final", "date": KICK + timedelta(days=7 * (w - 1))}
        for w in range(1, n_weeks + 1)
    ]
    rows.append({"canonical_game_id": "gp1", "season": 2022, "week": 1, "postseason": True,
                 "status_state": "final", "date": KICK + timedelta(days=140)})
    return pl.DataFrame(rows)


def _player(pid: str, first: str, last: str, *, group: str = "OTHER",
            college: str | None = "State") -> dict:
    return {"canonical_player_id": pid, "provider_player_id": pid.split(":")[-1],
            "first_name": first, "last_name": last, "college": college,
            "position": "Unknown" if group == "OTHER" else group, "position_group": group}


def _app(pid: str, week: int, team: str = T_A, *, attempts: int = 0, game: str | None = None) -> dict:
    return {"canonical_game_id": game or f"g{week}", "canonical_player_id": pid,
            "canonical_team_id": team, "passing_attempts": attempts}


def _ros(gsis: str, week: int, position: str, *, team: str = "AAA", full: str = "",
         last: str = "", college: str | None = "State", football: str | None = None) -> dict:
    return {"season": 2022, "week": week, "team": team, "gsis_id": gsis, "full_name": full,
            "football_name": football or full.split(" ")[0], "last_name": last or full.split(" ")[-1],
            "college": college, "position": position}


def build(rosters: list[dict], stats: list[dict], players: list[dict]):
    r = pl.DataFrame(rosters)
    xw = build_player_crosswalk(
        r, player_stats=pl.DataFrame(stats), games=_games(), teams=TEAMS,
        players=pl.DataFrame(players),
    )
    return xw, build_historical_positions(r, xw)


def group_at(positions: pl.DataFrame, players: list[dict], pid: str, week: int,
             *, postseason: int = 0) -> str:
    out = resolve_position_groups(
        pl.DataFrame(players), positions, target_slate=(2022, postseason, week)
    )
    return out.filter(pl.col("canonical_player_id") == pid)["position_group"].item()


QB = "bdl:p:1"
QB_PLAYERS = [_player(QB, "Derek", "Carr")]


# ----------------------------------------------------------- POS1 / POS6 / POS13


def test_pos1_historical_qb_overrides_current_unknown() -> None:
    xw, pos = build([_ros("00-1", 1, "QB", full="Derek Carr")], [_app(QB, 1)], QB_PLAYERS)
    assert xw.row(0, named=True)["match_class"] == MatchClass.MATCHED_EXACT
    assert group_at(pos, QB_PLAYERS, QB, 1) == "QB"


def test_pos6_already_correct_positions_unchanged() -> None:
    players = [_player(QB, "Derek", "Carr", group="QB"), _player("bdl:p:2", "Al", "Wide", group="WR")]
    _, pos = build(
        [_ros("00-1", 1, "QB", full="Derek Carr"), _ros("00-2", 1, "WR", full="Al Wide")],
        [_app(QB, 1), _app("bdl:p:2", 1)], players,
    )
    assert group_at(pos, players, QB, 1) == "QB"
    assert group_at(pos, players, "bdl:p:2", 1) == "WR"
    # A player with no historical observation keeps his dimension group.
    assert group_at(pos.head(0), players, "bdl:p:2", 1) == "WR"


def test_pos13_retired_player_resolves_and_carries_forward() -> None:
    # Unknown in the 2026 dimension; on a 2022 roster only in week 1.
    _, pos = build([_ros("00-1", 1, "QB", full="Derek Carr")], [_app(QB, 1), _app(QB, 3)], QB_PLAYERS)
    assert group_at(pos, QB_PLAYERS, QB, 3) == "QB"  # prior observation carried forward
    assert group_at(pos, QB_PLAYERS, QB, 1, postseason=1) == "QB"


# ----------------------------------------------------------- POS2 / POS3 / POS14


def test_pos2_traded_player_keeps_week_team_and_position() -> None:
    players = [_player("bdl:p:9", "Tim", "Trade")]
    _, pos = build(
        [_ros("00-9", 1, "WR", full="Tim Trade"), _ros("00-9", 3, "RB", team="BBB", full="Tim Trade")],
        [_app("bdl:p:9", 1), _app("bdl:p:9", 3, T_B)], players,
    )
    assert pos.select("week", "team", "position_group").rows() == [(1, "AAA", "WR"), (3, "BBB", "RB")]
    assert group_at(pos, players, "bdl:p:9", 2) == "WR"
    assert group_at(pos, players, "bdl:p:9", 3) == "RB"


def test_pos3_pos18_future_position_never_reaches_an_earlier_target() -> None:
    _, pos = build(
        [_ros("00-1", 1, "TE", full="Derek Carr"), _ros("00-1", 3, "QB", full="Derek Carr")],
        [_app(QB, 1), _app(QB, 3)], QB_PLAYERS,
    )
    assert group_at(pos, QB_PLAYERS, QB, 2) == "TE"
    _, only_future = build([_ros("00-1", 3, "QB", full="Derek Carr")], [_app(QB, 3)], QB_PLAYERS)
    assert group_at(only_future, QB_PLAYERS, QB, 2) == "OTHER"  # unresolved, not QB


def test_pos14_utility_player_follows_weekly_roster_not_blanket_qb() -> None:
    players = [_player("bdl:p:7", "Taysom", "Hill", group="QB")]  # 2026 dimension: QB
    rosters = [_ros("00-7", w, "QB" if w <= 5 else "TE", full="Taysom Hill") for w in range(1, 8)]
    _, pos = build(rosters, [_app("bdl:p:7", w) for w in range(1, 8)], players)
    assert [group_at(pos, players, "bdl:p:7", w) for w in (5, 6, 7)] == ["QB", "TE", "TE"]


# ----------------------------------------------------------- POS4 / POS5 / POS9


def test_pos4_pos10_target_outcomes_and_passing_never_decide_position() -> None:
    players = [_player("bdl:p:5", "Wes", "Catch", group="WR")]
    rosters = [_ros("00-5", 1, "WR", full="Wes Catch"), _ros("00-5", 2, "WR", full="Wes Catch")]
    quiet = build(rosters, [_app("bdl:p:5", 1), _app("bdl:p:5", 2)], players)
    throwing = build(rosters, [_app("bdl:p:5", 1, attempts=40), _app("bdl:p:5", 2, attempts=35)], players)
    assert quiet[0].equals(throwing[0]) and quiet[1].equals(throwing[1])
    assert group_at(throwing[1], players, "bdl:p:5", 2) == "WR"  # a passer is not promoted


def test_pos5_storage_order_never_changes_resolution() -> None:
    players = [_player(QB, "Derek", "Carr"), _player("bdl:p:2", "Al", "Wide"),
               _player("bdl:p:9", "Tim", "Trade")]
    rosters = [
        _ros("00-1", 1, "QB", full="Derek Carr"), _ros("00-1", 2, "QB", full="Derek Carr"),
        _ros("00-2", 1, "WR", full="Al Wide"), _ros("00-9", 1, "WR", full="Tim Trade"),
        _ros("00-9", 2, "RB", team="BBB", full="Tim Trade"),
    ]
    stats = [_app(QB, 1), _app(QB, 2), _app("bdl:p:2", 1), _app("bdl:p:9", 1), _app("bdl:p:9", 2, T_B)]
    base = build(rosters, stats, players)
    for seed in range(5):
        rng = random.Random(seed)
        shuffled = [rosters[:], stats[:], players[:]]
        for part in shuffled:
            rng.shuffle(part)
        other = build(*shuffled)
        assert base[0].equals(other[0]) and base[1].equals(other[1])
        resolved = resolve_position_groups(pl.DataFrame(shuffled[2]), other[1], target_slate=(2022, 0, 2))
        assert dict(resolved.select("canonical_player_id", "position_group").iter_rows()) == {
            QB: "QB", "bdl:p:2": "WR", "bdl:p:9": "RB"}


# ----------------------------------------------------------- crosswalk: POS10-POS12


def test_pos10_ambiguous_crosswalk_fails_closed() -> None:
    players = [_player("bdl:p:3", "Mike", "Edwards", college=None)]
    xw, pos = build(
        [_ros("00-31", 1, "DB", full="Mike Edwards", college="Kentucky"),
         _ros("00-32", 1, "QB", full="Mike Edwards", college="Campbell")],
        [_app("bdl:p:3", 1)], players,
    )
    row = xw.row(0, named=True)
    assert row["match_class"] == MatchClass.AMBIGUOUS and row["nflverse_gsis_id"] is None
    assert pos.is_empty()
    assert group_at(pos, players, "bdl:p:3", 1) == "OTHER"


def test_pos11_exact_identity_beats_corroboration() -> None:
    # An exact-name candidate exists; a different same-last-name, same-college
    # player on the team is never preferred over it.
    players = [_player("bdl:p:4", "Josh", "Palmer")]
    xw, _ = build(
        [_ros("00-41", 1, "WR", full="Josh Palmer"), _ros("00-42", 1, "TE", full="Joshua Palmer")],
        [_app("bdl:p:4", 1)], players,
    )
    row = xw.row(0, named=True)
    assert (row["match_class"], row["match_rule"], row["nflverse_gsis_id"]) == (
        MatchClass.MATCHED_EXACT, "TEAM_WEEK+EXACT_NAME", "00-41")
    # Without an exact-name candidate, a unique last name + college corroborates.
    players = [_player("bdl:p:6", "Kenny", "Gainwell")]
    xw, _ = build([_ros("00-6", 1, "RB", full="Kenneth Gainwell")], [_app("bdl:p:6", 1)], players)
    row = xw.row(0, named=True)
    assert (row["match_class"], row["match_rule"]) == (
        MatchClass.MATCHED_CORROBORATED, "TEAM_WEEK+LAST_NAME+COLLEGE")
    # ...but a college mismatch is never a match.
    xw, _ = build([_ros("00-6", 1, "RB", full="Kenneth Gainwell", college="Other")],
                  [_app("bdl:p:6", 1)], players)
    assert xw["match_class"].item() == MatchClass.UNMATCHED


def test_pos12_same_names_never_collapse() -> None:
    players = [_player("bdl:p:10", "Michael", "Carter", college="North Carolina"),
               _player("bdl:p:11", "Michael", "Carter II", college="Duke")]
    xw, pos = build(
        [_ros("00-10", 1, "RB", full="Michael Carter", college="North Carolina"),
         _ros("00-11", 1, "DB", full="Michael Carter", college="Duke")],
        [_app("bdl:p:10", 1), _app("bdl:p:11", 1)], players,
    )
    assert dict(xw.select("canonical_player_id", "nflverse_gsis_id").iter_rows()) == {
        "bdl:p:10": "00-10", "bdl:p:11": "00-11"}
    assert group_at(pos, players, "bdl:p:10", 1) == "RB"
    assert group_at(pos, players, "bdl:p:11", 1) == "OTHER"  # DB -> OTHER, not RB
    # Two BDL players claiming one gsis id both fail closed.
    twins = [_player("bdl:p:20", "Sam", "Same"), _player("bdl:p:21", "Sam", "Same")]
    xw, pos = build([_ros("00-20", 1, "WR", full="Sam Same")],
                    [_app("bdl:p:20", 1), _app("bdl:p:21", 1)], twins)
    assert set(xw["match_class"]) == {MatchClass.AMBIGUOUS} and pos.is_empty()


def test_conflicting_week_is_not_evidence() -> None:
    _, pos = build(
        [_ros("00-1", 1, "WR", full="Derek Carr"),
         _ros("00-1", 2, "QB", full="Derek Carr"), _ros("00-1", 2, "TE", team="BBB", full="Derek Carr")],
        [_app(QB, 1), _app(QB, 2)], QB_PLAYERS,
    )
    assert set(pos.filter(pl.col("week") == 2)["conflict_status"]) == {"CONFLICT"}
    assert group_at(pos, QB_PLAYERS, QB, 2) == "WR"  # falls back to the prior week


def test_postseason_week_mapping_fails_closed() -> None:
    assert [nflverse_week(postseason=True, week=w) for w in (1, 2, 3, 5)] == [19, 20, 21, 22]
    assert nflverse_week(postseason=False, week=18) == 18
    with pytest.raises(HistoricalPositionError):
        nflverse_week(postseason=True, week=4)


def test_pos15_source_files_are_pinned(tmp_path: Path) -> None:
    pl.DataFrame([_ros("00-1", 1, "QB", full="Derek Carr")]).write_parquet(
        tmp_path / "roster_weekly_2022.parquet")
    with pytest.raises(HistoricalPositionError, match="sha256"):
        hp.load_nflverse_weekly_rosters(tmp_path, (2022,))
    with pytest.raises(HistoricalPositionError, match="no pinned"):
        hp.load_nflverse_weekly_rosters(tmp_path, (2019,))


# ----------------------------------------------------------- model input: POS7/8/13-15

#: The Gate 1 world (season 2024, target slate week 3). HOME_RB is Unknown
#: in the 2026 dimension but was rostered as his team's QB in week 1.
UNKNOWN_DIM = PLAYERS.with_columns(
    pl.when(pl.col("canonical_player_id") == HOME_RB).then(pl.lit("OTHER"))
    .otherwise(pl.col("position_group")).alias("position_group")
)


def _positions(rows: list[tuple[str, int, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"canonical_player_id": p, "season": 2024, "week": w, "team": "X",
          "position_group": g, "conflict_status": "NONE"} for p, w, g in rows],
        schema_overrides={"season": pl.Int64, "week": pl.Int64},
    )


HOME_RB_QB = _positions([(HOME_RB, 1, "QB")])


def _inputs(world, mode, *, players=UNKNOWN_DIM, positions=HOME_RB_QB,
            profile=ModelProfile.STRUCTURAL_CORE):
    return build_model_inputs(
        model_profile=profile, evidence_mode=mode, games=world["games"],
        player_stats=world["player_stats"], team_stats=world["team_stats"], players=players,
        roster=world["roster"], injuries=world["injuries"], game_odds=world["game_odds"],
        as_of=datetime(2024, 9, 22, 16, 30, tzinfo=UTC), target_slate=TARGET_SLATE,
        target_game_id="g1:game:target" if mode is EvidenceMode.HISTORICAL_WALK_FORWARD else None,
        historical_positions=positions,
    )


def _sim_team(model_inputs, team: str):
    """The players the simulator actually allocates for `team`."""
    states = model_inputs.player_states
    return _ensure_players(TeamSimulationInput(
        team_id=team, state=model_inputs.team_states[team],
        opponent_state=model_inputs.team_states[AWAY if team == HOME else HOME],
        players=tuple(p for p in states.values() if p.team_id == team),
    ))


def test_pos7_real_historical_qb_removes_metadata_placeholder() -> None:
    hwf = EvidenceMode.HISTORICAL_WALK_FORWARD
    before = _inputs(historical_world(), hwf, positions=None)
    after = _inputs(historical_world(), hwf)
    assert before.player_states[HOME_RB].position_group == "OTHER"
    assert f"__QB__:{HOME}" in {p.player_id for p in _sim_team(before, HOME)}
    assert after.player_states[HOME_RB].position_group == "QB"
    sim = _sim_team(after, HOME)
    assert f"__QB__:{HOME}" not in {p.player_id for p in sim}
    assert [p.player_id for p in sim if p.position_group == "QB"] == [HOME_RB]


def test_pos8_genuinely_unresolved_qb_keeps_the_fallback() -> None:
    after = _inputs(historical_world(), EvidenceMode.HISTORICAL_WALK_FORWARD)
    assert f"__QB__:{AWAY}" in {p.player_id for p in _sim_team(after, AWAY)}


def test_pos13_pos3_future_roster_week_never_changes_earlier_input(monkeypatch) -> None:
    hwf = EvidenceMode.HISTORICAL_WALK_FORWARD
    base = _inputs(historical_world(), hwf)
    future = _inputs(historical_world(), hwf, positions=pl.concat([
        HOME_RB_QB, _positions([(HOME_RB, 4, "TE"), (HOME_WR, 4, "QB"), (TRADED, 9, "QB")])]))
    assert base.player_states == future.player_states
    a, _ = simulator_input(base, monkeypatch)
    b, _ = simulator_input(future, monkeypatch)
    assert dataclasses.asdict(a) == dataclasses.asdict(b)


def test_pos15_historical_and_live_identical_with_equivalent_positions(monkeypatch) -> None:
    historical = _inputs(historical_world(), EvidenceMode.HISTORICAL_WALK_FORWARD)
    live = _inputs(live_world(), EvidenceMode.LIVE_PIT)
    assert historical.player_states == live.player_states
    assert historical.player_states[HOME_RB].position_group == "QB"
    h_in, h_cfg = simulator_input(historical, monkeypatch)
    l_in, l_cfg = simulator_input(live, monkeypatch)
    assert dataclasses.asdict(h_in) == dataclasses.asdict(l_in) and h_cfg == l_cfg
    assert prepared(historical).simulation_input_sha256 == prepared(live).simulation_input_sha256


def test_live_enhanced_never_takes_historical_positions() -> None:
    with pytest.raises(ModelProfileError):
        _inputs(live_world(), EvidenceMode.LIVE_PIT, profile=ModelProfile.LIVE_ENHANCED)


def test_positions_table_is_certified_identity_evidence() -> None:
    world = historical_world()
    tables = {"games": world["games"], "player_game_stats": world["player_stats"],
              "team_game_stats": world["team_stats"], "players": PLAYERS,
              "historical_player_positions": HOME_RB_QB}
    evidence, _ = classify_model_evidence(
        tables, evidence_mode=EvidenceMode.HISTORICAL_WALK_FORWARD,
        model_profile=ModelProfile.STRUCTURAL_CORE)
    assert evidence is EvidenceClass.CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY
