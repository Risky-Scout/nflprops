"""v3 roster-eligible player universe (P1-P15).

Every real player -- not only QBs -- receives target-game opportunity only
as a structural member of the target team at the target cutoff. Before
this, a departed/retired non-QB kept the team of his last box-score line
and kept absorbing that team's targets/carries (about 16% of 2026 Week 5
projected receiving/rushing volume). Membership decides only WHICH team's
opportunity a player may receive; his admissible prior games still form
his history. Synthetic data and small-draw simulations only.
"""

from __future__ import annotations

import dataclasses
import random
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gate1_model_profiles import (
    AS_OF,
    AWAY,
    G_TARGET,
    G_W4,
    HOME,
    HOME_RB,
    HOME_WR,
    OTH_A,
    TARGET_MEMBERS,
    TARGET_SLATE,
    TRADED,
    W4,
    _box,
    historical_membership,
    historical_world,
    inputs,
    live_world,
    prepared,
)
from test_qb_eligibility import (
    BACKUP,
    MEMBERS,
    STALE,
    STARTER,
    T_A,
    WR,
    selected,
    states,
)

from nflprops.features.historical_evidence import EvidenceMode
from nflprops.features.team_membership import (
    build_historical_team_membership,
    historical_team_membership_at,
)
from nflprops.pipelines.model_inputs import ModelInputs
from nflprops.simulation.game import TeamSimulationInput, _ensure_players

HWF, LIVE = EvidenceMode.HISTORICAL_WALK_FORWARD, EvidenceMode.LIVE_PIT
AWAY_WR = "g1:p:away-wr"
OPPORTUNITY = ("targets", "receptions", "receiving_yards", "rush_attempts", "rushing_yards")


def _live_roster(members: dict[str, str], *, at=None) -> pl.DataFrame:
    """Each team's latest live roster batch: identity only, no depth."""
    return pl.DataFrame(
        [{"canonical_player_id": pid, "canonical_team_id": team, "depth": None,
          "available_at": at or AS_OF - timedelta(hours=3), "available_at_is_estimated": False}
         for pid, team in sorted(members.items())],
        schema_overrides={"depth": pl.Int64},
    )


def model(mode: EvidenceMode, members: dict[str, str], *, world=None) -> ModelInputs:
    """STRUCTURAL_CORE inputs with the SAME membership evidence in either
    mode: historical weekly-roster rows, or the live roster batch."""
    if mode is HWF:
        table = historical_membership(members) if members else (
            historical_membership({"x": "y"}).head(0))
        return inputs(world or historical_world(), mode=HWF, membership=table)
    roster = _live_roster(members) if members else pl.DataFrame()
    return inputs(dict(world or live_world(), roster=roster))


def without(*players: str) -> dict[str, str]:
    return {p: t for p, t in TARGET_MEMBERS.items() if p not in players}


def player_opportunity(m: ModelInputs, player: str) -> dict[str, int]:
    draws = prepared(m).result.player_draws.filter(pl.col("player_id") == player)
    return {stat: int(draws[stat].sum()) for stat in OPPORTUNITY}


def real_ids(m: ModelInputs) -> set[str]:
    return set(prepared(m).result.real_player_draws()["player_id"].unique().to_list())


# ------------------------------------------------------- P1-P5: off-roster players


@pytest.mark.parametrize("mode", [HWF, LIVE])
def test_p1_departed_wr_receives_no_receiving_opportunity(mode) -> None:
    full, gone = model(mode, TARGET_MEMBERS), model(mode, without(HOME_WR))
    assert player_opportunity(full, HOME_WR)["targets"] > 0  # he would absorb targets
    assert HOME_WR not in gone.player_states
    assert HOME_WR not in real_ids(gone)


@pytest.mark.parametrize("mode", [HWF, LIVE])
def test_p2_departed_rb_receives_no_rushing_opportunity(mode) -> None:
    full, gone = model(mode, TARGET_MEMBERS), model(mode, without(HOME_RB))
    assert player_opportunity(full, HOME_RB)["rush_attempts"] > 0
    assert HOME_RB not in gone.player_states
    assert HOME_RB not in real_ids(gone)


def test_p3_retired_player_is_not_a_candidate() -> None:
    """nflverse lists retired players (status RET): a roster row is not
    membership; only a membership status is."""
    xw = pl.DataFrame({"canonical_player_id": [HOME_WR, HOME_RB],
                       "nflverse_gsis_id": ["G-WR", "G-RB"],
                       "match_class": ["MATCHED_EXACT"] * 2})
    teams = pl.DataFrame({"abbreviation": ["HOM"], "canonical_team_id": [HOME]})
    table = build_historical_team_membership(pl.DataFrame(
        [{"season": 2024, "week": 3, "team": "HOM", "gsis_id": "G-WR", "status": "RET"},
         {"season": 2024, "week": 3, "team": "HOM", "gsis_id": "G-RB", "status": "ACT"}],
        schema_overrides={"season": pl.Int64, "week": pl.Int64}), xw, teams)
    assert historical_team_membership_at(table, target_slate=TARGET_SLATE) == {HOME_RB: HOME}
    retired = inputs(historical_world(), mode=HWF, membership=table)
    assert HOME_WR not in retired.player_states
    assert HOME_WR not in real_ids(retired)


@pytest.mark.parametrize("mode", [HWF, LIVE])
def test_p4_player_on_another_team_is_not_a_candidate(mode) -> None:
    """AWAY_WR's box score says AWAY; at the target he is OTH_A's."""
    moved = model(mode, {**TARGET_MEMBERS, AWAY_WR: OTH_A})
    assert moved.player_states[AWAY_WR].team_id == OTH_A
    assert AWAY_WR not in real_ids(moved)


def test_p5_future_week_membership_never_reaches_back() -> None:
    later = historical_membership(TARGET_MEMBERS).with_columns(pl.lit(4, dtype=pl.Int64).alias("week"))
    target = historical_membership(without(TRADED))
    joined_later = pl.concat([target, later.filter(pl.col("canonical_player_id") == TRADED)
                              .with_columns(pl.lit(HOME).alias("canonical_team_id"))])
    m = inputs(historical_world(), mode=HWF, membership=joined_later)
    assert TRADED not in m.player_states
    assert m.player_states == inputs(historical_world(), mode=HWF, membership=target).player_states
    # Live: a roster batch received after the cutoff is never read.
    world = live_world()
    roster = pl.concat([_live_roster(without(TRADED)),
                        _live_roster({TRADED: HOME}, at=AS_OF + timedelta(seconds=1))])
    assert TRADED not in inputs(dict(world, roster=roster)).player_states


# ------------------------------------------------------- P6-P7: trades


@pytest.mark.parametrize("mode", [HWF, LIVE])
def test_p6_eligible_player_keeps_prior_history_from_previous_teams(mode) -> None:
    """TRADED played W1 for HOME and W2 for AWAY. As an AWAY member his
    state still uses his HOME game: dropping it changes the state."""
    world = historical_world() if mode is HWF else live_world()
    gated = model(mode, TARGET_MEMBERS, world=world).player_states[TRADED]
    no_home_game = dict(world, player_stats=world["player_stats"].filter(
        ~((pl.col("canonical_player_id") == TRADED)
          & (pl.col("canonical_game_id") == "g1:game:w1"))))
    without_history = model(mode, TARGET_MEMBERS, world=no_home_game).player_states[TRADED]
    assert gated.team_id == without_history.team_id == AWAY
    assert gated.opportunities > without_history.opportunities > 0
    assert gated != without_history


@pytest.mark.parametrize("mode", [HWF, LIVE])
def test_p7_trade_assigns_opportunity_only_to_the_current_team(mode) -> None:
    to_home = model(mode, {**TARGET_MEMBERS, TRADED: HOME})
    stays = model(mode, TARGET_MEMBERS)
    assert to_home.player_states[TRADED].team_id == HOME
    assert stays.player_states[TRADED].team_id == AWAY
    # Same admissible history either way: only the team changes.
    assert dataclasses.replace(to_home.player_states[TRADED], team_id=AWAY) == (
        stays.player_states[TRADED])
    draws = prepared(to_home).result.player_draws.filter(pl.col("player_id") == TRADED)
    assert set(draws["team_id"].unique().to_list()) == {HOME}
    assert int(draws["targets"].sum()) > 0


# ------------------------------------------------------- P8-P10: determinism / leakage


def test_p8_storage_order_never_changes_the_candidate_universe() -> None:
    members = without(HOME_WR)
    base = model(HWF, members)
    base_live = model(LIVE, members)
    rng = random.Random(5)
    table = historical_membership(members)
    roster = _live_roster(members)
    for _ in range(4):
        world = historical_world()
        shuffled = {k: (v[rng.sample(range(v.height), v.height)]
                        if isinstance(v, pl.DataFrame) and v.height else v)
                    for k, v in world.items()}
        t = table[rng.sample(range(table.height), table.height)]
        assert inputs(shuffled, mode=HWF, membership=t).player_states == base.player_states
        r = roster[rng.sample(range(roster.height), roster.height)]
        assert inputs(dict(live_world(), roster=r)).player_states == base_live.player_states


@pytest.mark.parametrize("mode", [HWF, LIVE])
def test_p9_target_game_outcome_cannot_create_membership(mode) -> None:
    """A target-game box line for a non-member never makes him a candidate."""
    members = without(HOME_WR)
    world = historical_world() if mode is HWF else live_world()
    _teams, players = _box(G_TARGET, {HOME_WR: HOME},
                          available_at=AS_OF - timedelta(minutes=1), scale=9)
    leaked = dict(world, player_stats=pl.concat([world["player_stats"], pl.DataFrame(players)],
                                                how="diagonal_relaxed"))
    m = model(mode, members, world=leaked)
    assert HOME_WR not in m.player_states
    assert m.player_states == model(mode, members).player_states


@pytest.mark.parametrize("mode", [HWF, LIVE])
def test_p10_future_stats_cannot_create_membership(mode) -> None:
    """W4 (after the target) has HOME_WR on HOME and TRADED on OTH_A; with
    neither a member, neither is a candidate, and W4 changes nothing."""
    members = without(HOME_WR, TRADED)
    world = historical_world() if mode is HWF else live_world()
    _teams, players = _box(G_W4, {HOME_WR: HOME, TRADED: HOME},
                          available_at=W4 + timedelta(hours=5), scale=5)
    more = dict(world, player_stats=pl.concat([world["player_stats"], pl.DataFrame(players)],
                                              how="diagonal_relaxed"))
    m = model(mode, members, world=more)
    assert HOME_WR not in m.player_states and TRADED not in m.player_states
    assert m.player_states == model(mode, members).player_states


# ------------------------------------------------------- P11: historical == live


@pytest.mark.parametrize("members", [
    TARGET_MEMBERS, without(HOME_WR), without(HOME_RB, TRADED),
    {**TARGET_MEMBERS, TRADED: HOME}, {**TARGET_MEMBERS, AWAY_WR: OTH_A},
])
def test_p11_historical_and_live_share_the_candidate_universe(members) -> None:
    historical, live = model(HWF, members), model(LIVE, members)
    assert historical.team_membership == live.team_membership
    assert historical.player_states == live.player_states
    assert prepared(historical).simulation_input_sha256 == prepared(live).simulation_input_sha256


# ------------------------------------------------------- P12-P13: QB repair, fallback


def test_p12_qb_repair_is_intact_and_now_universal() -> None:
    s = states(MEMBERS)
    assert STALE not in s and selected(s, T_A) == STARTER
    assert WR not in s  # a non-member WR is now removed exactly like a non-member QB
    with_wr = states({**MEMBERS, WR: T_A})
    assert with_wr[WR].team_id == T_A
    assert selected(with_wr, T_A) == STARTER
    assert {k: v for k, v in with_wr.items() if k != WR} == s
    assert states({BACKUP: T_A, WR: T_A}).keys() == {BACKUP, WR}


@pytest.mark.parametrize("mode", [HWF, LIVE])
def test_p13_no_eligible_real_candidate_keeps_the_synthetic_fallback(mode) -> None:
    empty = model(mode, {})
    assert empty.player_states == {}
    for team in (HOME, AWAY):
        sim = _ensure_players(TeamSimulationInput(
            team_id=team, state=empty.team_states[team], opponent_state=empty.team_states[team],
            players=()))
        assert [p.player_id for p in sim] == [f"__OTHER__:{team}", f"__QB__:{team}"]
    result = prepared(empty).result
    assert result.real_player_draws().is_empty()
    assert set(result.player_draws["player_id"].unique().to_list()) == {
        f"__OTHER__:{HOME}", f"__QB__:{HOME}", f"__OTHER__:{AWAY}", f"__QB__:{AWAY}"}


# ------------------------------------------------------- P14-P15: conservation, zero


TEAM_VOLUME = ("plays", "dropbacks", "pass_attempts", "sacks", "rush_attempts",
               "directed_targets")


@pytest.mark.parametrize("members", [without(HOME_WR), without(HOME_WR, HOME_RB), {}])
def test_p14_team_opportunity_is_conserved_after_filtering(members) -> None:
    full, gated = prepared(model(HWF, TARGET_MEMBERS)).result, prepared(model(HWF, members)).result
    # Team play volume does not depend on who is eligible...
    assert full.team_draws.select("draw_id", "team_id", *TEAM_VOLUME).equals(
        gated.team_draws.select("draw_id", "team_id", *TEAM_VOLUME))
    # ...and is fully re-allocated among the eligible players (+ synthetic).
    for result in (full, gated):
        per_team = result.player_draws.group_by("draw_id", "team_id").agg(
            pl.col("targets").sum(), pl.col("rush_attempts").sum(),
            pl.col("passing_attempts").sum())
        joined = result.team_draws.join(per_team, on=["draw_id", "team_id"])
        assert joined.height == result.team_draws.height
        assert (joined["targets"] == joined["directed_targets"]).all()
        assert (joined["rush_attempts_right"] == joined["rush_attempts"]).all()
        assert (joined["passing_attempts"] == joined["pass_attempts"]).all()


@pytest.mark.parametrize("mode", [HWF, LIVE])
def test_p15_off_roster_real_player_opportunity_is_exactly_zero(mode) -> None:
    members = without(HOME_WR, HOME_RB)
    m = model(mode, members)
    draws = prepared(m).result.real_player_draws()
    simulated = set(draws["player_id"].unique().to_list())
    assert simulated <= set(members)
    off = draws.filter(~pl.col("player_id").is_in(list(members)))
    for stat in OPPORTUNITY:
        assert float(off[stat].sum()) == 0.0
    # Every simulated real player is a member of the team he plays for.
    pairs = draws.select("player_id", "team_id").unique()
    assert all(members[p] == t for p, t in pairs.iter_rows())


# ------------------------------------------------------- identity: v3, old artifacts refused


def test_v3_identity_and_round3_calibrator_fails_closed() -> None:
    from nflprops.calibration.artifact import compute_compatibility_digest
    from nflprops.calibration.joint_feature_contract import FEATURE_CONTRACT_VERSION
    from nflprops.domain.model_profile import ModelProfile, profile_base_model_version
    from nflprops.pipelines import production_forecast as pf

    new = profile_base_model_version("2026.1.0", "STRUCTURAL_CORE")
    old = "2026.1.0+structural-core.v2-qb-roster-membership"
    assert new == "2026.1.0+structural-core.v3-roster-eligible-player-universe"
    common = dict(model_profile="STRUCTURAL_CORE", simulation_config_version="sim-v1",
                  feature_contract_version="f", prop_contract_version="p",
                  calibration_contract_version="c", scope_type="JOINT_GAME",
                  checkpoint_scope="ALL_PREGAME_CHECKPOINTS")
    assert compute_compatibility_digest(base_model_version=old, **common) != (
        compute_compatibility_digest(base_model_version=new, **common))
    round3 = {"model_profile": "STRUCTURAL_CORE", "base_model_version": old,
              "feature_contract_version": FEATURE_CONTRACT_VERSION}
    assert pf.calibration_status(round3, profile=ModelProfile.STRUCTURAL_CORE,
                                 model_version="2026.1.0") == pf.CALIBRATION_PROFILE_MISMATCH
    assert pf.calibration_status({**round3, "base_model_version": new},
                                 profile=ModelProfile.STRUCTURAL_CORE,
                                 model_version="2026.1.0") == pf.CALIBRATION_APPLIED
