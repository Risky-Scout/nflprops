"""QB eligibility + starter selection (Gate 1 QB repair).

QB candidates are the target team's structural roster members, and
`qb_attempt_share` is team-relative: the candidate's prior attempts over
ALL of the team's prior games. Before this, a departed or retired QB kept
his last team and a personal share computed over his own games only
(retired Matt Ryan selected for IND 2025; a four-start Anthony Brown kept a
BAL share for years). Synthetic data only.
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
    TARGET_SLATE,
    historical_world,
    live_world,
    prepared,
    simulator_input,
)
from test_historical_positions import (
    G1_AS_OF,
    HOME_RB_MEMBER,
    HOME_RB_QB,
    HOME_RB_ROSTER,
    UNKNOWN_DIM,
    _sim_team,
)

from nflprops.calibration.artifact import compute_compatibility_digest
from nflprops.data.evidence_policy import (
    EvidenceClass,
    classify_model_evidence,
    profile_input_tables,
)
from nflprops.domain.model_profile import (
    PROFILE_SCIENCE_VERSIONS,
    ModelProfile,
    ModelProfileError,
    profile_base_model_version,
)
from nflprops.features.historical_evidence import EvidenceMode
from nflprops.features.team_membership import (
    MEMBER_ROSTER_STATUSES,
    build_historical_team_membership,
    eligible_qb_ids,
    historical_team_membership_at,
    live_team_membership_at,
)
from nflprops.pipelines.model_inputs import build_model_inputs
from nflprops.simulation.game import TeamSimulationInput, _ensure_players
from nflprops.simulation.selection import selected_starter_id
from nflprops.state.player import (
    PlayerStateConfig,
    _posterior_rate,
    build_player_states,
)

# ----------------------------------------------------------- synthetic league

T_A, T_B, T_C = "t:a", "t:b", "t:c"
START = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)
AS_OF = START + timedelta(days=7 * 8) - timedelta(hours=1)  # before game 9

STALE = "qb:stale"      # A's starter games 1-2 only, then gone (retired / departed)
STARTER = "qb:starter"  # A's starter games 3-8
BACKUP = "qb:backup"    # A's backup: mop-up attempts in game 8 only
MOVER = "qb:mover"      # B's starter, then a member of A with no A history
B_QB = "qb:b"           # B's backup who takes over when MOVER leaves
WR = "wr:a"

PLAYERS = pl.DataFrame({
    "canonical_player_id": [STALE, STARTER, BACKUP, MOVER, B_QB, WR],
    "position_group": ["QB", "QB", "QB", "QB", "QB", "WR"],
})


def _row(game: str, team: str, pid: str, when: datetime, attempts: int) -> dict:
    return {
        "canonical_game_id": game, "canonical_team_id": team, "canonical_player_id": pid,
        "available_at": when, "available_at_is_estimated": False,
        "passing_attempts": attempts, "passing_completions": attempts * 2 // 3,
        "passing_interceptions": 0, "receiving_targets": 0 if pid != WR else 8,
        "receptions": 0 if pid != WR else 5, "receiving_yards": 0 if pid != WR else 60,
        "receiving_touchdowns": 0, "rushing_attempts": 1, "rushing_yards": 3,
        "rushing_touchdowns": 0, "field_goal_attempts": 0, "field_goals_made": 0,
    }


def league(*, extra: list[dict] | None = None, a_attempts: dict | None = None):
    """A's games a1..a8 and B's games b1..b8, one per week, received 5 hours
    after kickoff. Each team-game's official passing totals reconcile."""
    rows = []
    for w in range(1, 9):
        when = START + timedelta(days=7 * (w - 1), hours=5)
        a = {STALE: 32 if w <= 2 else 0, STARTER: 0 if w <= 2 else 34, WR: 0}
        if w == 8:
            a[BACKUP] = 6
        for pid, att in (a_attempts or {}).get(w, {}).items():
            a[pid] = att
        rows += [_row(f"a{w}", T_A, pid, when, att) for pid, att in a.items()]
        rows += [_row(f"b{w}", T_B, MOVER, when, 30), _row(f"b{w}", T_B, B_QB, when, 0)]
    rows += extra or []
    ps = pl.DataFrame(rows)
    ts = ps.group_by("canonical_game_id", "canonical_team_id").agg(
        pl.col("available_at").max(), pl.col("passing_attempts").sum(),
        pl.col("passing_completions").sum(),
        pl.col("passing_interceptions").sum().alias("interceptions_thrown"),
        pl.col("rushing_attempts").sum(),
    ).sort("canonical_game_id")
    return ps, ts


def states(membership=None, *, ps=None, ts=None, roster=None, injuries=None, as_of=AS_OF):
    if ps is None:
        ps, ts = league()
    return build_player_states(
        ps, ts, PLAYERS, as_of=as_of, strict=False, roster=roster, injuries=injuries,
        qb_membership=membership,
    )


MEMBERS = {STARTER: T_A, BACKUP: T_A, B_QB: T_B, MOVER: T_A}


def team_qbs(player_states, team: str) -> list[str]:
    return sorted(p.player_id for p in player_states.values()
                  if p.team_id == team and p.position_group == "QB")


def selected(player_states, team: str) -> str | None:
    return selected_starter_id(
        sorted((p for p in player_states.values() if p.team_id == team),
               key=lambda p: p.player_id),
        "QB",
    )


# ----------------------------------------------------------- root cause


def test_root_cause_stale_personal_share_is_gone() -> None:
    """Old semantics: STALE's share was his attempts over the team attempts
    of HIS OWN games -- 100% -- so it persisted after he left. Team-relative,
    his share is over all of A's games, and STARTER's is the larger."""
    s = states()  # no membership rule: share semantics alone
    assert s[STALE].team_id == T_A
    assert s[STALE].qb_attempt_share < 0.5 < s[STARTER].qb_attempt_share
    assert selected(s, T_A) == STARTER


def test_team_relative_denominator_includes_zero_attempt_games() -> None:
    s = states(MEMBERS)
    cfg = PlayerStateConfig()
    w = [0.5 ** (((AS_OF - (START + timedelta(days=7 * (g - 1), hours=5))).total_seconds()
                  / 86400.0) / cfg.role_half_life_days) for g in range(1, 9)]
    team_total = sum(wi * (32 if g <= 2 else 34 + (6 if g == 8 else 0))
                     for g, wi in zip(range(1, 9), w, strict=True))
    backup_attempts = w[7] * 6  # his only attempts; the other 7 games are 0
    expected = _posterior_rate(backup_attempts, team_total, 0.97,
                               cfg.role_prior_opportunities)
    assert s[BACKUP].qb_attempt_share == pytest.approx(expected, rel=1e-12)
    # A game where he threw nothing still sits in his denominator.
    assert s[BACKUP].qb_attempt_share < 0.1


# ----------------------------------------------------------- eligibility


def test_retired_or_departed_qb_is_removed_not_penalised() -> None:
    s = states(MEMBERS)
    assert STALE not in s  # removed from the candidate set entirely
    assert team_qbs(s, T_A) == [BACKUP, MOVER, STARTER]
    assert selected(s, T_A) == STARTER


def test_other_team_qb_is_not_a_candidate_for_his_old_team() -> None:
    s = states({**MEMBERS, MOVER: T_B})
    assert s[MOVER].team_id == T_B
    assert MOVER not in team_qbs(s, T_A)


def test_traded_qb_joins_his_new_team_deterministically() -> None:
    ps, ts = league()
    base = states(MEMBERS, ps=ps, ts=ts)
    assert base[MOVER].team_id == T_A
    # No A history: his A share is the prior shrunk by A's evidence.
    assert base[MOVER].qb_attempt_share < base[STARTER].qb_attempt_share
    rng = random.Random(7)
    for _ in range(3):
        shuffled = ps.sample(fraction=1.0, shuffle=True, seed=rng.randrange(10**6))
        again = states(dict(reversed(list(MEMBERS.items()))), ps=shuffled, ts=ts)
        assert again == base
    # B's remaining member takes B over.
    assert selected(base, T_B) == B_QB


def test_legitimate_backup_becomes_selectable_when_starter_leaves() -> None:
    s = states({BACKUP: T_A})
    assert selected(s, T_A) == BACKUP


def test_no_eligible_qb_keeps_the_generic_fallback() -> None:
    s = states({STARTER: T_A})
    team = tuple(p for p in s.values() if p.team_id == T_B)
    assert not [p for p in team if p.position_group == "QB"]
    sim = _ensure_players(TeamSimulationInput(team_id=T_B, state=None, opponent_state=None,
                                              players=team))  # type: ignore[arg-type]
    assert selected_starter_id(sim, "QB") == f"__QB__:{T_B}"


def test_future_game_and_target_outcome_stats_never_change_selection() -> None:
    late = AS_OF + timedelta(hours=8)
    for extra in (
        [_row("a9", T_A, BACKUP, late, 60), _row("a9", T_A, STARTER, late, 0)],  # target game
        [_row("a10", T_A, STALE, late + timedelta(days=7), 70)],  # a later game
    ):
        ps, ts = league(extra=extra)
        assert states(MEMBERS, ps=ps, ts=ts) == states(MEMBERS)


def test_live_enhanced_depth_and_injury_never_restore_a_non_member() -> None:
    old, new = AS_OF - timedelta(days=30), AS_OF - timedelta(hours=2)
    roster = pl.DataFrame([
        # A's older batch still lists STALE (depth 1) and a roster-only QB.
        {"canonical_player_id": STALE, "canonical_team_id": T_A, "depth": 1, "available_at": old},
        {"canonical_player_id": "qb:ghost", "canonical_team_id": T_A, "depth": 1,
         "available_at": old},
        {"canonical_player_id": STARTER, "canonical_team_id": T_A, "depth": 2, "available_at": old},
        # A's latest batch.
        {"canonical_player_id": STARTER, "canonical_team_id": T_A, "depth": 1, "available_at": new},
        {"canonical_player_id": BACKUP, "canonical_team_id": T_A, "depth": 2, "available_at": new},
    ]).with_columns(pl.lit(False).alias("available_at_is_estimated"))
    membership = live_team_membership_at(roster, as_of=AS_OF)
    assert membership == {STARTER: T_A, BACKUP: T_A}
    players = pl.concat([PLAYERS, pl.DataFrame({"canonical_player_id": ["qb:ghost"],
                                                "position_group": ["QB"]})])
    ps, ts = league()
    s = build_player_states(ps, ts, players, as_of=AS_OF, strict=False, roster=roster,
                            qb_membership=membership)
    assert STALE not in s and "qb:ghost" not in s
    assert s[STARTER].depth == 1
    # An injured starter is inactive -- the enhancement acts on members only.
    inj = pl.DataFrame([{"canonical_player_id": STARTER, "status": "Out",
                         "available_at": new, "available_at_is_estimated": False}])
    s2 = build_player_states(ps, ts, players, as_of=AS_OF, strict=False, roster=roster,
                             injuries=inj, qb_membership=membership)
    assert s2[STARTER].active is False and STALE not in s2
    assert selected(s2, T_A) == BACKUP


# ----------------------------------------------------------- membership evidence


def _status(rows: list[tuple[str, int, str, str]], season: int = 2025) -> pl.DataFrame:
    return pl.DataFrame(
        [{"season": season, "week": w, "team": t, "gsis_id": g, "status": st}
         for g, w, t, st in rows],
        schema_overrides={"season": pl.Int64, "week": pl.Int64},
    )


XW = pl.DataFrame({
    "canonical_player_id": ["p:ryan", "p:richardson", "p:jones", "p:brown", "p:jackson",
                            "p:late"],
    "nflverse_gsis_id": ["G-RYAN", "G-RICH", "G-JONES", "G-BROWN", "G-LJ", "G-LATE"],
    "match_class": ["MATCHED_EXACT"] * 6,
})
TEAMS = pl.DataFrame({"abbreviation": ["IND", "ATL", "BAL", "ARI", "NYJ"],
                      "canonical_team_id": ["t:ind", "t:atl", "t:bal", "t:ari", "t:nyj"]})


def test_member_statuses_are_identity_not_availability() -> None:
    assert frozenset({"ACT", "INA", "RES", "DEV"}) == MEMBER_ROSTER_STATUSES
    table = build_historical_team_membership(_status([
        ("G-RICH", 3, "IND", "ACT"), ("G-JONES", 3, "IND", "INA"),
        ("G-BROWN", 3, "BAL", "RES"), ("G-LJ", 3, "BAL", "DEV"),
    ]), XW, TEAMS)
    # Active, game-day inactive, reserve and practice squad are one class.
    assert historical_team_membership_at(table, target_slate=(2025, 0, 3)) == {
        "p:richardson": "t:ind", "p:jones": "t:ind", "p:brown": "t:bal", "p:jackson": "t:bal"}


def test_matt_ryan_ind_2025_regression() -> None:
    """Retired after 2022 (IND); listed RET on ATL in 2024; no 2025 row."""
    table = build_historical_team_membership(pl.concat([
        _status([("G-RYAN", 1, "IND", "ACT")], season=2022),
        _status([("G-RYAN", 1, "ATL", "RET")], season=2024),
        _status([("G-RICH", 1, "IND", "ACT"), ("G-JONES", 1, "IND", "ACT")]),
    ]), XW, TEAMS)
    groups = {"p:ryan": "QB", "p:richardson": "QB", "p:jones": "QB"}
    at_2025 = historical_team_membership_at(table, target_slate=(2025, 0, 1))
    assert eligible_qb_ids(at_2025, groups, "t:ind") == ("p:jones", "p:richardson")
    assert "p:ryan" not in at_2025
    # Retired list membership is not membership: never an ATL candidate.
    assert historical_team_membership_at(table, target_slate=(2024, 0, 1)) == {}


def test_anthony_brown_stale_share_regression() -> None:
    """Four BAL starts in 2022, cut after 2023 week 1: no BAL candidacy in
    2023 week 2 or later, and a later ARI week never reaches back."""
    table = build_historical_team_membership(pl.concat([
        _status([("G-BROWN", 1, "BAL", "ACT")], season=2023),
        _status([("G-BROWN", 2, "BAL", "CUT"), ("G-LJ", 2, "BAL", "ACT")], season=2023),
        _status([("G-BROWN", 8, "ARI", "ACT")], season=2024),
    ]), XW, TEAMS)
    assert historical_team_membership_at(table, target_slate=(2023, 0, 1)) == {
        "p:brown": "t:bal"}
    assert historical_team_membership_at(table, target_slate=(2023, 0, 2)) == {
        "p:jackson": "t:bal"}
    assert historical_team_membership_at(table, target_slate=(2024, 0, 7)) == {}


def test_departed_traded_and_future_team_are_excluded_and_never_leak_backward() -> None:
    rows = [("G-LATE", 5, "NYJ", "ACT"), ("G-JONES", 4, "IND", "TRD"),
            ("G-JONES", 4, "ATL", "ACT"), ("G-RICH", 4, "IND", "CUT")]
    table = build_historical_team_membership(_status(rows), XW, TEAMS)
    at4 = historical_team_membership_at(table, target_slate=(2025, 0, 4))
    assert at4 == {"p:jones": "t:atl"}  # traded away: only his new team
    assert "p:late" not in at4  # joins NYJ only in week 5
    # A future roster week can never change an earlier target.
    more = build_historical_team_membership(
        _status([*rows, ("G-RICH", 5, "IND", "ACT"), ("G-LATE", 6, "IND", "ACT")]),
        XW, TEAMS)
    assert historical_team_membership_at(more, target_slate=(2025, 0, 4)) == at4


def test_ambiguous_membership_fails_closed() -> None:
    table = build_historical_team_membership(_status([
        ("G-JONES", 4, "IND", "ACT"), ("G-JONES", 4, "ATL", "ACT")]), XW, TEAMS)
    assert historical_team_membership_at(table, target_slate=(2025, 0, 4)) == {}
    with pytest.raises(ValueError, match="no canonical team"):
        build_historical_team_membership(_status([("G-JONES", 4, "XXX", "ACT")]), XW, TEAMS)


def test_live_membership_uses_latest_batch_and_never_a_future_one() -> None:
    t0 = AS_OF - timedelta(days=2)
    roster = pl.DataFrame([
        {"canonical_player_id": MOVER, "canonical_team_id": T_B, "available_at": t0},
        {"canonical_player_id": MOVER, "canonical_team_id": T_A,
         "available_at": t0 + timedelta(hours=1)},
        {"canonical_player_id": B_QB, "canonical_team_id": T_B, "available_at": t0},
        # Received after the cutoff: never read.
        {"canonical_player_id": STALE, "canonical_team_id": T_A,
         "available_at": AS_OF + timedelta(seconds=1)},
    ]).with_columns(pl.lit(False).alias("available_at_is_estimated"))
    assert live_team_membership_at(roster, as_of=AS_OF) == {MOVER: T_A, B_QB: T_B}
    assert live_team_membership_at(pl.DataFrame(), as_of=AS_OF) == {}


# ----------------------------------------------------------- Gate 1 equivalence

AS_OF_G1 = G1_AS_OF


def g1_inputs(mode, *, membership=HOME_RB_MEMBER, roster=HOME_RB_ROSTER,
              profile=ModelProfile.STRUCTURAL_CORE, world=None):
    hwf = mode is EvidenceMode.HISTORICAL_WALK_FORWARD
    world = world or (historical_world() if hwf else live_world())
    live_roster = pl.concat([world["roster"], roster], how="diagonal_relaxed") if (
        not hwf and roster is not None) else world["roster"]
    return build_model_inputs(
        model_profile=profile, evidence_mode=mode, games=world["games"],
        player_stats=world["player_stats"], team_stats=world["team_stats"],
        players=UNKNOWN_DIM, roster=live_roster, injuries=world["injuries"],
        game_odds=world["game_odds"], as_of=AS_OF_G1, target_slate=TARGET_SLATE,
        target_game_id="g1:game:target" if hwf else None,
        historical_positions=HOME_RB_QB,
        historical_team_membership=membership if hwf else None,
    )


def _qb_candidates(model_inputs, team):
    return [p.player_id for p in _sim_team(model_inputs, team) if p.position_group == "QB"]


def test_gate1_historical_live_structural_inputs_identical(monkeypatch) -> None:
    historical = g1_inputs(EvidenceMode.HISTORICAL_WALK_FORWARD)
    live = g1_inputs(EvidenceMode.LIVE_PIT)
    groups = {HOME_RB: "QB"}
    assert eligible_qb_ids(historical.qb_membership, groups, HOME) == (
        eligible_qb_ids(live.qb_membership, groups, HOME)) == (HOME_RB,)
    assert historical.player_states == live.player_states
    assert historical.team_states == live.team_states
    for team in (HOME, AWAY):
        assert _qb_candidates(historical, team) == _qb_candidates(live, team)
        assert selected_starter_id(_sim_team(historical, team), "QB") == (
            selected_starter_id(_sim_team(live, team), "QB"))
    assert selected_starter_id(_sim_team(live, HOME), "QB") == HOME_RB
    h_in, h_cfg = simulator_input(historical, monkeypatch)
    l_in, l_cfg = simulator_input(live, monkeypatch)
    assert dataclasses.asdict(h_in) == dataclasses.asdict(l_in) and h_cfg == l_cfg
    assert prepared(historical).simulation_input_sha256 == prepared(live).simulation_input_sha256
    # Live roster depth is never read by STRUCTURAL_CORE.
    assert live.player_states[HOME_RB].depth is None and live.roster.is_empty()


def test_gate1_no_membership_evidence_means_generic_qb_in_both_modes() -> None:
    historical = g1_inputs(EvidenceMode.HISTORICAL_WALK_FORWARD, membership=None)
    live = g1_inputs(EvidenceMode.LIVE_PIT, roster=None)
    assert HOME_RB not in historical.player_states and HOME_RB not in live.player_states
    assert _qb_candidates(historical, HOME) == _qb_candidates(live, HOME) == [f"__QB__:{HOME}"]


def test_gate1_membership_inputs_are_mode_bound() -> None:
    with pytest.raises(ModelProfileError):
        build_model_inputs(
            model_profile=ModelProfile.STRUCTURAL_CORE, evidence_mode=EvidenceMode.LIVE_PIT,
            games=live_world()["games"], player_stats=live_world()["player_stats"],
            team_stats=live_world()["team_stats"], players=UNKNOWN_DIM,
            roster=live_world()["roster"], injuries=live_world()["injuries"],
            game_odds=live_world()["game_odds"], as_of=AS_OF_G1, target_slate=TARGET_SLATE,
            historical_team_membership=HOME_RB_MEMBER,
        )
    with pytest.raises(ModelProfileError):
        g1_inputs(EvidenceMode.LIVE_PIT, profile=ModelProfile.LIVE_ENHANCED, membership=None)


def test_structural_core_live_never_reads_roster_depth_or_injury(monkeypatch) -> None:
    base, _ = simulator_input(g1_inputs(EvidenceMode.LIVE_PIT), monkeypatch)
    deeper = HOME_RB_ROSTER.with_columns(pl.lit(1).alias("depth"),
                                         pl.lit("Out").alias("injury_status"))
    changed, _ = simulator_input(g1_inputs(EvidenceMode.LIVE_PIT, roster=deeper), monkeypatch)
    assert dataclasses.asdict(changed) == dataclasses.asdict(base)


# ----------------------------------------------------------- evidence + identity


def test_membership_table_is_certified_identity_evidence() -> None:
    world = historical_world()
    tables = {"games": world["games"], "player_game_stats": world["player_stats"],
              "team_game_stats": world["team_stats"], "players": UNKNOWN_DIM,
              "historical_team_membership": HOME_RB_MEMBER,
              # 2026 receipts the historical run never consumes.
              "roster_snapshots": HOME_RB_ROSTER.with_columns(
                  pl.lit(True).alias("available_at_is_estimated"))}
    evidence, _ = classify_model_evidence(
        tables, evidence_mode=EvidenceMode.HISTORICAL_WALK_FORWARD,
        model_profile=ModelProfile.STRUCTURAL_CORE)
    assert evidence is EvidenceClass.CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY
    assert "roster_snapshots" in profile_input_tables(
        ModelProfile.STRUCTURAL_CORE, EvidenceMode.LIVE_PIT)
    assert "roster_snapshots" not in profile_input_tables(
        ModelProfile.STRUCTURAL_CORE, EvidenceMode.HISTORICAL_WALK_FORWARD)
    live, _ = classify_model_evidence(
        tables, evidence_mode=EvidenceMode.LIVE_PIT, model_profile=ModelProfile.STRUCTURAL_CORE)
    assert live is EvidenceClass.RESEARCH_ONLY  # an estimated live roster row is not PIT


def test_science_identity_changed_and_old_calibrator_fails_closed() -> None:
    assert set(PROFILE_SCIENCE_VERSIONS) == set(ModelProfile)
    old, new = "2026.1.0", profile_base_model_version("2026.1.0", "STRUCTURAL_CORE")
    assert new == "2026.1.0+structural-core.v2-qb-roster-membership"
    assert new != profile_base_model_version("2026.1.0", "LIVE_ENHANCED")
    common = dict(model_profile="STRUCTURAL_CORE", simulation_config_version="sim-v1",
                  feature_contract_version="f", prop_contract_version="p",
                  calibration_contract_version="c", scope_type="JOINT_GAME",
                  checkpoint_scope="ALL_PREGAME_CHECKPOINTS")
    assert compute_compatibility_digest(base_model_version=old, **common) != (
        compute_compatibility_digest(base_model_version=new, **common))
