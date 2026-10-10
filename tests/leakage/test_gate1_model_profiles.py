"""Gate 1: STRUCTURAL_CORE vs LIVE_ENHANCED model profiles.

STRUCTURAL_CORE must be the SAME model under historical walk-forward replay
and live execution; LIVE_ENHANCED keeps the live market/injury/roster
behaviour; calibration artifacts are bound to a profile. Synthetic data and
small-draw simulations only.
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

from test_calibration_registry import (
    _artifact,
    _backend,
    _fully_promote,
    _resolve_kwargs,
)
from test_challenger import _labeled_game
from test_historical_runner import _rb_row, _team_row, _wr_row
from test_phase10c3a_runner import _build_two_season_warehouse

from nflprops.calibration.artifact import (
    CalibrationArtifactError,
    compute_compatibility_digest,
)
from nflprops.calibration.challenger import fit_challenger_theta
from nflprops.calibration.challenger_registration import register_challenger
from nflprops.calibration.historical_runner import (
    EVIDENCE_MODE,
    MODEL_PROFILE,
    build_labeled_game,
)
from nflprops.calibration.phase10c3a_runner import (
    ConfigurationError,
    RunnerConfig,
    parse_args,
    run,
)
from nflprops.calibration.registry import (
    require_calibrator_applicable,
    resolve_calibration_champion,
)
from nflprops.config import Config, config_sha256, load
from nflprops.data.evidence_policy import (
    EvidenceClass,
    EvidencePolicyError,
    classify_model_evidence,
    official_view,
    promotion_evidence_allowed,
    require_official,
)
from nflprops.data.warehouse import Warehouse
from nflprops.domain.model_profile import (
    ModelProfile,
    ModelProfileError,
    resolve_model_profile,
)
from nflprops.features.historical_evidence import EvidenceMode
from nflprops.market.consensus import game_market_consensus
from nflprops.market.timing import MarketTimingError
from nflprops.orchestration.calibration_store import CALIBRATION_ARTIFACTS_TABLE
from nflprops.orchestration.run_store import compute_run_id
from nflprops.pipelines import pregame
from nflprops.pipelines.model_inputs import ModelInputs, build_model_inputs
from nflprops.simulation.game import (
    SimulationConfig,
    TeamSimulationInput,
    _team_td_rate,
)

HOME, AWAY = "g1:team:home", "g1:team:away"
OTH_A, OTH_B = "g1:team:oth-a", "g1:team:oth-b"
HOME_WR, HOME_RB, AWAY_WR = "g1:p:home-wr", "g1:p:home-rb", "g1:p:away-wr"
TRADED, ROOKIE = "g1:p:traded", "g1:p:rookie"

W1 = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)
W2 = datetime(2024, 9, 15, 17, 0, tzinfo=UTC)
THU3 = datetime(2024, 9, 20, 0, 15, tzinfo=UTC)  # same slate as target, earlier
TARGET_KICKOFF = datetime(2024, 9, 22, 17, 0, tzinfo=UTC)
W4 = datetime(2024, 9, 29, 17, 0, tzinfo=UTC)  # after the target
AS_OF = TARGET_KICKOFF - timedelta(minutes=30)
TARGET_SLATE = (2024, 0, 3)
IMPORTED_2026 = datetime(2026, 8, 21, 3, 0, tzinfo=UTC)

G_W1, G_W2, G_THU3, G_TARGET, G_W4 = (
    "g1:game:w1", "g1:game:w2", "g1:game:thu3", "g1:game:target", "g1:game:w4"
)
#: (game, kickoff, week, home, away, {player: team})
GAMES = (
    (G_W1, W1, 1, HOME, AWAY, {HOME_WR: HOME, HOME_RB: HOME, AWAY_WR: AWAY, TRADED: HOME}),
    (G_W2, W2, 2, AWAY, HOME, {HOME_WR: HOME, HOME_RB: HOME, AWAY_WR: AWAY, TRADED: AWAY}),
    (G_THU3, THU3, 3, OTH_A, OTH_B, {"g1:p:thu-wr": OTH_A}),
    (G_W4, W4, 4, OTH_A, HOME, {HOME_WR: HOME, TRADED: OTH_A}),
)
PLAYERS = pl.DataFrame(
    [
        {"canonical_player_id": p, "position_group": g}
        for p, g in (
            (HOME_WR, "WR"), (HOME_RB, "RB"), (AWAY_WR, "WR"), (TRADED, "WR"),
            ("g1:p:thu-wr", "WR"), (ROOKIE, "WR"),
        )
    ]
)


def _game(gid, kickoff, week, home, away, *, status, available_at):
    return {
        "canonical_game_id": gid, "date": kickoff, "season": 2024, "week": week,
        "postseason": False, "status_state": status, "available_at": available_at,
        "available_at_is_estimated": False,
        "home_canonical_team_id": home, "visitor_canonical_team_id": away,
    }


def _box(gid, roster: dict, *, available_at, estimated=False, scale=1):
    team_rows, player_rows = [], []
    for team in sorted(set(roster.values())):
        team_rows.append({**_team_row(game_id=gid, team_id=team, available_at=available_at),
                          "available_at_is_estimated": estimated})
    for player, team in roster.items():
        row = (_rb_row if player == HOME_RB else _wr_row)(
            game_id=gid, team_id=team, player_id=player, available_at=available_at
        )
        row["available_at_is_estimated"] = estimated
        row["receiving_yards"] *= scale
        row["receiving_targets"] *= scale
        player_rows.append(row)
    return team_rows, player_rows


def _stats(receipt, *, estimated=False, w2_scale=1, w4_scale=1):
    teams, players = [], []
    for gid, kickoff, _week, _h, _a, roster in GAMES:
        scale = {G_W2: w2_scale, G_W4: w4_scale}.get(gid, 1)
        t, p = _box(gid, roster, available_at=receipt(gid, kickoff), estimated=estimated,
                    scale=scale)
        teams += t
        players += p
    return pl.DataFrame(teams), pl.DataFrame(players)


def _live_receipt(gid, kickoff):
    # Genuine receipts a few hours after each game; W4 is after the cutoff.
    return kickoff + timedelta(hours=5)


def live_world(*, w2_scale=1, w4_scale=1, receipt=_live_receipt):
    """What the live warehouse holds at AS_OF: genuine receipts, schedule +
    final game rows, and genuine pregame observations."""
    games = []
    for gid, kickoff, week, home, away, _ in GAMES:
        games.append(_game(gid, kickoff, week, home, away, status="scheduled",
                           available_at=kickoff - timedelta(days=7)))
        games.append(_game(gid, kickoff, week, home, away, status="final",
                           available_at=kickoff + timedelta(hours=4)))
    games.append(_game(G_TARGET, TARGET_KICKOFF, 3, HOME, AWAY, status="scheduled",
                       available_at=TARGET_KICKOFF - timedelta(days=7)))
    team_stats, player_stats = _stats(receipt, w2_scale=w2_scale, w4_scale=w4_scale)
    return {
        "games": pl.DataFrame(games),
        "team_stats": team_stats,
        "player_stats": player_stats,
        "roster": pl.DataFrame([
            {"canonical_player_id": ROOKIE, "canonical_team_id": HOME, "depth": 1,
             "available_at": AS_OF - timedelta(hours=3), "available_at_is_estimated": False},
            {"canonical_player_id": HOME_WR, "canonical_team_id": HOME, "depth": 2,
             "available_at": AS_OF - timedelta(hours=3), "available_at_is_estimated": False},
        ]),
        "injuries": pl.DataFrame([
            {"canonical_player_id": HOME_RB, "status": "Out",
             "available_at": AS_OF - timedelta(hours=2), "available_at_is_estimated": False},
        ]),
        "game_odds": pl.DataFrame([
            {"canonical_game_id": G_TARGET, "vendor": "book", "spread_home_value": -6.5,
             "total_value": 51.5, "available_at": AS_OF - timedelta(hours=1),
             "available_at_is_estimated": False,
             "collector_received_at": AS_OF - timedelta(hours=1)},
        ]),
    }


def historical_world():
    """The same football as `live_world`, as the 2026 historical import holds
    it: every row received in August 2026, half of them estimated, final
    game rows only, and no genuine pregame observation."""
    games = [
        _game(gid, kickoff, week, home, away, status="final", available_at=IMPORTED_2026)
        for gid, kickoff, week, home, away, _ in GAMES
    ] + [_game(G_TARGET, TARGET_KICKOFF, 3, HOME, AWAY, status="final",
               available_at=IMPORTED_2026)]
    team_stats, player_stats = _stats(lambda gid, kickoff: IMPORTED_2026)
    flip = pl.int_range(pl.len()) % 2 == 0
    return {
        "games": pl.DataFrame(games),
        "team_stats": team_stats.with_columns(flip.alias("available_at_is_estimated")),
        "player_stats": player_stats.with_columns(flip.alias("available_at_is_estimated")),
        "roster": pl.DataFrame(),
        "injuries": pl.DataFrame(),
        "game_odds": pl.DataFrame(),
    }


def inputs(world, *, profile=ModelProfile.STRUCTURAL_CORE, mode=EvidenceMode.LIVE_PIT,
           as_of=AS_OF) -> ModelInputs:
    return build_model_inputs(
        model_profile=profile, evidence_mode=mode, games=world["games"],
        player_stats=world["player_stats"], team_stats=world["team_stats"], players=PLAYERS,
        roster=world["roster"], injuries=world["injuries"], game_odds=world["game_odds"],
        as_of=as_of, target_slate=TARGET_SLATE,
        target_game_id=G_TARGET if mode is EvidenceMode.HISTORICAL_WALK_FORWARD else None,
    )


def hwf_inputs(world=None) -> ModelInputs:
    return inputs(world or historical_world(), mode=EvidenceMode.HISTORICAL_WALK_FORWARD)


TARGET_ROW = {"canonical_game_id": G_TARGET, "home_canonical_team_id": HOME,
              "visitor_canonical_team_id": AWAY, "date": TARGET_KICKOFF}


class _CapturedError(Exception):
    pass


def simulator_input(model_inputs: ModelInputs, monkeypatch: pytest.MonkeyPatch,
                    cfg: SimulationConfig | None = None):
    """The exact `GameSimulationInput` + config the simulator would receive."""
    captured: dict = {}

    def _capture(sim_input, sim_cfg):
        captured["input"], captured["cfg"] = sim_input, sim_cfg
        raise _CapturedError

    with monkeypatch.context() as m:
        m.setattr(pregame, "simulate_game", _capture)
        with pytest.raises(_CapturedError):
            pregame.simulate_game_for_prediction(
                game=TARGET_ROW, team_states=model_inputs.team_states,
                player_states=model_inputs.player_states, game_odds=model_inputs.game_odds,
                as_of=AS_OF, model_version="gate1", market_mode="live",
                simulation_config=cfg, n_draws=200,
            )
    return captured["input"], captured["cfg"]


def prepared(model_inputs: ModelInputs, cfg: SimulationConfig | None = None):
    out = pregame.simulate_game_for_prediction(
        game=TARGET_ROW, team_states=model_inputs.team_states,
        player_states=model_inputs.player_states, game_odds=model_inputs.game_odds,
        as_of=AS_OF, model_version="gate1", market_mode="live",
        simulation_config=cfg, n_draws=200,
    )
    assert out is not None
    return out


# ------------------------------------------------------------------ E: equivalence


def test_structural_core_historical_and_live_model_input_identical(monkeypatch) -> None:
    historical, live = hwf_inputs(), inputs(live_world())
    assert historical.team_states == live.team_states
    assert historical.player_states == live.player_states
    hist_input, hist_cfg = simulator_input(historical, monkeypatch)
    live_input, live_cfg = simulator_input(live, monkeypatch)
    assert dataclasses.asdict(hist_input) == dataclasses.asdict(live_input)
    assert hist_cfg == live_cfg
    hist_run, live_run = prepared(historical), prepared(live)
    assert hist_run.simulation_input_sha256 == live_run.simulation_input_sha256
    assert hist_run.result.real_player_draws().equals(live_run.result.real_player_draws())
    # The same-slate Thursday game and the post-target game never enter.
    assert "g1:p:thu-wr" not in live.player_states
    assert live.player_states[HOME_WR].opportunities == historical.player_states[
        HOME_WR].opportunities


# ------------------------------------------------------------------ RE: recency


def test_re1_recency_is_event_time_in_both_modes() -> None:
    late = live_world(receipt=lambda gid, kickoff: kickoff + timedelta(days=2))
    # Receipt hours/days later changes nothing in STRUCTURAL_CORE once eligible...
    assert inputs(late).player_states == inputs(live_world()).player_states
    assert inputs(late).team_states == hwf_inputs().team_states
    # ...whereas LIVE_ENHANCED keeps its receipt-anchored recency (unchanged).
    assert inputs(late, profile=ModelProfile.LIVE_ENHANCED).player_states != inputs(
        live_world(), profile=ModelProfile.LIVE_ENHANCED).player_states


def test_re2_not_yet_received_row_is_ineligible_whatever_its_event_time() -> None:
    def w2_late(gid, kickoff):
        return AS_OF + timedelta(minutes=1) if gid == G_W2 else _live_receipt(gid, kickoff)

    pending = inputs(live_world(receipt=w2_late))
    # W2 (a prior-slate final game) is not yet received: it does not count.
    assert pending.team_states != hwf_inputs().team_states
    assert pending.team_states == inputs(
        live_world(receipt=w2_late, w2_scale=9)).team_states


def test_re3_after_genuine_receipt_same_weight_as_historical() -> None:
    def w2_just_in(gid, kickoff):
        return AS_OF - timedelta(seconds=1) if gid == G_W2 else _live_receipt(gid, kickoff)

    arrived = inputs(live_world(receipt=w2_just_in))
    assert arrived.team_states == hwf_inputs().team_states
    assert arrived.player_states == hwf_inputs().player_states


# ------------------------------------------------------------------ TR: traded


def test_tr1_tr3_latest_eligible_team_never_future() -> None:
    for model in (inputs(live_world()), hwf_inputs(),
                  inputs(live_world(), profile=ModelProfile.LIVE_ENHANCED)):
        assert model.player_states[TRADED].team_id == AWAY  # W2 team; W4's OTH_A is future


def test_tr2_storage_order_never_assigns_team() -> None:
    world = live_world()
    rng = random.Random(11)
    for profile in ModelProfile:
        baseline = inputs(world, profile=profile)
        for _ in range(4):
            shuffled = dict(world)
            for key in ("player_stats", "team_stats", "games"):
                frame = world[key]
                shuffled[key] = frame[rng.sample(range(frame.height), frame.height)]
            # Oldest game physically last: the pre-fix rule picked HOME here.
            older_last = pl.concat([
                shuffled["player_stats"].filter(pl.col("canonical_game_id") != G_W1),
                shuffled["player_stats"].filter(pl.col("canonical_game_id") == G_W1),
            ])
            shuffled["player_stats"] = older_last
            result = inputs(shuffled, profile=profile)
            assert result.player_states[TRADED].team_id == AWAY
            assert result.player_states == baseline.player_states
            assert result.team_states == baseline.team_states


def test_tr4_tr5_historical_live_agree_and_untraded_unchanged() -> None:
    live, historical = inputs(live_world()), hwf_inputs()
    for player in (TRADED, HOME_WR, HOME_RB, AWAY_WR):
        assert live.player_states[player].team_id == historical.player_states[player].team_id
    assert live.player_states[HOME_WR].team_id == HOME
    assert live.player_states[AWAY_WR].team_id == AWAY


# ------------------------------------------------------------------ EM: no market


def test_em1_em2_empty_or_absent_market_is_no_market() -> None:
    for frame in (pl.DataFrame(), live_world()["game_odds"].head(0)):
        market = game_market_consensus(frame, G_TARGET, as_of=AS_OF)
        assert (market.home_spread, market.total, market.n_books_spread) == (None, None, 0)
    structural = inputs(live_world())
    assert structural.game_odds is None
    no_table = dict(live_world(), game_odds=pl.DataFrame())
    assert prepared(inputs(no_table)).game_market_available_at is None
    assert prepared(inputs(no_table, profile=ModelProfile.LIVE_ENHANCED)) is not None


def test_em3_em4_em5_no_market_means_no_market_effect(monkeypatch) -> None:
    structural = inputs(live_world())
    sim_input, _ = simulator_input(structural, monkeypatch)
    for side in (sim_input.home, sim_input.away):
        assert side.implied_points is None
        assert side.team_spread == 0.0  # zero contribution to pass tendency
    team: TeamSimulationInput = sim_input.home
    cfg = SimulationConfig(n_draws=200)
    structural_rate = (
        max(team.state.offensive_td_rate, 1e-5)
        * max(team.opponent_state.td_rate_allowed, 1e-5)
    ) ** 0.5
    assert _team_td_rate(team, cfg) == pytest.approx(structural_rate, rel=0, abs=0)
    # Market knobs cannot move a STRUCTURAL_CORE simulation at all.
    knobs = SimulationConfig(n_draws=200, market_td_weight=0.0, pregame_spread_logit_per_point=0.5)
    assert prepared(structural).result.real_player_draws().equals(
        prepared(structural, knobs).result.real_player_draws())


def test_em6_live_enhanced_market_unchanged(monkeypatch) -> None:
    enhanced = inputs(live_world(), profile=ModelProfile.LIVE_ENHANCED)
    sim_input, _ = simulator_input(enhanced, monkeypatch)
    expected_home, expected_away = pregame._implied_points(51.5, -6.5)
    assert (sim_input.home.implied_points, sim_input.away.implied_points) == (
        expected_home, expected_away)
    assert sim_input.home.team_spread == -6.5
    assert prepared(enhanced).game_market_available_at == AS_OF - timedelta(hours=1)


def test_em7_malformed_live_enhanced_market_fails_closed() -> None:
    world = live_world()
    world["game_odds"] = world["game_odds"].drop("collector_received_at")
    with pytest.raises(MarketTimingError):
        prepared(inputs(world, profile=ModelProfile.LIVE_ENHANCED))


# ------------------------------------------------------------------ IR: injury/roster


def test_ir1_ir2_structural_core_ignores_injuries_and_roster(monkeypatch) -> None:
    base, _ = simulator_input(inputs(live_world()), monkeypatch)
    for changed in (
        dict(live_world(), injuries=pl.DataFrame()),
        dict(live_world(), roster=pl.DataFrame()),
        dict(live_world(), injuries=live_world()["injuries"].with_columns(
            pl.lit(HOME_WR).alias("canonical_player_id"))),
    ):
        sim_input, _ = simulator_input(inputs(changed), monkeypatch)
        assert dataclasses.asdict(sim_input) == dataclasses.asdict(base)
    structural = inputs(live_world())
    assert ROOKIE not in structural.player_states  # roster-only player never added
    assert all(p.active and p.depth is None for p in structural.player_states.values())
    assert structural.roster.is_empty() and structural.injuries.is_empty()


def test_ir3_ir4_live_enhanced_consumes_only_known_observations() -> None:
    enhanced = inputs(live_world(), profile=ModelProfile.LIVE_ENHANCED)
    assert enhanced.player_states[HOME_RB].active is False
    assert enhanced.player_states[ROOKIE].depth == 1
    assert enhanced.player_states[HOME_WR].depth == 2
    future = dict(live_world())
    future["injuries"] = future["injuries"].with_columns(
        pl.lit(AS_OF + timedelta(seconds=1)).alias("available_at"))
    future["roster"] = future["roster"].with_columns(
        pl.lit(AS_OF + timedelta(seconds=1)).alias("available_at"))
    later = inputs(future, profile=ModelProfile.LIVE_ENHANCED)
    assert later.player_states[HOME_RB].active is True
    assert ROOKIE not in later.player_states


# ------------------------------------------------------------------ MP: profile identity


def test_mp1_mp2_mp3_calibrator_profile_binding(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    structural = _artifact()
    _fully_promote(backend, structural)
    resolved = resolve_calibration_champion(backend, **_resolve_kwargs())
    assert resolved is not None and resolved.model_profile == "STRUCTURAL_CORE"  # MP1
    require_calibrator_applicable(resolved, prediction_profile="STRUCTURAL_CORE")
    # MP3: LIVE_ENHANCED predictions never resolve/accept the STRUCTURAL_CORE one.
    assert resolve_calibration_champion(
        backend, **_resolve_kwargs(model_profile="LIVE_ENHANCED")) is None
    with pytest.raises(ModelProfileError):
        require_calibrator_applicable(resolved, prediction_profile="LIVE_ENHANCED")
    # MP2: a LIVE_ENHANCED calibrator is never attached to STRUCTURAL_CORE.
    enhanced = _artifact(model_profile="LIVE_ENHANCED", object_uri="mem://artifact-2")
    _fully_promote(backend, enhanced)
    with pytest.raises(ModelProfileError):
        require_calibrator_applicable(enhanced, prediction_profile="STRUCTURAL_CORE")
    assert resolve_calibration_champion(
        backend, **_resolve_kwargs()).calibration_artifact_id == structural.calibration_artifact_id
    assert resolve_calibration_champion(
        backend, **_resolve_kwargs(model_profile="LIVE_ENHANCED")
    ).calibration_artifact_id == enhanced.calibration_artifact_id
    # Missing/unknown profile: nothing resolves.
    assert resolve_calibration_champion(backend, **_resolve_kwargs(model_profile="")) is None
    assert resolve_calibration_champion(backend, **_resolve_kwargs(model_profile="X")) is None


def test_mp_legacy_artifact_without_profile_is_never_applied(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    _fully_promote(backend, _artifact())
    legacy = backend.read(CALIBRATION_ARTIFACTS_TABLE).with_columns(
        pl.lit(None, dtype=pl.String).alias("model_profile"))
    backend.write(CALIBRATION_ARTIFACTS_TABLE, legacy)
    assert resolve_calibration_champion(backend, **_resolve_kwargs()) is None
    with pytest.raises(CalibrationArtifactError):
        _artifact(model_profile="")


def test_mp4_profile_changes_config_and_science_identity() -> None:
    structural = load(cli_overrides={"model.profile": "STRUCTURAL_CORE"})
    enhanced = load(cli_overrides={"model.profile": "LIVE_ENHANCED"})
    assert resolve_model_profile(load()) is ModelProfile.LIVE_ENHANCED  # shipped default
    assert config_sha256(structural) != config_sha256(enhanced)
    run_ids = {
        compute_run_id(game_id="g", checkpoint_name="T30M", scheduled_as_of=AS_OF,
                       kickoff_at=TARGET_KICKOFF, model_version="v", config_sha256=config_sha256(c),
                       source_sha256="s")
        for c in (structural, enhanced)
    }
    assert len(run_ids) == 2
    a, b = _artifact(), _artifact(model_profile="LIVE_ENHANCED")
    assert a.calibration_artifact_id != b.calibration_artifact_id
    assert compute_compatibility_digest(**dict(a.compatibility_items())) != (
        compute_compatibility_digest(**dict(b.compatibility_items())))
    with pytest.raises(ModelProfileError):
        resolve_model_profile(Config(data={}))


# ------------------------------------------------------------------ EP: evidence policy


def _hist_tables(**pregame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    world = historical_world()
    return {"games": world["games"], "player_game_stats": world["player_stats"],
            "team_game_stats": world["team_stats"], "players": PLAYERS, **pregame}


def _estimated(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.lit(True).alias("available_at_is_estimated"))


def test_ep1_ep2_completed_event_stats_certified_under_hwf() -> None:
    evidence, rows = classify_model_evidence(
        _hist_tables(), evidence_mode=EvidenceMode.HISTORICAL_WALK_FORWARD,
        model_profile=ModelProfile.STRUCTURAL_CORE)
    assert evidence is EvidenceClass.CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY
    assert rows["player_game_stats"] > 0 and rows["team_game_stats"] > 0
    assert promotion_evidence_allowed(evidence, evidence_mode=EvidenceMode.HISTORICAL_WALK_FORWARD)


@pytest.mark.parametrize("table,key", [
    ("injury_snapshots", "injuries"), ("roster_snapshots", "roster"),
    ("game_odds_snapshots", "game_odds"),
])
def test_ep3_ep4_ep5_unproven_pregame_is_never_promotion_evidence(table, key) -> None:
    observed = _estimated(live_world()[key])
    evidence, rows = classify_model_evidence(
        _hist_tables(**{table: observed}), evidence_mode=EvidenceMode.HISTORICAL_WALK_FORWARD,
        model_profile=ModelProfile.LIVE_ENHANCED)
    assert evidence is EvidenceClass.RESEARCH_ONLY and table in rows
    assert not promotion_evidence_allowed(
        evidence, evidence_mode=EvidenceMode.HISTORICAL_WALK_FORWARD)


def test_ep6_live_receipt_rule_unchanged() -> None:
    world = live_world()
    live_tables = {"games": world["games"], "player_game_stats": world["player_stats"],
                   "team_game_stats": world["team_stats"], "players": PLAYERS,
                   "injury_snapshots": world["injuries"]}
    for profile in ModelProfile:
        evidence, _ = classify_model_evidence(
            live_tables, evidence_mode=EvidenceMode.LIVE_PIT, model_profile=profile)
        assert evidence is EvidenceClass.OFFICIAL_PIT_FAITHFUL
    estimated = dict(live_tables, player_game_stats=_estimated(world["player_stats"]))
    evidence, _ = classify_model_evidence(
        estimated, evidence_mode=EvidenceMode.LIVE_PIT, model_profile=ModelProfile.STRUCTURAL_CORE)
    assert evidence is EvidenceClass.RESEARCH_ONLY  # BLOCK 4 unchanged


def test_ep7_event_chronology_is_never_live_receipt_evidence() -> None:
    tables = _hist_tables()
    assert not promotion_evidence_allowed(
        EvidenceClass.CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY, evidence_mode=EvidenceMode.LIVE_PIT)
    assert not promotion_evidence_allowed(
        EvidenceClass.OFFICIAL_PIT_FAITHFUL, evidence_mode=EvidenceMode.HISTORICAL_WALK_FORWARD)
    with pytest.raises(EvidencePolicyError):
        require_official(tables, context="live checkpoint")
    assert official_view(tables["player_game_stats"]).height < tables["player_game_stats"].height


def test_ep_consumed_table_of_unknown_semantics_fails_closed(monkeypatch) -> None:
    from nflprops.data import evidence_policy

    monkeypatch.setitem(
        evidence_policy.PROFILE_INPUT_TABLES,  # type: ignore[arg-type]
        ModelProfile.STRUCTURAL_CORE,
        frozenset({"mystery_feed"}),
    )
    with pytest.raises(EvidencePolicyError):
        classify_model_evidence(
            {"mystery_feed": pl.DataFrame({"a": [1]})},
            evidence_mode=EvidenceMode.HISTORICAL_WALK_FORWARD,
            model_profile=ModelProfile.STRUCTURAL_CORE,
        )


def _register(tmp_path: Path, **provenance: str):
    games = [
        _labeled_game(game_id=f"g{i}", as_of=W1 + timedelta(days=i),
                      outcome_available_at=W1 + timedelta(days=i, hours=4))
        for i in range(3)
    ]
    fit = fit_challenger_theta(games, regularization_lambda=0.02)
    return register_challenger(
        Warehouse(tmp_path / "wh"), _MemStore(), fit=fit, optimizer="L-BFGS-B",
        tolerance=1e-8, calibration_schema_version="2026.1.0", base_model_version="v",
        simulation_config_version="sim-v1", prop_contract_version="2026.1.0",
        calibration_contract_version="2026.1.0", checkpoint_scope="ALL_PREGAME_CHECKPOINTS",
        training_cutoff=W2, training_start=W1, training_end=W2,
        training_manifest_sha256="m", code_sha="c", payload_key="k.json", created_at=W2,
        scored_from=W1, scored_through=W2, validation_schema_version="v1",
        validation_manifest_sha256="vm", training_games=games, metrics={"x": 1.0},
        chronology_checks_passed=True, leakage_checks_passed=True,
        simulation_invariants_passed=True, reproducibility_passed=True,
        support_preservation_passed=True, first_td_simplex_passed=True,
        promotion_gate_passed=False, **provenance,
    )


class _MemStore:
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes) -> None:
        self._objects[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self._objects[key]

    def exists(self, key: str) -> bool:
        return key in self._objects


def test_ep8_registration_records_profile_and_evidence(tmp_path: Path) -> None:
    result = _register(tmp_path / "a", model_profile="STRUCTURAL_CORE",
                       evidence_mode="HISTORICAL_WALK_FORWARD",
                       evidence_class="CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY")
    assert result.register_result.artifact.model_profile == "STRUCTURAL_CORE"
    metrics = json.loads(result.validation.metrics_json)
    assert metrics["model_profile"] == "STRUCTURAL_CORE"
    assert metrics["evidence_mode"] == "HISTORICAL_WALK_FORWARD"
    assert metrics["evidence_class"] == "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY"
    with pytest.raises(ModelProfileError):
        _register(tmp_path / "b", model_profile="LIVE_ENHANCED",
                  evidence_mode="HISTORICAL_WALK_FORWARD",
                  evidence_class="CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY")


# ------------------------------------------------------------------ HR: 10C3A lock


def _runner_config(tmp_path: Path, **overrides) -> RunnerConfig:
    base = dict(data_root=tmp_path, output_dir=tmp_path / "out", season_min=2023,
                season_max=2024, n_draws=60, mode="smoke", model_version="g1",
                regularization_lambda=0.01, max_fit_iterations=20,
                expect_data_manifest_sha256=None)
    base.update(overrides)
    return RunnerConfig(**base)


def test_hr1_hr2_runner_is_locked_to_structural_core(tmp_path: Path, monkeypatch) -> None:
    assert (MODEL_PROFILE, EVIDENCE_MODE) == (
        ModelProfile.STRUCTURAL_CORE, EvidenceMode.HISTORICAL_WALK_FORWARD)
    assert _runner_config(tmp_path).model_profile == "STRUCTURAL_CORE"
    with pytest.raises(ConfigurationError):
        _runner_config(tmp_path, model_profile="LIVE_ENHANCED")
    with pytest.raises(ConfigurationError):
        parse_args(["--data-root", str(tmp_path), "--output-dir", str(tmp_path),
                    "--mode", "smoke", "--n-draws", "40", "--model-profile", "LIVE_ENHANCED"])

    def _never(*args, **kwargs):
        raise AssertionError("simulation must not start")

    monkeypatch.setattr("nflprops.calibration.historical_runner.simulate_game_for_prediction",
                        _never)
    warehouse = _build_two_season_warehouse(tmp_path)
    row = warehouse.read("games").row(2, named=True)
    with pytest.raises(ModelProfileError):
        build_labeled_game(warehouse, row, model_version="g1", n_draws=40,
                           model_profile=ModelProfile.LIVE_ENHANCED)


def test_hr3_report_states_profile_and_evidence(tmp_path: Path) -> None:
    warehouse = _build_two_season_warehouse(tmp_path)
    report = run(_runner_config(tmp_path, data_root=warehouse.root))
    assert report["model_profile"] == "STRUCTURAL_CORE"
    assert report["evidence_mode"] == "HISTORICAL_WALK_FORWARD"
    assert report["evidence_class"] == "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY"
    assert report["promotion_decision"] == "INSUFFICIENT_EVIDENCE"  # smoke


# ------------------------------------------------------------------ D: live execution


def _live_warehouse(tmp_path: Path) -> Warehouse:
    world = live_world()
    warehouse = Warehouse(tmp_path / "live")
    warehouse.write("games", world["games"])
    warehouse.write("player_game_stats", world["player_stats"])
    warehouse.write("team_game_stats", world["team_stats"])
    warehouse.write("players", PLAYERS)
    warehouse.write("roster_snapshots", world["roster"])
    warehouse.write("injury_snapshots", world["injuries"])
    warehouse.write("game_odds_snapshots", world["game_odds"])
    warehouse.write("player_prop_snapshots", pl.DataFrame([{
        "canonical_game_id": G_TARGET, "canonical_player_id": HOME_WR, "vendor": "book",
        "prop_type": "receiving_yards", "line_value": 60.5, "market_type": "over_under",
        "over_odds": -110, "under_odds": -110, "available_at": AS_OF - timedelta(hours=1),
        "collector_received_at": AS_OF - timedelta(hours=1), "provider_updated_at": None,
        "opened_at": None,
    }]))
    return warehouse


def test_structural_core_runs_live_and_equals_historical(tmp_path: Path) -> None:
    warehouse = _live_warehouse(tmp_path)
    computation = pregame.compute_game_prediction(
        warehouse, season=2024, week=3, game_id=G_TARGET, as_of=AS_OF,
        model_version="gate1", n_draws=200, model_profile=ModelProfile.STRUCTURAL_CORE,
    )
    assert computation is not None
    assert computation.model_profile is ModelProfile.STRUCTURAL_CORE
    assert computation.prepared.game_market_available_at is None
    assert computation.simulation_input_sha256 == prepared(hwf_inputs()).simulation_input_sha256
    priced = computation.price_markets()  # pricing stays a downstream read
    assert priced and {row["player_id"] for row in priced} == {HOME_WR}
    enhanced = pregame.compute_game_prediction(
        warehouse, season=2024, week=3, game_id=G_TARGET, as_of=AS_OF,
        model_version="gate1", n_draws=200, model_profile=ModelProfile.LIVE_ENHANCED,
    )
    assert enhanced is not None
    assert enhanced.simulation_input_sha256 != computation.simulation_input_sha256


def test_official_live_runs_require_an_explicit_profile(tmp_path: Path) -> None:
    warehouse = _live_warehouse(tmp_path)
    with pytest.raises(ModelProfileError):
        pregame.predict_week(warehouse, season=2024, week=3, as_of=AS_OF, n_draws=50,
                             persist=False, official_run_id="run-x")
    pytest.importorskip("prefect")  # the `leakage` CI job installs no orchestration extra
    from nflprops.orchestration.flows.checkpoints import checkpoint_dispatch_flow

    with pytest.raises(ModelProfileError):
        checkpoint_dispatch_flow(warehouse=warehouse, config=Config(data={}), season=2024,
                                 week=3, now=AS_OF)
    assert not warehouse.exists("prediction_runs")
