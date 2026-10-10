"""HISTORICAL_WALK_FORWARD evidence certification (HW1-HW15).

Two clocks: completed-game stats (Class A) are eligible history by EVENT
chronology -- their slate precedes the target's -- regardless of when
NFLProps imported them; pregame observations (Class B: injuries, rosters,
game odds) need a genuine availability time at or before the cutoff, exactly
as LIVE_PIT. Synthetic data only.
"""

from __future__ import annotations

import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

from test_challenger import _labeled_game
from test_historical_runner import _rb_row, _team_row, _wr_row

from nflprops.calibration.challenger import (
    LabeledGame,
    PropLabel,
    run_walk_forward_challenger,
)
from nflprops.calibration.historical_runner import (
    EVIDENCE_MODE,
    WarehouseTables,
    build_labeled_game,
    list_final_games,
    load_warehouse_tables,
)
from nflprops.calibration.phase10c3a_runner import (
    build_season_boundary_folds,
)
from nflprops.data.warehouse import Warehouse
from nflprops.features.asof import filter_pit
from nflprops.features.historical_evidence import (
    EVENT_CHRONOLOGY_COL,
    HISTORICAL_REPLAY_INPUT_SURFACE,
    LABEL_NEVER_TRAINABLE_AT,
    ChronologyEvidence,
    EvidenceMode,
    HistoricalChronologyError,
    InputCertification,
    build_slate_chronology,
    certify_event_derived,
    certify_pregame_observations,
    classify_pregame_rows,
    schedule_identity,
)
from nflprops.market.consensus import game_market_consensus
from nflprops.market.timing import quote_time_source
from nflprops.state.player import build_player_states
from nflprops.state.team import build_team_states

HOME = "hw:team:home"
AWAY = "hw:team:away"
OTHER_HOME = "hw:team:other-home"
OTHER_AWAY = "hw:team:other-away"
HOME_WR = "hw:player:home-wr"
HOME_RB = "hw:player:home-rb"
AWAY_WR = "hw:player:away-wr"
TRADED = "hw:player:traded"

S1_W1 = datetime(2023, 9, 10, 17, 0, tzinfo=UTC)  # prior season
W1 = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)
W2_THU = datetime(2024, 9, 13, 0, 15, tzinfo=UTC)  # same slate as target, earlier kickoff
W2 = datetime(2024, 9, 15, 17, 0, tzinfo=UTC)  # target
W3 = datetime(2024, 9, 22, 17, 0, tzinfo=UTC)  # future
W4 = datetime(2024, 9, 29, 17, 0, tzinfo=UTC)  # scheduled, not played

G_PRIOR_SEASON = "hw:game:2023w1"
G_W1 = "hw:game:2024w1"
G_W2_THU = "hw:game:2024w2-thu"
G_TARGET = "hw:game:2024w2"
G_FUTURE = "hw:game:2024w3"
G_SCHEDULED = "hw:game:2024w4"

IMPORTED_2026 = datetime(2026, 8, 21, 3, 0, tzinfo=UTC)


def _game(gid, season, week, kickoff, status="final", home=HOME, away=AWAY, **extra):
    return {
        "canonical_game_id": gid,
        "available_at": IMPORTED_2026,
        "date": kickoff,
        "season": season,
        "week": week,
        "status_state": status,
        "postseason": False,
        "home_canonical_team_id": home,
        "visitor_canonical_team_id": away,
        **extra,
    }


def _games() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _game(G_PRIOR_SEASON, 2023, 1, S1_W1),
            _game(G_W1, 2024, 1, W1),
            _game(G_W2_THU, 2024, 2, W2_THU, home=OTHER_HOME, away=OTHER_AWAY),
            _game(G_TARGET, 2024, 2, W2, home_team_score=31, visitor_team_score=3),
            _game(G_FUTURE, 2024, 3, W3),
            _game(G_SCHEDULED, 2024, 4, W4, status="scheduled"),
        ]
    )


def _box(gid: str, *, available_at: datetime, estimated: bool, scale: int = 1):
    team = [
        {**_team_row(game_id=gid, team_id=t, available_at=available_at),
         "available_at_is_estimated": estimated,
         "rushing_attempts": 25 * scale}
        for t in (HOME, AWAY)
    ]
    players = [
        _wr_row(game_id=gid, team_id=HOME, player_id=HOME_WR, available_at=available_at),
        _rb_row(game_id=gid, team_id=HOME, player_id=HOME_RB, available_at=available_at),
        _wr_row(game_id=gid, team_id=AWAY, player_id=AWAY_WR, available_at=available_at),
    ]
    for p in players:
        p["available_at_is_estimated"] = estimated
        p["receiving_targets"] *= scale
        p["receiving_yards"] *= scale
    return team, players


def _stats(*, future_scale: int = 1, target_scale: int = 1, w1_scale: int = 1):
    """History imported in 2026: genuine 2026 receipts for some games,
    legacy kickoff+12h estimates for others -- neither may matter."""
    team: list[dict] = []
    players: list[dict] = []
    for gid, kickoff, estimated, scale in (
        (G_PRIOR_SEASON, S1_W1, True, 1),
        (G_W1, W1, False, w1_scale),
        (G_TARGET, W2, True, target_scale),
        (G_FUTURE, W3, False, future_scale),
    ):
        available_at = kickoff + timedelta(hours=12) if estimated else IMPORTED_2026
        t, p = _box(gid, available_at=available_at, estimated=estimated, scale=scale)
        team += t
        players += p
    thu_team = [
        {**_team_row(game_id=G_W2_THU, team_id=t, available_at=IMPORTED_2026)}
        for t in (OTHER_HOME, OTHER_AWAY)
    ]
    thu_players = [
        _wr_row(game_id=G_W2_THU, team_id=OTHER_HOME, player_id="hw:player:thu-wr",
                available_at=IMPORTED_2026)
    ]
    return pl.DataFrame(team + thu_team), pl.DataFrame(players + thu_players)


PLAYERS = pl.DataFrame(
    [
        {"canonical_player_id": HOME_WR, "position_group": "WR"},
        {"canonical_player_id": HOME_RB, "position_group": "RB"},
        {"canonical_player_id": AWAY_WR, "position_group": "WR"},
        {"canonical_player_id": "hw:player:thu-wr", "position_group": "WR"},
        {"canonical_player_id": TRADED, "position_group": "WR"},
    ]
)


def _hwf_states(team_stats, player_stats, *, games=None, target=G_TARGET, as_of=W2):
    chronology = build_slate_chronology(games if games is not None else _games())
    ps = certify_event_derived(player_stats, chronology, target_game_id=target, as_of=as_of)
    ts = certify_event_derived(team_stats, chronology, target_game_id=target, as_of=as_of)
    return (
        build_team_states(ts, ps, as_of=as_of, history_time_col=EVENT_CHRONOLOGY_COL),
        build_player_states(ps, ts, PLAYERS, as_of=as_of, history_time_col=EVENT_CHRONOLOGY_COL),
        ps,
    )


# ------------------------------------------------------------ Class A chronology


def test_hw1_prior_completed_stats_eligible_despite_2026_import() -> None:
    team_stats, player_stats = _stats()
    chronology = build_slate_chronology(_games())
    certified = certify_event_derived(
        player_stats, chronology, target_game_id=G_TARGET, as_of=W2
    )
    w1 = certified.filter(pl.col("canonical_game_id") == G_W1)
    assert w1.height == 3
    assert (w1["available_at"] == IMPORTED_2026).all()  # genuine 2026 import, untouched
    assert (w1["available_at_is_estimated"] == False).all()  # noqa: E712
    assert set(certified["evidence_class"]) == {
        ChronologyEvidence.CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY.value
    }
    # LIVE_PIT on genuine receipt sees none of it -- the two clocks differ.
    assert filter_pit(player_stats, W2).filter(pl.col("canonical_game_id") == G_W1).is_empty()
    _, player_states, _ = _hwf_states(team_stats, player_stats)
    assert HOME_WR in player_states


def test_hw2_target_game_stats_never_visible_to_its_prediction() -> None:
    team_stats, player_stats = _stats()
    team_states, player_states, certified = _hwf_states(team_stats, player_stats)
    assert G_TARGET not in set(certified["canonical_game_id"])
    # Same-slate earlier kickoff (Thursday) is not proven complete either.
    assert G_W2_THU not in set(certified["canonical_game_id"])
    t2, p2 = _stats(target_scale=9)
    team_states_b, player_states_b, _ = _hwf_states(t2, p2)
    assert team_states == team_states_b
    assert player_states == player_states_b


def test_hw3_future_game_stats_never_visible() -> None:
    team_stats, player_stats = _stats()
    team_states, player_states, certified = _hwf_states(team_stats, player_stats)
    assert G_FUTURE not in set(certified["canonical_game_id"])
    t2, p2 = _stats(future_scale=9)
    assert _hwf_states(t2, p2)[:2] == (team_states, player_states)
    # Dropping the future rows altogether changes nothing either (no
    # population prior or normalization is fitted on them).
    t3 = team_stats.filter(pl.col("canonical_game_id") != G_FUTURE)
    p3 = player_stats.filter(pl.col("canonical_game_id") != G_FUTURE)
    assert _hwf_states(t3, p3)[:2] == (team_states, player_states)


def test_hw4_prior_season_stats_eligible_for_later_season() -> None:
    _, player_stats = _stats()
    chronology = build_slate_chronology(_games())
    certified = certify_event_derived(
        player_stats, chronology, target_game_id=G_W1, as_of=W1
    )
    assert set(certified["canonical_game_id"]) == {G_PRIOR_SEASON}


def test_hw5_storage_and_ingest_order_never_control_chronology() -> None:
    team_stats, player_stats = _stats()
    # A player traded HOME -> AWAY: his current team must come from event
    # order. Stored newest-first with the older game imported later.
    traded = pl.DataFrame(
        [
            _wr_row(game_id=G_W1, team_id=AWAY, player_id=TRADED, available_at=IMPORTED_2026),
            _wr_row(game_id=G_PRIOR_SEASON, team_id=HOME, player_id=TRADED,
                    available_at=IMPORTED_2026 + timedelta(days=1)),
        ]
    )
    player_stats = pl.concat([player_stats, traded], how="diagonal_relaxed")
    baseline = _hwf_states(team_stats, player_stats)
    assert baseline[1][TRADED].team_id == AWAY
    rng = random.Random(7)
    for _ in range(5):
        t_idx = list(range(team_stats.height))
        p_idx = list(range(player_stats.height))
        rng.shuffle(t_idx)
        rng.shuffle(p_idx)
        # Shuffle storage order AND scramble the import timestamps.
        shuffled_p = player_stats[p_idx].with_columns(
            pl.Series("available_at", [IMPORTED_2026 + timedelta(minutes=rng.randint(0, 9999))
                                       for _ in p_idx])
        )
        shuffled_t = team_stats[t_idx]
        games = _games()
        shuffled_g = games[rng.sample(range(games.height), games.height)]
        result = _hwf_states(shuffled_t, shuffled_p, games=shuffled_g)
        assert result[0] == baseline[0]
        assert result[1] == baseline[1]
        assert result[2].drop("available_at").equals(baseline[2].drop("available_at"))


def test_slate_chronology_fails_closed_on_overlapping_slates() -> None:
    games = pl.concat(
        [_games(), pl.DataFrame([_game("hw:game:late-w1", 2024, 1, W2_THU + timedelta(hours=1),
                                       home=OTHER_HOME, away=OTHER_AWAY)])],
        how="diagonal_relaxed",
    )
    with pytest.raises(HistoricalChronologyError):
        build_slate_chronology(games)


def test_unknown_or_unfinished_source_games_are_excluded() -> None:
    _, player_stats = _stats()
    orphan = player_stats.head(1).with_columns(pl.lit("hw:game:unknown").alias("canonical_game_id"))
    games = _games().with_columns(
        pl.when(pl.col("canonical_game_id") == G_W1).then(pl.lit("scheduled"))
        .otherwise(pl.col("status_state")).alias("status_state")
    )
    certified = certify_event_derived(
        pl.concat([player_stats, orphan]), build_slate_chronology(games),
        target_game_id=G_TARGET, as_of=W2,
    )
    assert set(certified["canonical_game_id"]) == {G_PRIOR_SEASON}


def test_event_chronology_filter_rejects_uncertified_frames() -> None:
    _, player_stats = _stats()
    forged = player_stats.with_columns(pl.col("available_at").alias(EVENT_CHRONOLOGY_COL))
    with pytest.raises(HistoricalChronologyError):
        filter_pit(forged, W2, time_col=EVENT_CHRONOLOGY_COL)
    forged = forged.with_columns(pl.lit("CERTIFIED_LIVE_PIT").alias("evidence_class"))
    with pytest.raises(HistoricalChronologyError):
        filter_pit(forged, W2, time_col=EVENT_CHRONOLOGY_COL)


def test_target_row_reduced_to_schedule_identity() -> None:
    row = _games().filter(pl.col("canonical_game_id") == G_TARGET).row(0, named=True)
    identity = schedule_identity(row)
    assert "home_team_score" not in identity and "status_state" not in identity
    assert identity["home_canonical_team_id"] == HOME


# ------------------------------------------------------- Class B observations


def _injury(status: str, *, available_at: datetime, estimated: bool) -> dict:
    return {
        "canonical_player_id": HOME_WR,
        "status": status,
        "available_at": available_at,
        "available_at_is_estimated": estimated,
    }


def _roster(depth: int, *, player: str, available_at: datetime, estimated: bool) -> dict:
    return {
        "canonical_player_id": player,
        "canonical_team_id": HOME,
        "depth": depth,
        "available_at": available_at,
        "available_at_is_estimated": estimated,
    }


def _player_states_with(roster=None, injuries=None):
    team_stats, player_stats = _stats()
    chronology = build_slate_chronology(_games())
    ps = certify_event_derived(player_stats, chronology, target_game_id=G_TARGET, as_of=W2)
    ts = certify_event_derived(team_stats, chronology, target_game_id=G_TARGET, as_of=W2)
    return build_player_states(
        ps, ts, PLAYERS, as_of=W2, strict=True, history_time_col=EVENT_CHRONOLOGY_COL,
        roster=certify_pregame_observations(roster, as_of=W2) if roster is not None else None,
        injuries=certify_pregame_observations(injuries, as_of=W2) if injuries is not None else None,
    )


def test_hw6_estimated_injury_observation_not_silently_eligible() -> None:
    injuries = pl.DataFrame(
        [_injury("Out", available_at=W2 - timedelta(days=2), estimated=True)]
    )
    assert certify_pregame_observations(injuries, as_of=W2).is_empty()
    assert classify_pregame_rows(injuries).to_list() == [
        ChronologyEvidence.UNVERIFIED_OR_ESTIMATED_PREGAME.value
    ]
    assert _player_states_with(injuries=injuries)[HOME_WR].active is True
    # Even passed straight to the state builder, strict mode refuses it.
    team_stats, player_stats = _stats()
    chronology = build_slate_chronology(_games())
    ps = certify_event_derived(player_stats, chronology, target_game_id=G_TARGET, as_of=W2)
    ts = certify_event_derived(team_stats, chronology, target_game_id=G_TARGET, as_of=W2)
    raw = build_player_states(ps, ts, PLAYERS, as_of=W2, injuries=injuries,
                              history_time_col=EVENT_CHRONOLOGY_COL)
    assert raw[HOME_WR].active is True
    # A genuine but 2026 receipt is no evidence for a 2024 cutoff.
    late = pl.DataFrame([_injury("Out", available_at=IMPORTED_2026, estimated=False)])
    assert _player_states_with(injuries=late)[HOME_WR].active is True


def test_hw7_estimated_roster_observation_not_silently_eligible() -> None:
    rookie = "hw:player:rookie"
    roster = pl.DataFrame(
        [
            _roster(1, player=rookie, available_at=W2 - timedelta(days=1), estimated=True),
            _roster(3, player=HOME_WR, available_at=W2 - timedelta(days=1), estimated=True),
        ]
    )
    assert certify_pregame_observations(roster, as_of=W2).is_empty()
    states = _player_states_with(roster=roster)
    assert rookie not in states
    assert states[HOME_WR].depth is None


def test_hw8_game_odds_without_genuine_pregame_receipt_not_eligible() -> None:
    odds = pl.DataFrame(
        [
            {"canonical_game_id": G_TARGET, "vendor": "est", "spread_home_value": -3.0,
             "total_value": 44.0, "available_at": W2 - timedelta(days=3),
             "available_at_is_estimated": True,
             "collector_received_at": W2 - timedelta(days=3)},
            {"canonical_game_id": G_TARGET, "vendor": "late", "spread_home_value": -7.0,
             "total_value": 51.0, "available_at": IMPORTED_2026,
             "available_at_is_estimated": False, "collector_received_at": IMPORTED_2026},
            {"canonical_game_id": G_TARGET, "vendor": "noreceipt", "spread_home_value": -1.0,
             "total_value": 40.0, "available_at": W2 - timedelta(days=3),
             "available_at_is_estimated": False, "collector_received_at": None},
        ],
        schema_overrides={"collector_received_at": pl.Datetime("us", "UTC")},
    )
    eligible = certify_pregame_observations(odds, as_of=W2, time_col="collector_received_at")
    assert eligible.is_empty()
    market = game_market_consensus(eligible, G_TARGET, as_of=W2)
    assert market.home_spread is None and market.total is None


def test_hw9_genuine_historical_pregame_observation_eligible_when_known() -> None:
    injuries = pl.DataFrame(
        [_injury("Out", available_at=W2 - timedelta(days=1), estimated=False)]
    )
    assert classify_pregame_rows(injuries).to_list() == [
        ChronologyEvidence.CERTIFIED_LIVE_PIT.value
    ]
    assert _player_states_with(injuries=injuries)[HOME_WR].active is False
    after = pl.DataFrame([_injury("Out", available_at=W2 + timedelta(minutes=1), estimated=False)])
    assert _player_states_with(injuries=after)[HOME_WR].active is True
    odds = pl.DataFrame(
        [{"canonical_game_id": G_TARGET, "vendor": "book", "spread_home_value": -3.0,
          "total_value": 44.0, "available_at": W2 - timedelta(hours=2),
          "available_at_is_estimated": False, "collector_received_at": W2 - timedelta(hours=2)}]
    )
    eligible = certify_pregame_observations(odds, as_of=W2, time_col="collector_received_at")
    assert game_market_consensus(eligible, G_TARGET, as_of=W2).total == 44.0


# ---------------------------------------------------------- historical replay


def _warehouse(root: Path, *, target_scale: int = 1, w1_scale: int = 1) -> Warehouse:
    warehouse = Warehouse(root / "wh")
    team_stats, player_stats = _stats(target_scale=target_scale, w1_scale=w1_scale)
    warehouse.write("games", _games())
    warehouse.write("team_game_stats", team_stats)
    warehouse.write("player_game_stats", player_stats)
    warehouse.write("players", PLAYERS)
    # As in the real warehouse: game odds exist, but only as 2026 receipts.
    warehouse.write(
        "game_odds_snapshots",
        pl.DataFrame(
            [{"canonical_game_id": G_TARGET, "vendor": "book", "spread_home_value": -9.5,
              "total_value": 58.5, "available_at": IMPORTED_2026,
              "available_at_is_estimated": False, "collector_received_at": IMPORTED_2026}]
        ),
    )
    return warehouse


def _replay(warehouse: Warehouse, game_id: str = G_TARGET) -> LabeledGame:
    games = list_final_games(warehouse, season_min=2024, season_max=2024)
    row = games.filter(pl.col("canonical_game_id") == game_id).row(0, named=True)
    result = build_labeled_game(warehouse, row, model_version="hw-v1", n_draws=200)
    assert isinstance(result, LabeledGame)
    return result


def test_historical_replay_is_explicitly_hwf() -> None:
    assert EVIDENCE_MODE is EvidenceMode.HISTORICAL_WALK_FORWARD


def test_hw10_player_prop_markets_never_reach_the_fundamental_model(tmp_path: Path) -> None:
    without = _replay(_warehouse(tmp_path / "a"))
    warehouse = _warehouse(tmp_path / "b")
    props = pl.DataFrame(
        [{"canonical_game_id": G_TARGET, "canonical_player_id": HOME_WR, "vendor": "book",
          "prop_type": "receiving_yards", "line_value": 250.5, "available_at": W2,
          "collector_received_at": W2 - timedelta(hours=1)}]
    )
    warehouse.write("player_prop_snapshots", props)
    warehouse.write("player_prop_openings", props)
    with_props = _replay(warehouse)
    assert with_props.simulation.real_player_draws().equals(without.simulation.real_player_draws())

    read: list[str] = []

    class Spy:
        def __init__(self, inner: Warehouse) -> None:
            self.inner = inner

        def read(self, table: str) -> pl.DataFrame:
            read.append(table)
            return self.inner.read(table)

        def exists(self, table: str) -> bool:
            return self.inner.exists(table)

    load_warehouse_tables(Spy(warehouse))  # type: ignore[arg-type]
    assert not any("prop" in t for t in read)
    assert set(read) <= set(HISTORICAL_REPLAY_INPUT_SURFACE)


def test_every_replay_input_is_classified() -> None:
    assert len(WarehouseTables.__dataclass_fields__) == len(HISTORICAL_REPLAY_INPUT_SURFACE)
    assert not any("prop" in name for name in HISTORICAL_REPLAY_INPUT_SURFACE)
    for audit in HISTORICAL_REPLAY_INPUT_SURFACE.values():
        unsafe = audit.certification is InputCertification.UNSAFE_OR_UNPROVABLE_HISTORICAL_PREGAME
        assert unsafe is not audit.historical_allowed


def test_hw11_fold_label_cannot_affect_its_own_prediction(tmp_path: Path) -> None:
    base = _replay(_warehouse(tmp_path / "a"))
    perturbed = _replay(_warehouse(tmp_path / "b", target_scale=7))
    assert base.labels != perturbed.labels  # the label really changed
    assert base.simulation.real_player_draws().equals(perturbed.simulation.real_player_draws())


def test_hw12_completed_outcome_feeds_the_next_slate(tmp_path: Path) -> None:
    chronology = build_slate_chronology(_games())
    w1_label_at = chronology.label_available_at(G_W1)
    assert W1 < w1_label_at < W2_THU  # after its own slate, before every later cutoff
    base = _replay(_warehouse(tmp_path / "a"))
    perturbed = _replay(_warehouse(tmp_path / "b", w1_scale=7))
    assert not base.simulation.real_player_draws().equals(
        perturbed.simulation.real_player_draws()
    )
    # A slate with no successor in the schedule is scoreable, never trainable.
    games = _games().filter(pl.col("canonical_game_id") != G_SCHEDULED)
    assert build_slate_chronology(games).label_available_at(G_FUTURE) == LABEL_NEVER_TRAINABLE_AT


# ------------------------------------------------------- calibration chronology


def _season_games(chronology_games: pl.DataFrame) -> tuple[list[LabeledGame], dict[str, int]]:
    chronology = build_slate_chronology(chronology_games)
    final = chronology_games.filter(pl.col("status_state") == "final")
    games = [
        _labeled_game(
            game_id=row["canonical_game_id"], as_of=row["date"],
            outcome_available_at=chronology.label_available_at(row["canonical_game_id"]),
        )
        for row in final.iter_rows(named=True)
    ]
    return games, {r["canonical_game_id"]: r["season"] for r in final.iter_rows(named=True)}


def _three_season_schedule() -> pl.DataFrame:
    rows = []
    for season in (2022, 2023, 2024):
        start = datetime(season, 9, 8, 17, 0, tzinfo=UTC)
        for week in (1, 2, 3):
            rows.append(_game(f"cal:{season}:w{week}", season, week,
                              start + timedelta(days=7 * (week - 1))))
    rows.append(_game("cal:2025:w1", 2025, 1, datetime(2025, 9, 7, 17, 0, tzinfo=UTC),
                      status="scheduled"))
    return pl.DataFrame(rows)


def _relabel(game: LabeledGame, factor: float) -> LabeledGame:
    return LabeledGame(
        game_id=game.game_id, simulation=game.simulation, as_of=game.as_of,
        outcome_available_at=game.outcome_available_at,
        injury_data_available=game.injury_data_available,
        labels=tuple(PropLabel(lab.player_id, lab.prop_type, lab.observed_value * factor + 3)
                     for lab in game.labels),
    )


def test_hw13_fold_calibrator_fits_on_prior_folds_only() -> None:
    games, season_by_id = _season_games(_three_season_schedule())
    folds = build_season_boundary_folds(tuple(games), season_by_id)
    results = run_walk_forward_challenger(games, folds)
    for result in results:
        score_season = season_by_id[result.scoring_game_ids[0]]
        assert {season_by_id[g] for g in result.scoring_game_ids} == {score_season}
        # Every strictly-prior-season game -- incl. the last slate before
        # the boundary -- and nothing from the scored or later seasons.
        assert set(result.training_game_ids) == {
            g for g, s in season_by_id.items() if s < score_season
        }
        assert not set(result.training_game_ids) & set(result.scoring_game_ids)
    # The fold's own labels cannot fit its calibrator.
    first_score = set(results[0].scoring_game_ids)
    relabeled = [_relabel(g, 3.0) if g.game_id in first_score else g for g in games]
    again = run_walk_forward_challenger(relabeled, folds)
    assert again[0].fit.theta == results[0].fit.theta
    assert again[1].fit.theta != results[1].fit.theta  # ...but they may fit the NEXT fold's


def test_hw14_future_folds_cannot_affect_earlier_calibration() -> None:
    games, season_by_id = _season_games(_three_season_schedule())
    folds = build_season_boundary_folds(tuple(games), season_by_id)
    results = run_walk_forward_challenger(games, folds)
    relabeled = [_relabel(g, 5.0) if season_by_id[g.game_id] == 2024 else g for g in games]
    again = run_walk_forward_challenger(relabeled, folds)
    assert again[0].fit.theta == results[0].fit.theta
    assert again[1].fit.theta == results[1].fit.theta  # 2024 is only ever scored


# ------------------------------------------------------------------- LIVE_PIT


def test_hw15_live_pit_remains_genuine_receipt_based() -> None:
    as_of = datetime(2026, 9, 20, 17, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        [
            {"canonical_game_id": "g1", "available_at": as_of - timedelta(days=1),
             "available_at_is_estimated": False},
            {"canonical_game_id": "g2", "available_at": as_of - timedelta(days=1),
             "available_at_is_estimated": True},
            {"canonical_game_id": "g3", "available_at": as_of + timedelta(seconds=1),
             "available_at_is_estimated": False},
        ]
    )
    assert filter_pit(frame, as_of)["canonical_game_id"].to_list() == ["g1"]
    assert filter_pit(frame, as_of, strict=False)["canonical_game_id"].to_list() == ["g1", "g2"]
    # Event chronology is never an implicit fallback for live state.
    live_like = frame.with_columns(pl.lit(as_of - timedelta(days=30)).alias(EVENT_CHRONOLOGY_COL))
    assert filter_pit(live_like, as_of)["canonical_game_id"].to_list() == ["g1"]
    assert quote_time_source("live") == "collector_received_at"
    assert quote_time_source("opening") == "collector_received_at"
    # Default state builders are LIVE_PIT: 2026-imported history is invisible.
    team_stats, player_stats = _stats()
    assert build_player_states(player_stats, team_stats, PLAYERS, as_of=W2) == {}
    assert build_team_states(team_stats, player_stats, as_of=W2) == {}
