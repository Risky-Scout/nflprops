"""PHASE 8D: the canonical `player_game_threshold_events` product is
produced inside the official checkpoint execution path -- from the ONE
coherent `GameSimulationResult` that also feeds Phase-7 projections and
current sportsbook pricing, persisted AFTER the complete Phase-7
projections and BEFORE any current-market price, and never conditioned on
pricing success.

Reuses the Phase-7D checkpoint fixture/helpers (`_build_warehouse`,
`_execute`, `_ctx`, `_claim_run`, ...) -- same real
`game_checkpoint_flow` / `_run_game_checkpoint_task` path, no
reimplementation.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _fixtures import (
    AWAY_PLAYER_ID,
    AWAY_TEAM_ID,
    HOME_PLAYER_ID,
    HOME_TEAM_ID,
)
from test_phase7d_checkpoint_projection_integration import (
    AS_OF,
    AWAY_BACKUP_ID,
    KICKOFF,
    N_DRAWS,
    SEASON,
    WEEK,
    _build_warehouse,
    _claim_run,
    _ctx,
    _execute,
    _games_only_warehouse,
    _projections,
)

import nflprops.pipelines.pregame as pregame_module
from nflprops.data.warehouse import Warehouse
from nflprops.domain.enums import PropType
from nflprops.orchestration.flows import checkpoints as checkpoints_flow
from nflprops.orchestration.flows.checkpoints import _run_game_checkpoint_task
from nflprops.orchestration.run_store import PredictionRunStatus, PublicationStatus
from nflprops.orchestration.threshold_event_store import (
    PLAYER_GAME_THRESHOLD_EVENTS_TABLE,
    compute_threshold_event_id,
)
from nflprops.simulation.props import prop_values
from nflprops.simulation.results import player_distribution
from nflprops.thresholds import load_threshold_catalog

CATALOG = load_threshold_catalog()
EVENT_COUNT = CATALOG.event_count  # 131


def _raises(*_a, **_k):
    raise RuntimeError("injected failure")


def _thresholds(warehouse: Warehouse, *, run_id: str | None = None) -> pl.DataFrame:
    frame = warehouse.read(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)
    if run_id is not None and not frame.is_empty():
        frame = frame.filter(pl.col("run_id") == run_id)
    return frame


# --------------------------------------------------------------- full roster fixture


def _full_roster_warehouse(tmp_path: Path, **kw) -> tuple[Warehouse, set[str]]:
    """The Phase-7D WR fixture plus a fuller multi-position roster:
    starting + backup QB, starting + backup K, RBs, and a bench WR.

    The backup QB, backup K and bench WR each carry a real prior-game row
    AND an ``injury_snapshots`` status of ``out`` -- so the certified
    `build_player_states` marks them ``active=False`` and the certified
    `eligible_player_states` rule excludes them. (Share-based exclusion of
    an *active* zero-opportunity player through the empirical-Bayes state
    fit is certified separately with hand-built states in
    `tests/projections/test_eligibility.py`; the eligibility function
    itself is reused here unchanged.) Returns the warehouse and the
    player_ids Phase-7 eligibility should accept.
    """
    warehouse = _build_warehouse(tmp_path, **kw)  # HOME_WR, AWAY_WR, AWAY_BACKUP_WR

    ps = warehouse.read("player_game_stats")
    template = {col: ps[col][0] for col in ps.columns}
    hist_available_at = ps["available_at"][0]

    def _p(player_id, team_id, **stats):
        row = dict(template)
        row["canonical_player_id"] = player_id
        row["canonical_team_id"] = team_id
        for field in (
            "receiving_targets",
            "receiving_touchdowns",
            "rushing_touchdowns",
            "rushing_attempts",
            "rushing_yards",
            "receptions",
            "receiving_yards",
            "passing_attempts",
            "passing_completions",
            "passing_interceptions",
            "field_goal_attempts",
            "field_goals_made",
        ):
            row[field] = 0
        row.update(stats)
        return row

    extra = [
        _p("h:qb:starter", HOME_TEAM_ID, passing_attempts=34, passing_completions=22,
           passing_interceptions=1),
        _p("h:qb:backup", HOME_TEAM_ID, passing_attempts=6, passing_completions=4),
        _p("h:rb:1", HOME_TEAM_ID, rushing_attempts=16, rushing_yards=64,
           receiving_targets=2, receptions=2, receiving_yards=15),
        _p("h:k1:starter", HOME_TEAM_ID, field_goal_attempts=3, field_goals_made=2),
        _p("h:k2:backup", HOME_TEAM_ID, field_goal_attempts=1, field_goals_made=1),
        _p("h:wr:bench", HOME_TEAM_ID, receiving_targets=1, receptions=1),
        _p("a:qb:starter", AWAY_TEAM_ID, passing_attempts=31, passing_completions=20,
           passing_interceptions=1),
        _p("a:rb:1", AWAY_TEAM_ID, rushing_attempts=14, rushing_yards=55,
           receiving_targets=1, receptions=1, receiving_yards=8),
        _p("a:k1:starter", AWAY_TEAM_ID, field_goal_attempts=2, field_goals_made=2),
    ]
    warehouse.write(
        "player_game_stats",
        pl.concat([ps, pl.DataFrame(extra)], how="diagonal_relaxed"),
    )

    players = warehouse.read("players")
    pos = {
        "h:qb:starter": "QB", "h:qb:backup": "QB", "h:rb:1": "RB",
        "h:k1:starter": "K", "h:k2:backup": "K", "h:wr:bench": "WR",
        "a:qb:starter": "QB", "a:rb:1": "RB", "a:k1:starter": "K",
    }
    warehouse.write(
        "players",
        pl.concat(
            [players, pl.DataFrame(
                [{"canonical_player_id": p, "position_group": g} for p, g in pos.items()]
            )],
            how="diagonal_relaxed",
        ),
    )

    out_players = ("h:qb:backup", "h:k2:backup", "h:wr:bench")
    warehouse.write(
        "injury_snapshots",
        pl.DataFrame(
            [
                {
                    "canonical_player_id": p,
                    "available_at": hist_available_at,
                    "status": "out",
                }
                for p in out_players
            ]
        ),
    )

    expected_eligible = {
        HOME_PLAYER_ID, AWAY_PLAYER_ID, AWAY_BACKUP_ID,
        "h:qb:starter", "h:rb:1", "h:k1:starter",
        "a:qb:starter", "a:rb:1", "a:k1:starter",
    }
    return warehouse, expected_eligible


# --------------------------------------------------------------- happy path


def test_checkpoint_persists_projections_thresholds_and_pricing_and_publishes(
    tmp_path: Path,
) -> None:
    warehouse = _build_warehouse(tmp_path)
    record = _execute(warehouse, run_id="p8d-happy")

    assert record.status is PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.PUBLISHED

    proj = _projections(warehouse, run_id="p8d-happy")
    thr = _thresholds(warehouse, run_id="p8d-happy")
    e = proj["player_id"].n_unique()
    assert e == 3
    assert proj.height == e * 30
    assert thr.height == e * EVENT_COUNT == 3 * 131 == 393
    assert set(thr["player_id"].to_list()) == set(proj["player_id"].to_list())
    assert thr["event_type"].unique().to_list() == ["AT_LEAST"]
    assert thr["n_draws"].unique().to_list() == [N_DRAWS]
    assert thr["catalog_version"].unique().to_list() == [CATALOG.version]
    # every eligible player carries the full 131 canonical events
    per_player = thr.group_by("player_id").len().sort("player_id")
    assert per_player["len"].to_list() == [131, 131, 131]
    canonical = {(s, "AT_LEAST", t) for s, t in CATALOG.iter_events()}
    for pid in set(thr["player_id"].to_list()):
        got = set(zip(
            thr.filter(pl.col("player_id") == pid)["stat_name"].to_list(),
            thr.filter(pl.col("player_id") == pid)["event_type"].to_list(),
            thr.filter(pl.col("player_id") == pid)["threshold"].to_list(),
            strict=True,
        ))
        assert got == canonical
    # not present anywhere: p_miss / odds / EV
    for forbidden in ("p_miss", "american_odds", "p_push", "ev_per_unit", "vendor"):
        assert forbidden not in thr.columns
    # deterministic id
    for r in thr.iter_rows(named=True):
        assert r["threshold_event_id"] == compute_threshold_event_id(
            run_id="p8d-happy", player_id=r["player_id"], stat_name=r["stat_name"],
            event_type=r["event_type"], threshold=r["threshold"],
        )


def test_full_roster_eligibility_and_e_times_131(tmp_path: Path) -> None:
    warehouse, expected = _full_roster_warehouse(tmp_path)
    record = _execute(warehouse, run_id="p8d-roster")
    assert record.status is PredictionRunStatus.SUCCESS

    proj = _projections(warehouse, run_id="p8d-roster")
    thr = _thresholds(warehouse, run_id="p8d-roster")

    eligible = set(proj["player_id"].to_list())
    assert eligible == expected
    e = len(eligible)
    assert e == 9
    assert proj.height == e * 30 == 270
    assert thr.height == e * 131 == 1179
    # inclusions / exclusions
    assert {"h:qb:starter", "h:k1:starter", "a:qb:starter", "a:k1:starter"} <= eligible
    assert {"h:qb:backup", "h:k2:backup", "h:wr:bench"}.isdisjoint(eligible)
    # every eligible player -- incl. ones with all-zero stat vectors -- has 131
    assert thr.group_by("player_id").len()["len"].to_list() == [131] * 9
    # a bench-adjacent eligible player still gets its all-zero events as rows
    kicker_pass = thr.filter(
        (pl.col("player_id") == "a:k1:starter") & (pl.col("stat_name") == "passing_yards")
    )
    assert kicker_pass.height == 10
    assert kicker_pass["p_hit"].to_list() == [0.0] * 10


# --------------------------------------------------------------- one simulation


def test_exactly_one_simulation_feeds_projections_thresholds_and_pricing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _build_warehouse(tmp_path, extra_prop_vendors=("book2", "book3"))

    sim_calls = {"n": 0}
    real_simulate_game = pregame_module.simulate_game

    def _sim_spy(game, config=None):
        sim_calls["n"] += 1
        return real_simulate_game(game, config)

    monkeypatch.setattr(pregame_module, "simulate_game", _sim_spy)

    seen: dict[str, int] = {}
    real_proj = checkpoints_flow.build_player_game_projections
    real_thr = checkpoints_flow.build_player_game_threshold_events
    real_price = pregame_module.price_current_markets

    def _proj_spy(simulation, *, player_states):
        seen["proj_sim"] = id(simulation)
        return real_proj(simulation, player_states=player_states)

    def _thr_spy(simulation, *, player_states, catalog=None):
        seen["thr_sim"] = id(simulation)
        return real_thr(simulation, player_states=player_states, catalog=catalog)

    def _price_spy(game, result, quotes, **kwargs):
        seen["price_result"] = id(result)
        return real_price(game, result, quotes, **kwargs)

    monkeypatch.setattr(checkpoints_flow, "build_player_game_projections", _proj_spy)
    monkeypatch.setattr(checkpoints_flow, "build_player_game_threshold_events", _thr_spy)
    monkeypatch.setattr(pregame_module, "price_current_markets", _price_spy)

    record = _execute(warehouse, run_id="p8d-onesim")
    assert record.status is PredictionRunStatus.SUCCESS

    assert sim_calls["n"] == 1
    assert seen["proj_sim"] == seen["thr_sim"] == seen["price_result"]
    # pricing still consumed every book
    assert set(warehouse.read("predictions")["vendor"].to_list()) == {
        "fakebook", "book2", "book3",
    }


# ----------------------------------------------------- same-draw numerical proof


def test_threshold_p_hit_matches_shared_simulation_vector_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _build_warehouse(tmp_path)

    captured: dict[str, object] = {}
    real_thr = checkpoints_flow.build_player_game_threshold_events

    def _thr_spy(simulation, *, player_states, catalog=None):
        captured["simulation"] = simulation
        return real_thr(simulation, player_states=player_states, catalog=catalog)

    real_price = pregame_module.price_current_markets

    def _price_spy(game, result, quotes, **kwargs):
        captured["price_result"] = result
        return real_price(game, result, quotes, **kwargs)

    monkeypatch.setattr(checkpoints_flow, "build_player_game_threshold_events", _thr_spy)
    monkeypatch.setattr(pregame_module, "price_current_markets", _price_spy)

    _execute(warehouse, run_id="p8d-samedraw")

    sim = captured["simulation"]
    assert captured["price_result"] is sim  # pricing got the same object

    pid, stat, threshold = HOME_PLAYER_ID, "receiving_yards", 60
    vec = np.asarray(player_distribution(sim, pid, stat))
    expected_p = float(np.count_nonzero(vec >= threshold) / sim.n_draws)

    row = _thresholds(warehouse, run_id="p8d-samedraw").filter(
        (pl.col("player_id") == pid)
        & (pl.col("stat_name") == stat)
        & (pl.col("threshold") == threshold)
    )
    assert row.height == 1
    assert row["p_hit"][0] == expected_p  # exact, no tolerance

    # current pricing reads that stat via prop_values off the SAME vector
    priced_vec = np.asarray(prop_values(sim, pid, PropType.RECEIVING_YARDS))
    assert np.array_equal(priced_vec, vec)


# --------------------------------------------------------------- zero quote


def test_zero_quote_run_persists_projections_and_thresholds_model_only(
    tmp_path: Path,
) -> None:
    warehouse = _build_warehouse(
        tmp_path,
        quote_visible_at=AS_OF + timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(minutes=5),
    )
    record = _execute(warehouse, run_id="p8d-zeroquote")

    assert record.status is PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.MODEL_ONLY

    proj = _projections(warehouse, run_id="p8d-zeroquote")
    thr = _thresholds(warehouse, run_id="p8d-zeroquote")
    e = proj["player_id"].n_unique()
    assert proj.height == e * 30
    assert thr.height == e * 131
    assert warehouse.read("predictions").is_empty()  # zero prices, not a failure


def test_model_only_requires_both_complete_model_artifacts(tmp_path: Path) -> None:
    """MODEL_ONLY (zero quotes) has BOTH complete canonical artifacts;
    a run with no usable model has NEITHER and is never MODEL_ONLY."""
    zq = _build_warehouse(
        tmp_path / "zq",
        quote_visible_at=AS_OF + timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(minutes=5),
    )
    rec_a = _execute(zq, run_id="p8d-mo-a")
    assert rec_a.publication_status is PublicationStatus.MODEL_ONLY
    ea = _projections(zq, run_id="p8d-mo-a")["player_id"].n_unique()
    assert _projections(zq, run_id="p8d-mo-a").height == ea * 30
    assert _thresholds(zq, run_id="p8d-mo-a").height == ea * 131

    none = _games_only_warehouse(tmp_path / "none")
    rec_b = _execute(none, run_id="p8d-mo-b")
    assert rec_b.status is PredictionRunStatus.FAILED
    assert rec_b.failure_code == "GAME_NOT_MODELED"
    assert rec_b.publication_status is not PublicationStatus.MODEL_ONLY
    assert not none.exists("player_game_projections")
    assert not none.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)
    assert not none.exists("predictions")


# --------------------------------------------------------------- failure injection


def test_projection_persistence_failure_blocks_thresholds_and_pricing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _build_warehouse(tmp_path)

    thr_called = {"n": 0}
    price_called = {"n": 0}

    def _thr_spy(*_a, **_k):
        thr_called["n"] += 1

    def _price_spy(*_a, **_k):
        price_called["n"] += 1

    monkeypatch.setattr(
        checkpoints_flow, "build_player_game_threshold_events", _thr_spy
    )
    monkeypatch.setattr(pregame_module, "price_current_markets", _price_spy)
    monkeypatch.setattr(
        checkpoints_flow, "persist_player_game_projections", _raises
    )

    record = _execute(warehouse, run_id="p8d-projfail")
    assert record.status is PredictionRunStatus.FAILED
    assert record.publication_status is PublicationStatus.NOT_PUBLISHED
    assert thr_called["n"] == 0
    assert price_called["n"] == 0
    assert not warehouse.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)


def test_threshold_build_failure_is_partial_projections_retained_no_pricing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _build_warehouse(tmp_path)

    price_called = {"n": 0}
    real_price = pregame_module.price_current_markets

    def _count_price(*a, **k):
        price_called["n"] += 1
        return real_price(*a, **k)

    monkeypatch.setattr(pregame_module, "price_current_markets", _count_price)
    monkeypatch.setattr(
        checkpoints_flow, "build_player_game_threshold_events", _raises
    )

    record = _execute(warehouse, run_id="p8d-thrbuildfail")
    assert record.status is PredictionRunStatus.PARTIAL
    assert record.publication_status is PublicationStatus.NOT_PUBLISHED
    assert record.failure_code == "THRESHOLD_ERROR"
    # Phase-7 projections retained, no threshold artifact, pricing not run
    e = _projections(warehouse, run_id="p8d-thrbuildfail")["player_id"].n_unique()
    assert _projections(warehouse, run_id="p8d-thrbuildfail").height == e * 30
    assert not warehouse.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)
    assert price_called["n"] == 0
    assert not warehouse.exists("predictions")  # pricing never ran


def test_threshold_persistence_failure_is_partial_no_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _build_warehouse(tmp_path)

    price_called = {"n": 0}

    def _price_spy(*_a, **_k):
        price_called["n"] += 1

    monkeypatch.setattr(pregame_module, "price_current_markets", _price_spy)
    monkeypatch.setattr(
        checkpoints_flow, "persist_player_game_threshold_events", _raises
    )

    record = _execute(warehouse, run_id="p8d-thrpersistfail")
    assert record.status is PredictionRunStatus.PARTIAL
    assert record.publication_status is PublicationStatus.NOT_PUBLISHED
    e = _projections(warehouse, run_id="p8d-thrpersistfail")["player_id"].n_unique()
    assert _projections(warehouse, run_id="p8d-thrpersistfail").height == e * 30
    assert not warehouse.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)  # atomic: nothing
    assert price_called["n"] == 0


def test_retry_after_threshold_failure_completes_cleanly_without_duplication(
    tmp_path: Path,
) -> None:
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p8d-thr-retry")
    _claim_run(warehouse, run_id="p8d-thr-retry")

    real_build = checkpoints_flow.build_player_game_threshold_events
    boom = {"armed": True}

    def _maybe(*a, **k):
        if boom["armed"]:
            raise TimeoutError("transient threshold blip")
        return real_build(*a, **k)

    checkpoints_flow.build_player_game_threshold_events = _maybe
    try:
        first = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
        assert first.threshold_failed is True
        proj_after_first = _projections(warehouse, run_id="p8d-thr-retry").sort(
            "projection_id"
        )
        assert not warehouse.exists(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)

        boom["armed"] = False
        second = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    finally:
        checkpoints_flow.build_player_game_threshold_events = real_build

    assert second.threshold_failed is False
    assert second.pricing_failed is False
    proj_after_retry = _projections(warehouse, run_id="p8d-thr-retry").sort(
        "projection_id"
    )
    assert proj_after_retry.equals(proj_after_first)  # projections not duplicated
    e = proj_after_retry["player_id"].n_unique()
    assert _thresholds(warehouse, run_id="p8d-thr-retry").height == e * 131
    assert not warehouse.read("predictions").is_empty()  # pricing continued


def test_pricing_failure_after_both_artifacts_is_partial_and_retainable(
    tmp_path: Path,
) -> None:
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p8d-price-retry")
    _claim_run(warehouse, run_id="p8d-price-retry")

    real_price = pregame_module.price_current_markets
    boom = {"armed": True}

    def _maybe(*a, **k):
        if boom["armed"]:
            raise TimeoutError("transient pricing blip")
        return real_price(*a, **k)

    pregame_module.price_current_markets = _maybe
    try:
        first = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
        assert first.pricing_failed is True
        assert first.threshold_failed is False
        e = _projections(warehouse, run_id="p8d-price-retry")["player_id"].n_unique()
        proj_before = _projections(warehouse, run_id="p8d-price-retry").sort("projection_id")
        thr_before = _thresholds(warehouse, run_id="p8d-price-retry").sort("threshold_event_id")
        assert proj_before.height == e * 30
        assert thr_before.height == e * 131

        boom["armed"] = False
        second = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    finally:
        pregame_module.price_current_markets = real_price

    assert second.pricing_failed is False
    assert _projections(warehouse, run_id="p8d-price-retry").sort("projection_id").equals(
        proj_before
    )
    assert _thresholds(warehouse, run_id="p8d-price-retry").sort(
        "threshold_event_id"
    ).equals(thr_before)
    assert not warehouse.read("predictions").is_empty()


# --------------------------------------------------------------- idempotency


def test_official_retry_is_fully_idempotent_no_duplicate_threshold_rows(
    tmp_path: Path,
) -> None:
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p8d-idem")
    _claim_run(warehouse, run_id="p8d-idem")

    first = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    thr_before = _thresholds(warehouse, run_id="p8d-idem").sort("threshold_event_id")
    proj_before = _projections(warehouse, run_id="p8d-idem").sort("projection_id")
    e = proj_before["player_id"].n_unique()

    second = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    thr_after = _thresholds(warehouse, run_id="p8d-idem").sort("threshold_event_id")
    proj_after = _projections(warehouse, run_id="p8d-idem").sort("projection_id")

    assert first.threshold_rows_persisted == second.threshold_rows_persisted == e * 131
    assert thr_after.equals(thr_before)  # not 2 * E * 131, identical rows + ids
    assert proj_after.equals(proj_before)
    assert thr_after.height == e * 131
    assert thr_after["threshold_event_id"].n_unique() == e * 131
    # created_at untouched
    assert thr_after["created_at"].to_list() == thr_before["created_at"].to_list()


def test_different_created_at_on_retry_does_not_move_threshold_rows(
    tmp_path: Path,
) -> None:
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p8d-created-at")
    _claim_run(warehouse, run_id="p8d-created-at")

    _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    first = _thresholds(warehouse, run_id="p8d-created-at").sort("threshold_event_id")

    _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    second = _thresholds(warehouse, run_id="p8d-created-at").sort("threshold_event_id")
    assert second.equals(first)
    assert second["created_at"].to_list() == first["created_at"].to_list()


# --------------------------------------------------------------- catch-up / reschedule


def test_catch_up_execution_produces_identical_threshold_artifact(tmp_path: Path) -> None:
    on_time = _build_warehouse(tmp_path / "on_time")
    catch_up = _build_warehouse(tmp_path / "catch_up")

    _execute(on_time, run_id="p8d-catchup", now=AS_OF)  # exactly on schedule
    _execute(catch_up, run_id="p8d-catchup", now=AS_OF + timedelta(minutes=35))  # 19:05

    cols = [
        "run_id", "season", "week", "game_id", "player_id", "team_id",
        "position_group", "stat_name", "event_type", "threshold", "p_hit",
        "n_draws", "catalog_version",
    ]
    a = _thresholds(on_time, run_id="p8d-catchup").select(cols).sort(
        ["player_id", "stat_name", "threshold"]
    )
    b = _thresholds(catch_up, run_id="p8d-catchup").select(cols).sort(
        ["player_id", "stat_name", "threshold"]
    )
    assert a.equals(b)
    assert set(
        _thresholds(on_time, run_id="p8d-catchup")["threshold_event_id"].to_list()
    ) == set(
        _thresholds(catch_up, run_id="p8d-catchup")["threshold_event_id"].to_list()
    )


def test_kickoff_reschedule_gives_distinct_threshold_history(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)

    old_kick = KICKOFF
    old_sched = old_kick - timedelta(minutes=30)
    _execute(
        warehouse, run_id="p8d-old-kick", scheduled_as_of=old_sched, kickoff_at=old_kick,
        now=old_sched + timedelta(minutes=1),
    )
    old_ids = set(_thresholds(warehouse, run_id="p8d-old-kick")["threshold_event_id"].to_list())
    old_rows = _thresholds(warehouse, run_id="p8d-old-kick").sort("threshold_event_id")

    new_kick = KICKOFF + timedelta(hours=3)
    new_sched = new_kick - timedelta(minutes=30)
    _execute(
        warehouse, run_id="p8d-new-kick", scheduled_as_of=new_sched, kickoff_at=new_kick,
        now=new_sched + timedelta(minutes=1),
    )
    new_ids = set(_thresholds(warehouse, run_id="p8d-new-kick")["threshold_event_id"].to_list())

    assert old_ids.isdisjoint(new_ids)  # distinct parent run -> distinct ids
    # old threshold rows untouched, still attached to their own parent run
    still_old = _thresholds(warehouse, run_id="p8d-old-kick").sort("threshold_event_id")
    assert still_old.equals(old_rows)
    all_runs = set(warehouse.read(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)["run_id"].to_list())
    assert {"p8d-old-kick", "p8d-new-kick"} <= all_runs


# --------------------------------------------------------------- book independence


def test_bet365_absence_and_book_order_do_not_change_threshold_artifact(
    tmp_path: Path,
) -> None:
    with_book = _build_warehouse(tmp_path / "with", extra_prop_vendors=("bet365", "dk"))
    without_book = _build_warehouse(tmp_path / "without")  # zero extra books

    _execute(with_book, run_id="p8d-book")
    _execute(without_book, run_id="p8d-book")  # same run_id on purpose

    sci = [
        "run_id", "season", "week", "game_id", "player_id", "team_id",
        "position_group", "stat_name", "event_type", "threshold", "p_hit",
        "n_draws", "catalog_version",
    ]
    a = _thresholds(with_book, run_id="p8d-book").select(sci).sort(
        ["player_id", "stat_name", "threshold"]
    )
    b = _thresholds(without_book, run_id="p8d-book").select(sci).sort(
        ["player_id", "stat_name", "threshold"]
    )
    assert a.equals(b)
    assert a.height == b.height  # same row count
    assert set(a["player_id"].to_list()) == set(b["player_id"].to_list())  # same universe
    assert set(
        _thresholds(with_book, run_id="p8d-book")["threshold_event_id"].to_list()
    ) == set(
        _thresholds(without_book, run_id="p8d-book")["threshold_event_id"].to_list()
    )
    # only current pricing differs
    assert "bet365" in set(with_book.read("predictions")["vendor"].to_list())
    assert (
        not without_book.exists("predictions")
        or without_book.read("predictions").is_empty()
        or "bet365" not in set(without_book.read("predictions")["vendor"].to_list())
    )


def test_zero_books_still_full_threshold_artifact(tmp_path: Path) -> None:
    warehouse = _build_warehouse(
        tmp_path,
        quote_visible_at=AS_OF + timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(minutes=9),
    )
    record = _execute(warehouse, run_id="p8d-nobooks")
    assert record.publication_status is PublicationStatus.MODEL_ONLY
    e = _projections(warehouse, run_id="p8d-nobooks")["player_id"].n_unique()
    assert _thresholds(warehouse, run_id="p8d-nobooks").height == e * 131


# --------------------------------------------------------------- dispatcher e2e


def test_dispatcher_end_to_end_persists_threshold_artifact_for_due_checkpoints(
    tmp_path: Path,
) -> None:
    from nflprops.config import Config
    from nflprops.orchestration.checkpoints import CheckpointName
    from nflprops.orchestration.flows.checkpoints import checkpoint_dispatch_flow
    from nflprops.orchestration.run_store import get_run

    warehouse = _build_warehouse(tmp_path)
    results = checkpoint_dispatch_flow(
        warehouse=warehouse,
        config=Config(data={}),
        season=SEASON,
        week=WEEK,
        now=AS_OF,
        n_draws=N_DRAWS,
        model_version="phase7d-test",
    )
    t30m = [r for r in results if r.checkpoint_name == CheckpointName.T30M.value]
    assert len(t30m) == 1
    assert t30m[0].status is PredictionRunStatus.SUCCESS

    thr = warehouse.read(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)
    assert not thr.is_empty()
    proj = warehouse.read("player_game_projections")
    # every threshold run has a matching projections run + real parent + E*131
    for run_id in set(thr["run_id"].to_list()):
        assert get_run(warehouse, run_id) is not None
        proj_players = set(proj.filter(pl.col("run_id") == run_id)["player_id"].to_list())
        thr_players = set(thr.filter(pl.col("run_id") == run_id)["player_id"].to_list())
        assert thr_players == proj_players
        assert thr.filter(pl.col("run_id") == run_id).height == len(proj_players) * 131
