"""PHASE 6 §34/§35/§51/§52: the Phase-5 official-checkpoint dispatcher
continues to work through the Phase-6 simulation/pricing boundary --
`scheduled_as_of` (never actual execution time) drives the coherent
simulation, exactly one simulation occurs per claimed checkpoint, all
priced quotes derive from it, and `run_id`/`checkpoint_name` persist on
the resulting prediction rows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from _fixtures import TARGET_GAME_ID, build_pit_fixture_warehouse

from nflprops.config import Config
from nflprops.orchestration.checkpoints import CheckpointName
from nflprops.orchestration.flows.checkpoints import checkpoint_dispatch_flow
from nflprops.orchestration.run_store import PredictionRunStatus, PublicationStatus

SEASON = 2025
WEEK = 2


def test_official_checkpoint_uses_one_simulation_and_persists_run_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    """§51: scheduled_as_of flows into simulation; exactly one
    `simulate_game` call for the claimed checkpoint; run_id/checkpoint_name
    persist on the resulting prediction rows."""
    kickoff = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)
    as_of = kickoff - timedelta(minutes=30)  # T30M
    quote_visible_at = as_of - timedelta(seconds=1)
    quote_hidden_at = as_of + timedelta(seconds=1)

    warehouse = build_pit_fixture_warehouse(
        tmp_path, kickoff_at=kickoff, quote_visible_at=quote_visible_at, quote_hidden_at=quote_hidden_at
    )

    import nflprops.pipelines.pregame as pregame_module

    calls = {"n": 0, "as_of_seen": []}
    real_simulate_game = pregame_module.simulate_game

    def _spy(game, config=None):
        calls["n"] += 1
        calls["as_of_seen"].append(game.as_of)
        return real_simulate_game(game, config)

    monkeypatch.setattr(pregame_module, "simulate_game", _spy)

    results = checkpoint_dispatch_flow(
        warehouse=warehouse, config=Config(data={}), season=SEASON, week=WEEK, now=as_of
    )

    t30m = [r for r in results if r.checkpoint_name == CheckpointName.T30M.value]
    assert len(t30m) == 1
    record = t30m[0]
    assert record.status is PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.PUBLISHED
    assert record.scheduled_as_of == as_of

    # At this frozen `now`, every official checkpoint's scheduled_as_of is
    # already <= now (first-ever dispatch for this game), so all five are
    # legitimately due and each gets its own single simulation -- one
    # simulate_game call per checkpoint, not per game overall. The
    # T30M-specific claim under test here still got exactly one.
    assert len(results) == 5
    calls_for_t30m = [c for c in calls["as_of_seen"] if c == as_of]
    assert len(calls_for_t30m) == 1

    predictions = warehouse.read("predictions")
    game_predictions = predictions.filter(predictions["game_id"] == TARGET_GAME_ID)
    assert not game_predictions.is_empty()
    assert set(game_predictions["run_id"].to_list()) == {record.run_id}
    assert set(game_predictions["checkpoint_name"].to_list()) == {CheckpointName.T30M.value}


def test_catch_up_simulation_uses_scheduled_as_of_not_actual_execution_time(
    tmp_path: Path, monkeypatch
) -> None:
    """§52: scheduled_as_of=18:30, actual dispatch (catch-up) at 19:05.
    Football simulation must be seeded/cutoff at 18:30, never 19:05, and a
    quote/state row available only after 18:30 must not enter that
    checkpoint's simulation math."""
    kickoff = datetime(2025, 9, 15, 20, 0, 0, tzinfo=UTC)
    scheduled_as_of = kickoff - timedelta(minutes=90)  # T90M -> 18:30
    late_now = scheduled_as_of + timedelta(minutes=35)  # worker recovers at 19:05

    quote_visible_at = scheduled_as_of - timedelta(seconds=1)
    # A quote that becomes available strictly AFTER scheduled_as_of but
    # BEFORE the late catch-up execution time -- must never leak in.
    quote_hidden_at = scheduled_as_of + timedelta(minutes=10)

    warehouse = build_pit_fixture_warehouse(
        tmp_path, kickoff_at=kickoff, quote_visible_at=quote_visible_at, quote_hidden_at=quote_hidden_at
    )

    import nflprops.pipelines.pregame as pregame_module

    seen_as_of = []
    real_simulate_game = pregame_module.simulate_game

    def _spy(game, config=None):
        seen_as_of.append(game.as_of)
        return real_simulate_game(game, config)

    monkeypatch.setattr(pregame_module, "simulate_game", _spy)

    results = checkpoint_dispatch_flow(
        warehouse=warehouse, config=Config(data={}), season=SEASON, week=WEEK, now=late_now
    )

    t90m = [r for r in results if r.checkpoint_name == CheckpointName.T90M.value]
    assert len(t90m) == 1
    record = t90m[0]
    assert record.scheduled_as_of == scheduled_as_of
    assert record.flow_started_at == late_now

    # T90M's own simulation must have used scheduled_as_of (18:30), never
    # late_now (19:05) -- exactly once, regardless of how many other
    # checkpoints were also due in this same dispatch call.
    calls_for_t90m = [c for c in seen_as_of if c == scheduled_as_of]
    assert len(calls_for_t90m) == 1
    assert late_now not in seen_as_of

    predictions = warehouse.read("predictions")
    game_predictions = predictions.filter(predictions["game_id"] == TARGET_GAME_ID)
    if not game_predictions.is_empty():
        assert "hiddenbook" not in set(game_predictions["vendor"].to_list())
