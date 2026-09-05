"""§28-§32, §53: checkpoint-dispatch idempotency, catch-up, post-kickoff
missed, game-level isolation, manual-checkpoint exclusion, retry/run_id
reuse.

Games-only fixtures (no team/player history) are sufficient here: as
`test_predict_game_and_pit_wiring.py` shows, `predict_game` finds the game
and builds state (so the dispatcher reaches SUCCESS/MODEL_ONLY with empty
predictions) purely from a `games` row -- these tests are about dispatcher
bookkeeping (claiming, status, isolation), not prediction math, so the
lighter fixture keeps them fast and focused.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nflprops.config import Config
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.checkpoints import CheckpointName
from nflprops.orchestration.flows import checkpoints as checkpoints_flow
from nflprops.orchestration.flows.checkpoints import (
    checkpoint_dispatch_flow,
)
from nflprops.orchestration.run_store import (
    PredictionRunStatus,
    PublicationStatus,
    runs_for_game,
)

SEASON = 2026
WEEK = 2


def _games_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _game_row(
    *, game_id: str, kickoff_at: datetime, available_at: datetime | None = None
) -> dict:
    return {
        "canonical_game_id": game_id,
        "available_at": available_at or (kickoff_at - timedelta(days=3)),
        "date": kickoff_at,
        "season": SEASON,
        "week": WEEK,
        "home_canonical_team_id": f"{game_id}-home",
        "visitor_canonical_team_id": f"{game_id}-away",
    }


def _minimal_config() -> Config:
    return Config(data={})


def test_dispatch_called_twice_at_same_frozen_now_produces_no_duplicates(
    tmp_path: Path,
) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    kickoff = datetime(2026, 9, 13, 20, 0, 0, tzinfo=UTC)
    warehouse.write("games", _games_frame([_game_row(game_id="g1", kickoff_at=kickoff)]))

    # now = exactly T30M's scheduled_as_of. All five official checkpoints'
    # scheduled_as_of times are already <= now (none have been claimed yet,
    # this being the first call ever), so a single dispatch call legitimately
    # catches up on all five -- that IS the correct behavior. The idempotency
    # property under test is that a SECOND call at the identical frozen `now`
    # produces zero new rows, for every one of those five checkpoints.
    now = kickoff - timedelta(minutes=30)

    first = checkpoint_dispatch_flow(
        warehouse=warehouse, config=_minimal_config(), season=SEASON, week=WEEK, now=now
    )
    second = checkpoint_dispatch_flow(
        warehouse=warehouse, config=_minimal_config(), season=SEASON, week=WEEK, now=now
    )

    assert len(first) == 5
    assert len(second) == 0  # nothing new due -- already claimed by the first call

    rows = runs_for_game(warehouse, game_id="g1")
    assert rows.height == 5
    assert set(rows["checkpoint_name"].to_list()) == {c.value for c in CheckpointName if c != CheckpointName.MANUAL}


def test_catch_up_uses_original_scheduled_as_of_not_now(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    kickoff = datetime(2026, 9, 13, 20, 0, 0, tzinfo=UTC)
    warehouse.write("games", _games_frame([_game_row(game_id="g1", kickoff_at=kickoff)]))

    # T90M scheduled_as_of = 18:30; worker recovers late at 19:05, still < kickoff.
    late_now = datetime(2026, 9, 13, 19, 5, 0, tzinfo=UTC)
    expected_scheduled_as_of = kickoff - timedelta(minutes=90)  # 18:30

    results = checkpoint_dispatch_flow(
        warehouse=warehouse, config=_minimal_config(), season=SEASON, week=WEEK, now=late_now
    )

    t90m = [r for r in results if r.checkpoint_name == CheckpointName.T90M.value]
    assert len(t90m) == 1
    record = t90m[0]
    assert record.scheduled_as_of == expected_scheduled_as_of
    assert record.flow_started_at == late_now
    assert record.status is PredictionRunStatus.SUCCESS


def test_post_kickoff_discovery_is_checkpoint_missed_without_executing_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    kickoff = datetime(2026, 9, 13, 20, 0, 0, tzinfo=UTC)
    warehouse.write("games", _games_frame([_game_row(game_id="g1", kickoff_at=kickoff)]))

    called = {"n": 0}

    def _spy(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("must never be called for a post-kickoff checkpoint")

    monkeypatch.setattr(checkpoints_flow, "predict_game", _spy)

    now = datetime(2026, 9, 13, 20, 1, 0, tzinfo=UTC)  # T30M never claimed, kickoff has passed
    results = checkpoint_dispatch_flow(
        warehouse=warehouse, config=_minimal_config(), season=SEASON, week=WEEK, now=now
    )

    t30m = [r for r in results if r.checkpoint_name == CheckpointName.T30M.value]
    assert len(t30m) == 1
    record = t30m[0]
    assert record.status is PredictionRunStatus.FAILED
    assert record.publication_status is PublicationStatus.NOT_PUBLISHED
    assert record.failure_code == "CHECKPOINT_MISSED"
    assert called["n"] == 0


def test_manual_checkpoint_is_never_produced_by_dispatch(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    kickoff = datetime(2026, 9, 13, 20, 0, 0, tzinfo=UTC)
    warehouse.write("games", _games_frame([_game_row(game_id="g1", kickoff_at=kickoff)]))

    now = kickoff - timedelta(minutes=1)  # inside every remaining window
    results = checkpoint_dispatch_flow(
        warehouse=warehouse, config=_minimal_config(), season=SEASON, week=WEEK, now=now
    )
    assert CheckpointName.MANUAL.value not in {r.checkpoint_name for r in results}


def test_game_level_isolation_one_failure_does_not_abort_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    kickoff = datetime(2026, 9, 13, 20, 0, 0, tzinfo=UTC)
    warehouse.write(
        "games",
        _games_frame(
            [
                _game_row(game_id="gA", kickoff_at=kickoff),
                _game_row(game_id="gB", kickoff_at=kickoff),
                _game_row(game_id="gC", kickoff_at=kickoff),
            ]
        ),
    )

    real_predict_game = checkpoints_flow.predict_game

    def _flaky(warehouse, *, season, week, game_id, as_of, **kwargs):
        if game_id == "gB":
            raise AssertionError("INV001 failed")
        return real_predict_game(
            warehouse, season=season, week=week, game_id=game_id, as_of=as_of, **kwargs
        )

    monkeypatch.setattr(checkpoints_flow, "predict_game", _flaky)

    now = kickoff - timedelta(minutes=90)  # T90M due for all three
    results = checkpoint_dispatch_flow(
        warehouse=warehouse, config=_minimal_config(), season=SEASON, week=WEEK, now=now
    )

    by_game = {r.game_id: r for r in results}
    assert len(by_game) == 3
    assert by_game["gA"].status is PredictionRunStatus.SUCCESS
    assert by_game["gC"].status is PredictionRunStatus.SUCCESS
    assert by_game["gB"].status is PredictionRunStatus.FAILED
    assert by_game["gB"].failure_code == "INVARIANT_VIOLATION"


def test_predict_game_retry_with_fixed_run_id_never_fabricates_a_new_identity_or_duplicate(
    tmp_path: Path,
) -> None:
    """§16: a Prefect retry re-invokes `predict_game` again with the exact
    same `ctx` (same `run_id`, same `scheduled_as_of`) -- it must never
    create a second official identity, and repeated persistence must not
    duplicate prediction rows (idempotent via the `predictions` table's
    existing `prediction_id` natural key). Uses the real non-empty fixture
    so there are actual prediction rows whose `run_id` column can be
    checked, not just an empty frame.
    """
    from _fixtures import TARGET_GAME_ID, build_pit_fixture_warehouse

    warehouse = build_pit_fixture_warehouse(
        tmp_path,
        kickoff_at=datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC),
        quote_visible_at=datetime(2025, 9, 15, 11, 59, 59, tzinfo=UTC),
        quote_hidden_at=datetime(2025, 9, 15, 12, 0, 1, tzinfo=UTC),
    )
    as_of = datetime(2025, 9, 15, 12, 0, 0, tzinfo=UTC)
    fixed_run_id = "fixed-run-id-simulating-a-retry"

    from nflprops.pipelines.pregame import predict_game

    first_attempt = predict_game(
        warehouse,
        season=2025,
        week=2,
        game_id=TARGET_GAME_ID,
        as_of=as_of,
        persist=True,
        official_run_id=fixed_run_id,
        checkpoint_name=CheckpointName.T30M.value,
    )
    second_attempt = predict_game(
        warehouse,
        season=2025,
        week=2,
        game_id=TARGET_GAME_ID,
        as_of=as_of,
        persist=True,
        official_run_id=fixed_run_id,
        checkpoint_name=CheckpointName.T30M.value,
    )

    assert not first_attempt.is_empty()
    assert not second_attempt.is_empty()
    assert set(first_attempt["run_id"].to_list()) == {fixed_run_id}
    assert set(second_attempt["run_id"].to_list()) == {fixed_run_id}

    persisted = warehouse.read("predictions")
    # Persisting the "retried" attempt again does not duplicate rows --
    # prediction_id-keyed append-only dedup absorbs the identical retry.
    assert persisted.height == first_attempt.height
