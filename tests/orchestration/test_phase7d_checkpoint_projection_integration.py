"""PHASE 7D: the approved sportsbook-independent `player_game_projections`
artifact is produced inside the existing Phase-5/6 official-checkpoint
execution path -- from the ONE coherent `GameSimulationResult` that also
prices current sportsbook markets, persisted before pricing, and never
conditioned on pricing success.

All fixtures are temporary local warehouses (no live data, no BDL, no
Prefect Cloud). The checkpoint is driven through the real
`game_checkpoint_flow` / `_run_game_checkpoint_task` -- not a
reimplementation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from _fixtures import (
    AWAY_PLAYER_ID,
    AWAY_TEAM_ID,
    HOME_PLAYER_ID,
    TARGET_GAME_ID,
    build_pit_fixture_warehouse,
)

import nflprops.pipelines.pregame as pregame_module
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.checkpoints import CheckpointName
from nflprops.orchestration.flows import checkpoints as checkpoints_flow
from nflprops.orchestration.flows.checkpoints import (
    CheckpointRunContext,
    _run_game_checkpoint_task,
    game_checkpoint_flow,
)
from nflprops.orchestration.projection_store import (
    PLAYER_GAME_PROJECTIONS_TABLE,
    compute_projection_id,
)
from nflprops.orchestration.run_store import (
    PredictionRunRecord,
    PredictionRunStatus,
    PublicationStatus,
    claim_checkpoint,
    get_run,
)
from nflprops.projections import REGISTRY_SIZE

SEASON = 2025
WEEK = 2
KICKOFF = datetime(2025, 9, 15, 17, 0, 0, tzinfo=UTC)
AS_OF = KICKOFF - timedelta(minutes=30)  # T30M
N_DRAWS = 200

AWAY_BACKUP_ID = "fake:player:away-wr-backup"


# --------------------------------------------------------------------------- fixtures


def _augment_with_backup_player(warehouse: Warehouse) -> None:
    """Add one rotational/backup WR on the away team: real prior-game
    opportunity (a few targets), on the roster, active -- and never quoted.
    A direct Phase-7D product requirement (§13/§14): its full 30-row
    projection must still be produced and persisted."""
    player_stats = warehouse.read("player_game_stats")
    backup_row = {
        col: player_stats[col][1] for col in player_stats.columns
    }  # clone AWAY_PLAYER_ID's row shape
    backup_row["canonical_player_id"] = AWAY_BACKUP_ID
    backup_row["canonical_team_id"] = AWAY_TEAM_ID
    backup_row["receiving_targets"] = 3
    backup_row["receptions"] = 2
    backup_row["receiving_yards"] = 21
    backup_row["receiving_touchdowns"] = 0
    warehouse.write(
        "player_game_stats",
        pl.concat([player_stats, pl.DataFrame([backup_row])], how="diagonal_relaxed"),
    )
    players = warehouse.read("players")
    warehouse.write(
        "players",
        pl.concat(
            [players, pl.DataFrame([{"canonical_player_id": AWAY_BACKUP_ID, "position_group": "WR"}])],
            how="diagonal_relaxed",
        ),
    )


def _build_warehouse(
    tmp_path: Path,
    *,
    quote_visible_at: datetime | None = None,
    quote_hidden_at: datetime | None = None,
    with_backup: bool = True,
    extra_prop_vendors: tuple[str, ...] = (),
) -> Warehouse:
    warehouse = build_pit_fixture_warehouse(
        tmp_path,
        kickoff_at=KICKOFF,
        quote_visible_at=quote_visible_at or (AS_OF - timedelta(seconds=1)),
        quote_hidden_at=quote_hidden_at or (AS_OF + timedelta(seconds=1)),
    )
    if with_backup:
        _augment_with_backup_player(warehouse)
    for vendor in extra_prop_vendors:
        props = warehouse.read("player_prop_snapshots")
        new = {col: props[col][0] for col in props.columns}
        new["vendor"] = vendor
        new["line_value"] = 66.5
        new["available_at"] = AS_OF - timedelta(seconds=1)
        new["collector_received_at"] = AS_OF - timedelta(seconds=1)
        warehouse.write(
            "player_prop_snapshots",
            pl.concat([props, pl.DataFrame([new])], how="diagonal_relaxed"),
        )
    return warehouse


def _claim_run(
    warehouse: Warehouse,
    *,
    run_id: str,
    scheduled_as_of: datetime = AS_OF,
    kickoff_at: datetime = KICKOFF,
    game_id: str = TARGET_GAME_ID,
    n_draws: int = N_DRAWS,
    week: int = WEEK,
) -> PredictionRunRecord:
    record = PredictionRunRecord(
        run_id=run_id,
        season=SEASON,
        week=week,
        game_id=game_id,
        checkpoint_name=CheckpointName.T30M.value,
        scheduled_as_of=scheduled_as_of,
        kickoff_at=kickoff_at,
        flow_started_at=scheduled_as_of,
        flow_completed_at=None,
        status=PredictionRunStatus.SCHEDULED,
        model_version="phase7d-test",
        config_sha256=f"cfg-{run_id}",
        source_sha256="src",
        data_manifest_sha256=f"manifest-{run_id}",
        n_draws=n_draws,
        retained_joint_draws=0,
        publication_status=PublicationStatus.NOT_PUBLISHED,
        is_final_forecast=False,
        fallback_from_checkpoint=None,
        failure_code=None,
        failure_detail=None,
        created_at=scheduled_as_of,
    )
    assert claim_checkpoint(warehouse, record) is True
    return record


def _ctx(
    warehouse: Warehouse,
    *,
    run_id: str,
    scheduled_as_of: datetime = AS_OF,
    kickoff_at: datetime = KICKOFF,
    game_id: str = TARGET_GAME_ID,
    n_draws: int = N_DRAWS,
    week: int = WEEK,
) -> CheckpointRunContext:
    return CheckpointRunContext(
        warehouse=warehouse,
        season=SEASON,
        week=week,
        game_id=game_id,
        checkpoint=CheckpointName.T30M,
        kickoff_at=kickoff_at,
        scheduled_as_of=scheduled_as_of,
        run_id=run_id,
        model_version="phase7d-test",
        n_draws=n_draws,
        retain_joint_draws=0,
        max_confidence_tier=2,
        market_mode="live",
    )


def _execute(
    warehouse: Warehouse,
    *,
    run_id: str,
    now: datetime | None = None,
    scheduled_as_of: datetime = AS_OF,
    kickoff_at: datetime = KICKOFF,
    game_id: str = TARGET_GAME_ID,
    n_draws: int = N_DRAWS,
    week: int = WEEK,
) -> PredictionRunRecord:
    _claim_run(
        warehouse,
        run_id=run_id,
        scheduled_as_of=scheduled_as_of,
        kickoff_at=kickoff_at,
        game_id=game_id,
        n_draws=n_draws,
        week=week,
    )
    ctx = _ctx(
        warehouse,
        run_id=run_id,
        scheduled_as_of=scheduled_as_of,
        kickoff_at=kickoff_at,
        game_id=game_id,
        n_draws=n_draws,
        week=week,
    )
    return game_checkpoint_flow(ctx, now=now or (scheduled_as_of + timedelta(minutes=1)))


def _projections(warehouse: Warehouse, *, run_id: str | None = None) -> pl.DataFrame:
    frame = warehouse.read(PLAYER_GAME_PROJECTIONS_TABLE)
    if run_id is not None and not frame.is_empty():
        frame = frame.filter(pl.col("run_id") == run_id)
    return frame


_SCIENTIFIC_COLS = [
    "run_id", "season", "week", "game_id", "player_id", "team_id",
    "position_group", "stat_name", "n_draws", "mean",
    "p05", "p10", "p25", "p50", "p75", "p90", "p95",
]


# --------------------------------------------------------------------------- tests


def test_official_checkpoint_persists_projections_and_publishes(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    record = _execute(warehouse, run_id="p7d-happy")

    assert record.status is PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.PUBLISHED

    proj = _projections(warehouse, run_id="p7d-happy")
    eligible_players = set(proj["player_id"].to_list())
    # Player A (quoted), Player B (unquoted), Player C (unquoted backup).
    assert eligible_players == {HOME_PLAYER_ID, AWAY_PLAYER_ID, AWAY_BACKUP_ID}
    e_count = len(eligible_players)
    assert e_count == 3
    assert proj.height == e_count * 30 == e_count * REGISTRY_SIZE

    # Every eligible player carries all 30 registry stats.
    per_player = proj.group_by("player_id").len().sort("player_id")
    assert per_player["len"].to_list() == [30, 30, 30]

    # priced predictions exist only for the quoted player; projections are
    # unaffected by that.
    predictions = warehouse.read("predictions")
    assert set(predictions["player_id"].to_list()) == {HOME_PLAYER_ID}


def test_exactly_one_simulation_and_same_result_feeds_projections_and_pricing(
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
    real_build = checkpoints_flow.build_player_game_projections
    real_price = pregame_module.price_current_markets

    def _proj_spy(simulation, *, player_states):
        seen["projection_sim_id"] = id(simulation)
        return real_build(simulation, player_states=player_states)

    def _price_spy(game, result, quotes, **kwargs):
        seen["pricing_result_id"] = id(result)
        return real_price(game, result, quotes, **kwargs)

    monkeypatch.setattr(checkpoints_flow, "build_player_game_projections", _proj_spy)
    monkeypatch.setattr(pregame_module, "price_current_markets", _price_spy)

    record = _execute(warehouse, run_id="p7d-spy")

    assert record.status is PredictionRunStatus.SUCCESS
    assert sim_calls["n"] == 1  # one football simulation for the game/checkpoint
    assert seen["projection_sim_id"] == seen["pricing_result_id"]  # identical instance

    predictions = warehouse.read("predictions")
    assert set(predictions["vendor"].to_list()) == {"fakebook", "book2", "book3"}


def test_zero_quote_checkpoint_persists_projections_and_is_model_only(
    tmp_path: Path,
) -> None:
    # every quote becomes knowable strictly AFTER scheduled_as_of.
    warehouse = _build_warehouse(
        tmp_path,
        quote_visible_at=AS_OF + timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(minutes=5),
    )
    record = _execute(warehouse, run_id="p7d-zero-quote")

    assert record.status is PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.MODEL_ONLY

    proj = _projections(warehouse, run_id="p7d-zero-quote")
    assert set(proj["player_id"].to_list()) == {
        HOME_PLAYER_ID,
        AWAY_PLAYER_ID,
        AWAY_BACKUP_ID,
    }
    assert proj.height == 3 * 30
    # zero-quote is not a failure and not a pricing exception.
    assert warehouse.read("predictions").is_empty()


def test_pricing_exception_after_projection_persist_is_partial_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _build_warehouse(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic downstream pricing failure")

    monkeypatch.setattr(pregame_module, "price_current_markets", _boom)

    record = _execute(warehouse, run_id="p7d-price-boom")

    assert record.status is PredictionRunStatus.PARTIAL
    assert record.publication_status is PublicationStatus.NOT_PUBLISHED

    # Projection artifact was persisted BEFORE pricing and is untouched.
    proj = _projections(warehouse, run_id="p7d-price-boom")
    assert proj.height == 3 * 30
    assert warehouse.read("predictions").is_empty()


def test_projection_failure_prevents_pricing_and_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _build_warehouse(tmp_path)

    priced_called = {"n": 0}
    real_price = pregame_module.price_current_markets

    def _count_price(*args, **kwargs):
        priced_called["n"] += 1
        return real_price(*args, **kwargs)

    def _bad_projection_build(simulation, *, player_states):
        from nflprops.errors import ProjectionError

        raise ProjectionError("synthetic projection construction failure")

    monkeypatch.setattr(pregame_module, "price_current_markets", _count_price)
    monkeypatch.setattr(
        checkpoints_flow, "build_player_game_projections", _bad_projection_build
    )

    record = _execute(warehouse, run_id="p7d-proj-fail")

    assert record.status is PredictionRunStatus.FAILED
    assert record.publication_status is PublicationStatus.NOT_PUBLISHED
    assert priced_called["n"] == 0  # pricing never continued
    assert not warehouse.exists(PLAYER_GAME_PROJECTIONS_TABLE)


def test_incomplete_projection_product_fails_before_pricing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§22: a build that is not exactly E*30 rows must fail before pricing."""
    warehouse = _build_warehouse(tmp_path)

    real_build = checkpoints_flow.build_player_game_projections

    def _truncated(simulation, *, player_states):
        return real_build(simulation, player_states=player_states).head(29)

    priced_called = {"n": 0}
    real_price = pregame_module.price_current_markets
    monkeypatch.setattr(
        pregame_module,
        "price_current_markets",
        lambda *a, **k: (priced_called.__setitem__("n", priced_called["n"] + 1), real_price(*a, **k))[1],
    )
    monkeypatch.setattr(checkpoints_flow, "build_player_game_projections", _truncated)

    record = _execute(warehouse, run_id="p7d-incomplete")
    assert record.status is PredictionRunStatus.FAILED
    assert record.failure_code == "INVARIANT_VIOLATION"
    assert priced_called["n"] == 0


def test_unquoted_players_and_backup_get_full_thirty_rows(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    _execute(warehouse, run_id="p7d-unquoted")
    proj = _projections(warehouse, run_id="p7d-unquoted")

    for player_id in (AWAY_PLAYER_ID, AWAY_BACKUP_ID):
        rows = proj.filter(pl.col("player_id") == player_id)
        assert rows.height == 30
        assert rows["stat_name"].n_unique() == 30
        # not present in the priced table at all
    predictions = warehouse.read("predictions")
    assert AWAY_PLAYER_ID not in set(predictions["player_id"].to_list())
    assert AWAY_BACKUP_ID not in set(predictions["player_id"].to_list())


def test_bet365_absence_leaves_projections_byte_identical(tmp_path: Path) -> None:
    with_book = _build_warehouse(tmp_path / "with", extra_prop_vendors=("bet365",))
    without_book = _build_warehouse(tmp_path / "without")

    rec_with = _execute(with_book, run_id="p7d-book")
    rec_without = _execute(without_book, run_id="p7d-book")  # SAME run_id on purpose

    assert rec_with.status is PredictionRunStatus.SUCCESS
    assert rec_without.status is PredictionRunStatus.SUCCESS

    proj_with = _projections(with_book, run_id="p7d-book").select(_SCIENTIFIC_COLS).sort(
        ["player_id", "stat_name"]
    )
    proj_without = _projections(without_book, run_id="p7d-book").select(
        _SCIENTIFIC_COLS
    ).sort(["player_id", "stat_name"])
    assert proj_with.equals(proj_without)

    id_with = set(_projections(with_book, run_id="p7d-book")["projection_id"].to_list())
    id_without = set(
        _projections(without_book, run_id="p7d-book")["projection_id"].to_list()
    )
    assert id_with == id_without

    # current pricing DOES differ: bet365 row present only in the first.
    assert "bet365" in set(with_book.read("predictions")["vendor"].to_list())
    assert "bet365" not in set(without_book.read("predictions")["vendor"].to_list())


def test_official_retry_is_idempotent(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p7d-retry")
    _claim_run(warehouse, run_id="p7d-retry")

    first = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    before = _projections(warehouse, run_id="p7d-retry").sort("projection_id")
    stored_created_at = before["created_at"].unique().to_list()

    # simulate a Prefect retry of the exact same task: same ctx, same run_id.
    second = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    after = _projections(warehouse, run_id="p7d-retry").sort("projection_id")

    assert first.projection_rows_persisted == second.projection_rows_persisted == 3 * 30
    assert after.equals(before)  # no duplicate rows, identical scientific values
    assert after["created_at"].unique().to_list() == stored_created_at
    assert after.height == 3 * 30
    assert after["projection_id"].n_unique() == 3 * 30


def test_retry_after_projections_already_exist_is_a_noop_and_continues(
    tmp_path: Path,
) -> None:
    """§17: projections persisted, then a transient orchestration failure
    before run completion, then the same official run retries -> projection
    persistence is an idempotent no-op and execution proceeds to pricing."""
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p7d-retry-after")
    _claim_run(warehouse, run_id="p7d-retry-after")

    # First attempt: force a failure AFTER projections are persisted but
    # before the task returns, by making pricing raise a transient error.
    import nflprops.pipelines.pregame as pm

    original_price = pm.price_current_markets
    boom = {"armed": True}

    def _maybe_boom(*args, **kwargs):
        if boom["armed"]:
            raise TimeoutError("transient orchestration blip")
        return original_price(*args, **kwargs)

    pm.price_current_markets = _maybe_boom
    try:
        first = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
        assert first.pricing_failed is True
        assert first.projection_rows_persisted == 3 * 30
        proj_after_first = _projections(warehouse, run_id="p7d-retry-after").sort(
            "projection_id"
        )

        # Retry: projections already exist -> idempotent no-op; pricing now
        # succeeds and the task completes.
        boom["armed"] = False
        second = _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    finally:
        pm.price_current_markets = original_price

    assert second.pricing_failed is False
    assert second.projection_rows_persisted == 3 * 30
    proj_after_retry = _projections(warehouse, run_id="p7d-retry-after").sort(
        "projection_id"
    )
    assert proj_after_retry.equals(proj_after_first)  # no conflict, no duplicates
    assert not warehouse.read("predictions").is_empty()  # execution continued


def test_different_created_at_on_retry_is_harmless(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    ctx = _ctx(warehouse, run_id="p7d-created-at")
    _claim_run(warehouse, run_id="p7d-created-at")

    _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    first = _projections(warehouse, run_id="p7d-created-at").sort("projection_id")
    original_created_at = first["created_at"].to_list()

    # A retry happens later in wall-clock time; created_at of the stored
    # canonical rows must not move.
    _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)
    second = _projections(warehouse, run_id="p7d-created-at").sort("projection_id")
    assert second["created_at"].to_list() == original_created_at
    assert second.equals(first)


def test_catch_up_execution_produces_identical_projection_artifact(
    tmp_path: Path,
) -> None:
    """§18: identical PIT data through scheduled_as_of -> identical run_id,
    simulation, eligible players, 30-stat projections, projection_ids,
    whether executed on-time or as a late catch-up."""
    on_time = _build_warehouse(tmp_path / "on_time")
    catch_up = _build_warehouse(tmp_path / "catch_up")

    on_time_rec = _execute(
        on_time, run_id="p7d-catchup", now=AS_OF  # executed exactly on schedule
    )
    catch_up_rec = _execute(
        catch_up, run_id="p7d-catchup", now=AS_OF + timedelta(minutes=35)  # 19:05
    )

    assert on_time_rec.status is catch_up_rec.status is PredictionRunStatus.SUCCESS

    a = _projections(on_time, run_id="p7d-catchup").select(_SCIENTIFIC_COLS).sort(
        ["player_id", "stat_name"]
    )
    b = _projections(catch_up, run_id="p7d-catchup").select(_SCIENTIFIC_COLS).sort(
        ["player_id", "stat_name"]
    )
    assert a.equals(b)
    assert set(_projections(on_time, run_id="p7d-catchup")["projection_id"].to_list()) == set(
        _projections(catch_up, run_id="p7d-catchup")["projection_id"].to_list()
    )


def test_post_scheduled_as_of_data_is_excluded_from_projection_artifact(
    tmp_path: Path,
) -> None:
    """A prop quote and its player exist only AFTER scheduled_as_of. The
    quote must not be priced and must not change the projection product."""
    baseline = _build_warehouse(tmp_path / "baseline")
    with_late = _build_warehouse(tmp_path / "with_late")

    # Add a late-arriving quote for a NEW player on the away team, knowable
    # only after scheduled_as_of.
    props = with_late.read("player_prop_snapshots")
    late_quote = {col: props[col][0] for col in props.columns}
    late_quote["canonical_player_id"] = AWAY_PLAYER_ID
    late_quote["vendor"] = "latebook"
    late_quote["available_at"] = AS_OF + timedelta(minutes=1)
    late_quote["collector_received_at"] = AS_OF + timedelta(minutes=1)
    with_late.write(
        "player_prop_snapshots",
        pl.concat([props, pl.DataFrame([late_quote])], how="diagonal_relaxed"),
    )

    base_rec = _execute(baseline, run_id="p7d-cutoff", now=AS_OF + timedelta(minutes=10))
    late_rec = _execute(with_late, run_id="p7d-cutoff", now=AS_OF + timedelta(minutes=10))

    assert base_rec.status is late_rec.status is PredictionRunStatus.SUCCESS

    a = _projections(baseline, run_id="p7d-cutoff").select(_SCIENTIFIC_COLS).sort(
        ["player_id", "stat_name"]
    )
    b = _projections(with_late, run_id="p7d-cutoff").select(_SCIENTIFIC_COLS).sort(
        ["player_id", "stat_name"]
    )
    assert a.equals(b)
    assert "latebook" not in set(with_late.read("predictions")["vendor"].to_list())


def test_kickoff_reschedule_creates_distinct_projection_history(tmp_path: Path) -> None:
    """§19: a new kickoff -> a new run_id -> new projection_ids. The old
    run's projections remain and are never overwritten or treated as
    satisfying the new schedule revision."""
    warehouse = _build_warehouse(tmp_path)

    old_kickoff = KICKOFF
    old_scheduled = old_kickoff - timedelta(minutes=30)
    _execute(
        warehouse,
        run_id="p7d-old-kickoff",
        scheduled_as_of=old_scheduled,
        kickoff_at=old_kickoff,
        now=old_scheduled + timedelta(minutes=1),
    )
    old_proj = _projections(warehouse, run_id="p7d-old-kickoff")
    old_ids = set(old_proj["projection_id"].to_list())
    assert old_proj.height == 3 * 30

    # Kickoff moves 3 hours later; the dispatcher would compute a new T30M
    # scheduled_as_of and a new run_id for the revised schedule.
    new_kickoff = KICKOFF + timedelta(hours=3)
    new_scheduled = new_kickoff - timedelta(minutes=30)
    _execute(
        warehouse,
        run_id="p7d-new-kickoff",
        scheduled_as_of=new_scheduled,
        kickoff_at=new_kickoff,
        now=new_scheduled + timedelta(minutes=1),
    )
    new_proj = _projections(warehouse, run_id="p7d-new-kickoff")
    new_ids = set(new_proj["projection_id"].to_list())

    assert new_proj.height == 3 * 30
    assert old_ids.isdisjoint(new_ids)
    # old rows still present and unchanged
    still_old = _projections(warehouse, run_id="p7d-old-kickoff")
    assert still_old.sort("projection_id").equals(old_proj.sort("projection_id"))
    # both run histories coexist in the canonical table
    all_runs = set(warehouse.read(PLAYER_GAME_PROJECTIONS_TABLE)["run_id"].to_list())
    assert {"p7d-old-kickoff", "p7d-new-kickoff"} <= all_runs


def test_projection_parent_provenance_remains_valid(tmp_path: Path) -> None:
    warehouse = _build_warehouse(tmp_path)
    _execute(warehouse, run_id="p7d-provenance")

    parent = get_run(warehouse, "p7d-provenance")
    assert parent is not None
    proj = _projections(warehouse, run_id="p7d-provenance")
    assert set(proj["season"].to_list()) == {parent.season}
    assert set(proj["week"].to_list()) == {parent.week}
    assert set(proj["game_id"].to_list()) == {parent.game_id}
    assert set(proj["n_draws"].to_list()) == {parent.n_draws}
    # projection_id is exactly SHA256(run_id | player_id | stat_name)
    for row in proj.iter_rows(named=True):
        assert row["projection_id"] == compute_projection_id(
            run_id="p7d-provenance",
            player_id=row["player_id"],
            stat_name=row["stat_name"],
        )


def test_full_n_draws_used_for_projection_summaries(tmp_path: Path) -> None:
    """§23: projection summaries are computed from the full simulation
    n_draws, never the retained joint-draw subset."""
    warehouse = _build_warehouse(tmp_path)
    ctx = replace(_ctx(warehouse, run_id="p7d-ndraws"), n_draws=256, retain_joint_draws=8)
    _claim_run(warehouse, run_id="p7d-ndraws", n_draws=256)

    _run_game_checkpoint_task.fn(ctx, on_state_context=lambda _c: None)

    proj = _projections(warehouse, run_id="p7d-ndraws")
    assert set(proj["n_draws"].to_list()) == {256}  # full draw count, not 8

    # the retained joint-draw table, if written, only keeps the small subset
    if warehouse.exists("simulation_player_results"):
        joint = warehouse.read("simulation_player_results")
        assert joint["draw_id"].max() < 8


def test_game_level_failure_isolation_for_projection_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§26: Game A succeeds, Game B's projection persistence fails
    deliberately, Game C succeeds -- A and C artifacts survive."""
    warehouse = _build_warehouse(tmp_path)

    # Three games sharing the same two teams / player universe.
    games = warehouse.read("games")
    target = games.filter(pl.col("canonical_game_id") == TARGET_GAME_ID)
    extra = []
    for gid in ("p7d:gameB", "p7d:gameC"):
        row = {col: target[col][0] for col in target.columns}
        row["canonical_game_id"] = gid
        extra.append(row)
    warehouse.write(
        "games", pl.concat([games, pl.DataFrame(extra)], how="diagonal_relaxed")
    )
    # player/team history keyed by team id already covers all three games.

    real_persist = checkpoints_flow.persist_player_game_projections

    def _selective_persist(backend, projections, **kwargs):
        if kwargs.get("run_id") == "p7d-iso-B":
            raise RuntimeError("deliberate Game B projection persistence failure")
        return real_persist(backend, projections, **kwargs)

    monkeypatch.setattr(
        checkpoints_flow, "persist_player_game_projections", _selective_persist
    )

    rec_a = _execute(warehouse, run_id="p7d-iso-A", game_id=TARGET_GAME_ID)
    rec_b = _execute(warehouse, run_id="p7d-iso-B", game_id="p7d:gameB")
    rec_c = _execute(warehouse, run_id="p7d-iso-C", game_id="p7d:gameC")

    assert rec_a.status is PredictionRunStatus.SUCCESS
    assert rec_b.status is PredictionRunStatus.FAILED
    assert rec_c.status is PredictionRunStatus.SUCCESS

    runs_with_projections = set(
        warehouse.read(PLAYER_GAME_PROJECTIONS_TABLE)["run_id"].to_list()
    )
    assert "p7d-iso-A" in runs_with_projections
    assert "p7d-iso-C" in runs_with_projections
    assert "p7d-iso-B" not in runs_with_projections  # B's artifact never landed


def test_current_pricing_is_regression_equivalent_to_pre_7d(tmp_path: Path) -> None:
    """§24: Phase 7D adds an artifact; it must not perturb any priced value.
    `compute_game_prediction(...).price_markets()` must equal what
    `predict_game(...)` produced pre-7D, to exact equality."""
    warehouse = _build_warehouse(tmp_path, extra_prop_vendors=("book2",))

    from nflprops.pipelines.pregame import compute_game_prediction, predict_game

    legacy = predict_game(
        warehouse,
        season=SEASON,
        week=WEEK,
        game_id=TARGET_GAME_ID,
        as_of=AS_OF,
        model_version="phase7d-test",
        n_draws=N_DRAWS,
        persist=False,
    )
    computation = compute_game_prediction(
        warehouse,
        season=SEASON,
        week=WEEK,
        game_id=TARGET_GAME_ID,
        as_of=AS_OF,
        model_version="phase7d-test",
        n_draws=N_DRAWS,
    )
    assert computation is not None
    integrated = pl.DataFrame(computation.price_markets())

    compare_cols = [
        "game_id", "player_id", "prop_type", "side", "line", "vendor",
        "model_mean", "model_median", "p05", "p50", "p95",
        "p_model_raw", "p_market_fair", "edge", "ev_per_unit", "model_fair_american",
    ]
    left = legacy.select(compare_cols).sort(["player_id", "prop_type", "vendor", "side"])
    right = integrated.select(compare_cols).sort(
        ["player_id", "prop_type", "vendor", "side"]
    )
    assert left.height == right.height > 0
    for col in compare_cols:
        lv, rv = left[col].to_list(), right[col].to_list()
        if left.schema[col] in (pl.Float64, pl.Float32):
            for a, b in zip(lv, rv, strict=True):
                if a is None or b is None:
                    assert a is b
                else:
                    assert abs(a - b) <= 1e-12, (col, a, b)
        else:
            assert lv == rv, col


def test_phase5_data_manifest_and_sim_input_sha_unchanged_by_projection_persist(
    tmp_path: Path,
) -> None:
    """§20/§21: `data_manifest_sha256` (a hash of PIT INPUTS) and
    `simulation_input_sha256` are independent of the projection output
    artifact and of player-prop lines/prices/vendors."""
    from nflprops.orchestration.manifest import compute_data_manifest_sha256
    from nflprops.pipelines.pregame import compute_game_prediction

    no_book = _build_warehouse(tmp_path / "no_book")
    with_book = _build_warehouse(tmp_path / "with_book", extra_prop_vendors=("bet365",))

    # simulation_input_sha256 must not depend on player-prop quotes at all.
    comp_no = compute_game_prediction(
        no_book, season=SEASON, week=WEEK, game_id=TARGET_GAME_ID, as_of=AS_OF,
        model_version="phase7d-test", n_draws=N_DRAWS,
    )
    comp_with = compute_game_prediction(
        with_book, season=SEASON, week=WEEK, game_id=TARGET_GAME_ID, as_of=AS_OF,
        model_version="phase7d-test", n_draws=N_DRAWS,
    )
    assert comp_no is not None and comp_with is not None
    assert comp_no.simulation_input_sha256 == comp_with.simulation_input_sha256

    # data_manifest_sha256 before vs after running the checkpoint (which
    # persists player_game_projections) is unchanged: projections are an
    # output, never a manifest input.
    manifest_before = compute_data_manifest_sha256(
        no_book, game_id=TARGET_GAME_ID, scheduled_as_of=AS_OF
    )
    _execute(no_book, run_id="p7d-manifest")
    assert no_book.exists(PLAYER_GAME_PROJECTIONS_TABLE)
    manifest_after = compute_data_manifest_sha256(
        no_book, game_id=TARGET_GAME_ID, scheduled_as_of=AS_OF
    )
    assert manifest_before == manifest_after

    # zero-quote checkpoint: the player-props manifest component is a
    # deterministic empty representation, not a crash / missing key.
    zero_quote = _build_warehouse(
        tmp_path / "zero",
        quote_visible_at=AS_OF + timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(minutes=5),
    )
    m1 = compute_data_manifest_sha256(
        zero_quote, game_id=TARGET_GAME_ID, scheduled_as_of=AS_OF
    )
    m2 = compute_data_manifest_sha256(
        zero_quote, game_id=TARGET_GAME_ID, scheduled_as_of=AS_OF
    )
    assert m1 == m2


def test_dispatcher_end_to_end_persists_projections_for_due_checkpoints(
    tmp_path: Path,
) -> None:
    """The unmodified `checkpoint_dispatch_flow` (claim -> manifest ->
    state -> ONE simulation -> projections -> pricing -> terminal status)
    persists the projection product for every due checkpoint it runs."""
    from nflprops.config import Config
    from nflprops.orchestration.flows.checkpoints import checkpoint_dispatch_flow

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

    proj = warehouse.read(PLAYER_GAME_PROJECTIONS_TABLE)
    assert not proj.is_empty()
    # every persisted projection row is tied to a real prediction_runs row
    run_ids = set(proj["run_id"].to_list())
    for run_id in run_ids:
        assert get_run(warehouse, run_id) is not None
    # E*30 for each successfully modeled checkpoint run
    for _run_id, count in proj.group_by("run_id").len().iter_rows():
        assert count % 30 == 0 and count > 0


# ------------------------------------------------------- PHASE 7E certification


def _games_only_warehouse(tmp_path: Path) -> Warehouse:
    """A scheduled, PIT-visible game with NO team/player structural history:
    state context builds, but no coherent simulation (and therefore no
    projection artifact) can be produced."""
    warehouse = Warehouse(tmp_path / "wh")
    warehouse.write(
        "games",
        pl.DataFrame(
            [
                {
                    "canonical_game_id": TARGET_GAME_ID,
                    "available_at": KICKOFF - timedelta(days=3),
                    "date": KICKOFF,
                    "season": SEASON,
                    "week": WEEK,
                    "home_canonical_team_id": "no-history-home",
                    "visitor_canonical_team_id": "no-history-away",
                }
            ]
        ),
    )
    return warehouse


def test_no_model_result_is_failed_not_success_and_not_model_only(
    tmp_path: Path,
) -> None:
    """PHASE 7E §13 hard acceptance gate: a checkpoint that executes but
    produces no usable game model result (hence no projection artifact) is
    NOT SUCCESS and NOT MODEL_ONLY -- it is FAILED / NOT_PUBLISHED /
    GAME_NOT_MODELED (no new run status, no DATA_HOLD gate exists)."""
    warehouse = _games_only_warehouse(tmp_path)
    record = _execute(warehouse, run_id="p7e-no-model")

    assert record.status is PredictionRunStatus.FAILED
    assert record.status is not PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.NOT_PUBLISHED
    assert record.publication_status is not PublicationStatus.MODEL_ONLY
    assert record.failure_code == "GAME_NOT_MODELED"
    # no projection artifact was created
    assert not warehouse.exists(PLAYER_GAME_PROJECTIONS_TABLE)


def test_model_only_status_always_implies_a_complete_projection_artifact(
    tmp_path: Path,
) -> None:
    """PHASE 7E §2 certification invariant: publication_status == MODEL_ONLY
    <=> a complete, valid E*30 player_game_projections artifact exists."""
    # (a) valid model + zero quotes -> MODEL_ONLY, WITH a complete artifact.
    zero_quote = _build_warehouse(
        tmp_path / "zero",
        quote_visible_at=AS_OF + timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(minutes=5),
    )
    rec_a = _execute(zero_quote, run_id="p7e-model-only")
    assert rec_a.publication_status is PublicationStatus.MODEL_ONLY
    proj = _projections(zero_quote, run_id="p7e-model-only")
    e_count = proj["player_id"].n_unique()
    assert proj.height == e_count * 30 > 0

    # (b) no usable model -> the artifact does not exist, so the run is
    # NEVER MODEL_ONLY.
    no_model = _games_only_warehouse(tmp_path / "none")
    rec_b = _execute(no_model, run_id="p7e-no-model-2")
    assert rec_b.publication_status is not PublicationStatus.MODEL_ONLY
    assert not no_model.exists(PLAYER_GAME_PROJECTIONS_TABLE)
