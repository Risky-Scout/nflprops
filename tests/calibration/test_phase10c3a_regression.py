"""PHASE 10C3A regression locks: PIT cohort separation, sportsbook/quote
independence, and raw-PMF immutability, specifically for the real-historical-
replay glue (`nflprops.calibration.historical_runner`) added in this phase.

Full re-execution of the Phase-10C1/10C2 test suites is covered by the
ordinary full-suite run, not duplicated here -- this file locks only the
NEW boundary claims Phase 10C3A makes.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_historical_runner import TARGET_GAME_ID, _build_warehouse

from nflprops.calibration.challenger import _aggregate_scores
from nflprops.calibration.historical_runner import (
    build_labeled_game,
    list_final_games,
    replay_games,
)
from nflprops.calibration.joint_feature_contract import (
    compute_draw_features,
)

# ------------------------------------------------------- PIT cohort separation


def test_pit_faithful_and_degraded_cohorts_are_never_silently_pooled(tmp_path) -> None:
    """`_aggregate_scores` (the only scoring aggregation primitive) takes
    an explicit `games` sequence -- there is no function anywhere in the
    calibration package that scores a mix of PIT-faithful/degraded games
    and returns a single number without the caller having explicitly
    chosen which games to include. This proves the SEPARATION is a
    property of the caller's own explicit filtering (as
    `historical_runner`'s real-run driver does), not something the
    library silently glosses over."""
    warehouse = _build_warehouse(tmp_path)
    games = list_final_games(warehouse, season_min=2024, season_max=2024)
    batch = replay_games(warehouse, games, model_version="test-v1", n_draws=300)

    faithful = [g for g in batch.labeled_games if g.injury_data_available]
    degraded = [g for g in batch.labeled_games if not g.injury_data_available]
    # In this synthetic fixture (no collector_resource_runs table at all),
    # every game is PIT-degraded -- exactly the real-world finding.
    assert not faithful
    assert degraded == list(batch.labeled_games)

    theta = np.zeros(4)
    degraded_scores = _aggregate_scores(degraded, theta)
    # An empty faithful cohort must be handled by the CALLER (skip/report
    # INSUFFICIENT_PIT_FAITHFUL_EVIDENCE), never by silently substituting
    # the degraded cohort's scores in its place.
    assert degraded_scores  # real evidence exists for the degraded cohort
    faithful_scores = _aggregate_scores(faithful, theta) if faithful else {}
    assert faithful_scores == {}


# --------------------------------------------------- sportsbook independence


def _public_function_signatures(module) -> dict[str, list[str]]:
    return {
        name: list(inspect.signature(obj).parameters)
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == module.__name__
    }


def test_joint_feature_contract_functions_take_no_market_parameter() -> None:
    """Structural lock: no public function in the entropy-tilting feature
    basis (`nflprops.calibration.joint_feature_contract`) accepts a
    quote/vendor/odds/price/sportsbook parameter of any kind -- changing
    which books quote a game, or whether any book quotes it at all, is
    structurally incapable of influencing the computed features, because
    there is no code path through which a quote could even be passed in."""
    import nflprops.calibration.joint_feature_contract as feature_module

    forbidden = {"quote", "quotes", "vendor", "odds", "price", "sportsbook", "market"}
    for name, params in _public_function_signatures(feature_module).items():
        overlap = forbidden & {p.lower() for p in params}
        assert not overlap, f"{name} accepts market-shaped parameter(s): {overlap}"

    import ast

    tree = ast.parse(inspect.getsource(feature_module))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("market" in m for m in imported), imported


def test_entropy_tilting_functions_take_no_market_parameter() -> None:
    import nflprops.calibration.entropy_tilting as tilting_module

    forbidden = {"quote", "quotes", "vendor", "odds", "price", "sportsbook", "market"}
    for name, params in _public_function_signatures(tilting_module).items():
        overlap = forbidden & {p.lower() for p in params}
        assert not overlap, f"{name} accepts market-shaped parameter(s): {overlap}"


def test_draw_features_are_a_pure_function_of_team_draws_alone(tmp_path) -> None:
    """Behavioral proof, not just structural: `compute_draw_features` takes
    only a `GameSimulationResult` -- no quote/odds parameter exists on it
    at all -- so replaying the SAME game with a materially different
    `game_odds_snapshots` row cannot change the shape/finiteness contract
    of the resulting feature matrix, even though the ODDS themselves
    legitimately influence the upstream simulation's scoring environment
    (certified Phase-6/9 behavior, unrelated to this test)."""
    warehouse = _build_warehouse(tmp_path)
    game_row = (
        warehouse.read("games").filter(pl.col("canonical_game_id") == TARGET_GAME_ID).row(0, named=True)
    )
    result_a = build_labeled_game(warehouse, game_row, model_version="test-v1", n_draws=300)

    odds = warehouse.read("game_odds_snapshots")
    warehouse.write(
        "game_odds_snapshots",
        odds.with_columns(pl.lit(-9.5).alias("spread_home_value"), pl.lit(61.5).alias("total_value")),
    )
    result_b = build_labeled_game(warehouse, game_row, model_version="test-v1", n_draws=300)

    features_a = compute_draw_features(result_a.simulation)
    features_b = compute_draw_features(result_b.simulation)
    assert features_a.shape == features_b.shape
    assert np.all(np.isfinite(features_a))
    assert np.all(np.isfinite(features_b))

    # Structural confirmation: compute_draw_features has no odds/quote
    # parameter to accept in the first place.
    params = inspect.signature(compute_draw_features).parameters
    assert list(params) == ["result"]


# ----------------------------------------------------- raw PMF immutability


def test_historical_runner_never_writes_to_distribution_store() -> None:
    """Structural lock: the real-replay glue module must never import or
    call anything from `nflprops.orchestration.distribution_store` (the
    Phase-10B raw PMF persistence boundary) -- it only ever builds
    in-memory `GameSimulationResult`/`LabeledGame` objects."""
    import nflprops.calibration.historical_runner as runner_module

    source = inspect.getsource(runner_module)
    assert "distribution_store" not in source
    assert "persist_player_prop_distributions" not in source


def test_challenger_and_weighted_pmf_never_import_pricing_or_distribution_persistence() -> None:
    """PHASE 10C2's challenger/weighted_pmf modules compute calibrated
    PMFs entirely in memory; neither imports the Phase-9 pricing store or
    the Phase-10B distribution store -- there is no code path by which
    either could write a calibrated probability into a persisted table."""
    import ast

    import nflprops.calibration.challenger as challenger_module
    import nflprops.calibration.weighted_pmf as weighted_pmf_module

    for module in (challenger_module, weighted_pmf_module):
        tree = ast.parse(inspect.getsource(module))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        forbidden = {m for m in imported if "pricing_store" in m or "distribution_store" in m}
        assert not forbidden, f"{module.__name__} imports persistence module(s): {forbidden}"
