"""Versioned joint-game calibration feature contract (PHASE 10C2).

Defines the exact, ordered, compact feature basis the entropy-tilting
calibrator (`nflprops.calibration.entropy_tilting`) scores each coherent
simulation draw with. Every feature is:

* deterministic -- a pure function of one `GameSimulationResult`;
* sportsbook/quote independent -- computed only from `result.team_draws`,
  which is produced entirely by `nflprops.simulation.game.simulate_game`
  before any market or settlement data ever enters the pipeline;
* free of realized-result/future information -- `GameSimulationResult` is
  itself a pregame artifact; nothing here reads a settled outcome;
* stable-ordered -- `FEATURE_NAMES` is a fixed tuple, `compute_draw_features`
  always returns columns in that exact order;
* finite -- `FeatureContractError` is raised rather than ever emitting
  NaN/inf.

`theta = 0` (see `nflprops.calibration.entropy_tilting.softmax_weights`)
is the explicit raw/unweighted evaluation baseline for every feature
version -- this holds regardless of what these features are, since a
zero dot-product is zero no matter the input.

v1 feature basis (systematic joint-game environment/dispersion signals,
never player-specific corrections): for each draw, sum the two teams'
`plays`, `points`, and `interceptions` (turnovers), plus the absolute
score margin, then standardize each of the four raw quantities to a
game-local z-score (mean/std computed over that SAME game's own
`n_draws`, population statistics). Standardizing within-game is what
makes one shared theta comparable in scale across many different games
(a 55-play shootout and a 44-play defensive struggle both express
"one standard deviation of pace above this game's own mean" the same
way).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nflprops.simulation.game import GameSimulationResult

#: Bump whenever the ordered feature basis changes. A payload records this
#: string; a payload's feature list must match currently-known versions
#: exactly (`nflprops.calibration.payload`) or loading fails closed.
FEATURE_CONTRACT_VERSION = "joint_game_calibration_features/v1"

#: Stable, ordered feature names. `compute_draw_features` returns columns
#: in exactly this order.
FEATURE_NAMES: tuple[str, ...] = (
    "total_plays_z",
    "total_points_z",
    "abs_margin_z",
    "total_turnovers_z",
)

_REQUIRED_TEAM_DRAW_COLUMNS: tuple[str, ...] = (
    "draw_id",
    "team_id",
    "plays",
    "points",
    "interceptions",
)


class FeatureContractError(ValueError):
    """A `GameSimulationResult` violated a structural precondition of the
    v1 joint-game feature contract (missing column, wrong team count,
    non-contiguous draw ids, or a non-finite computed feature)."""


def _require_two_teams(result: GameSimulationResult) -> None:
    missing = [c for c in _REQUIRED_TEAM_DRAW_COLUMNS if c not in result.team_draws.columns]
    if missing:
        raise FeatureContractError(f"team_draws missing required column(s): {missing}")
    team_ids = result.team_draws["team_id"].unique().to_list()
    if len(team_ids) != 2:
        raise FeatureContractError(
            "joint-game calibration features require exactly two teams in "
            f"team_draws, found {len(team_ids)}: {sorted(team_ids)}"
        )


def compute_raw_draw_totals(result: GameSimulationResult) -> dict[str, np.ndarray]:
    """The four RAW (unstandardized) per-draw joint-game quantities, each a
    length-`n_draws` vector in ascending `draw_id` order: total plays,
    total points, absolute point margin, and total turnovers (both teams
    combined). A pure read over `result.team_draws` -- never resamples,
    never recomputes simulation state.
    """
    _require_two_teams(result)
    n = result.n_draws

    per_draw = (
        result.team_draws.group_by("draw_id")
        .agg(
            pl.col("plays").sum().alias("total_plays"),
            pl.col("points").sum().alias("total_points"),
            pl.col("interceptions").sum().alias("total_turnovers"),
            (pl.col("points").max() - pl.col("points").min()).alias("abs_margin"),
        )
        .sort("draw_id")
    )

    if per_draw.height != n:
        raise FeatureContractError(
            f"team_draws does not have exactly n_draws={n} distinct draw_id "
            f"values (found {per_draw.height})"
        )

    draw_ids = per_draw["draw_id"].to_numpy()
    if not np.array_equal(draw_ids, np.arange(n)):
        raise FeatureContractError(
            "team_draws draw_id values must be exactly 0..n_draws-1 with no gaps"
        )

    return {
        "total_plays": per_draw["total_plays"].to_numpy().astype(np.float64),
        "total_points": per_draw["total_points"].to_numpy().astype(np.float64),
        "abs_margin": per_draw["abs_margin"].to_numpy().astype(np.float64),
        "total_turnovers": per_draw["total_turnovers"].to_numpy().astype(np.float64),
    }


def _game_local_zscore(values: np.ndarray) -> np.ndarray:
    """Standardize to this game's own draw-population mean/std (ddof=0).

    A degenerate zero-variance feature (every draw identical -- possible
    only in a pathological/deterministic fixture) standardizes to all
    zeros rather than dividing by zero: a feature with no within-game
    information content contributes nothing to the tilt score, which is
    the correct, finite, deterministic fallback.
    """
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std <= 0.0:
        return np.zeros_like(values)
    z = (values - mean) / std
    if not np.all(np.isfinite(z)):
        raise FeatureContractError("standardized joint-game feature produced a non-finite value")
    return z


def compute_draw_features(result: GameSimulationResult) -> np.ndarray:
    """The `(n_draws, len(FEATURE_NAMES))` v1 feature matrix, columns in
    exactly `FEATURE_NAMES` order, each game-locally standardized
    (`_game_local_zscore`). Raises `FeatureContractError` on any
    structural violation; never returns NaN/inf.
    """
    raw = compute_raw_draw_totals(result)
    columns = (
        _game_local_zscore(raw["total_plays"]),
        _game_local_zscore(raw["total_points"]),
        _game_local_zscore(raw["abs_margin"]),
        _game_local_zscore(raw["total_turnovers"]),
    )
    matrix = np.column_stack(columns)
    if matrix.shape != (result.n_draws, len(FEATURE_NAMES)):
        raise FeatureContractError(
            f"feature matrix shape {matrix.shape} does not match "
            f"(n_draws={result.n_draws}, n_features={len(FEATURE_NAMES)})"
        )
    if not np.all(np.isfinite(matrix)):
        raise FeatureContractError("joint-game feature matrix contains non-finite values")
    return matrix
