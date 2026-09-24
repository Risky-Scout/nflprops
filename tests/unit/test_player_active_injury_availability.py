"""Historical-availability semantics check (pre-Phase-3 pause).

Verifies `build_player_states` continues to apply the configured
`features.injury.missing_row_means = "active"` default identically whether
or not injury data exists — `PlayerState.active` numerically does NOT change
between Case A and Case B, on purpose (see
docs/PRODUCTION_BASELINE_AUDIT.md and the historical-availability audit).
The machine-readable Case A/B distinction lives one layer up, in
`StateProvenanceContext.injury_data_available` /
`PredictionProvenance.injury_data_available` (see
test_pregame_provenance_wiring.py and test_historical_fold_adapter.py) — this
file only proves the state layer's numeric behavior is unchanged and
identical in both cases, which is what makes the provenance flag necessary
in the first place (nothing else would tell the two cases apart).
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from nflprops.state.player import build_player_states

AS_OF = datetime(2026, 9, 10, tzinfo=UTC)
OBSERVED = datetime(2026, 9, 1, tzinfo=UTC)


def _player_stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": ["g1", "g1"],
            "canonical_team_id": ["t1", "t1"],
            "canonical_player_id": ["healthy-wr", "other-wr"],
            "available_at": [OBSERVED, OBSERVED],
            "receiving_targets": [5, 3],
            "receptions": [3, 2],
            "receiving_yards": [40.0, 20.0],
            "receiving_touchdowns": [0, 0],
            "rushing_attempts": [0, 0],
            "rushing_yards": [0.0, 0.0],
            "rushing_touchdowns": [0, 0],
            "passing_attempts": [0, 0],
            "passing_completions": [0, 0],
            "passing_interceptions": [0, 0],
            "field_goal_attempts": [0, 0],
            "field_goals_made": [0, 0],
        }
    )


def _team_stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": ["g1"],
            "canonical_team_id": ["t1"],
            "available_at": [OBSERVED],
            "rushing_attempts": [0],
            "passing_attempts": [0],
            "passing_completions": [0],
            "interceptions_thrown": [0],
        }
    )


def _players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_player_id": ["healthy-wr", "other-wr"],
            "position_group": ["WR", "WR"],
        }
    )


def _injuries_with_other_player_designation() -> pl.DataFrame:
    """A non-empty, valid PIT injury snapshot exists for this team/time — it
    just doesn't mention `healthy-wr` because that player has no designation.
    This is Case A."""
    return pl.DataFrame(
        {
            "canonical_player_id": ["other-wr"],
            "status": ["Questionable"],
            "available_at": [OBSERVED],
        }
    )


def test_case_a_snapshot_exists_player_absent_defaults_to_active() -> None:
    """CASE A: PIT injury snapshot exists for the relevant team/time, target
    player absent from it -> configured missing_row_means='active' applies."""
    states = build_player_states(
        _player_stats(),
        _team_stats(),
        _players(),
        as_of=AS_OF,
        injuries=_injuries_with_other_player_designation(),
        strict=False,
    )

    assert states["healthy-wr"].active is True
    # Regression: the player who DOES have a non-OUT/IR designation ("other-wr"
    # is Questionable, not Out/IR/etc.) is also still active — Questionable is
    # not guaranteed-inactive (blueprint §5 / Phase 2).
    assert states["other-wr"].active is True


def test_case_b_no_snapshot_at_all_still_defaults_to_active_numerically() -> None:
    """CASE B: no PIT injury snapshot exists at all for this team/time.

    The state layer intentionally keeps the SAME numeric default as Case A —
    "do not fabricate historical active/inactive states" means don't invent a
    new, different point estimate for Case B either. What must differ is
    whether this was a verified "checked and found nothing" read (Case A) or
    an "we have no idea" read (Case B); that distinction is recorded one
    layer up (see StateProvenanceContext.injury_data_available), not by
    changing this number.
    """
    states_no_frame = build_player_states(
        _player_stats(),
        _team_stats(),
        _players(),
        as_of=AS_OF,
        injuries=None,
        strict=False,
    )
    states_empty_frame = build_player_states(
        _player_stats(),
        _team_stats(),
        _players(),
        as_of=AS_OF,
        injuries=pl.DataFrame(),
        strict=False,
    )

    assert states_no_frame["healthy-wr"].active is True
    assert states_empty_frame["healthy-wr"].active is True

    # Case A and Case B produce IDENTICAL `active` values at this layer —
    # proving the state layer alone cannot and does not distinguish them.
    case_a = build_player_states(
        _player_stats(),
        _team_stats(),
        _players(),
        as_of=AS_OF,
        injuries=_injuries_with_other_player_designation(),
        strict=False,
    )
    assert (
        states_no_frame["healthy-wr"].active
        == case_a["healthy-wr"].active
        is True
    )


def test_out_status_still_produces_inactive_regardless_of_availability_case() -> None:
    """Regression: an explicit OUT designation still overrides the default in
    both cases — this behavior predates and is unaffected by this check."""
    injuries = pl.DataFrame(
        {
            "canonical_player_id": ["healthy-wr"],
            "status": ["Out"],
            "available_at": [OBSERVED],
        }
    )

    states = build_player_states(
        _player_stats(),
        _team_stats(),
        _players(),
        as_of=AS_OF,
        injuries=injuries,
        strict=False,
    )

    assert states["healthy-wr"].active is False
