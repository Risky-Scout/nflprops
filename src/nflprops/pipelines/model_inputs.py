"""The one place the fundamental model's inputs are assembled, per model
profile and evidence mode. Historical replay and live execution both call
`build_model_inputs`, so `STRUCTURAL_CORE` is the same model in both.

STRUCTURAL_CORE (either mode)
    history = completed player/team game stats of FINAL games from a slate
    strictly before the target slate; recency aged by the source-game
    kickoff (`PERFORMANCE_EVENT_COL`). Eligibility differs only in the
    clock that proves it:
      * HISTORICAL_WALK_FORWARD -- event chronology (slate boundary);
      * LIVE_PIT -- the row's genuine `available_at` (unchanged live gate),
        additionally restricted to prior-slate final games.
    No game odds (market = None), no injury rows, no roster rows.

LIVE_ENHANCED (LIVE_PIT only)
    The existing live behaviour: history gated by `available_at` alone and
    aged by it; roster depth, injuries and game odds by their LIVE_PIT
    rules. Refused under HISTORICAL_WALK_FORWARD: 2022-2025 pregame
    observations have no certified historical availability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from nflprops.domain.model_profile import ModelProfile, ModelProfileError
from nflprops.features.historical_evidence import (
    EVENT_CHRONOLOGY_COL,
    PERFORMANCE_EVENT_COL,
    EvidenceMode,
    attach_performance_event_time,
    build_slate_chronology,
    certify_event_derived,
    restrict_to_prior_slates,
)
from nflprops.state.player import PlayerState, PlayerStateConfig, build_player_states
from nflprops.state.team import TeamState, TeamStateConfig, build_team_states


@dataclass(frozen=True)
class ModelInputs:
    model_profile: ModelProfile
    evidence_mode: EvidenceMode
    team_states: dict[str, TeamState]
    player_states: dict[str, PlayerState]
    #: None = the profile consumes no game market.
    game_odds: pl.DataFrame | None
    #: Pregame observations the model consumed (empty for STRUCTURAL_CORE);
    #: pricing provenance reads them, so they must match what was consumed.
    roster: pl.DataFrame
    injuries: pl.DataFrame


def _games_known_at(games: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    if games.is_empty() or "available_at" not in games.columns:
        return games
    return games.filter(pl.col("available_at") <= as_of)


def build_model_inputs(
    *,
    model_profile: ModelProfile,
    evidence_mode: EvidenceMode,
    games: pl.DataFrame,
    player_stats: pl.DataFrame,
    team_stats: pl.DataFrame,
    players: pl.DataFrame,
    roster: pl.DataFrame,
    injuries: pl.DataFrame,
    game_odds: pl.DataFrame,
    as_of: datetime,
    target_slate: tuple[int, int, int],
    target_game_id: str | None = None,
    player_state_config: PlayerStateConfig | None = None,
    team_state_config: TeamStateConfig | None = None,
) -> ModelInputs:
    profile = ModelProfile(model_profile)
    mode = EvidenceMode(evidence_mode)
    player_cfg = player_state_config or PlayerStateConfig()
    team_cfg = team_state_config or TeamStateConfig()

    if profile is ModelProfile.LIVE_ENHANCED:
        if mode is not EvidenceMode.LIVE_PIT:
            raise ModelProfileError(
                "LIVE_ENHANCED cannot run under HISTORICAL_WALK_FORWARD: historical "
                "game odds, injuries and roster depth have no certified availability"
            )
        known_games = _games_known_at(games, as_of)
        ps = attach_performance_event_time(player_stats, known_games)
        ts = attach_performance_event_time(team_stats, known_games)
        # Existing live behaviour: estimated historical-backfill outcome
        # rows are admitted (their availability is set after game end).
        return ModelInputs(
            model_profile=profile,
            evidence_mode=mode,
            team_states=build_team_states(ts, ps, as_of=as_of, strict=False, config=team_cfg),
            player_states=build_player_states(
                ps, ts, players, as_of=as_of,
                roster=roster if not roster.is_empty() else None,
                injuries=injuries if not injuries.is_empty() else None,
                strict=False, config=player_cfg,
            ),
            game_odds=game_odds,
            roster=roster,
            injuries=injuries,
        )

    if mode is EvidenceMode.HISTORICAL_WALK_FORWARD:
        if target_game_id is None:
            raise ModelProfileError("historical replay requires target_game_id")
        chronology = build_slate_chronology(games)
        if chronology.slate_of(target_game_id) != target_slate:
            raise ModelProfileError("target_slate disagrees with the schedule")
        ps = certify_event_derived(
            player_stats, chronology, target_game_id=target_game_id, as_of=as_of
        )
        ts = certify_event_derived(
            team_stats, chronology, target_game_id=target_game_id, as_of=as_of
        )
        history_time_col, strict = EVENT_CHRONOLOGY_COL, True
    else:
        chronology = build_slate_chronology(_games_known_at(games, as_of))
        ps = restrict_to_prior_slates(player_stats, chronology, target_slate=target_slate)
        ts = restrict_to_prior_slates(team_stats, chronology, target_slate=target_slate)
        # Same live receipt gate as LIVE_ENHANCED; the slate restriction
        # above only removes rows.
        history_time_col, strict = "available_at", False

    return ModelInputs(
        model_profile=profile,
        evidence_mode=mode,
        team_states=build_team_states(
            ts, ps, as_of=as_of, strict=strict, config=team_cfg,
            history_time_col=history_time_col, recency_time_col=PERFORMANCE_EVENT_COL,
        ),
        player_states=build_player_states(
            ps, ts, players, as_of=as_of, roster=None, injuries=None,
            strict=strict, config=player_cfg,
            history_time_col=history_time_col, recency_time_col=PERFORMANCE_EVENT_COL,
        ),
        game_odds=None,
        roster=roster.head(0),
        injuries=injuries.head(0),
    )

