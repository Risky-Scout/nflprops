"""BLOCK 4 (PR #21): the pre-simulation science readiness gate.

The first production T90M canary transported correctly but simulated zero
of its 20,000 draws: state chronology could not be proven because 32 games
referenced by PIT player/team stats had no `games`-table metadata, so the
flow raised inside state construction and the run was recorded FAILED
(PREDICTION_ERROR) after a full GitHub execution.

That condition is a deterministic property of the immutable snapshot -- the
snapshot can never acquire the missing rows -- so it is decided BEFORE any
simulation, from exactly the PIT universe the state build will see
(`backtest.provenance.state_game_universe_at`, shared with
`build_state_provenance_context`), and it is a SCIENTIFIC refusal
(`MISSING_REQUIRED_GAME_METADATA`) carrying every missing id as evidence.

PR #21 (second gate): a snapshot can carry complete metadata for an EMPTY
state universe (no PIT stats at all) and pass the first gate vacuously,
only to end GAME_NOT_MODELED after a full execution. `check_state_universe`
builds the exact PIT football state the model simulates from
(`pipelines.pregame.build_pit_model_state` -- never a market table) and
applies the model's own pre-simulation rules: the game is modeled only when
it is PIT-visible and both teams have learned structural state
(`pipelines.pregame.missing_game_team_states`), and each eligible player
(`projections.eligible_player_states_for_teams`, Phase 7A) yields one output
per supported registry stat. Zero generatable outputs is a SCIENTIFIC
refusal (`INSUFFICIENT_STATE_UNIVERSE`) -- no league-wide minimum of games,
rows or players is invented here, and sportsbook lines/prices never decide
it.

Read-only: nothing here writes to any warehouse.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from nflprops.backtest.provenance import (
    MISSING_REQUIRED_GAME_METADATA,
    state_game_universe_at,
)
from nflprops.data.warehouse import Warehouse

if TYPE_CHECKING:
    from nflprops.state.player import PlayerStateConfig
    from nflprops.state.team import TeamStateConfig

READINESS_SCHEMA_VERSION = "nflprops.platform.science_readiness/v1"
STATE_UNIVERSE_SCHEMA_VERSION = "nflprops.platform.science_readiness.state_universe/v1"

#: The scientific refusal when the immutable snapshot's PIT state lets the
#: model generate zero supported player-prop outputs for the game.
INSUFFICIENT_STATE_UNIVERSE = "INSUFFICIENT_STATE_UNIVERSE"

#: Why zero outputs are generatable (sorted into the evidence). Each is a
#: pre-simulation skip condition the prediction path itself applies.
REASON_TARGET_GAME_NOT_PIT_VISIBLE = "TARGET_GAME_NOT_PIT_VISIBLE"  # GAME_NOT_FOUND
REASON_TEAM_STATE_MISSING = "TEAM_STATE_MISSING"  # GAME_NOT_MODELED
REASON_NO_ELIGIBLE_PLAYERS = "NO_ELIGIBLE_PLAYERS"  # zero Phase-7A players
STATE_UNIVERSE_REASONS: frozenset[str] = frozenset({
    REASON_TARGET_GAME_NOT_PIT_VISIBLE,
    REASON_TEAM_STATE_MISSING,
    REASON_NO_ELIGIBLE_PLAYERS,
})

#: The PIT tables `build_pit_model_state` reads -- football state only. No
#: game-odds or player-prop quote table is ever read by this gate.
STATE_UNIVERSE_TABLES = (
    "games", "player_game_stats", "team_game_stats", "players",
    "roster_snapshots", "injury_snapshots", "collector_resource_runs",
)

#: The only tables the gate reads: the state-game universe is defined by
#: PIT player/team stat rows and proven by PIT `games` rows.
READINESS_TABLES = ("games", "player_game_stats", "team_game_stats")


def _ids_sha256(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ScienceReadiness:
    """Whether a warehouse's PIT state universe at `state_as_of` can be
    built at all. `ready` is False iff some referenced game has no PIT
    `games` row, or its latest PIT row lacks season/week."""

    state_as_of: datetime
    state_game_count: int
    missing_game_ids: tuple[str, ...]
    incomplete_game_ids: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing_game_ids and not self.incomplete_game_ids

    @property
    def refusal_code(self) -> str | None:
        return None if self.ready else MISSING_REQUIRED_GAME_METADATA

    def message(self) -> str:
        if self.ready:
            return (
                f"science-ready: all {self.state_game_count} state games carry "
                f"PIT games metadata at {self.state_as_of.isoformat()}"
            )
        parts = []
        if self.missing_game_ids:
            parts.append(
                f"{len(self.missing_game_ids)} state games have no PIT games row"
            )
        if self.incomplete_game_ids:
            parts.append(
                f"{len(self.incomplete_game_ids)} PIT games lack season/week"
            )
        return (
            "cannot prove state chronology at "
            f"{self.state_as_of.isoformat()}: " + "; ".join(parts)
            + f" (of {self.state_game_count} state games); simulation never started"
        )

    def evidence(self) -> dict[str, Any]:
        """Complete, machine-readable refusal evidence (every id, never a
        preview), with a digest of the sorted id lists."""
        return {
            "schema_version": READINESS_SCHEMA_VERSION,
            "ready": self.ready,
            "refusal_code": self.refusal_code,
            "state_as_of": self.state_as_of.astimezone(UTC).isoformat(),
            "state_game_count": self.state_game_count,
            "missing_game_count": len(self.missing_game_ids),
            "missing_game_ids": list(self.missing_game_ids),
            "missing_game_ids_sha256": _ids_sha256(self.missing_game_ids),
            "incomplete_game_count": len(self.incomplete_game_ids),
            "incomplete_game_ids": list(self.incomplete_game_ids),
        }


def _read(warehouse: Warehouse, table: str) -> pl.DataFrame:
    return warehouse.read(table) if warehouse.exists(table) else pl.DataFrame()


def check_science_readiness(
    warehouse: Warehouse, *, scheduled_as_of: datetime
) -> ScienceReadiness:
    """Evaluate the gate on `warehouse` at the run's `scheduled_as_of`
    (the as-of the checkpoint flow builds state at)."""
    universe = state_game_universe_at(
        games=_read(warehouse, "games"),
        player_stats=_read(warehouse, "player_game_stats"),
        team_stats=_read(warehouse, "team_game_stats"),
        as_of=scheduled_as_of,
    )
    return ScienceReadiness(
        state_as_of=scheduled_as_of,
        state_game_count=len(universe.state_game_ids),
        missing_game_ids=universe.missing_game_ids,
        # build_state_provenance_context refuses ANY PIT game whose latest
        # row lacks season/week (state game or not), so every one counts.
        incomplete_game_ids=universe.incomplete_game_ids,
    )


@dataclass(frozen=True)
class TeamEligibilityCensus:
    """One game team's pre-simulation eligibility, from the model's rules."""

    team_id: str
    team_state_present: bool
    player_state_count: int
    #: Phase-7A ineligibility reason -> player count (sorted by reason).
    ineligible_counts: tuple[tuple[str, int], ...]
    eligible_player_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "team_state_present": self.team_state_present,
            "player_state_count": self.player_state_count,
            "ineligible_counts": dict(self.ineligible_counts),
            "eligible_player_count": len(self.eligible_player_ids),
            "eligible_player_ids": list(self.eligible_player_ids),
        }


@dataclass(frozen=True)
class StateUniverseReadiness:
    """Whether the model can generate at least one supported player-prop
    output for `game_id` from the PIT football state at `state_as_of`."""

    game_id: str
    state_as_of: datetime
    target_game_pit_visible: bool
    home_team_id: str | None
    away_team_id: str | None
    team_state_count: int
    player_state_count: int
    teams: tuple[TeamEligibilityCensus, ...]
    supported_stat_count: int

    @property
    def missing_team_state_ids(self) -> tuple[str, ...]:
        return tuple(t.team_id for t in self.teams if not t.team_state_present)

    @property
    def eligible_player_count(self) -> int:
        return sum(len(t.eligible_player_ids) for t in self.teams)

    @property
    def game_modeled(self) -> bool:
        return self.target_game_pit_visible and not self.missing_team_state_ids

    @property
    def generatable_output_count(self) -> int:
        """Eligible players x supported registry stats, when the game is
        modeled at all (otherwise the model simulates nothing)."""
        if not self.game_modeled:
            return 0
        return self.eligible_player_count * self.supported_stat_count

    @property
    def reasons(self) -> tuple[str, ...]:
        if not self.target_game_pit_visible:
            return (REASON_TARGET_GAME_NOT_PIT_VISIBLE,)
        found = set()
        if self.missing_team_state_ids:
            found.add(REASON_TEAM_STATE_MISSING)
        if self.eligible_player_count == 0:
            found.add(REASON_NO_ELIGIBLE_PLAYERS)
        return tuple(sorted(found))

    @property
    def ready(self) -> bool:
        return self.generatable_output_count > 0

    @property
    def refusal_code(self) -> str | None:
        return None if self.ready else INSUFFICIENT_STATE_UNIVERSE

    def message(self) -> str:
        if self.ready:
            return (
                f"science-ready: {self.eligible_player_count} eligible players x "
                f"{self.supported_stat_count} supported stats for game {self.game_id} at "
                f"{self.state_as_of.isoformat()}"
            )
        return (
            f"insufficient PIT state universe for game {self.game_id} at "
            f"{self.state_as_of.isoformat()}: zero supported player-prop outputs are "
            f"generatable ({', '.join(self.reasons)}; {self.team_state_count} team states, "
            f"{self.player_state_count} player states, {self.eligible_player_count} eligible "
            "players); simulation never started"
        )

    def evidence(self) -> dict[str, Any]:
        """Complete, machine-readable, deterministically ordered evidence."""
        return {
            "schema_version": STATE_UNIVERSE_SCHEMA_VERSION,
            "ready": self.ready,
            "refusal_code": self.refusal_code,
            "game_id": self.game_id,
            "state_as_of": self.state_as_of.astimezone(UTC).isoformat(),
            "eligibility_rule": (
                "game PIT-visible AND both teams have learned state "
                "(pipelines.pregame.missing_game_team_states); outputs = Phase-7A eligible "
                "players (projections.eligible_player_states_for_teams) x supported "
                "registry stats (projections.REGISTRY)"
            ),
            "market_tables_read": [],
            "state_tables_read": list(STATE_UNIVERSE_TABLES),
            "target_game_pit_visible": self.target_game_pit_visible,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "team_state_count": self.team_state_count,
            "player_state_count": self.player_state_count,
            "missing_team_state_ids": list(self.missing_team_state_ids),
            "teams": [t.as_dict() for t in self.teams],
            "eligible_player_count": self.eligible_player_count,
            "supported_stat_count": self.supported_stat_count,
            "generatable_output_count": self.generatable_output_count,
            "reasons": list(self.reasons),
        }


def check_state_universe(
    warehouse: Warehouse,
    *,
    season: int,
    week: int,
    game_id: str,
    scheduled_as_of: datetime,
    model_version: str,
    player_state_config: PlayerStateConfig | None,
    team_state_config: TeamStateConfig | None,
) -> StateUniverseReadiness:
    """Evaluate the second gate exactly as `compute_game_prediction` would
    build state for this checkpoint -- same tables, same PIT cutoff, same
    configs, same skip rules -- WITHOUT simulating and without reading any
    market table. Exceptions from the state build propagate (the caller
    treats them as model-code failures, as the flow would)."""
    from nflprops.pipelines.pregame import (
        build_pit_model_state,
        game_team_ids,
        missing_game_team_states,
    )
    from nflprops.projections import REGISTRY_SIZE
    from nflprops.projections.summarize import player_eligibility

    state = build_pit_model_state(
        warehouse,
        season=season,
        week=week,
        as_of=scheduled_as_of,
        model_version=model_version,
        player_state_config=player_state_config,
        team_state_config=team_state_config,
        game_ids={game_id},
    )
    matches = (
        state.current_games.filter(pl.col("canonical_game_id") == game_id)
        if state is not None
        else None
    )
    if state is None or matches is None or matches.is_empty():
        return StateUniverseReadiness(
            game_id=game_id, state_as_of=scheduled_as_of, target_game_pit_visible=False,
            home_team_id=None, away_team_id=None, team_state_count=0, player_state_count=0,
            teams=(), supported_stat_count=REGISTRY_SIZE,
        )
    game = matches.row(0, named=True)
    team_ids = game_team_ids(game)
    missing = set(missing_game_team_states(game, state.team_states))
    by_team: dict[str, list[tuple[str, str | None]]] = {team: [] for team in team_ids}
    for player, reason in player_eligibility(team_ids, state.player_states):
        if player.team_id in by_team:
            by_team[player.team_id].append((player.player_id, reason))
    teams = []
    for team_id in team_ids:
        ineligible: dict[str, int] = {}
        for _player_id, reason in by_team[team_id]:
            if reason is not None:
                ineligible[reason] = ineligible.get(reason, 0) + 1
        teams.append(TeamEligibilityCensus(
            team_id=team_id,
            team_state_present=team_id not in missing,
            player_state_count=len(by_team[team_id]),
            ineligible_counts=tuple(sorted(ineligible.items())),
            eligible_player_ids=tuple(p for p, reason in by_team[team_id] if reason is None),
        ))
    return StateUniverseReadiness(
        game_id=game_id, state_as_of=scheduled_as_of, target_game_pit_visible=True,
        home_team_id=team_ids[0], away_team_id=team_ids[1],
        team_state_count=len(state.team_states),
        player_state_count=len(state.player_states),
        teams=tuple(teams), supported_stat_count=REGISTRY_SIZE,
    )
