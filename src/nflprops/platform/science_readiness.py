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

Read-only: nothing here writes to any warehouse.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

from nflprops.backtest.provenance import (
    MISSING_REQUIRED_GAME_METADATA,
    state_game_universe_at,
)
from nflprops.data.warehouse import Warehouse

READINESS_SCHEMA_VERSION = "nflprops.platform.science_readiness/v1"

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
