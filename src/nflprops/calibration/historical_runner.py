"""Real-historical-data glue for coherent joint-game calibration (PHASE
10C3A).

Assembles `nflprops.calibration.challenger.LabeledGame` objects from an
existing `StorageBackend`'s SETTLED history (`games`/`player_game_stats`),
using the exact same production state-building and simulation entry
points `nflprops.pipelines.pregame.predict_week` uses
(`build_team_states`/`build_player_states`/`simulate_game_for_prediction`).

This module never:

* runs a second, simplified, or approximate simulation -- one
  `GameSimulationResult` per game, from the certified pregame path;
* introduces realized-outcome information into simulation inputs -- state
  is built under `EvidenceMode.HISTORICAL_WALK_FORWARD`
  (`nflprops.features.historical_evidence`): completed-game stats enter
  only when their slate precedes the target's slate (event chronology,
  never NFLProps' 2026 import date and never a `kickoff + lag` estimate),
  and pregame observations (rosters, injuries, game odds) only with a
  genuine availability time at or before `as_of` -- the LIVE_PIT rule;
* fabricates a label for an unlabeled (PBP-gated) PropType -- only
  `nflprops.calibration.artifact.DIRECTLY_LABELED_PROP_TYPES` are ever
  scored, via the SAME settlement rules
  (`nflprops.market.rules`/`nflprops.pipelines.settle`) production
  settlement uses;
* fabricates PIT faithfulness -- `LabeledGame.injury_data_available`
  comes from the real `nflprops.data.injury_availability.
  injury_feed_available_at` mechanism (the one
  `build_state_provenance_context` uses), never hardcoded.

`as_of` is each game's own kickoff time (`games.date`), matching the T30M
production checkpoint's spirit (the latest possible pregame cutoff). A
game's OWN stats are never visible to it (same slate, and excluded by id);
they are its labels, and enter history for later slates at its slate
boundary (`LabeledGame.outcome_available_at`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from nflprops.calibration.artifact import DIRECTLY_LABELED_PROP_TYPES
from nflprops.calibration.challenger import LabeledGame, PropLabel
from nflprops.data.injury_availability import injury_feed_available_at
from nflprops.domain.enums import PropType
from nflprops.domain.model_profile import ModelProfile
from nflprops.features.historical_evidence import (
    EvidenceMode,
    build_slate_chronology,
    schedule_identity,
)
from nflprops.features.historical_positions import HISTORICAL_POSITIONS_TABLE
from nflprops.features.team_membership import HISTORICAL_TEAM_MEMBERSHIP_TABLE
from nflprops.market.rules import (
    SettlementRuleError,
    SettlementRuleSet,
    evaluate_actual_value,
    load_settlement_rules,
)
from nflprops.pipelines.model_inputs import build_model_inputs
from nflprops.pipelines.pregame import simulate_game_for_prediction
from nflprops.simulation.game import SimulationConfig
from nflprops.state.player import PlayerStateConfig
from nflprops.state.team import TeamStateConfig

if TYPE_CHECKING:
    from nflprops.data.storage.base import StorageBackend

#: Historical replay's only evidence mode and the only model profile it can
#: certify; both are recorded in the 10C3A report and calibration artifact.
EVIDENCE_MODE = EvidenceMode.HISTORICAL_WALK_FORWARD
MODEL_PROFILE = ModelProfile.STRUCTURAL_CORE

_REQUIRED_WAREHOUSE_TABLES: tuple[str, ...] = (
    "games",
    "player_game_stats",
    "team_game_stats",
    "players",
)


class HistoricalReplayError(ValueError):
    """A structural precondition of real-historical replay was violated:
    a missing required warehouse table, a naive datetime, or a game row
    missing an expected column."""


@dataclass(frozen=True)
class WarehouseTables:
    """Every warehouse table `build_labeled_game` reads, loaded once.

    `build_labeled_game` re-reads all ten tables from `backend` on
    every call when `tables` is omitted -- correct, but O(n_games) redundant
    disk I/O over an unchanging historical warehouse (the dominant real-run
    cost, unrelated to `n_draws`). A caller replaying many games loads this
    ONCE via `load_warehouse_tables` and passes it to every `build_labeled_game`
    / `replay_games` call instead -- same frames, same values, purely fewer
    reads; no scoring/simulation semantics differ either way.
    """

    games: pl.DataFrame
    player_stats: pl.DataFrame
    team_stats: pl.DataFrame
    players: pl.DataFrame
    roster: pl.DataFrame
    injuries: pl.DataFrame
    injury_runs: pl.DataFrame
    game_odds: pl.DataFrame
    #: Week-versioned historical positions (empty = none built; position
    #: groups then come from the `players` dimension alone).
    historical_positions: pl.DataFrame
    #: Week-versioned roster membership (empty = none built; no QB is then
    #: a structural candidate and the simulator's generic QB stands in).
    historical_team_membership: pl.DataFrame

    def as_table_mapping(self) -> dict[str, pl.DataFrame]:
        """These frames keyed by their warehouse table name."""
        return {
            "games": self.games,
            "player_game_stats": self.player_stats,
            "team_game_stats": self.team_stats,
            "players": self.players,
            "roster_snapshots": self.roster,
            "injury_snapshots": self.injuries,
            "collector_resource_runs": self.injury_runs,
            "game_odds_snapshots": self.game_odds,
            HISTORICAL_POSITIONS_TABLE: self.historical_positions,
            HISTORICAL_TEAM_MEMBERSHIP_TABLE: self.historical_team_membership,
        }


def load_warehouse_tables(backend: StorageBackend) -> WarehouseTables:
    """Read every table `build_labeled_game` needs exactly once."""
    return WarehouseTables(
        games=backend.read("games"),
        player_stats=backend.read("player_game_stats"),
        team_stats=backend.read("team_game_stats"),
        players=backend.read("players"),
        roster=_empty_or(backend, "roster_snapshots"),
        injuries=_empty_or(backend, "injury_snapshots"),
        injury_runs=_empty_or(backend, "collector_resource_runs"),
        game_odds=_empty_or(backend, "game_odds_snapshots"),
        historical_positions=_empty_or(backend, HISTORICAL_POSITIONS_TABLE),
        historical_team_membership=_empty_or(backend, HISTORICAL_TEAM_MEMBERSHIP_TABLE),
    )


@dataclass(frozen=True)
class GameReplaySkip:
    """Why one historical game did not produce a `LabeledGame` -- an
    honest accounting record, never silently dropped."""

    game_id: str
    reason: str


def _aware(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise HistoricalReplayError(f"{field} must be a datetime, got {type(value)!r}")
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _empty_or(backend: StorageBackend, table: str) -> pl.DataFrame:
    return backend.read(table) if backend.exists(table) else pl.DataFrame()


def list_final_games(
    backend: StorageBackend, *, season_min: int, season_max: int
) -> pl.DataFrame:
    """Every `status_state == 'final'` game in `[season_min, season_max]`,
    sorted chronologically by kickoff (`date`)."""
    for table in _REQUIRED_WAREHOUSE_TABLES:
        if not backend.exists(table):
            raise HistoricalReplayError(f"warehouse is missing required table {table!r}")
    games = backend.read("games")
    return games.filter(
        (pl.col("status_state") == "final")
        & (pl.col("season") >= season_min)
        & (pl.col("season") <= season_max)
    ).sort(["season", "week", "date"])


def build_prop_labels_for_game(
    player_game_stats_for_game: pl.DataFrame,
    *,
    rules: SettlementRuleSet | None = None,
) -> tuple[PropLabel, ...]:
    """Real observed labels for `DIRECTLY_LABELED_PROP_TYPES` only, one per
    (player, prop) with a defined actual value under the EXISTING
    production settlement rules (`nflprops.market.rules`). A PropType with
    no registered rule, or a row with no defined actual value for it
    (`evaluate_actual_value` returns `None`), simply contributes no label
    -- never a fabricated one.
    """
    active_rules = rules if rules is not None else load_settlement_rules()
    labels: list[PropLabel] = []
    for row in player_game_stats_for_game.iter_rows(named=True):
        player_id = str(row["canonical_player_id"])
        for prop_value in sorted(DIRECTLY_LABELED_PROP_TYPES):
            try:
                rule = active_rules.rule_for(prop_value, None)
            except SettlementRuleError:
                continue
            actual = evaluate_actual_value(rule, row)
            if actual is None:
                continue
            labels.append(
                PropLabel(
                    player_id=player_id,
                    prop_type=PropType(prop_value),
                    observed_value=float(actual),
                )
            )
    return tuple(labels)


def build_labeled_game(
    backend: StorageBackend,
    game_row: dict[str, object],
    *,
    model_version: str,
    n_draws: int,
    simulation_config: SimulationConfig | None = None,
    player_state_config: PlayerStateConfig | None = None,
    team_state_config: TeamStateConfig | None = None,
    settlement_rules: SettlementRuleSet | None = None,
    tables: WarehouseTables | None = None,
    model_profile: ModelProfile = ModelProfile.STRUCTURAL_CORE,
) -> LabeledGame | GameReplaySkip:
    """Real historical replay of one game: build PIT-safe states from data
    strictly available at kickoff, run the certified
    `simulate_game_for_prediction`, and attach only genuinely observed
    settlement labels. Returns a `GameReplaySkip` (never raises, never
    silently omits the game from an honest coverage accounting) when the
    game cannot be simulated (untrustworthy team structural state -- the
    same pre-existing `simulate_game_for_prediction` skip condition
    `predict_week` has always had) or has no scoreable evidence at all.

    `tables`, if given (`load_warehouse_tables(backend)`), is used instead of
    re-reading `backend` -- identical values, purely fewer disk reads for a
    caller replaying many games against the same unchanging warehouse.
    """
    game_id = str(game_row["canonical_game_id"])
    as_of = _aware(game_row["date"], field="game.date")

    loaded = tables if tables is not None else load_warehouse_tables(backend)
    games = loaded.games
    player_stats = loaded.player_stats
    team_stats = loaded.team_stats
    players = loaded.players
    roster = loaded.roster
    injuries = loaded.injuries
    injury_runs = loaded.injury_runs
    game_odds = loaded.game_odds
    historical_positions = (
        None if loaded.historical_positions.is_empty() else loaded.historical_positions
    )
    historical_team_membership = (
        None
        if loaded.historical_team_membership.is_empty()
        else loaded.historical_team_membership
    )

    # STRUCTURAL_CORE under HISTORICAL_WALK_FORWARD, through the SAME
    # assembly live execution uses (nflprops.pipelines.model_inputs): history
    # by slate order, recency by source kickoff, no market/injury/roster
    # input. LIVE_ENHANCED is refused there, before any simulation.
    chronology = build_slate_chronology(games)
    inputs = build_model_inputs(
        model_profile=model_profile,
        evidence_mode=EVIDENCE_MODE,
        games=games, player_stats=player_stats, team_stats=team_stats, players=players,
        roster=roster, injuries=injuries, game_odds=game_odds,
        as_of=as_of, target_slate=chronology.slate_of(game_id), target_game_id=game_id,
        player_state_config=player_state_config, team_state_config=team_state_config,
        historical_positions=historical_positions,
        historical_team_membership=historical_team_membership,
    )

    prepared = simulate_game_for_prediction(
        game=schedule_identity(game_row), team_states=inputs.team_states,
        player_states=inputs.player_states,
        game_odds=inputs.game_odds, as_of=as_of, model_version=model_version,
        market_mode="live", simulation_config=simulation_config, n_draws=n_draws,
    )
    if prepared is None:
        return GameReplaySkip(game_id=game_id, reason="UNTRUSTWORTHY_TEAM_STRUCTURAL_STATE")

    # Not `build_state_provenance_context`: it re-derives the state universe
    # on the LIVE_PIT clock (`available_at`), which historical replay does
    # not use. Only its injury-feed flag is needed, from the same mechanism.
    injury_data_available = injury_feed_available_at(injury_runs, as_of=as_of)

    # Not every player with a real box-score line is part of the coherent
    # simulated player universe (e.g. a rarely-used player the eligibility
    # rule excluded, or a defensive/return player the offense-only
    # simulator never models at all) -- `canonical_outcome_values` requires
    # the player to appear in every draw, so a label for a player outside
    # `result.player_draws` must never be constructed (it is not a
    # fabrication to omit one; there is simply no calibrated PMF for that
    # player to score against).
    simulated_player_ids = set(
        prepared.result.real_player_draws()["player_id"].unique().to_list()
    )
    this_game_stats = player_stats.filter(
        (pl.col("canonical_game_id") == game_id)
        & pl.col("canonical_player_id").is_in(list(simulated_player_ids))
    )
    labels = build_prop_labels_for_game(this_game_stats, rules=settlement_rules)
    if not labels:
        return GameReplaySkip(game_id=game_id, reason="NO_SCOREABLE_EVIDENCE")

    # The label enters history for later targets at its slate boundary --
    # event chronology, not the stats' (2026 or estimated) `available_at`.
    return LabeledGame(
        game_id=game_id,
        simulation=prepared.result,
        as_of=as_of,
        outcome_available_at=chronology.label_available_at(game_id),
        injury_data_available=injury_data_available,
        labels=labels,
    )


@dataclass(frozen=True)
class ReplayBatchResult:
    labeled_games: tuple[LabeledGame, ...]
    skips: tuple[GameReplaySkip, ...]

    @property
    def replayable_game_count(self) -> int:
        return len(self.labeled_games)

    @property
    def total_game_count(self) -> int:
        return len(self.labeled_games) + len(self.skips)


def replay_games(
    backend: StorageBackend,
    game_rows: pl.DataFrame,
    *,
    model_version: str,
    n_draws: int,
    simulation_config: SimulationConfig | None = None,
    player_state_config: PlayerStateConfig | None = None,
    team_state_config: TeamStateConfig | None = None,
    settlement_rules: SettlementRuleSet | None = None,
    on_progress: object | None = None,
    tables: WarehouseTables | None = None,
    model_profile: ModelProfile = ModelProfile.STRUCTURAL_CORE,
) -> ReplayBatchResult:
    """Replay every row of `game_rows` (as returned by `list_final_games`).
    Honest bookkeeping: every game becomes exactly one `LabeledGame` or one
    `GameReplaySkip`, never silently disappears. `on_progress`, if given,
    is called with `(index, total, game_id)` after each game.

    `tables`, if given, is loaded once and reused for every game instead of
    re-reading `backend` per game (see `WarehouseTables`); omitted, this
    reads fresh (and therefore redundantly, once per game) exactly as
    before.
    """
    rules = settlement_rules if settlement_rules is not None else load_settlement_rules()
    loaded = tables if tables is not None else load_warehouse_tables(backend)
    labeled: list[LabeledGame] = []
    skips: list[GameReplaySkip] = []
    total = game_rows.height
    for i, row in enumerate(game_rows.iter_rows(named=True)):
        result = build_labeled_game(
            backend, row, model_version=model_version, n_draws=n_draws,
            simulation_config=simulation_config,
            player_state_config=player_state_config,
            team_state_config=team_state_config,
            settlement_rules=rules,
            tables=loaded,
            model_profile=model_profile,
        )
        if isinstance(result, LabeledGame):
            labeled.append(result)
        else:
            skips.append(result)
        if on_progress is not None:
            on_progress(i + 1, total, str(row["canonical_game_id"]))
    return ReplayBatchResult(labeled_games=tuple(labeled), skips=tuple(skips))


def compute_training_manifest_sha256(games: list[LabeledGame] | tuple[LabeledGame, ...]) -> str:
    """Deterministic, order-independent SHA-256 fingerprint of the exact
    training evidence one fit consumed: every game's id/as_of/
    outcome_available_at/injury_data_available plus every one of its
    labels' (player_id, prop_type, observed_value). Two independent
    replays over identical underlying data must produce this SAME hash
    (`tests/calibration/test_historical_runner.py::
    test_compute_training_manifest_sha256_is_deterministic`); it changes
    if -- and only if -- the actual training evidence changes.
    """

    def _serialize_game(game: LabeledGame) -> str:
        label_parts = sorted(
            f"{label.player_id}|{label.prop_type.value}|{label.observed_value!r}"
            for label in game.labels
        )
        return "\x1f".join(
            [
                game.game_id,
                game.as_of.astimezone(UTC).isoformat(),
                game.outcome_available_at.astimezone(UTC).isoformat(),
                "true" if game.injury_data_available else "false",
                *label_parts,
            ]
        )

    ordered = sorted(games, key=lambda g: g.game_id)
    payload = "\x1e".join(["training_manifest/v1", *(_serialize_game(g) for g in ordered)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_data_root_manifest(data_root: Path) -> dict[str, str]:
    """Deterministic `{relative_parquet_path: sha256_hex}` manifest of every
    `*.parquet` file under `data_root`, sorted by path.

    This is a manifest of the RAW WAREHOUSE INPUT (what real-historical
    replay reads), distinct from `compute_training_manifest_sha256` (a
    manifest of the DERIVED training evidence one fit consumed). A remote
    runner invocation records/verifies this so a promotion decision can be
    tied to an exact, reproducible input data snapshot -- never to "whatever
    happened to be on disk."
    """
    root = Path(data_root)
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*.parquet")):
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        manifest[str(path.relative_to(root))] = digest.hexdigest()
    return manifest


def compute_data_root_manifest_sha256(data_root: Path) -> str:
    """Single deterministic SHA-256 digest of `compute_data_root_manifest`
    (sorted-key canonical JSON) -- one value to record/compare instead of
    the full per-file mapping."""
    manifest = compute_data_root_manifest(data_root)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
