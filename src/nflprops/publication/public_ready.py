"""PUBLIC_READY: the minimal consumer-facing artifact for one checkpoint.

Built on the GitHub executor (`remote_checkpoint.execute_checkpoint`) ONLY
from a run that is SUCCESS / PUBLISHED / production n_draws, on the
production profile, whose calibration gate is APPLIED with a certified
calibrator -- otherwise `PublicReadyError` and no artifact exists. The
artifact ships inside the immutable result bundle (its bytes are covered by
the bundle manifest), so the consumer only fetches, validates and renders it
(`nflprops.publication.consumer`); it never recomputes a probability.

One row per actual book quote (never a cross-book consensus): the side with
the larger calibrated expected value is the pick. Every number is read from
the persisted calibrated product (`player_prop_calibrated_prices`); book
lines/prices are downstream only.

Published fields are deliberately limited (`PUBLIC_ROW_FIELDS`): no science
/ config SHA, seed, calibrator hash, scoring metric, receipt timestamp,
warehouse hash, run id or debug field. The run's audit identity lives in
`result.json`, which records this file's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from nflprops.simulation.props import AT_LEAST_HIT_PROPS

PUBLIC_READY_SCHEMA_VERSION = "nflprops.public_ready/v1"
PUBLIC_READY_FILE = "public_ready.json"

#: The ONLY fields a published row may carry (contracts/public_ready.schema.json).
PUBLIC_ROW_FIELDS: tuple[str, ...] = (
    "opportunity_id",
    "player",
    "position",
    "team",
    "opponent",
    "matchup",
    "kickoff_utc",
    "prop",
    "prop_type",
    "line",
    "pick",
    "model_probability",
    "fair_odds",
    "book",
    "book_odds",
    "edge",
    "projection",
    "updated_at",
)
PUBLIC_DOCUMENT_FIELDS: tuple[str, ...] = (
    "schema_version", "generated_at", "season", "week", "game", "rows",
)
PUBLIC_GAME_FIELDS: tuple[str, ...] = (
    "matchup", "home_team", "away_team", "kickoff_utc", "expires_at",
)

#: Internal fields that must never appear anywhere in the artifact.
FORBIDDEN_FIELDS: frozenset[str] = frozenset({
    "science_sha", "execution_sha", "config_sha256", "scientific_config_sha256",
    "seed", "simulation_seed", "payload_sha256", "calibration_artifact_id",
    "calibrator_id", "brier", "crps", "pit", "log_loss", "collector_received_at",
    "quote_available_at", "provider_updated_at", "data_manifest_sha256",
    "snapshot_manifest_sha256", "run_id", "request_bundle_sha256", "prediction_id",
    "p_model_raw", "debug",
})

EXCLUDED_STALE = "STALE_QUOTE"
EXCLUDED_FUTURE_RECEIPT = "POST_CUTOFF_RECEIPT"
EXCLUDED_INVALID_MAPPING = "INVALID_PLAYER_TEAM_MAPPING"
EXCLUDED_INCOMPATIBLE = "INCOMPATIBLE_CALIBRATION"
EXCLUDED_PLACEHOLDER = "PLACEHOLDER_PLAYER"
EXCLUDED_MISSING_FIELDS = "MISSING_REQUIRED_FIELD"

PROP_LABELS: dict[str, str] = {
    "anytime_td": "Anytime TD",
    "anytime_td_1h": "1st Half Anytime TD",
    "anytime_td_1q": "1st Quarter Anytime TD",
    "anytime_td_2h": "2nd Half Anytime TD",
    "first_td": "First TD",
    "fg_made": "Field Goals Made",
    "fg_made_1h": "1st Half Field Goals Made",
    "interceptions": "Interceptions Thrown",
    "kicking_points": "Kicking Points",
    "longest_pass": "Longest Completion",
    "longest_reception": "Longest Reception",
    "longest_rush": "Longest Rush",
    "passing_attempts": "Pass Attempts",
    "passing_completions": "Pass Completions",
    "passing_tds": "Passing TDs",
    "passing_tds_1h": "1st Half Passing TDs",
    "passing_yards": "Passing Yards",
    "passing_yards_1h": "1st Half Passing Yards",
    "receiving_yards": "Receiving Yards",
    "receiving_yards_1h": "1st Half Receiving Yards",
    "receptions": "Receptions",
    "rushing_attempts": "Rush Attempts",
    "rushing_receiving_yards": "Rush + Rec Yards",
    "rushing_yards": "Rushing Yards",
    "rushing_yards_1h": "1st Half Rushing Yards",
}
_HIT_PROP_VALUES = frozenset(p.value for p in AT_LEAST_HIT_PROPS)
_PICK_LABEL = {"OVER": "OVER", "UNDER": "UNDER", "HIT": "YES"}


class PublicReadyError(ValueError):
    """The run may not produce a PUBLIC_READY artifact (fail closed)."""


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(float(value), digits)


def _latest(frame: pl.DataFrame, key: str) -> pl.DataFrame:
    return frame.unique(key, keep="last", maintain_order=True)


def build_public_ready(
    *,
    run: Any,
    calibration_gate: dict[str, Any],
    production_profile: str,
    production_n_draws: int,
    calibrated: pl.DataFrame,
    projections: pl.DataFrame,
    exclusions: pl.DataFrame,
    games: pl.DataFrame,
    teams: pl.DataFrame,
    players: pl.DataFrame,
    max_quote_age: timedelta | None,
    generated_at: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(PUBLIC_READY document, internal publication audit). Raises
    `PublicReadyError` unless the run is publishable at all."""
    from nflprops.orchestration.run_store import PredictionRunStatus, PublicationStatus

    reasons: list[str] = []
    if run.status is not PredictionRunStatus.SUCCESS:
        reasons.append(f"run status {run.status.value}")
    if run.publication_status is not PublicationStatus.PUBLISHED:
        reasons.append(f"publication status {run.publication_status.value}")
    if int(run.n_draws) != production_n_draws:
        reasons.append(f"n_draws {run.n_draws} is not {production_n_draws}")
    if calibration_gate.get("status") != "APPLIED" or not calibration_gate.get("approved"):
        reasons.append(f"calibration {calibration_gate.get('status')}")
    if calibration_gate.get("model_profile") != production_profile:
        reasons.append(f"profile {calibration_gate.get('model_profile')} is not production")
    if max_quote_age is None:
        reasons.append(f"checkpoint {run.checkpoint_name} has no freshness rule")
    if reasons:
        raise PublicReadyError("; ".join(reasons))
    assert max_quote_age is not None
    artifact_id = calibration_gate.get("calibration_artifact_id")

    game_rows = _latest(games, "canonical_game_id").filter(
        pl.col("canonical_game_id") == run.game_id)
    if game_rows.height != 1:
        raise PublicReadyError(f"game {run.game_id} is not in the games table")
    game = game_rows.row(0, named=True)
    abbr = dict(_latest(teams, "canonical_team_id")
                .select("canonical_team_id", "abbreviation").iter_rows())
    home_id, away_id = game["home_canonical_team_id"], game["visitor_canonical_team_id"]
    home, away = abbr.get(home_id), abbr.get(away_id)
    if not home or not away:
        raise PublicReadyError(f"game {run.game_id} teams have no abbreviation")
    kickoff = run.kickoff_at
    matchup = f"{away} @ {home}"
    names = {
        r["canonical_player_id"]: r for r in _latest(players, "canonical_player_id")
        .select("canonical_player_id", "first_name", "last_name", "position_abbreviation")
        .iter_rows(named=True)
    }
    player_team = dict(
        projections.select("player_id", "team_id").unique("player_id").iter_rows()
    ) if projections.height else {}

    excluded: Counter[str] = Counter()
    if exclusions.height:
        for reason, count in exclusions.group_by("reason").len().iter_rows():
            excluded[f"QUOTE_{reason}"] += int(count)

    if calibrated.is_empty():
        quotes: list[list[dict[str, Any]]] = []
    else:
        quotes = [
            group.to_dicts() for _, group in calibrated.group_by(
                "player_id", "prop_type", "market_type", "vendor", "line", maintain_order=True)
        ]
    rows: list[dict[str, Any]] = []
    for sides in quotes:
        pick = max(sides, key=lambda r: (float(r["ev_per_unit_calibrated"]), r["side"]))
        player_id = str(pick["player_id"])
        prop_type = str(pick["prop_type"])
        received: datetime = pick["quote_available_at"]
        if artifact_id is None or pick.get("calibration_artifact_id") != artifact_id or (
            pick.get("calibration_status") != "APPLIED"
        ):
            excluded[EXCLUDED_INCOMPATIBLE] += 1
            continue
        if player_id.startswith("__"):
            excluded[EXCLUDED_PLACEHOLDER] += 1
            continue
        if received > run.scheduled_as_of:
            excluded[EXCLUDED_FUTURE_RECEIPT] += 1
            continue
        if run.scheduled_as_of - received > max_quote_age:
            excluded[EXCLUDED_STALE] += 1
            continue
        info = names.get(player_id)
        team_id = player_team.get(player_id)
        if info is None or team_id not in (home_id, away_id):
            excluded[EXCLUDED_INVALID_MAPPING] += 1
            continue
        name = " ".join(p for p in (info["first_name"], info["last_name"]) if p)
        is_hit = prop_type in _HIT_PROP_VALUES
        row = {
            "opportunity_id": hashlib.sha256(
                f"{run.game_id}|{player_id}|{prop_type}|{pick['vendor']}|{pick['line']}"
                .encode()).hexdigest()[:20],
            "player": name,
            "position": info["position_abbreviation"],
            "team": abbr.get(team_id),
            "opponent": abbr.get(away_id if team_id == home_id else home_id),
            "matchup": matchup,
            "kickoff_utc": _iso(kickoff),
            "prop": PROP_LABELS.get(prop_type, prop_type.replace("_", " ").title()),
            "prop_type": prop_type,
            "line": None if is_hit else _round(pick["line"], 1),
            "pick": _PICK_LABEL.get(str(pick["side"])),
            "model_probability": _round(pick["p_model_calibrated"], 4),
            "fair_odds": None if pick["model_fair_american_calibrated"] is None
            else round(float(pick["model_fair_american_calibrated"])),
            "book": str(pick["vendor"]),
            "book_odds": int(pick["american_odds"]),
            "edge": _round(pick["ev_per_unit_calibrated"], 4),
            "projection": None if is_hit else _round(pick["model_mean_calibrated"], 1),
            "updated_at": _iso(received),
        }
        required = ("player", "team", "opponent", "prop", "pick", "model_probability",
                    "fair_odds", "book", "book_odds", "edge", "updated_at")
        if any(row[f] in (None, "") for f in required) or (not is_hit and row["line"] is None):
            excluded[EXCLUDED_MISSING_FIELDS] += 1
            continue
        rows.append(row)

    rows.sort(key=lambda r: (-r["edge"], r["player"], r["prop_type"], r["book"]))
    document = {
        "schema_version": PUBLIC_READY_SCHEMA_VERSION,
        "generated_at": _iso(generated_at),
        "season": int(run.season),
        "week": int(run.week),
        "game": {
            "matchup": matchup,
            "home_team": home,
            "away_team": away,
            "kickoff_utc": _iso(kickoff),
            "expires_at": _iso(kickoff),
        },
        "rows": rows,
    }
    audit = {
        "rows": len(rows),
        "books": sorted({r["book"] for r in rows}),
        "excluded": dict(sorted(excluded.items())),
        "calibrated_quotes": len(quotes),
    }
    return document, audit


def public_ready_bytes(document: dict[str, Any]) -> bytes:
    """Canonical, deterministic serialization."""
    return (json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()
