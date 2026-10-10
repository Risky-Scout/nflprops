"""Round 4: production-shaped current forecast with the STRUCTURAL_CORE
production-candidate calibrator (shadow; never published, never Wizard).

Per target game, in order:

1. FUNDAMENTAL: the production `compute_game_prediction` path (LIVE_PIT
   receipt rules, the configured profile, `model.version`, 20,000 draws, the
   deterministic `(model_version, game_id, as_of)` seed). It reads no
   player-prop quote.
2. CALIBRATION: the candidate calibrator's theta becomes per-draw weights
   (`softmax_weights(theta, compute_draw_features(result))`) -- the same
   coherent joint-game reweighting Phase 10C3A validated. Applied only when
   the artifact's profile/base model version/feature contract match the
   prediction exactly; otherwise the game is priced raw and labelled
   `UNAVAILABLE_PROFILE_MISMATCH` -- never a cross-profile fallback.
3. DOWNSTREAM: book-specific quotes priced against the finished
   distribution (`GamePredictionComputation.price_markets`, the existing
   raw pricing), plus the calibrated probability of the identical event.
   Nothing priced here feeds back into steps 1-2.

Writes, per run: `forecast_manifest.json`, `games.parquet` (one row per
target game: status, prospective classification, hashes),
`distributions.parquet` (raw + calibrated summaries per player/prop) and
`prices.parquet` (one row per book/side).
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nflprops.calibration.artifact import DIRECTLY_LABELED_PROP_TYPES
from nflprops.calibration.entropy_tilting import softmax_weights
from nflprops.calibration.joint_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    compute_draw_features,
)
from nflprops.calibration.payload import deserialize_calibration_payload
from nflprops.config import load, scientific_config_sha256
from nflprops.data.warehouse import Warehouse
from nflprops.domain.enums import PropType
from nflprops.domain.model_profile import (
    ModelProfile,
    parse_model_profile,
    profile_base_model_version,
)
from nflprops.market.odds import (
    american_to_decimal,
    expected_value,
    fair_american_odds,
)
from nflprops.orchestration.manifest import compute_data_manifest_sha256
from nflprops.pipelines.pregame import (
    compute_game_prediction,
    simulation_config_from_app_config,
    state_configs_from_app_config,
)
from nflprops.simulation.game import GameSimulationResult
from nflprops.simulation.props import prop_values

PRODUCTION_N_DRAWS = 20_000
CHECKPOINT_NAME = "ROUND4_SHADOW_CURRENT"
CALIBRATION_APPLIED = "APPLIED"
CALIBRATION_PROFILE_MISMATCH = "UNAVAILABLE_PROFILE_MISMATCH"
CALIBRATION_NO_CALIBRATOR = "UNAVAILABLE_NO_CALIBRATOR"

GENUINE_PROSPECTIVE = "GENUINE_PROSPECTIVE"
HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
NOT_EVALUABLE = "NOT_EVALUABLE"

#: Every prop the joint calibrator reweights coherently; summaries are kept
#: for the directly-labeled set plus the anytime-TD product books quote.
SUMMARY_PROPS: tuple[str, ...] = (*sorted(DIRECTLY_LABELED_PROP_TYPES), PropType.ANYTIME_TD.value)
_HIT_PROPS = frozenset({PropType.ANYTIME_TD, PropType.ANYTIME_TD_1H, PropType.ANYTIME_TD_1Q,
                        PropType.ANYTIME_TD_2H, PropType.FIRST_TD})


#: Milestone markets the certified pricer can price (an AT_LEAST hit
#: distribution exists). It raises -- failing the whole game closed -- on any
#: other milestone (count-style ladders such as `receiving_yards 60+`).
PRICEABLE_MILESTONE_PROPS = frozenset(p.value for p in _HIT_PROPS)
EXCLUDED_COUNT_MILESTONE = "EXCLUDED_UNSUPPORTED_COUNT_MILESTONE"


def split_priceable_quotes(quotes: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Quotes the certified pricer supports, and an explicit count of every
    excluded quote by reason (never a silent drop)."""
    if quotes.is_empty():
        return quotes, {}
    unsupported = (pl.col("market_type") == "milestone") & ~pl.col("prop_type").is_in(
        sorted(PRICEABLE_MILESTONE_PROPS))
    excluded = quotes.filter(unsupported)
    counts = (
        {f"{EXCLUDED_COUNT_MILESTONE}:{r['vendor']}:{r['prop_type']}": int(r["len"])
         for r in excluded.group_by("vendor", "prop_type").len().iter_rows(named=True)}
        if not excluded.is_empty() else {}
    )
    return quotes.filter(~unsupported), counts


class ForecastError(Exception):
    """Fail-closed configuration/identity error."""


def classify_prediction(*, as_of: datetime, completed_at: datetime, kickoff: datetime) -> str:
    """Only a forecast whose cutoff AND completion both precede kickoff is
    prospective evidence; anything else is a replay."""
    if as_of < kickoff and completed_at < kickoff:
        return GENUINE_PROSPECTIVE
    return HISTORICAL_REPLAY


def load_calibrator(manifest_path: Path) -> dict[str, Any]:
    """The candidate manifest plus its theta, re-read from the payload bytes
    and verified against the registered payload SHA."""
    manifest = json.loads(manifest_path.read_text())
    payload_bytes = (manifest_path.parent / manifest["payload_path"]).read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != manifest["payload_sha256"]:
        raise ForecastError("calibrator payload bytes do not match payload_sha256")
    payload = deserialize_calibration_payload(payload_bytes)
    if list(payload.theta) != list(manifest["theta"]):
        raise ForecastError("calibrator payload theta differs from its manifest")
    return {**manifest, "theta_array": np.asarray(payload.theta, dtype=np.float64)}


def calibration_status(
    calibrator: dict[str, Any] | None, *, profile: ModelProfile, model_version: str
) -> str:
    """APPLIED only on an exact profile / science / feature-contract match."""
    if calibrator is None:
        return CALIBRATION_NO_CALIBRATOR
    if (
        calibrator["model_profile"] == profile.value
        and calibrator["base_model_version"] == profile_base_model_version(model_version, profile)
        and calibrator["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    ):
        return CALIBRATION_APPLIED
    return CALIBRATION_PROFILE_MISMATCH


def draw_weights(result: GameSimulationResult, theta: np.ndarray | None) -> np.ndarray:
    n = result.n_draws
    if theta is None:
        return np.full(n, 1.0 / n)
    w = softmax_weights(theta, compute_draw_features(result))
    if not (np.all(np.isfinite(w)) and np.all(w > 0) and abs(float(w.sum()) - 1.0) <= 1e-12):
        raise ForecastError(f"{result.game_id}: calibrated weights fail strict health")
    return w


def _player_values(result: GameSimulationResult) -> dict[str, GameSimulationResult]:
    by_player = result.player_draws.partition_by("player_id", as_dict=True)
    return {
        str(pid[0]): GameSimulationResult(
            game_id=result.game_id, model_version=result.model_version, as_of=result.as_of,
            n_draws=result.n_draws, player_draws=frame, team_draws=result.team_draws,
            first_td_player=result.first_td_player,
        )
        for pid, frame in by_player.items()
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    return float(values[order][min(int(np.searchsorted(cdf, q, side="left")), len(values) - 1)])


def summarize_distributions(
    result: GameSimulationResult, weights: np.ndarray, *, real_players: set[str]
) -> list[dict[str, Any]]:
    """Raw and calibrated summaries of the SAME draws per player/prop."""
    rows: list[dict[str, Any]] = []
    uniform = np.full(result.n_draws, 1.0 / result.n_draws)
    for pid, sub in sorted(_player_values(result).items()):
        if pid not in real_players:
            continue
        position = str(sub.player_draws["position_group"][0])
        for prop_name in SUMMARY_PROPS:
            prop = PropType(prop_name)
            values = prop_values(sub, pid, prop).astype(float)
            raw_probs, cal_probs = (
                np.bincount(np.unique(values, return_inverse=True)[1], weights=w)
                for w in (uniform, weights)
            )
            support = np.unique(values)
            rows.append({
                "game_id": result.game_id, "player_id": pid, "position_group": position,
                "prop_type": prop_name,
                "mean_raw": float(values @ uniform), "mean_calibrated": float(values @ weights),
                "p50_raw": _weighted_quantile(values, uniform, 0.5),
                "p50_calibrated": _weighted_quantile(values, weights, 0.5),
                "p_zero_raw": float(uniform[values == 0].sum()),
                "p_zero_calibrated": float(weights[values == 0].sum()),
                "outcomes": support.astype(np.int64).tolist(),
                "probabilities_raw": raw_probs.tolist(),
                "probabilities_calibrated": cal_probs.tolist(),
            })
    return rows


def calibrate_price_rows(
    result: GameSimulationResult, weights: np.ndarray, priced: list[dict[str, Any]],
    *, status: str,
) -> list[dict[str, Any]]:
    """Add the calibrated probability of each priced event (same draws,
    reweighted). Quote values are read only to define the event."""
    subs = _player_values(result)
    out: list[dict[str, Any]] = []
    for row in priced:
        row = dict(row)
        row["calibration_status"] = status
        p_cal: float | None = None
        p_push_cal = 0.0
        if status == CALIBRATION_APPLIED:
            prop = PropType(row["prop_type"])
            values = prop_values(subs[row["player_id"]], row["player_id"], prop).astype(float)
            line = row["line"]
            if row["side"] == "HIT":
                p_cal = float(weights[values >= 1].sum()) if prop in _HIT_PROPS else None
            elif line is not None:
                p_push_cal = float(weights[values == line].sum())
                mask = values > line if row["side"] == "OVER" else values < line
                p_cal = float(weights[mask].sum())
        row["p_model_calibrated"] = p_cal
        row["p_push_calibrated"] = p_push_cal if p_cal is not None else None
        if p_cal is not None:
            row["model_fair_american_calibrated"] = fair_american_odds(p_cal, p_push_cal)
            row["edge_calibrated"] = (
                p_cal - row["p_market_fair"] if row.get("p_market_fair") is not None else None
            )
            row["ev_per_unit_calibrated"] = expected_value(
                p_cal, american_to_decimal(int(row["american_odds"])), p_push_cal
            )
        else:
            row["model_fair_american_calibrated"] = None
            row["edge_calibrated"] = None
            row["ev_per_unit_calibrated"] = None
        out.append(row)
    return out


def persist_draw_evidence(
    directory: Path, result: GameSimulationResult, real_players: set[str]
) -> str:
    """The joint-game feature matrix plus every non-degenerate per-draw
    outcome vector (player x SUMMARY_PROPS), so any later calibrator can be
    applied to THIS forecast's exact draws without re-simulating. Returns
    the file's SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"__features__": compute_draw_features(result)}
    for pid, sub in sorted(_player_values(result).items()):
        if pid not in real_players:
            continue
        for prop_name in SUMMARY_PROPS:
            values = prop_values(sub, pid, PropType(prop_name)).astype(np.int32)
            if values.any():
                arrays[f"{pid}|{prop_name}"] = values
    path = directory / f"{result.game_id}.npz"
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution_hash(rows: list[dict[str, Any]], column: str) -> str:
    digest = hashlib.sha256()
    for r in sorted(rows, key=lambda r: (r["player_id"], r["prop_type"])):
        digest.update(json.dumps([r["player_id"], r["prop_type"], r["outcomes"], r[column]],
                                 separators=(",", ":")).encode())
    return digest.hexdigest()


def run_forecast(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = parse_model_profile(args.profile)
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        raise ForecastError("--as-of must carry a timezone")
    if args.n_draws != PRODUCTION_N_DRAWS and not args.smoke:
        raise ForecastError(f"production forecast requires {PRODUCTION_N_DRAWS} draws")

    cfg = load(cli_overrides={"model.profile": profile.value})
    model_version = str(cfg.get_path("model.version"))
    sim_cfg = simulation_config_from_app_config(cfg, n_draws=args.n_draws)
    player_cfg, team_cfg = state_configs_from_app_config(cfg)
    calibrator = load_calibrator(Path(args.calibrator)) if args.calibrator else None
    status = calibration_status(calibrator, profile=profile, model_version=model_version)
    if args.require_calibrated and status != CALIBRATION_APPLIED:
        raise ForecastError(f"calibrator is not applicable to {profile.value}: {status}")
    theta = calibrator["theta_array"] if status == CALIBRATION_APPLIED and calibrator else None

    root = Path(args.data_root)
    warehouse = Warehouse(root / "canonical", root / "nflprops.duckdb")
    games = warehouse.read("games")
    targets = (
        games.filter((pl.col("season") == args.season) & (pl.col("week") == args.week))
        .sort("available_at").group_by("canonical_game_id").last().sort("date", "canonical_game_id")
    )
    if args.game_ids:
        targets = targets.filter(pl.col("canonical_game_id").is_in(args.game_ids))

    game_rows: list[dict[str, Any]] = []
    all_dists: list[pl.DataFrame] = []
    all_prices: list[pl.DataFrame] = []
    for target in targets.iter_rows(named=True):
        game_id = str(target["canonical_game_id"])
        kickoff = target["date"]
        base = {"game_id": game_id, "kickoff": kickoff, "status_state": target["status_state"],
                "home_team_id": target["home_canonical_team_id"],
                "visitor_team_id": target["visitor_canonical_team_id"]}
        if kickoff <= as_of:
            game_rows.append({**base, "status": "NOT_TARGET_KICKOFF_BEFORE_AS_OF",
                              "classification": NOT_EVALUABLE})
            continue
        started = datetime.now(UTC)
        comp = compute_game_prediction(
            warehouse, season=args.season, week=args.week, game_id=game_id, as_of=as_of,
            model_version=model_version, n_draws=args.n_draws, simulation_config=sim_cfg,
            player_state_config=player_cfg, team_state_config=team_cfg,
            max_confidence_tier=int(cfg.get_path("market.max_confidence_tier", 2)),
            market_mode="live", model_profile=profile,
        )
        if comp is None:
            game_rows.append({**base, "status": "BLOCKED_UNTRUSTWORTHY_OR_NOT_PIT_VISIBLE",
                              "classification": NOT_EVALUABLE})
            continue
        result = comp.simulation
        if result.n_draws != args.n_draws:
            raise ForecastError(f"{game_id}: simulated {result.n_draws} draws")
        weights = draw_weights(result, theta)
        real_players = {str(p) for p in comp.player_states} & {
            str(p) for p in result.player_draws["player_id"].unique().to_list()}
        dists = summarize_distributions(result, weights, real_players=real_players)
        draws_sha = persist_draw_evidence(out / "draws", result, real_players)
        quotes = comp.latest_quotes.filter(pl.col("canonical_game_id") == game_id)
        priceable, excluded = split_priceable_quotes(quotes)
        priced = calibrate_price_rows(
            result, weights, dataclasses.replace(comp, latest_quotes=priceable).price_markets(),
            status=status,
        )
        completed = datetime.now(UTC)
        game_rows.append({
            **base, "status": "SUCCESS",
            "classification": classify_prediction(as_of=as_of, completed_at=completed,
                                                  kickoff=kickoff),
            "simulation_draws_actual": result.n_draws,
            "simulation_input_sha256": comp.simulation_input_sha256,
            "data_manifest_sha256": compute_data_manifest_sha256(
                warehouse, game_id=game_id, scheduled_as_of=as_of),
            "raw_distribution_sha256": distribution_hash(dists, "probabilities_raw"),
            "calibrated_distribution_sha256": distribution_hash(dists, "probabilities_calibrated"),
            "calibration_status": status,
            "draw_evidence_sha256": draws_sha,
            "players_simulated": len(real_players),
            "book_quotes": quotes.height,
            "book_quotes_priceable": priceable.height,
            "excluded_quotes": json.dumps(excluded, sort_keys=True),
            "priced_rows": len(priced),
            "flow_started_at": started, "flow_completed_at": completed,
        })
        all_dists.append(pl.DataFrame(dists))
        if priced:
            all_prices.append(pl.DataFrame(priced, infer_schema_length=None))
        print(f"{game_id} SUCCESS draws={result.n_draws} players={len(real_players)} "
              f"priced={len(priced)} {completed - started}", flush=True)
        del comp, result, weights
        gc.collect()

    games_df = pl.DataFrame(game_rows, infer_schema_length=None)
    games_df.write_parquet(out / "games.parquet")
    if all_dists:
        pl.concat(all_dists, how="diagonal_relaxed").write_parquet(out / "distributions.parquet")
    if all_prices:
        pl.concat(all_prices, how="diagonal_relaxed").write_parquet(out / "prices.parquet")
    ok = [r for r in game_rows if r["status"] == "SUCCESS"]
    manifest = {
        "checkpoint_name": CHECKPOINT_NAME,
        "publication": "SHADOW_NOT_PUBLISHED",
        "season": args.season, "week": args.week, "as_of": as_of.isoformat(),
        "model_profile": profile.value,
        "model_profile_id": profile_base_model_version(model_version, profile),
        "model_version": model_version,
        "scientific_config_sha256": scientific_config_sha256(cfg),
        "simulation_config": asdict(sim_cfg),
        "n_draws_requested": args.n_draws,
        "calibration_status": status,
        "calibrator_id": calibrator["production_calibrator_id"] if calibrator else None,
        "calibrator_payload_sha256": calibrator["payload_sha256"] if calibrator else None,
        "no_fallback_cross_profile": True,
        "seed_policy": "deterministic_seed(model_version, game_id, as_of)",
        "execution_sha": args.execution_sha,
        "data_root_tables": sorted(p.name for p in (root / "canonical").glob("*.parquet")),
        "games_targeted": len(game_rows),
        "games_succeeded": len(ok),
        "games": [{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in r.items()}
                  for r in game_rows],
    }
    (out / "forecast_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m nflprops.pipelines.production_forecast")
    p.add_argument("--data-root", required=True, help="Directory containing canonical/.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--as-of", required=True)
    p.add_argument("--profile", default=ModelProfile.STRUCTURAL_CORE.value)
    p.add_argument("--calibrator", default=None, help="production_calibrator.json")
    p.add_argument("--require-calibrated", action="store_true")
    p.add_argument("--n-draws", type=int, default=PRODUCTION_N_DRAWS)
    p.add_argument("--smoke", action="store_true", help="tests only: allow fewer draws")
    p.add_argument("--game-ids", nargs="*", default=None)
    p.add_argument("--execution-sha", default=None)
    args = p.parse_args(argv)
    try:
        manifest = run_forecast(args)
    except ForecastError as exc:
        print(f"FORECAST REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: manifest[k] for k in ("model_profile", "calibration_status",
                                               "games_targeted", "games_succeeded")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
