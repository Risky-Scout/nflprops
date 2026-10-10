"""Phase 10C3A OOF prediction rows, proper-score reports and §65 gate
evidence (evaluation/reporting plumbing only).

Consumes predictions and realized outcomes only -- it never simulates and
never changes a model input. Every metric reuses the repository's existing
definition:

* binary (`anytime_td`): `nflprops.backtest.metrics.brier_score` /
  `log_loss` / `calibration_error`, `nflprops.backtest.evaluation.reliability_curve`;
* count/yardage props: `nflprops.calibration.scoring.crps_from_pmf` (the
  calibration objective's own CRPS) and
  `nflprops.backtest.evaluation.distributional_summary` (MAE, CRPS, WIS from
  the p05..p95 quantiles, 50/80/90% coverage, PIT KS distance);
* quantiles: `nflprops.distributions.pmf.weighted_inverse_cdf_quantile`;
* gates: `nflprops.backtest.promotion.full_promotion_gate` (SPEC §65) and
  `market_superiority_gate`, with their locked default thresholds, where the
  "market" benchmark slot is the theta=0 raw model -- exactly how the
  existing 10C3A gate (`challenger.evaluate_promotion_gate`) frames the
  calibrated-vs-raw comparison. No sportsbook price is read anywhere.

The PIT of a discrete outcome is the standard randomized PIT
`F(y-1) + u * P(Y=y)` (Czado, Gneiting & Held 2009) with `u` a fixed
pseudo-uniform derived by SHA-256 from (game, player, prop) -- the same `u`
for raw and calibrated rows (common random numbers), deterministic across runs.

SPEC §65 names "stable across seasons" and "stable in weeks 1-4 and weeks
14+" without a numeric rule. This module evaluates them as the §64 slice
benchmarks (`evaluate_backtest` slices by season, weeks 1-4 and weeks 14+)
each passing the same locked `market_superiority_gate`; the report records
that operationalization explicitly (`STABILITY_RULE`).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl

from nflprops.backtest.evaluation import distributional_summary, reliability_curve
from nflprops.backtest.metrics import (
    MarketBenchmark,
    brier_score,
    calibration_error,
    compare_to_market,
    log_loss,
)
from nflprops.backtest.promotion import (
    FullPromotionEvidence,
    PromotionDecision,
    full_promotion_gate,
    market_superiority_gate,
)
from nflprops.calibration.compact_game import CompactGame, compact_weighted_pmf
from nflprops.calibration.entropy_tilting import softmax_weights
from nflprops.calibration.scoring import crps_from_pmf
from nflprops.calibration.weighted_pmf import WeightedPMF
from nflprops.distributions.pmf import (
    BINARY_COUNT_COLLAPSE_PROPS,
    weighted_inverse_cdf_quantile,
)

OOF_ROW_VERSION = "phase10c3a_oof_row/v1"
QUANTILE_LEVELS: tuple[tuple[str, float], ...] = (
    ("p05", 0.05), ("p10", 0.10), ("p25", 0.25), ("p50", 0.50),
    ("p75", 0.75), ("p90", 0.90), ("p95", 0.95),
)
STABILITY_RULE = (
    "SPEC §65 gives no numeric stability rule; evaluated as every §64 slice "
    "benchmark (each season / weeks 1-4 / weeks 14+) passing the locked "
    "market_superiority_gate (min_rows=500, log loss and Brier strictly better, "
    "ECE degradation <= 0.01), calibrated vs theta=0 raw"
)
_BINARY = frozenset(p.value for p in BINARY_COUNT_COLLAPSE_PROPS)


def is_binary_prop(prop_type: str) -> bool:
    return prop_type in _BINARY


def pit_uniform(game_id: str, player_id: str, prop_type: str) -> float:
    digest = hashlib.sha256(f"pit-u/v1|{game_id}|{player_id}|{prop_type}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2.0**64


def distribution_sha256(outcomes: Sequence[int], probabilities: Sequence[float]) -> str:
    payload = ",".join(f"{o}:{p!r}" for o, p in zip(outcomes, probabilities, strict=True))
    return hashlib.sha256(payload.encode()).hexdigest()


def _pmf_row(pmf: WeightedPMF, observed: float, u: float) -> dict[str, Any]:
    x = np.asarray(pmf.outcomes, dtype=np.float64)
    p = np.asarray(pmf.probabilities, dtype=np.float64)
    mean = float(np.sum(x * p))
    below = float(p[x < observed].sum())
    at = float(p[x == observed].sum())
    row: dict[str, Any] = {
        "outcomes": list(pmf.outcomes),
        "probabilities": list(pmf.probabilities),
        "support_min": pmf.support_min,
        "support_max": pmf.support_max,
        "model_mean": mean,
        "model_variance": float(np.sum(p * (x - mean) ** 2)),
        "p_zero": float(p[x == 0].sum()),
        "p_le_actual": below + at,
        "p_ge_actual": 1.0 - below,
        "pit": min(max(below + u * at, 0.0), 1.0),
        "distribution_sha256": distribution_sha256(pmf.outcomes, pmf.probabilities),
    }
    for name, level in QUANTILE_LEVELS:
        row[name] = weighted_inverse_cdf_quantile(pmf, level)  # type: ignore[arg-type]
    if is_binary_prop(pmf.prop_type.value):
        p_hit = sum(o_p for o, o_p in zip(pmf.outcomes, pmf.probabilities, strict=True) if o >= 1)
        row["p_final"] = p_hit
        row["log_loss"] = log_loss([observed], [p_hit])
        row["brier"] = brier_score([observed], [p_hit])
        row["crps"] = None
    else:
        row["p_final"] = None
        row["log_loss"] = None
        row["brier"] = None
        row["crps"] = crps_from_pmf(pmf.outcomes, pmf.probabilities, observed)
    return row


def game_rows(
    game: CompactGame,
    theta: np.ndarray,
    *,
    variant: str,
    season: int,
    week: int,
    fold_id: str,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """One OOF row per labeled (player, prop) of `game` under `theta`
    (`variant` "raw" means theta = 0). The probabilities are exactly the
    PMFs the calibration objective scores."""
    weights = softmax_weights(theta, game.draw_features())
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(game.labels):
        pmf = compact_weighted_pmf(game, index, weights)
        prop = label.prop_type.value
        row: dict[str, Any] = {
            "row_version": OOF_ROW_VERSION,
            "variant": variant,
            "prediction_id": f"{game.game_id}|{label.player_id}|{prop}",
            "game_id": game.game_id,
            "player_id": label.player_id,
            "prop_type": prop,
            "position_group": game.position_groups[index],
            "season": season,
            "week": week,
            "fold_id": fold_id,
            "cutoff_as_of": game.as_of.isoformat(),
            "outcome_available_at": game.outcome_available_at.isoformat(),
            "injury_data_available": game.injury_data_available,
            "n_draws": game.n_draws,
            "seed_lineage": f"child_rng({game.model_version}|{game.game_id}|{game.as_of.isoformat()}|<stream>)",
            "theta": [float(t) for t in theta],
            "actual_value": label.observed_value,
            "outcome": (1 if label.observed_value >= 1 else 0) if is_binary_prop(prop) else None,
            **identity,
        }
        row.update(_pmf_row(pmf, label.observed_value, pit_uniform(game.game_id, label.player_id, prop)))
        rows.append(row)
    return rows


# ------------------------------------------------------------------ scoring


def _finite_mean(values: Iterable[float | None]) -> float | None:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def binary_summary(rows: pl.DataFrame) -> dict[str, Any]:
    binary = rows.filter(pl.col("p_final").is_not_null())
    if binary.is_empty():
        return {"n": 0}
    y = binary["outcome"].to_numpy().astype(float)
    p = binary["p_final"].to_numpy().astype(float)
    return {
        "n": int(binary.height),
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "ece": calibration_error(y, p),
        "mean_probability": float(p.mean()),
        "observed_rate": float(y.mean()),
        "reliability": [asdict(b) for b in reliability_curve(binary)],
    }


def continuous_summary(rows: pl.DataFrame) -> dict[str, Any]:
    dist = rows.filter(pl.col("crps").is_not_null())
    if dist.is_empty():
        return {"n": 0}
    summary = asdict(distributional_summary(dist))
    y = dist["actual_value"].to_numpy().astype(float)
    summary.update(
        {
            "zero_mass_predicted": float(dist["p_zero"].mean()),  # type: ignore[arg-type]
            "zero_rate_observed": float((y == 0).mean()),
            "below_p05_rate": float((y < dist["p05"].to_numpy()).mean()),
            "above_p95_rate": float((y > dist["p95"].to_numpy()).mean()),
            "p_le_actual_le_005_rate": float((dist["p_le_actual"].to_numpy() <= 0.05).mean()),
            "p_ge_actual_le_005_rate": float((dist["p_ge_actual"].to_numpy() <= 0.05).mean()),
        }
    )
    return summary


def score_rows(rows: pl.DataFrame) -> dict[str, Any]:
    return {"n_rows": rows.height, "binary": binary_summary(rows),
            "distributional": continuous_summary(rows)}


def segment_scores(rows: pl.DataFrame, column: str) -> dict[str, Any]:
    return {
        str(value): score_rows(rows.filter(pl.col(column) == value))
        for value in sorted(rows[column].unique().to_list(), key=str)
    }


def zero_tail_report(rows: pl.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prop in sorted(rows["prop_type"].unique().to_list()):
        sub = rows.filter(pl.col("prop_type") == prop)
        y = sub["actual_value"].to_numpy().astype(float)
        entry: dict[str, Any] = {
            "n": sub.height,
            "zero_mass_predicted": float(sub["p_zero"].mean()),  # type: ignore[arg-type]
            "zero_rate_observed": float((y == 0).mean()),
            "below_p05_rate": float((y < sub["p05"].to_numpy()).mean()),
            "above_p95_rate": float((y > sub["p95"].to_numpy()).mean()),
            "mean_predicted": float(sub["model_mean"].mean()),  # type: ignore[arg-type]
            "mean_observed": float(y.mean()),
        }
        out[str(prop)] = entry
    return out


# -------------------------------------------------------------------- gates


def _paired(raw: pl.DataFrame, cal: pl.DataFrame) -> pl.DataFrame:
    keys = ["prediction_id"]
    left = raw.select([*keys, "season", "week", "prop_type", "outcome",
                       pl.col("p_final").alias("p_raw")])
    right = cal.select([*keys, pl.col("p_final").alias("p_cal")])
    paired = left.join(right, on=keys, how="inner", validate="1:1")
    if paired.height != cal.height or paired.height != raw.height:
        raise ValueError("raw/calibrated OOF rows are not one-to-one")
    return paired


def calibrated_vs_raw_benchmark(paired: pl.DataFrame) -> MarketBenchmark | None:
    binary = paired.filter(pl.col("p_cal").is_not_null())
    if binary.is_empty():
        return None
    return compare_to_market(
        binary["outcome"].to_numpy(), binary["p_cal"].to_numpy(), binary["p_raw"].to_numpy()
    )


def _slice_gate(paired: pl.DataFrame) -> tuple[bool | None, dict[str, Any]]:
    benchmark = calibrated_vs_raw_benchmark(paired)
    if benchmark is None:
        return None, {"n": 0}
    decision = market_superiority_gate(benchmark)
    return decision.promote, {**asdict(benchmark), "promote": decision.promote,
                              "reasons": list(decision.reasons)}


def promotion_gates(
    raw_scored: pl.DataFrame,
    cal_scored: pl.DataFrame,
    *,
    zero_leakage_failures: bool,
    simulation_invariants_pass: bool,
    reproducibility_pass: bool,
) -> dict[str, Any]:
    """SPEC §65 via the locked `full_promotion_gate`, calibrated (challenger)
    vs theta=0 raw (benchmark), over the rows both variants scored."""
    paired = _paired(raw_scored.filter(pl.col("p_final").is_not_null()),
                     cal_scored.filter(pl.col("p_final").is_not_null()))
    benchmark = calibrated_vs_raw_benchmark(paired)
    raw_dist = continuous_summary(raw_scored)
    cal_dist = continuous_summary(cal_scored)

    def not_worse(key: str) -> bool | None:
        r, c = raw_dist.get(key), cal_dist.get(key)
        return None if r is None or c is None else bool(c <= r)

    season_slices = {}
    season_pass: list[bool | None] = []
    for season in sorted(paired["season"].unique().to_list()):
        ok, detail = _slice_gate(paired.filter(pl.col("season") == season))
        season_slices[str(season)] = detail
        season_pass.append(ok)
    early_ok, early = _slice_gate(paired.filter(pl.col("week").is_between(1, 4)))
    late_ok, late = _slice_gate(paired.filter(pl.col("week") >= 14))
    stable_seasons: bool | None = (
        None if not season_pass or any(v is None for v in season_pass) else all(season_pass)
    )

    observed = {
        "benchmark": asdict(benchmark) if benchmark is not None else None,
        "crps": {"raw": raw_dist.get("mean_crps"), "calibrated": cal_dist.get("mean_crps")},
        "wis": {"raw": raw_dist.get("mean_wis"), "calibrated": cal_dist.get("mean_wis")},
        "pit_ks_uniform": {"raw": raw_dist.get("pit_ks_uniform"),
                           "calibrated": cal_dist.get("pit_ks_uniform")},
        "season_slices": season_slices,
        "weeks_1_4": early,
        "weeks_14_plus": late,
        "stability_rule": STABILITY_RULE,
    }
    if benchmark is None:
        return {"decision": {"promote": False, "reasons": ["no binary evidence"]},
                "rows": [], "observed": observed}
    evidence = FullPromotionEvidence(
        benchmark=benchmark,
        crps_not_worse=not_worse("mean_crps"),
        wis_not_worse=not_worse("mean_wis"),
        pit_not_worse=not_worse("pit_ks_uniform"),
        stable_across_seasons=stable_seasons,
        stable_weeks_1_4=early_ok,
        stable_weeks_14_plus=late_ok,
        zero_leakage_failures=zero_leakage_failures,
        simulation_invariants_pass=simulation_invariants_pass,
        reproducibility_pass=reproducibility_pass,
    )
    decision: PromotionDecision = full_promotion_gate(evidence)
    gate_rows = _gate_rows(evidence, observed)
    return {"decision": {"promote": decision.promote, "reasons": list(decision.reasons)},
            "rows": gate_rows, "observed": observed}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else str(value)
    return str(value)


def _gate_rows(evidence: FullPromotionEvidence, observed: dict[str, Any]) -> list[dict[str, str]]:
    b = evidence.benchmark

    def verdict(value: bool | None) -> str:
        return "PASS" if value is True else ("MISSING" if value is None else "FAIL")

    return [
        {"gate": "sample_size", "rule": "N >= 500 (market_superiority_gate min_rows)",
         "observed": f"N={b.n}", "result": verdict(b.n >= 500)},
        {"gate": "aggregate_log_loss", "rule": "calibrated log loss < raw log loss",
         "observed": f"{_fmt(b.model_log_loss)} vs {_fmt(b.market_log_loss)}",
         "result": verdict(b.beats_market_log_loss)},
        {"gate": "brier", "rule": "calibrated Brier < raw Brier (locked gate is strict)",
         "observed": f"{_fmt(b.model_brier)} vs {_fmt(b.market_brier)}",
         "result": verdict(b.beats_market_brier)},
        {"gate": "calibration_error", "rule": "calibrated ECE <= raw ECE + 0.01",
         "observed": f"{_fmt(b.model_ece)} vs {_fmt(b.market_ece)}",
         "result": verdict(b.model_ece <= b.market_ece + 0.01)},
        {"gate": "crps", "rule": "calibrated mean CRPS <= raw",
         "observed": f"{_fmt(observed['crps']['calibrated'])} vs {_fmt(observed['crps']['raw'])}",
         "result": verdict(evidence.crps_not_worse)},
        {"gate": "wis", "rule": "calibrated mean WIS <= raw",
         "observed": f"{_fmt(observed['wis']['calibrated'])} vs {_fmt(observed['wis']['raw'])}",
         "result": verdict(evidence.wis_not_worse)},
        {"gate": "pit", "rule": "calibrated PIT KS-to-uniform <= raw",
         "observed": f"{_fmt(observed['pit_ks_uniform']['calibrated'])} vs "
                     f"{_fmt(observed['pit_ks_uniform']['raw'])}",
         "result": verdict(evidence.pit_not_worse)},
        {"gate": "stable_across_seasons", "rule": STABILITY_RULE,
         "observed": ", ".join(f"{k}:{v.get('promote')}" for k, v in observed["season_slices"].items()),
         "result": verdict(evidence.stable_across_seasons)},
        {"gate": "stable_weeks_1_4", "rule": STABILITY_RULE,
         "observed": f"promote={observed['weeks_1_4'].get('promote')} "
                     f"reasons={observed['weeks_1_4'].get('reasons')}",
         "result": verdict(evidence.stable_weeks_1_4)},
        {"gate": "stable_weeks_14_plus", "rule": STABILITY_RULE,
         "observed": f"promote={observed['weeks_14_plus'].get('promote')} "
                     f"reasons={observed['weeks_14_plus'].get('reasons')}",
         "result": verdict(evidence.stable_weeks_14_plus)},
        {"gate": "zero_leakage_failures", "rule": "fold chronology/overlap checks pass",
         "observed": str(evidence.zero_leakage_failures),
         "result": verdict(evidence.zero_leakage_failures)},
        {"gate": "simulation_invariants", "rule": "weight positivity/normalization + first-TD simplex",
         "observed": str(evidence.simulation_invariants_pass),
         "result": verdict(evidence.simulation_invariants_pass)},
        {"gate": "reproducibility", "rule": "independent re-replay: identical compact hashes, theta, objective",
         "observed": str(evidence.reproducibility_pass),
         "result": verdict(evidence.reproducibility_pass)},
    ]
