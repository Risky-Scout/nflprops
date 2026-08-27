"""Proper scoring rules and market-relative diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def clip_probability(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def brier_score(y_true, p) -> float:
    y = np.asarray(y_true, dtype=float)
    pp = clip_probability(p)
    return float(np.mean((pp - y) ** 2))


def log_loss(y_true, p) -> float:
    y = np.asarray(y_true, dtype=float)
    pp = clip_probability(p)
    return float(-np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp)))


def calibration_error(y_true, p, bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins."""
    y = np.asarray(y_true, dtype=float)
    pp = clip_probability(p)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for i in range(bins):
        mask = (pp >= edges[i]) & (
            pp <= edges[i + 1] if i == bins - 1 else pp < edges[i + 1]
        )
        if not mask.any():
            continue
        error += mask.mean() * abs(float(pp[mask].mean()) - float(y[mask].mean()))
    return float(error)


def empirical_crps(draws: np.ndarray, observed: float) -> float:
    """CRPS from Monte Carlo draws: E|X-y| - 1/2 E|X-X'|."""
    x = np.asarray(draws, dtype=float)
    if x.size == 0:
        raise ValueError("empty draw set")
    first = np.mean(np.abs(x - observed))
    # Efficient pairwise absolute-difference expectation from sorted samples.
    xs = np.sort(x)
    n = xs.size
    weights = 2 * np.arange(1, n + 1) - n - 1
    pair = 2.0 * np.sum(weights * xs) / (n * n)
    return float(first - 0.5 * pair)


@dataclass(frozen=True)
class MarketBenchmark:
    n: int
    model_log_loss: float
    market_log_loss: float
    log_loss_improvement: float
    model_brier: float
    market_brier: float
    brier_improvement: float
    model_ece: float
    market_ece: float

    @property
    def beats_market_log_loss(self) -> bool:
        return self.model_log_loss < self.market_log_loss

    @property
    def beats_market_brier(self) -> bool:
        return self.model_brier < self.market_brier


def compare_to_market(y_true, p_model, p_market) -> MarketBenchmark:
    y = np.asarray(y_true, dtype=float)
    pm = np.asarray(p_model, dtype=float)
    pk = np.asarray(p_market, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pm) & np.isfinite(pk)
    y, pm, pk = y[mask], pm[mask], pk[mask]
    if y.size == 0:
        raise ValueError("no comparable model/market rows")
    mll = log_loss(y, pm)
    kll = log_loss(y, pk)
    mb = brier_score(y, pm)
    kb = brier_score(y, pk)
    return MarketBenchmark(
        n=int(y.size),
        model_log_loss=mll,
        market_log_loss=kll,
        log_loss_improvement=kll - mll,
        model_brier=mb,
        market_brier=kb,
        brier_improvement=kb - mb,
        model_ece=calibration_error(y, pm),
        market_ece=calibration_error(y, pk),
    )
