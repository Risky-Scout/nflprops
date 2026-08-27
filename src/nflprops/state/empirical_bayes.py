"""Empirical-Bayes state updating.

SPEC: docs/IMPLEMENTATION_SPEC.md §25, §26, §27
PHASE: 5
STATUS: IMPLEMENTED core mechanics. Hyperparameter FITTING is Phase 5 work.

The update:

    theta_t = w_t * x_t + (1 - w_t) * theta_prior_t
    w_t     = n_t / (n_t + k)

The temporal transition between observations:

    theta_prior_{t+1} = lambda * theta_t + (1 - lambda) * theta_population

`k` controls how fast a player earns his own estimate. `lambda` controls how long
that estimate persists. BOTH ARE LEARNED PER METRIC (SPEC §25) by maximizing
out-of-sample predictive likelihood on a walk-forward grid. Guessing them is the
difference between a model that reacts to a role change in one week and one that
takes six.

The most important consequence, and the highest-leverage idea in this system
(SPEC §26): ROLE state must have LOWER lambda than SKILL state. When a WR1 is ruled
out, the WR2's target share should move immediately while his catch rate and YAC
ability should not. Systems that project "receiving yards" as a single blob get this
wrong, and it is exactly where the market is beatable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EBHyperparams:
    """Per-metric hyperparameters. FITTED, not chosen."""

    k: float           # shrinkage strength (prior pseudo-observations)
    lam: float         # temporal persistence in [0, 1]
    metric: str
    fitted_on: str | None = None   # training window identifier
    is_role_metric: bool = False   # role metrics must have lower lam than skill

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive for metric {self.metric}")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"lambda must be in [0,1] for metric {self.metric}")


@dataclass(frozen=True)
class Posterior:
    """A point-in-time posterior for one (entity, metric)."""

    mean: float
    variance: float
    effective_n: float
    prior_mean: float
    as_of: str
    metric: str

    @property
    def shrinkage_weight(self) -> float:
        """How much of this estimate is the player's own data vs the prior."""
        return self.effective_n / (self.effective_n + 1e-12)


def shrinkage_weight(n: float, k: float) -> float:
    """w = n / (n + k). Monotone increasing in n. SPEC §25."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n / (n + k)


def update(
    observation: float,
    n_observations: float,
    prior_mean: float,
    hp: EBHyperparams,
) -> float:
    """One EB update step."""
    w = shrinkage_weight(n_observations, hp.k)
    return w * observation + (1.0 - w) * prior_mean


def transition(
    posterior_mean: float,
    population_mean: float,
    hp: EBHyperparams,
) -> float:
    """Decay the posterior toward the population between observations."""
    return hp.lam * posterior_mean + (1.0 - hp.lam) * population_mean


def posterior_variance(
    observation_variance: float,
    n_observations: float,
    prior_variance: float,
    hp: EBHyperparams,
) -> float:
    """Posterior variance under the conjugate-normal approximation.

    Tracked because the simulator NEEDS it: Dirichlet concentration kappa is driven
    by posterior uncertainty (SPEC §36). A rookie WR with two games of data and a
    veteran with sixty should not have the same share variance, and if you only track
    means, they will.
    """
    if n_observations <= 0:
        return prior_variance
    precision = (n_observations / max(observation_variance, 1e-12)) + (
        1.0 / max(prior_variance, 1e-12)
    )
    return 1.0 / precision


def kappa_from_uncertainty(
    posterior_var: float,
    base_kappa: float,
    scale: float,
) -> float:
    """Map posterior uncertainty to Dirichlet concentration. SPEC §36.

    Higher uncertainty -> LOWER kappa -> fatter share variance in the allocation.
    `base_kappa` and `scale` are fitted, not chosen.
    """
    if posterior_var < 0:
        raise ValueError("variance must be non-negative")
    return base_kappa / (1.0 + scale * posterior_var)


def fit_hyperparams(
    metric: str,
    walk_forward_data,
    k_grid,
    lam_grid,
    is_role_metric: bool = False,
) -> EBHyperparams:
    """Fit (k, lambda) by maximizing out-of-sample predictive likelihood.

    PHASE 5. Requirements:
      - Expanding-window walk-forward only. No random splits (SPEC §61).
      - Grid or coarse-to-fine search over (k, lambda).
      - Fitted values are stored in the model artifact and printed in the training
        report so a reviewer can see them.
      - After fitting all metrics, assert lambda_role < lambda_skill per position
        group (SPEC §26). If the fit does NOT produce that ordering, report it as a
        finding — do not force it silently, because it means either the role metric
        is mis-specified or the data disagrees with the premise, and both are worth
        knowing.
    """
    raise NotImplementedError("PHASE 5 — see SPEC §25, §26")
