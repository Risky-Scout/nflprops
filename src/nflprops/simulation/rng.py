"""Deterministic random number generation.

SPEC: docs/IMPLEMENTATION_SPEC.md §3 (determinism), §51 (RNG), §52 (draw counts)
PHASE: 7
STATUS: IMPLEMENTED — this is normative code, do not rewrite it.

Python's built-in hash() is salted per process (PYTHONHASHSEED) and therefore cannot
be used for seeding anything that must reproduce across runs. blake2b is used
instead: stable across processes, machines, and Python versions.

tests/determinism/test_no_builtin_hash.py greps this package for `hash(` to make sure
nobody reintroduces it.
"""

from __future__ import annotations

import hashlib

import numpy as np


def deterministic_seed(*parts: str) -> int:
    """Stable 64-bit seed from string parts.

    Parts are joined with '|' so that ("ab", "c") and ("a", "bc") do not collide.
    """
    key = "|".join(parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def make_rng(model_version: str, game_id: str, as_of: str) -> np.random.Generator:
    """Root generator for one (model_version, game, as_of) simulation.

    Same triple -> same random stream -> byte-identical predictions.
    """
    return np.random.default_rng(deterministic_seed(model_version, game_id, as_of))


def child_rng(
    model_version: str, game_id: str, as_of: str, stream: str
) -> np.random.Generator:
    """Independent named substream.

    Use this so that adding a new stochastic component (say, overtime) does not
    shift the draws of every existing component and silently change all historical
    outputs. Each component owns a named stream:

        child_rng(mv, gid, ts, "targets")
        child_rng(mv, gid, ts, "rush_gains")
        child_rng(mv, gid, ts, "overtime")

    This also makes common random numbers across model versions meaningful — see
    SPEC §52 and the attribution method in §69.
    """
    return np.random.default_rng(
        deterministic_seed(model_version, game_id, as_of, stream)
    )


def monte_carlo_se(p: float, n: int) -> float:
    """Monte Carlo standard error of a probability estimate. SPEC §52."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = min(max(p, 0.0), 1.0)
    return float(np.sqrt(p * (1.0 - p) / n))


def draws_needed(p: float, target_se: float) -> int:
    """Draws required to hit an absolute standard-error target.

    Note the asymmetry called out in SPEC §52: a first_td longshot at p=0.03 and a
    receptions line at p=0.50 need very different N for equal RELATIVE precision.
    The caller applies the relative-SE floor from config.
    """
    if target_se <= 0:
        raise ValueError("target_se must be positive")
    p = min(max(p, 0.0), 1.0)
    return int(np.ceil(p * (1.0 - p) / (target_se**2)))
