"""Deterministic RNG. SPEC §3, §51."""

import subprocess
import sys
from pathlib import Path

from nflprops.simulation.rng import (
    child_rng,
    deterministic_seed,
    draws_needed,
    make_rng,
    monte_carlo_se,
)

ROOT = Path(__file__).resolve().parents[2]


def test_seed_is_stable():
    assert deterministic_seed("a", "b") == deterministic_seed("a", "b")


def test_seed_join_is_unambiguous():
    """('ab','c') and ('a','bc') must not collide."""
    assert deterministic_seed("ab", "c") != deterministic_seed("a", "bc")


def test_same_triple_same_stream():
    a = make_rng("2026.1.0", "game-1", "2026-09-10T17:00:00Z").normal(size=50)
    b = make_rng("2026.1.0", "game-1", "2026-09-10T17:00:00Z").normal(size=50)
    assert (a == b).all()


def test_different_as_of_different_stream():
    a = make_rng("2026.1.0", "game-1", "2026-09-10T17:00:00Z").normal(size=50)
    b = make_rng("2026.1.0", "game-1", "2026-09-10T18:00:00Z").normal(size=50)
    assert not (a == b).all()


def test_named_substreams_are_independent():
    """Adding a new component must not shift every other component's draws."""
    t = child_rng("2026.1.0", "g", "ts", "targets").normal(size=20)
    r = child_rng("2026.1.0", "g", "ts", "rush_gains").normal(size=20)
    assert not (t == r).all()


def test_seed_stable_across_processes():
    """The real point: PYTHONHASHSEED must not matter. Built-in hash() would fail."""
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from nflprops.simulation.rng import deterministic_seed;"
        "print(deterministic_seed('model','game','asof'))" % (ROOT / "src")
    )
    out1 = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    ).stdout.strip()
    out2 = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    ).stdout.strip()
    assert out1 and out1 == out2


def test_monte_carlo_se_and_draws():
    n = draws_needed(0.5, 0.0025)
    assert monte_carlo_se(0.5, n) <= 0.0025 + 1e-9


def test_longshot_needs_relative_floor():
    """SPEC §52 — absolute SE is the wrong target for a p=0.03 first-TD price."""
    n_even = draws_needed(0.50, 0.0025)
    n_longshot = draws_needed(0.03, 0.0025)
    assert n_longshot < n_even          # absolute target is easily met...
    # ...but relative precision is terrible, which is why the config has a floor.
    assert monte_carlo_se(0.03, n_longshot) / 0.03 > 0.05
