"""Shared test fixtures.

Tests NEVER hit the live network. Provider tests run against saved fixtures in
tests/fixtures/bdl/. If you find yourself adding a real HTTP call to a test, stop —
that test will start failing for reasons unrelated to your code, and you will learn
to ignore it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def bdl_fixtures() -> Path:
    return FIXTURES / "bdl"


def phase(n: int):
    """Mark a test as belonging to a not-yet-built phase.

    Skipped rather than deleted so the acceptance criteria stay visible in the
    repository from day one. Un-skip as you implement.
    """
    return pytest.mark.skip(reason=f"PHASE {n} not yet implemented")
