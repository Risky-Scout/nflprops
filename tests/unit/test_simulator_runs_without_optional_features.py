"""ACCEPTANCE TEST — no optional feature is required

PHASE: 7
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 7.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 7 not yet implemented")


def test_simulator_runs_without_optional_features():
    """no optional feature is required"""
    raise NotImplementedError("PHASE 7")
