"""ACCEPTANCE TEST — season_stats never a PIT feature

PHASE: 4
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 4.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 4 not yet implemented")


def test_season_aggregate_not_used_pit():
    """season_stats never a PIT feature"""
    raise NotImplementedError("PHASE 4")
