"""ACCEPTANCE TEST — benchmark uses same-as_of market price

PHASE: 10
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 10.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 10 not yet implemented")


def test_market_benchmark_same_asof():
    """benchmark uses same-as_of market price"""
    raise NotImplementedError("PHASE 10")
