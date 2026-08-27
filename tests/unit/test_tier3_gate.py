"""ACCEPTANCE TEST — LOW-quality game cannot enter tier-3 labels

PHASE: 3
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 3.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 3 not yet implemented")


def test_tier3_gate():
    """LOW-quality game cannot enter tier-3 labels"""
    raise NotImplementedError("PHASE 3")
