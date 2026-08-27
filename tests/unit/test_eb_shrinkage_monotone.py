"""ACCEPTANCE TEST — more observations means less shrinkage

PHASE: 5
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 5.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 5 not yet implemented")


def test_eb_shrinkage_monotone():
    """more observations means less shrinkage"""
    raise NotImplementedError("PHASE 5")
