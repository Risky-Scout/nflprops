"""ACCEPTANCE TEST — every feature has a __is_missing twin

PHASE: 4
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 4.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 4 not yet implemented")


def test_is_missing_columns_present():
    """every feature has a __is_missing twin"""
    raise NotImplementedError("PHASE 4")
