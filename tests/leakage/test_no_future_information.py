"""ACCEPTANCE TEST — available_at <= as_of everywhere

PHASE: 4
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 4.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 4 not yet implemented")


def test_no_future_information():
    """available_at <= as_of everywhere"""
    raise NotImplementedError("PHASE 4")
