"""ACCEPTANCE TEST — state as_of never exceeds prediction as_of

PHASE: 5
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 5.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 5 not yet implemented")


def test_state_timestamp_not_future():
    """state as_of never exceeds prediction as_of"""
    raise NotImplementedError("PHASE 5")
