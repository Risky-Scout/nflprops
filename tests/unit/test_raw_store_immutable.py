"""ACCEPTANCE TEST — raw store rejects differing rewrite

PHASE: 2
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 2.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 2 not yet implemented")


def test_raw_store_immutable():
    """raw store rejects differing rewrite"""
    raise NotImplementedError("PHASE 2")
