"""ACCEPTANCE TEST — canonical IDs never rewritten

PHASE: 2
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 2.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 2 not yet implemented")


def test_entity_resolution_stable():
    """canonical IDs never rewritten"""
    raise NotImplementedError("PHASE 2")
