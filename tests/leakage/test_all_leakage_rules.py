"""ACCEPTANCE TEST — every rule in SPEC §66

PHASE: 10
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 10.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 10 not yet implemented")


def test_all_leakage_rules():
    """every rule in SPEC §66"""
    raise NotImplementedError("PHASE 10")
