"""ACCEPTANCE TEST — property-based over 200+ configs

PHASE: 7
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 7.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 7 not yet implemented")


def test_all_invariants_on_random_games():
    """property-based over 200+ configs"""
    raise NotImplementedError("PHASE 7")
