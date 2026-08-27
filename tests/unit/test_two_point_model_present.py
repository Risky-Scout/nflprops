"""ACCEPTANCE TEST — XP attempts are not approximated as TDs

PHASE: 6
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 6.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 6 not yet implemented")


def test_two_point_model_present():
    """XP attempts are not approximated as TDs"""
    raise NotImplementedError("PHASE 6")
