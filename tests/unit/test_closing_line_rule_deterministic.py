"""ACCEPTANCE TEST — close_buffer_seconds honored

PHASE: 9
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 9.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 9 not yet implemented")


def test_closing_line_rule_deterministic():
    """close_buffer_seconds honored"""
    raise NotImplementedError("PHASE 9")
