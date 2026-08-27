"""ACCEPTANCE TEST — run manifest write/read

PHASE: 0
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 0.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 0 not yet implemented")


def test_manifest_roundtrip():
    """run manifest write/read"""
    raise NotImplementedError("PHASE 0")
