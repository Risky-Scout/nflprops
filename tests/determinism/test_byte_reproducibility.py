"""ACCEPTANCE TEST — rebuild from raw + manifest, byte-identical

PHASE: 10
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 10.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 10 not yet implemented")


def test_byte_reproducibility():
    """rebuild from raw + manifest, byte-identical"""
    raise NotImplementedError("PHASE 10")
