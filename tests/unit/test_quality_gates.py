"""ACCEPTANCE TEST — BLOCK severity halts the pipeline

PHASE: 2
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 2.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 2 not yet implemented")


def test_quality_gates():
    """BLOCK severity halts the pipeline"""
    raise NotImplementedError("PHASE 2")
