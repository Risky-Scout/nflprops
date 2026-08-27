"""ACCEPTANCE TEST — every model implements fit/predict/sample

PHASE: 6
STATUS: not yet implemented. Skipped, NOT deleted: the acceptance criterion stays
visible in the repository from day one. Remove the skip and write the test as part
of Phase 6.
"""

import pytest

pytestmark = pytest.mark.skip(reason="PHASE 6 not yet implemented")


def test_component_model_interface():
    """every model implements fit/predict/sample"""
    raise NotImplementedError("PHASE 6")
