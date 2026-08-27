"""Leakage detectors run in CI.

SPEC: docs/IMPLEMENTATION_SPEC.md §61-§67
PHASE: 10
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 10. Leakage detectors run in CI.
#
# Every rule in SPEC §66. Any failure fails the build. Includes the check that the
# target prop's OWN price never enters p_fundamental inputs.
