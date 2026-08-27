"""QB-side CONDITIONING ONLY. Yards/completions/TDs are DERIVED.

SPEC: docs/IMPLEMENTATION_SPEC.md §62
PHASE: 6
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 6. QB-side CONDITIONING ONLY. Yards/completions/TDs are DERIVED.
#
# *** ARCHITECTURAL GUARD ***
# This module MUST NOT contain a passing-yards regressor. QB passing yards,
# completions, and TDs are computed by AGGREGATING the receiving events on that
# QB's targets (SPEC §40). If you fit a passing-yards model here, the QB prop and
# the receiver props for the same game will disagree — which is the exact failure
# this whole architecture exists to prevent.
# Enforced by tests/unit/test_no_passing_yards_regressor.py.
#
# Implements nflprops.models.base.ComponentModel. Imports NOTHING under providers/.
# Optional challenger: structural_GLM + residual_ML, registered as a CHALLENGER and
# promoted only through the Phase 10 gates.
