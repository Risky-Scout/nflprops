"""Scoring opportunities, TD split, and the two-point choice.

SPEC: docs/IMPLEMENTATION_SPEC.md §62
PHASE: 6
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 6. Scoring opportunities, TD split, and the two-point choice.
#
# Includes P(go_for_two | score differential, quarter, time). The common shortcut
# XP_attempts ~= team_TDs breaks kicking_points in exactly the late-game spots
# where the market is softest (SPEC §43). Model it.
#
# Implements nflprops.models.base.ComponentModel. Imports NOTHING under providers/.
# Optional challenger: structural_GLM + residual_ML, registered as a CHALLENGER and
# promoted only through the Phase 10 gates.
