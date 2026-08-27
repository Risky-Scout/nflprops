"""Dropback probability given game state.

SPEC: docs/IMPLEMENTATION_SPEC.md §62
PHASE: 6
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 6. Dropback probability given game state.
#
# MUST include current score differential and time remaining. This is the single
# mechanism that creates game script: trailing team passes more -> RB carries
# fall -> targets rise, WITHIN the same simulated game. Fit on realized in-game
# score states, not hand-tuned.
#
# Implements nflprops.models.base.ComponentModel. Imports NOTHING under providers/.
# Optional challenger: structural_GLM + residual_ML, registered as a CHALLENGER and
# promoted only through the Phase 10 gates.
