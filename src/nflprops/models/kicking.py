"""FG make and XP make probabilities.

SPEC: docs/IMPLEMENTATION_SPEC.md §62
PHASE: 6
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 6. FG make and XP make probabilities.
#
# v1 uses a distance-MARGINAL make rate plus an opportunity-quality adjustment.
# BDL does not expose per-attempt FG distance structurally. Distance-aware
# kicking is a v2 item gated on PBP parser validation, and this limitation goes
# in the model card (SPEC §45). Do not imply precision you do not have.
#
# Implements nflprops.models.base.ComponentModel. Imports NOTHING under providers/.
# Optional challenger: structural_GLM + residual_ML, registered as a CHALLENGER and
# promoted only through the Phase 10 gates.
