"""Team offensive plays per game/quarter.

SPEC: docs/IMPLEMENTATION_SPEC.md §62
PHASE: 6
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 6. Team offensive plays per game/quarter.
#
# Overdispersed count regression. DO NOT assume Poisson — NFL play counts are
# overdispersed and a Poisson assumption silently narrows every downstream prop
# distribution. Fit and validate dispersion explicitly.
#
# Implements nflprops.models.base.ComponentModel. Imports NOTHING under providers/.
# Optional challenger: structural_GLM + residual_ML, registered as a CHALLENGER and
# promoted only through the Phase 10 gates.
