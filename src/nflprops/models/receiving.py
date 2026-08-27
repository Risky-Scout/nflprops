"""Reception gain mean model + empirical residual pool.

SPEC: docs/IMPLEMENTATION_SPEC.md §62
PHASE: 6
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 6. Reception gain mean model + empirical residual pool.
#
# Residual pool stratified by (position, aDOT bucket, catch-depth bucket).
# Empirical rather than Gamma because completed NFL receptions produce negative
# yardage and explosive heavy tails (SPEC §39).
#
# Implements nflprops.models.base.ComponentModel. Imports NOTHING under providers/.
# Optional challenger: structural_GLM + residual_ML, registered as a CHALLENGER and
# promoted only through the Phase 10 gates.
