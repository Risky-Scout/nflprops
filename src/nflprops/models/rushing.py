"""Rush gain mean model + empirical residual pool.

SPEC: docs/IMPLEMENTATION_SPEC.md §62
PHASE: 6
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 6. Rush gain mean model + empirical residual pool.
#
# Residual pool is EMPIRICAL and must contain negative runs, zero-yard carries,
# ordinary gains, and explosive tails. Stratified by (position, context bucket).
# A fitted positive distribution cannot represent a 2-yard loss.
#
# Implements nflprops.models.base.ComponentModel. Imports NOTHING under providers/.
# Optional challenger: structural_GLM + residual_ML, registered as a CHALLENGER and
# promoted only through the Phase 10 gates.
