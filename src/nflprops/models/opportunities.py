"""Directed-target rate, share priors, Dirichlet concentration.

SPEC: docs/IMPLEMENTATION_SPEC.md §62
PHASE: 6
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 6. Directed-target rate, share priors, Dirichlet concentration.
#
# p_directed = sum(player targets) / team pass attempts, estimated from history.
# Guarantees directed_targets <= pass_attempts BY CONSTRUCTION (INV004).
# kappa comes from posterior uncertainty: higher uncertainty -> lower kappa ->
# fatter share variance (SPEC §36).
#
# Implements nflprops.models.base.ComponentModel. Imports NOTHING under providers/.
# Optional challenger: structural_GLM + residual_ML, registered as a CHALLENGER and
# promoted only through the Phase 10 gates.
