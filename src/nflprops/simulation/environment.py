"""Shared latent pace and scoring shocks.

SPEC: docs/IMPLEMENTATION_SPEC.md §29-§52
PHASE: 7
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 7. Shared latent pace and scoring shocks.
#
# Z_pace ~ N(0,1) and Z_score ~ N(0,1) shared by BOTH teams. Loadings are FITTED.
# This is the primary source of cross-player correlation within a game, which is
# what makes the joint distribution — and therefore SGP pricing — honest.
#
# Uses nflprops.simulation.rng for ALL randomness (named substreams). Calls
# nflprops.simulation.invariants after aggregation. An invariant failure ABORTS the
# run — there is no flag that downgrades it to a warning.
