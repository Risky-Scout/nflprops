"""Learned injury redistribution.

SPEC: docs/IMPLEMENTATION_SPEC.md §29-§52
PHASE: 7
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 7. Learned injury redistribution.
#
# NEVER proportionally redistribute a missing player's share across everyone —
# that is the classic error and it systematically misprices the backup who
# actually inherits the role. Sequence: depth-chart replacement bonus ->
# same-position -> cross-position -> OTHER -> normalize (SPEC §37).
#
# Uses nflprops.simulation.rng for ALL randomness (named substreams). Calls
# nflprops.simulation.invariants after aggregation. An invariant failure ABORTS the
# run — there is no flag that downgrades it to a warning.
