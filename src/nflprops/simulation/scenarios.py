"""Availability scenario mixture.

SPEC: docs/IMPLEMENTATION_SPEC.md §29-§52
PHASE: 7
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 7. Availability scenario mixture.
#
# P(prop) = sum_s P(s) * P(prop | s). A questionable player produces a genuine
# MIXTURE, not a point estimate with a haircut. The reported uncertainty widens
# correctly instead of pretending we know (SPEC §42).
#
# Uses nflprops.simulation.rng for ALL randomness (named substreams). Calls
# nflprops.simulation.invariants after aggregation. An invariant failure ABORTS the
# run — there is no flag that downgrades it to a warning.
