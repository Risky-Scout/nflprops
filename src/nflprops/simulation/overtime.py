"""Overtime period.

SPEC: docs/IMPLEMENTATION_SPEC.md §29-§52
PHASE: 7
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 7. Overtime period.
#
# Runs when the simulated score is tied at end of regulation. Full-game props
# settle including OT; ignoring it biases yardage and TD props downward in
# exactly the close games where lines are tightest (SPEC §47).
#
# Uses nflprops.simulation.rng for ALL randomness (named substreams). Calls
# nflprops.simulation.invariants after aggregation. An invariant failure ABORTS the
# run — there is no flag that downgrades it to a warning.
