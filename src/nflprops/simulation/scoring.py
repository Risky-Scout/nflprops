"""Scoring opportunities -> TD / FG / XP / 2PT events.

SPEC: docs/IMPLEMENTATION_SPEC.md §29-§52
PHASE: 7
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 7. Scoring opportunities -> TD / FG / XP / 2PT events.
#
# TDs are ASSIGNED to already-simulated completions and carries, never generated
# independently. That is what makes qb.passing_tds == sum(receiving_tds) hold by
# construction, and what prevents a receiver scoring in a game he caught nothing
# in (SPEC §43).
#
# Uses nflprops.simulation.rng for ALL randomness (named substreams). Calls
# nflprops.simulation.invariants after aggregation. An invariant failure ABORTS the
# run — there is no flag that downgrades it to a warning.
