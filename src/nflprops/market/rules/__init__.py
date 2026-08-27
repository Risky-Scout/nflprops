"""Versioned sportsbook settlement rules. SEPARATE from model rules.

SPEC: docs/IMPLEMENTATION_SPEC.md §44
PHASE: 9
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 9. Settlement rules are VERSIONED SEPARATELY from model rules (SPEC §44).
#
# Example: whether anytime_td includes return and defensive touchdowns varies by
# vendor. The simulator produces BOTH offensive_tds and all_tds; the settlement rule
# selects which one prices a given vendor's market.
#
# Never bake a settlement assumption into simulator code — when a book changes its
# rule you must be able to change one YAML file, not retrain.
