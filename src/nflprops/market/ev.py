"""Expected value and edge computation.

SPEC: docs/IMPLEMENTATION_SPEC.md §57 §60
PHASE: 9
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 9. Uses nflprops.market.odds.expected_value (push-aware).
# Reports edge in probability space and EV per unit staked, tagged with the
# devig_method and devig_confidence that produced the fair price.
