"""Closing line value tracking.

SPEC: docs/IMPLEMENTATION_SPEC.md §60
PHASE: 9
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 9. CLV in probability terms and in cents, by prop family, vendor, and
# time-to-kickoff bucket.
#
# PROP AVAILABILITY BIAS (SPEC §60): props that vanish from the board are NOT missing
# at random — they are often the ones the model liked. Every row records whether the
# prop was still quoted at close, and edge is reported CONDITIONAL on availability.
# Without this, the CLV number is flattering nonsense.
