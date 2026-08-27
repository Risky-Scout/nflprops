"""Human-readable weekly reports.

SPEC: docs/IMPLEMENTATION_SPEC.md §69
PHASE: 11
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 11. Shows the DISTRIBUTION, not just a point: median, percentile ladder,
# push probability, Monte Carlo standard error, and availability-scenario entropy.
#
# If you cannot say why a projection moved, you cannot trust it and you certainly
# cannot bet it.
