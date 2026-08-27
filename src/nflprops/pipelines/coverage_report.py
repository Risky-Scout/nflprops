"""Empirical provider coverage discovery.

SPEC: docs/IMPLEMENTATION_SPEC.md §72
PHASE: 12
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 2. Hardcodes NO counts. Discovers: earliest game, latest game, games per
# season, stats rows, advanced-stat weeks, PBP games, opening-odds coverage,
# opening-prop coverage, roster coverage. Its output is a required input to the
# training-window decision (SPEC §18).
# Empirical provider coverage discovery.
