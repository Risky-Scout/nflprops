"""The market snapshot collector.

SPEC: docs/IMPLEMENTATION_SPEC.md §58
PHASE: 9
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 9 — BUT DEPLOY AS SOON AS PHASE 1 LANDS.
#
# BDL retains NO historical live prop data. Every week you do not collect is a week
# of market history that can never be bought, backfilled, or recovered. This module
# is the moat.
#
# Retained checkpoints: OPEN, T-48H, T-24H, T-12H, T-6H, T-3H, T-1H, T-30M, T-10M,
# CLOSE. Poll frequency increases toward kickoff. DO NOT store only "latest".
#
# Collector health is monitored: a missing checkpoint raises an alert, because a
# silent collector failure destroys history permanently.
