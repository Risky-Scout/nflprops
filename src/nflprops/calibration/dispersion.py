"""PIT / rank histograms and variance inflation.

SPEC: docs/IMPLEMENTATION_SPEC.md §53 §63
PHASE: 8
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 8. PIT / rank histograms and variance inflation.
#
# MANDATORY, not optional. Structural simulators are almost always UNDER-dispersed:
# good means, bad tails, and a slow bleed on high-line overs. A U-shaped PIT
# histogram is the tell.
#
# Remedy order: (1) find the missing structural variance source — usually role or
# availability uncertainty; (2) widen residual pools / lower kappa where justified;
# (3) ONLY LAST, a fitted variance-inflation factor, which must be recorded in the
# model card as a known crutch (SPEC §53).
