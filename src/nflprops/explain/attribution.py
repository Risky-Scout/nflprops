"""Causal effect decomposition.

SPEC: docs/IMPLEMENTATION_SPEC.md §69
PHASE: 11
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 11. Attribution is computed by ABLATING each effect group through the
# simulator with COMMON RANDOM NUMBERS — a causal decomposition within the model,
# not a post-hoc SHAP narrative over a black box.
#
# Groups: baseline, team volume, player role, QB, opponent, game-market environment,
# injury redistribution.
#
# Components must sum to the total within a documented tolerance
# (tests/unit/test_explanation_additivity.py).
