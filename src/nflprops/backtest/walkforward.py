"""Expanding-window walk-forward orchestration.

SPEC: docs/IMPLEMENTATION_SPEC.md §61-§67
PHASE: 10
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 10. Expanding-window walk-forward orchestration.
#
# EXPANDING WINDOW ONLY. Random train/test splitting is a hard failure
# (tests/leakage/test_no_random_split.py). Train through T -> predict T+1 ->
# FREEZE -> observe -> add -> predict T+2.
