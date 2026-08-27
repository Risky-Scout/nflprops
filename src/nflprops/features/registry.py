"""Feature registry loader and validator.

SPEC: docs/IMPLEMENTATION_SPEC.md §23
PHASE: 4
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 4. Loads and validates contracts/feature_registry.yml.
#
# Enforces BOTH directions (tests/leakage/test_feature_registry_complete.py):
#   - every column written to player_features / game_features has a registry entry
#   - every registry entry has an implementation
#
# A feature that is not in the registry MUST NOT reach a model. This is what keeps
# the system from degenerating into feature soup that nobody can audit two seasons
# later.
