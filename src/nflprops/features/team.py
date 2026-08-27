"""Team-state features: pace, tendency, sack rate, third down.

SPEC: docs/IMPLEMENTATION_SPEC.md §24
PHASE: 4
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 4. Team-state features: pace, tendency, sack rate, third down.
#
# Every feature materializes as TWO columns: <name> and <name>__is_missing.
# BLANKET ZERO-FILLING IS PROHIBITED (SPEC §23). A zero target share and an unknown
# target share are completely different statements and the model must be able to
# tell them apart.
#
# Definitions, available_at rules, and null policies come from
# contracts/feature_registry.yml. Do not invent a feature here that is not there.
