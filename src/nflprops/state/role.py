"""Role posteriors. MUST decay faster than skill posteriors.

SPEC: docs/IMPLEMENTATION_SPEC.md §25 §26 §27
PHASE: 5
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 5. Role posteriors. MUST decay faster than skill posteriors.
#
# Uses nflprops.state.empirical_bayes for the mechanics. Hyperparameters (k, lambda)
# are FITTED per metric on a walk-forward grid, never guessed.
#
# SPEC §26: role state must have LOWER lambda than skill state. Asserted after
# fitting by tests/unit/test_role_faster_than_skill.py. If the fit does not produce
# that ordering, REPORT IT — do not force it silently.
