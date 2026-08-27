"""Reconciliation against structured stats.

SPEC: docs/IMPLEMENTATION_SPEC.md §22
PHASE: 3
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 3. Compare parser-reconstructed totals against canonical structured player-game stats for:
#   pass attempts, completions, pass yards, rush attempts, rush yards,
#   receptions, receiving yards, rushing TDs, receiving TDs, FG made
#
# Assign pbp_quality in {HIGH, MEDIUM, LOW, FAIL} using the CONFIG threshold
# ([pbp] minimum_reconciliation_score), not a hardcoded number.
#
# GATE: tier-3 props (halves, quarters, first_td, longest_pass) train ONLY on
# HIGH-quality games. Exclusions and their counts are written to the training
# manifest so the reduced effective sample size is always visible in the report.
