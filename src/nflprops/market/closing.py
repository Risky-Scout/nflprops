"""Deterministic closing-line rule.

SPEC: docs/IMPLEMENTATION_SPEC.md §59
PHASE: 9
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 9. Closing line = latest valid quote received at least
# [market] close_buffer_seconds before scheduled kickoff. FROZEN IN CONFIG.
#
# Never hand-pick whichever closing quote flatters a backtest. Games with no quote
# inside the window are marked closing_line_missing and EXCLUDED from CLV aggregates
# — not filled.
