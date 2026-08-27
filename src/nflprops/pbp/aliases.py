"""Player alias construction from canonical rosters.

SPEC: docs/IMPLEMENTATION_SPEC.md §21
PHASE: 3
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 3. Build alias sets from BOTH teams' canonical rosters for the game:
#   full name, "F.Last", "First Last Jr.", hyphen and apostrophe variants,
#   suffix stripping, punctuation normalization, diacritic folding.
#
# An alias that matches two players on the SAME team is a collision. The parser must
# treat it as a parse FAILURE, not a coin flip (SPEC §21).
