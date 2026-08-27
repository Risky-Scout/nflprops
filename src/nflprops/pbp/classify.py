"""Play family classification.

SPEC: docs/IMPLEMENTATION_SPEC.md §21
PHASE: 3
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 3. Classify play_family from type_slug / type_abbreviation / type_text
# BEFORE parsing free text. Structured type fields are far more reliable than regex
# over the description, so they lead.
