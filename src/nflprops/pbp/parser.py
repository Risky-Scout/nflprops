"""Player attribution from play text.

SPEC: docs/IMPLEMENTATION_SPEC.md §21
PHASE: 3
STATUS: SKELETON
"""

from __future__ import annotations

# PHASE 3. Extracts parsed_passer_id / parsed_rusher_id / parsed_receiver_id /
# parsed_kicker_id plus parser_confidence in [0,1].
#
# BLOCKED UNTIL YARD-LINE SEMANTICS ARE VALIDATED:
#   is_red_zone_candidate and is_goal_to_go_candidate emit None with is_missing=1
#   until tests/provider_contract/test_bdl_yardline_semantics.py passes. The BDL spec
#   does not define whether yard lines are offense-oriented 0-100 or absolute.
#   Guessing wrong silently corrupts every red-zone feature. SPEC §6, §21.
#
# NEVER silently trust regex output — everything is reconciled in reconcile.py.
