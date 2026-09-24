"""Make the Phase-7B projection fixture helper importable from Phase-8B
tests -- it hand-builds a coherent `GameSimulationResult` with a known
eligible/ineligible player mix and is the natural shared fixture for the
threshold engine too."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projections"))
