"""Provider contract: real pinned spec coverage.

This repository intentionally refuses to fake a pin.  The archive-building
environment had no outbound DNS, so the exact 2,939-line upstream document could
not be copied into the artifact.  Run `nflprops provider pin bdl` in a networked
environment, then this test becomes active.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "specs/providers/bdl/nfl.yml"


def test_spec_coverage():
    spec = yaml.safe_load(SPEC.read_text())
    if "PLACEHOLDER" in str(spec.get("info", {}).get("title", "")).upper():
        pytest.skip("real BDL spec has not yet been pinned into this artifact")
    result = subprocess.run(
        [sys.executable, str(ROOT/"tools/verify_spec_coverage.py"), "--strict-fields"],
        cwd=ROOT,
    )
    assert result.returncode == 0
