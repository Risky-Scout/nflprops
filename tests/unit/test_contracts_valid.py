"""The machine-readable contracts must stay internally consistent."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_validate_contracts_passes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "validate_contracts.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
