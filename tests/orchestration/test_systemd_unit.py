"""PHASE 5 §37: static validation of the systemd worker unit.

This never calls `systemctl` (the unit is a deployment artifact only, never
installed on a developer machine) -- it only parses the unit file's text
and checks its structural properties.
"""

from __future__ import annotations

import re
from pathlib import Path

UNIT_PATH = (
    Path(__file__).resolve().parents[2] / "deploy" / "systemd" / "nflprops-prefect-worker.service"
)


def _sections(text: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        header = re.fullmatch(r"\[(\w+)\]", line)
        if header:
            current = header.group(1)
            sections[current] = {}
            continue
        if current is not None and "=" in line:
            key, _, value = line.partition("=")
            sections[current][key.strip()] = value.strip()
    return sections


def test_unit_file_exists() -> None:
    assert UNIT_PATH.is_file()


def test_unit_defines_required_sections() -> None:
    sections = _sections(UNIT_PATH.read_text())
    assert {"Unit", "Service", "Install"} <= set(sections)


def test_environment_file_is_external_not_inline() -> None:
    sections = _sections(UNIT_PATH.read_text())
    env_file = sections["Service"]["EnvironmentFile"]
    assert env_file == "/etc/nflprops/nflprops.env"
    # Must not point inside the application checkout or be a relative path --
    # secrets are managed independently of the deployed code.
    assert not env_file.startswith("/opt/nflprops")


def test_no_secret_looking_values_in_unit_file() -> None:
    # Scoped to actual directive key=value lines (not prose comments, which
    # legitimately explain *where* secrets live) -- no directive's value may
    # itself look like a credential.
    sections = _sections(UNIT_PATH.read_text())
    for section in sections.values():
        for key, value in section.items():
            for forbidden in ("api_key", "password", "secret", "token", "dsn"):
                assert forbidden not in key.lower(), f"unit file must not set {key!r}"
                assert forbidden not in value.lower(), (
                    f"unit file directive {key}={value!r} looks like a credential"
                )


def test_restart_always_configured() -> None:
    sections = _sections(UNIT_PATH.read_text())
    assert sections["Service"]["Restart"] == "always"
    assert int(sections["Service"]["RestartSec"]) > 0


def test_expected_work_pool_and_working_directory() -> None:
    sections = _sections(UNIT_PATH.read_text())
    assert sections["Service"]["WorkingDirectory"] == "/opt/nflprops"
    exec_start = sections["Service"]["ExecStart"]
    assert "prefect worker start" in exec_start
    assert "--pool nflprops-production" in exec_start
    assert exec_start.startswith("/opt/nflprops/.venv/bin/prefect")


def test_runs_as_unprivileged_dedicated_user() -> None:
    sections = _sections(UNIT_PATH.read_text())
    assert sections["Service"]["User"] == "nflprops"
    assert sections["Service"]["Group"] == "nflprops"
    assert sections["Service"]["User"] != "root"
