"""Runtime path resolution for cloned-repo and installed-package execution.

The Git repository keeps human-editable configs/contracts/specs at the repository
root. A synchronized copy ships inside the Python package so an installed wheel
does not depend on repository-relative files.
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGED_RESOURCES = PACKAGE_ROOT / "resources"


def repository_root() -> Path | None:
    """Return the repository root when running from a checkout/editable install."""
    candidate = PACKAGE_ROOT.parents[1]
    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "nflprops").exists():
        return candidate
    return None


def runtime_resource(*parts: str) -> Path:
    """Resolve a config/contract/spec path in repo first, packaged resources second."""
    root = repository_root()
    if root is not None:
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    return PACKAGED_RESOURCES.joinpath(*parts)


def runtime_data_root(relative: str | Path) -> Path:
    """Resolve mutable data relative to the operator's working directory.

    Data must never be written into site-packages or source-control directories merely
    because the package was installed there.
    """
    p = Path(relative)
    return p if p.is_absolute() else Path.cwd() / p
