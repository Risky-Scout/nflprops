"""BLOCK 4: choose the ONE pending checkpoint request a GitHub executor
(`.github/workflows/checkpoint-execute.yml`) handles next.

Input is the Wizard runtime owner's read-only `checkpoint executable`
JSON (requests that are PENDING_REMOTE_EXECUTION and still pass the
fail-closed execution gate). Every identity value that will be used in a
path or SHA comparison is format-validated here, so nothing unexpected
from the host ever reaches a shell command.

Order: requests whose kickoff is still upcoming first (earliest kickoff,
then earliest cutoff) -- a live T30M must never wait behind a stale
backlog -- then past-kickoff requests, oldest cutoff first.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$")


class SelectionError(ValueError):
    """The executable-request listing is malformed, or the requested run
    is not executable."""


def _ts(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace(" ", "T", 1))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _validated(row: dict[str, Any]) -> dict[str, Any]:
    for field in ("run_id", "request_bundle_sha256", "snapshot_manifest_sha256"):
        if not _HEX64.match(str(row.get(field, ""))):
            raise SelectionError(f"{field} is not a 64-hex SHA-256: {row.get(field)!r}")
    if not _SNAPSHOT_ID.match(str(row.get("snapshot_id", ""))):
        raise SelectionError(f"snapshot_id format: {row.get('snapshot_id')!r}")
    published = row.get("published_result_manifest_sha256")
    if published is not None and not _HEX64.match(str(published)):
        raise SelectionError(f"published_result_manifest_sha256 format: {published!r}")
    return row


def select_request(
    rows: Sequence[dict[str, Any]], *, run_id: str | None, now: datetime
) -> dict[str, Any] | None:
    """The request to handle, or None when nothing is executable. An
    explicit `run_id` must be executable, else `SelectionError`."""
    validated = [_validated(dict(row)) for row in rows]
    if run_id:
        for row in validated:
            if row["run_id"] == run_id:
                return row
        raise SelectionError(f"run {run_id} is not an executable pending request")
    upcoming = sorted(
        (r for r in validated if _ts(r["kickoff_at"]) > now),
        key=lambda r: (_ts(r["kickoff_at"]), _ts(r["scheduled_as_of"]), r["run_id"]),
    )
    past = sorted(
        (r for r in validated if _ts(r["kickoff_at"]) <= now),
        key=lambda r: (_ts(r["scheduled_as_of"]), r["run_id"]),
    )
    ordered = upcoming + past
    return ordered[0] if ordered else None


def github_outputs(selected: dict[str, Any] | None) -> dict[str, str]:
    if selected is None:
        return {"found": "false"}
    return {
        "found": "true",
        "run_id": selected["run_id"],
        "bundle_id": f"checkpoint-result-{selected['run_id']}",
        "checkpoint_name": str(selected.get("checkpoint_name", "")),
        "request_bundle_sha256": selected["request_bundle_sha256"],
        "snapshot_id": selected["snapshot_id"],
        "snapshot_manifest_sha256": selected["snapshot_manifest_sha256"],
        "published_sha256": selected.get("published_result_manifest_sha256") or "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nflprops.platform.checkpoint_select")
    parser.add_argument("--executable-json", required=True, type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--github-output", required=True, type=Path)
    ns = parser.parse_args(argv)
    if ns.run_id and not _HEX64.match(ns.run_id):
        raise SelectionError("--run-id must be 64 hex")
    rows = json.loads(ns.executable_json.read_text())
    if not isinstance(rows, list):
        raise SelectionError("executable listing must be a JSON list")
    selected = select_request(rows, run_id=ns.run_id or None, now=datetime.now(UTC))
    outputs = github_outputs(selected)
    with ns.github_output.open("a") as handle:
        for key, value in outputs.items():
            handle.write(f"{key}={value}\n")
    print(json.dumps(outputs, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
