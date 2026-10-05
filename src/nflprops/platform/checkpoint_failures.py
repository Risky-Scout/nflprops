"""BLOCK 4: OPERATIONAL checkpoint-execution failures, recorded separately
from scientific request state.

A GitHub executor run can fail for reasons that say nothing about the
checkpoint's science: an executor/environment or config-runtime mismatch
(`remote_checkpoint.OPERATIONAL_REFUSAL_CODES`), a download or bundle
read failure, a crashed or timed-out job, a failed publish/ingest. None of
these may change the request: it stays PENDING_REMOTE_EXECUTION (its run
SCHEDULED) and can be executed again. Only a SCIENTIFIC verification
refusal becomes NOT_EXECUTABLE (`result_ingest.refuse_request`).

Each failure is appended as one JSON line to
`<runtime_root>/state/checkpoint_execute_operational_failures.jsonl`
(append-only, never rewritten; outside the warehouse, so recording needs
no writer lock and never races a writer). The request row is read only to
note its state at record time.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nflprops.data.warehouse import Warehouse
from nflprops.errors import NflpropsError
from nflprops.platform.checkpoint_prepare import _read_requests
from nflprops.platform.remote_checkpoint import OPERATIONAL_REFUSAL_CODES
from nflprops.platform.runtime_layout import RuntimeLayout

OPERATIONAL_FAILURES_FILE = "checkpoint_execute_operational_failures.jsonl"
OPERATIONAL_FAILURE_SCHEMA = "nflprops.platform.checkpoint_operational_failure/v1"

#: Workflow-stage failures (checkpoint-execute.yml) besides the classified
#: verification refusals.
FAILURE_DOWNLOAD_OR_BUNDLE_VERIFY = "DOWNLOAD_OR_BUNDLE_VERIFY_FAILED"
FAILURE_EXECUTOR = "EXECUTOR_FAILED"
FAILURE_TIMEOUT_OR_CANCELLED = "TIMEOUT_OR_CANCELLED"
FAILURE_PUBLISH = "PUBLISH_FAILED"
FAILURE_INGEST = "INGEST_FAILED"
WORKFLOW_FAILURE_CODES: frozenset[str] = frozenset({
    FAILURE_DOWNLOAD_OR_BUNDLE_VERIFY,
    FAILURE_EXECUTOR,
    FAILURE_TIMEOUT_OR_CANCELLED,
    FAILURE_PUBLISH,
    FAILURE_INGEST,
})
OPERATIONAL_FAILURE_CODES: frozenset[str] = OPERATIONAL_REFUSAL_CODES | WORKFLOW_FAILURE_CODES

_RUN_ID = re.compile(r"[0-9a-f]{64}")
_WORKFLOW_RUN = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/actions/runs/\d+")


class OperationalFailureError(NflpropsError):
    """The operational failure could not be recorded as given."""


def operational_failures_path(layout: RuntimeLayout) -> Path:
    return layout.state / OPERATIONAL_FAILURES_FILE


def record_operational_failure(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    *,
    run_id: str,
    workflow_run: str,
    failure_code: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append one operational failure. Never changes any request or run."""
    if not _RUN_ID.fullmatch(run_id):
        raise OperationalFailureError("run_id must be 64 hex")
    if not _WORKFLOW_RUN.fullmatch(workflow_run):
        raise OperationalFailureError("workflow_run must be a GitHub Actions run URL")
    if failure_code not in OPERATIONAL_FAILURE_CODES:
        raise OperationalFailureError(
            f"{failure_code!r} is not an operational failure code "
            f"({sorted(OPERATIONAL_FAILURE_CODES)})"
        )
    requests = _read_requests(warehouse)
    rows = requests.filter(requests["run_id"] == run_id) if requests.height else requests
    record = {
        "schema_version": OPERATIONAL_FAILURE_SCHEMA,
        "recorded_at": (now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "run_id": run_id,
        "workflow_run": workflow_run,
        "failure_class": "OPERATIONAL",
        "failure_code": failure_code,
        "request_state": rows["state"][0] if rows.height else None,
    }
    path = operational_failures_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)
    return record


def read_operational_failures(layout: RuntimeLayout) -> list[dict[str, Any]]:
    path = operational_failures_path(layout)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
