"""PR #21: same-science-SHA quarantine for MODEL_EXECUTION_FAILED requests.

A MODEL_EXECUTION_FAILED request (class MODEL_FAILURE -- neither SCIENTIFIC
nor OPERATIONAL) stays PENDING_REMOTE_EXECUTION and is never COMPLETED or
NOT_EXECUTABLE. Unlike an operational failure, re-running the SAME model
code on the SAME immutable inputs cannot be expected to succeed, so the
request is quarantined for the exact science/code identity it failed under:
the executor's `science_sha` (the 40-hex git commit the GitHub executor ran,
`$GITHUB_SHA` -- the same value stamped into every result bundle).

* Quarantine source: every MODEL_FAILURE record in the append-only
  operational failure log (`checkpoint_failures`), which carries its
  `science_sha`. A legacy record without one quarantines the request under
  EVERY science SHA (fail closed) until manually released.
* Automatic selection (`checkpoint_select`) skips a request quarantined for
  the executor's own science SHA; once the science SHA changes the request
  is automatically eligible again.
* Manual release (`release_model_failure_quarantine`) appends one audited
  record to `state/checkpoint_model_failure_releases.jsonl`, naming the
  exact failure it releases (run, science SHA, failing workflow run) and an
  incident id. It never deletes or rewrites failure history, never touches
  the warehouse, and is not a result: the request stays
  PENDING_REMOTE_EXECUTION and can only become COMPLETED through a new,
  successful execution and the ordinary result ingest.

Both logs live outside the warehouse (no writer lock needed) and are
append-only; nothing here ever changes a request or run.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nflprops.data.warehouse import Warehouse
from nflprops.errors import NflpropsError
from nflprops.platform.checkpoint_prepare import _read_requests
from nflprops.platform.remote_checkpoint import (
    MODEL_EXECUTION_FAILED,
    MODEL_FAILURE_CLASS,
)
from nflprops.platform.runtime_layout import RuntimeLayout

RELEASES_FILE = "checkpoint_model_failure_releases.jsonl"
RELEASE_SCHEMA = "nflprops.platform.model_failure_quarantine_release/v1"
RELEASE_KIND = "MANUAL_MODEL_FAILURE_QUARANTINE_RELEASE"

#: The science identity of a legacy model-failure record that carries none.
UNKNOWN_SCIENCE_SHA = "UNKNOWN"

SCIENCE_SHA = re.compile(r"[0-9a-f]{40}")
_RUN_ID = re.compile(r"[0-9a-f]{64}")
_WORKFLOW_RUN = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/actions/runs/\d+")
_INCIDENT_ID = re.compile(r"INC-[A-Za-z0-9-]{1,80}")


class QuarantineError(NflpropsError):
    """The quarantine release was refused (nothing was written)."""


@dataclass(frozen=True)
class ModelFailureQuarantine:
    """One active quarantine: `run_id` must not be auto-executed by
    `science_sha` (or by any SHA when it is `UNKNOWN_SCIENCE_SHA`)."""

    run_id: str
    science_sha: str
    workflow_run: str
    recorded_at: str

    def blocks(self, science_sha: str) -> bool:
        return self.science_sha in (science_sha, UNKNOWN_SCIENCE_SHA)

    def as_dict(self) -> dict[str, str]:
        return {
            "science_sha": self.science_sha,
            "workflow_run": self.workflow_run,
            "recorded_at": self.recorded_at,
        }


def releases_path(layout: RuntimeLayout) -> Path:
    return layout.state / RELEASES_FILE


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_releases(layout: RuntimeLayout) -> list[dict[str, Any]]:
    return _read_jsonl(releases_path(layout))


def _key(run_id: str, science_sha: str, workflow_run: str) -> tuple[str, str, str]:
    return (run_id, science_sha, workflow_run)


def active_quarantines(layout: RuntimeLayout) -> dict[str, list[ModelFailureQuarantine]]:
    """run_id -> its active quarantines (sorted by science SHA, then time).
    A model failure is active until a release names exactly it."""
    from nflprops.platform.checkpoint_failures import read_operational_failures

    released = {
        _key(r["run_id"], r["science_sha"], r["failure_workflow_run"])
        for r in read_releases(layout)
    }
    active: dict[str, list[ModelFailureQuarantine]] = {}
    for record in read_operational_failures(layout):
        if (
            record.get("failure_class") != MODEL_FAILURE_CLASS
            and record.get("failure_code") != MODEL_EXECUTION_FAILED
        ):
            continue
        quarantine = ModelFailureQuarantine(
            run_id=record["run_id"],
            science_sha=record.get("science_sha") or UNKNOWN_SCIENCE_SHA,
            workflow_run=record["workflow_run"],
            recorded_at=record["recorded_at"],
        )
        if _key(quarantine.run_id, quarantine.science_sha, quarantine.workflow_run) in released:
            continue
        active.setdefault(quarantine.run_id, []).append(quarantine)
    for entries in active.values():
        entries.sort(key=lambda q: (q.science_sha, q.recorded_at, q.workflow_run))
    return active


def release_model_failure_quarantine(
    layout: RuntimeLayout,
    warehouse: Warehouse,
    *,
    run_id: str,
    science_sha: str,
    failure_workflow_run: str,
    incident_id: str,
    released_by_release_sha: str | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Release exactly ONE active quarantine -- the model failure of
    `run_id` under `science_sha` recorded by `failure_workflow_run` --
    under an incident id. Fail closed: every identity must be well formed,
    the request must still be PENDING_REMOTE_EXECUTION, and that exact
    failure must be actively quarantining it. Appends one release record;
    deletes nothing; changes no request, run or result."""
    from nflprops.platform.checkpoint_prepare import STATE_PENDING_REMOTE_EXECUTION

    if not _RUN_ID.fullmatch(run_id):
        raise QuarantineError("run_id must be 64 hex")
    if not (SCIENCE_SHA.fullmatch(science_sha) or science_sha == UNKNOWN_SCIENCE_SHA):
        raise QuarantineError(f"science_sha must be 40 hex or {UNKNOWN_SCIENCE_SHA!r}")
    if not _WORKFLOW_RUN.fullmatch(failure_workflow_run):
        raise QuarantineError("failure_workflow_run must be a GitHub Actions run URL")
    if not _INCIDENT_ID.fullmatch(incident_id):
        raise QuarantineError("incident_id must look like INC-<id>")

    requests = _read_requests(warehouse)
    rows = requests.filter(requests["run_id"] == run_id) if requests.height else requests
    if rows.height != 1:
        raise QuarantineError(f"no live checkpoint request for run {run_id}")
    request_state = rows["state"][0]
    if request_state != STATE_PENDING_REMOTE_EXECUTION:
        raise QuarantineError(
            f"request is {request_state}; only a PENDING_REMOTE_EXECUTION request is "
            "ever quarantined"
        )
    target = [
        q for q in active_quarantines(layout).get(run_id, [])
        if q.science_sha == science_sha and q.workflow_run == failure_workflow_run
    ]
    if not target:
        raise QuarantineError(
            f"run {run_id} has no active model-failure quarantine for science SHA "
            f"{science_sha} from {failure_workflow_run}; nothing released"
        )
    quarantine = target[0]
    record = {
        "schema_version": RELEASE_SCHEMA,
        "release_kind": RELEASE_KIND,
        "recorded_at": (now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "run_id": run_id,
        "science_sha": science_sha,
        "failure_code": MODEL_EXECUTION_FAILED,
        "failure_workflow_run": failure_workflow_run,
        "failure_recorded_at": quarantine.recorded_at,
        "incident_id": incident_id,
        "released_by_release_sha": released_by_release_sha,
        "request_state": request_state,
        # Never a result: the request stays pending; only a new successful
        # execution + the ordinary result ingest can ever complete it.
        "is_science_result": False,
        "request_state_after": request_state,
    }
    path = releases_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)
    return record
