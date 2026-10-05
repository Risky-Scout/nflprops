"""BLOCK 4: the pinned, audited registry of PROVEN false NOT_EXECUTABLE
refusals -- requests the pre-PR-19 executor recorded NOT_EXECUTABLE for an
operational (non-scientific) reason.

Pure data, no imports beyond the standard library, so the snapshot
protection in `checkpoint_prepare` and the remediation in
`refusal_repair` can both read it. An entry here is the ONLY way a
NOT_EXECUTABLE request can ever be reopened, and only through
`refusal_repair.repair_false_refusal`, which re-proves every claim below
against the live warehouse, the immutable request bundle and snapshot
before changing anything. Adding an entry requires a reviewed code change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FalseRefusalIncident:
    incident_id: str
    run_id: str
    game_id: str
    checkpoint_name: str
    scheduled_as_of: str
    snapshot_id: str
    #: The workflow run whose verification produced the false refusal.
    refusal_workflow_run: str
    #: Exactly what the pre-classification `checkpoint refuse` wrote on the run.
    refusal_failure_code: str
    refusal_failure_detail: str
    #: The claimant's (Wizard) full resolved-config SHA in the request ...
    claimed_config_sha256: str
    #: ... and the executor's (GitHub) full resolved-config SHA it was
    #: compared with. They differ ONLY in operational config paths.
    executor_config_sha256: str
    cause: str
    remediation_reason: str
    #: The change that implements the remediation.
    remediation_change: str


INCIDENT_2026_10_05_CONFIG_HOST_PATH = FalseRefusalIncident(
    incident_id="INC-2026-10-05-config-sha-host-path",
    run_id="42d8bd5d080fbf95e61b35c495b161bede22db1e622c2ef17e89ef0709440757",
    game_id="984fa187-18f8-57b0-a760-917ab0a78f04",
    checkpoint_name="T6H",
    scheduled_as_of="2026-10-05T18:15:00+00:00",
    snapshot_id="20261005T181551Z-df892b330493",
    refusal_workflow_run="https://github.com/Risky-Scout/nflprops/actions/runs/37362214826",
    refusal_failure_code="REMOTE_EXECUTION_REFUSED",
    refusal_failure_detail=(
        "GitHub executor verification refused the request: "
        "https://github.com/Risky-Scout/nflprops/actions/runs/37362214826"
    ),
    claimed_config_sha256="a2df655ebffc4529b67f53e34561ba1cf1e6a58cbc4b5d2292d07e7cef0363ae",
    executor_config_sha256="303bdef39a616be18c9185f8151a3cde0557a7ad6c4f4aa97b6f1de731d5d332",
    cause=(
        "config_sha256 hashed the full resolved config including the host path "
        "run.data_root (Wizard NFLPROPS_DATA_ROOT=/home/wizard-deploy/nflprops/state; "
        "GitHub runner: shipped default), so the GitHub executor's verify_only refused a "
        "valid Wizard-prepared request, and checkpoint-execute.yml recorded every "
        "verification refusal as NOT_EXECUTABLE. Bundle and snapshot identity had passed; "
        "no simulation ran."
    ),
    remediation_reason=(
        "Operational host-config identity defect, not a scientific refusal: restore the "
        "request to PENDING_REMOTE_EXECUTION (run SCHEDULED) after re-proving it."
    ),
    remediation_change="PR #19 (fix/scientific-config-identity)",
)

KNOWN_FALSE_REFUSALS: dict[str, FalseRefusalIncident] = {
    INCIDENT_2026_10_05_CONFIG_HOST_PATH.run_id: INCIDENT_2026_10_05_CONFIG_HOST_PATH,
}


def incident_snapshot_ids() -> frozenset[str]:
    """Snapshots of pinned false refusals: kept (never pruned) so the
    remediation can still re-verify them."""
    return frozenset(incident.snapshot_id for incident in KNOWN_FALSE_REFUSALS.values())
