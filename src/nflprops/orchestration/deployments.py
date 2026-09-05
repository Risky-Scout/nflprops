"""Prefect deployment definitions and registration (PHASE 5, §33-§35).

Deployment *construction* (`build_deployments`) is pure Python object
construction: it never contacts a Prefect API and never creates a cloud
account, work pool, or any other paid/cloud resource. Only `register()` --
reached from the command line via::

    python -m nflprops.orchestration.deployments register

-- talks to a Prefect API, and only once an explicit `PREFECT_API_URL` is
confirmed to be configured (Prefect Cloud and a self-hosted Prefect server
are configured identically, via `PREFECT_API_URL`/`PREFECT_API_KEY`; this
module does not distinguish between them).

The two required deployments (§33) wrap `collection_dispatch_flow` /
`checkpoint_dispatch_flow` in small adapter flows
(`collection_dispatch_deployment_flow` / `checkpoint_dispatch_deployment_flow`)
that accept only JSON-serializable parameters (`provider_name`, `season`,
`week`) and construct the live `provider`/`warehouse`/`config` objects
in-process before delegating -- Prefect deployments must be able to
serialize their run parameters, so the underlying flows' own signatures
(which take live objects) are left untouched.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

from prefect import flow
from prefect.deployments.runner import RunnerDeployment

from nflprops.config import Config
from nflprops.config import load as load_config
from nflprops.orchestration.checkpoints import OrchestrationConfig
from nflprops.orchestration.flows.checkpoints import checkpoint_dispatch_flow
from nflprops.orchestration.flows.collection import collection_dispatch_flow

COLLECTION_DISPATCH_DEPLOYMENT_NAME = "collection-dispatch"
CHECKPOINT_DISPATCH_DEPLOYMENT_NAME = "checkpoint-dispatch"


class DeploymentRegistrationError(RuntimeError):
    """Raised when Prefect deployment registration is attempted without a
    configured Prefect API (§35). Registration must never silently create
    a cloud account or fall back to some other backend."""


@dataclass(frozen=True)
class PrefectApiSettings:
    api_url: str | None
    api_key: str | None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> PrefectApiSettings:
        source = os.environ if env is None else env
        return cls(
            api_url=source.get("PREFECT_API_URL") or None,
            api_key=source.get("PREFECT_API_KEY") or None,
        )

    def require_configured(self) -> None:
        if not self.api_url:
            raise DeploymentRegistrationError(
                "PREFECT_API_URL is not set. Deployment registration requires an "
                "explicit Prefect API -- Prefect Cloud or a self-hosted Prefect "
                "server -- configured via PREFECT_API_URL (and PREFECT_API_KEY for "
                "Prefect Cloud). It never silently falls back to an ephemeral "
                "local server, and never creates a cloud account or work pool "
                "on your behalf."
            )


@flow(name="collection-dispatch-deployment")
def collection_dispatch_deployment_flow(
    *, provider_name: str = "bdl", season: int, week: int
) -> None:
    """Deployment entry point for `collection_dispatch_flow` (§33): builds
    the live provider/warehouse/config the underlying flow needs from
    JSON-serializable parameters, exactly as the existing
    `nflprops collect once` CLI command already does."""
    from nflprops.pipelines.lean import LeanIngestor  # noqa: F401 -- registers "bdl"
    from nflprops.providers.registry import get_provider

    cfg = load_config()
    provider, warehouse = get_provider(provider_name, cfg)
    try:
        collection_dispatch_flow(
            provider=provider,
            season=season,
            week=week,
            warehouse=warehouse,
            config=cfg,
            now=datetime.now(UTC),
        )
    finally:
        client = getattr(provider, "client", None)
        if client is not None:
            client.close()


@flow(name="checkpoint-dispatch-deployment")
def checkpoint_dispatch_deployment_flow(*, season: int, week: int) -> None:
    """Deployment entry point for `checkpoint_dispatch_flow` (§33)."""
    from nflprops.pipelines.lean import open_warehouse

    cfg = load_config()
    warehouse = open_warehouse(cfg)
    checkpoint_dispatch_flow(
        warehouse=warehouse,
        config=cfg,
        season=season,
        week=week,
        now=datetime.now(UTC),
    )


def build_deployments(cfg: Config | None = None) -> list[RunnerDeployment]:
    """Construct (never register) the two required Phase-5 deployments.

    Pure object construction: no network call, no Prefect API contact, no
    resource creation of any kind.
    """
    cfg = cfg or load_config()
    orchestration_cfg = OrchestrationConfig.from_config(cfg)
    interval = orchestration_cfg.dispatcher_tick_seconds
    work_pool = orchestration_cfg.prefect_work_pool

    collection_deployment = collection_dispatch_deployment_flow.to_deployment(
        name=COLLECTION_DISPATCH_DEPLOYMENT_NAME,
        interval=interval,
        work_pool_name=work_pool,
        description=(
            "PHASE 5: run a Phase-4 collection cycle when the current "
            "game-relative cadence says one is due."
        ),
        tags=["nflprops", "collection"],
    )
    checkpoint_deployment = checkpoint_dispatch_deployment_flow.to_deployment(
        name=CHECKPOINT_DISPATCH_DEPLOYMENT_NAME,
        interval=interval,
        work_pool_name=work_pool,
        description=(
            "PHASE 5: dispatch due official pregame checkpoints "
            "(T48H/T24H/T6H/T90M/T30M)."
        ),
        tags=["nflprops", "checkpoints"],
    )
    return [collection_deployment, checkpoint_deployment]


def register(cfg: Config | None = None, *, env: dict[str, str] | None = None) -> list[str]:
    """Register/upsert the Phase-5 deployments against the configured
    Prefect API (§35). Raises `DeploymentRegistrationError` if
    `PREFECT_API_URL` is not set -- this function never contacts a Prefect
    API otherwise."""
    PrefectApiSettings.from_env(env).require_configured()
    deployments = build_deployments(cfg)
    return [str(deployment.apply()) for deployment in deployments]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] != "register":
        print("usage: python -m nflprops.orchestration.deployments register", file=sys.stderr)
        return 2
    try:
        deployment_ids = register()
    except DeploymentRegistrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for deployment_id in deployment_ids:
        print(f"registered deployment: {deployment_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
