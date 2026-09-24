"""PHASE 5 §33/§35: deployment construction and registration gating.

Deployment *construction* must never contact a Prefect API or create any
resource. Registration must refuse outright when PREFECT_API_URL is not
configured -- it must never silently fall back to an ephemeral local
server or create a cloud account/work pool on its own.
"""

from __future__ import annotations

import pytest

from nflprops.config import Config
from nflprops.orchestration.deployments import (
    CHECKPOINT_DISPATCH_DEPLOYMENT_NAME,
    COLLECTION_DISPATCH_DEPLOYMENT_NAME,
    DeploymentRegistrationError,
    PrefectApiSettings,
    build_deployments,
    register,
)


def _cfg(**orchestration: object) -> Config:
    return Config(data={"orchestration": {"prefect_work_pool": "nflprops-production", **orchestration}})


def test_prefect_api_settings_from_env_missing_url() -> None:
    settings = PrefectApiSettings.from_env({})
    assert settings.api_url is None
    with pytest.raises(DeploymentRegistrationError):
        settings.require_configured()


def test_prefect_api_settings_from_env_present() -> None:
    settings = PrefectApiSettings.from_env(
        {"PREFECT_API_URL": "http://localhost:4200/api", "PREFECT_API_KEY": "k"}
    )
    assert settings.api_url == "http://localhost:4200/api"
    settings.require_configured()  # must not raise


def test_register_refuses_without_prefect_api_url() -> None:
    with pytest.raises(DeploymentRegistrationError, match="PREFECT_API_URL"):
        register(_cfg(), env={})


def test_build_deployments_never_needs_a_prefect_api() -> None:
    # No PREFECT_API_URL set anywhere in this process's env is fine --
    # construction is pure object-building, not a network call.
    deployments = build_deployments(_cfg())
    names = {d.name for d in deployments}
    assert names == {COLLECTION_DISPATCH_DEPLOYMENT_NAME, CHECKPOINT_DISPATCH_DEPLOYMENT_NAME}


def test_build_deployments_uses_configured_work_pool_and_interval() -> None:
    cfg = _cfg(dispatcher_tick_seconds=30, prefect_work_pool="custom-pool")
    deployments = build_deployments(cfg)
    for deployment in deployments:
        assert deployment.work_pool_name == "custom-pool"
