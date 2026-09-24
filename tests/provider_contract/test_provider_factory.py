"""PHASE 3: centralized, name-driven provider construction.

`nflprops.providers.registry` is the single provider-construction mechanism
-- production code builds providers by name (`get_provider("bdl", cfg)`),
never by importing a concrete provider class outside its own bootstrap
module. An unregistered/misconfigured name fails explicitly; it never
silently falls back to BDL.
"""

from __future__ import annotations

import pytest

from nflprops.providers import registry


def test_valid_configured_provider_builds_correctly() -> None:
    sentinel = object()
    registry.register("test-valid-provider", lambda: sentinel)

    built = registry.get_provider("test-valid-provider")

    assert built is sentinel


def test_invalid_provider_name_fails_explicitly() -> None:
    with pytest.raises(KeyError, match="unknown provider"):
        registry.get_provider("definitely-not-a-registered-provider")


def test_invalid_provider_name_does_not_silently_default_to_bdl() -> None:
    """Registering bdl must not make lookups of OTHER unknown names resolve
    to it -- each name is looked up independently and unregistered names
    fail, they never fall through to whatever else happens to be registered."""
    import nflprops.pipelines.lean  # noqa: F401 -- registers "bdl"/"balldontlie"

    with pytest.raises(KeyError):
        registry.get_provider("nonexistent-provider-xyz")


def test_bdl_is_registered_by_name_and_alias() -> None:
    import nflprops.pipelines.lean  # noqa: F401 -- registers "bdl"/"balldontlie"

    assert registry.get("bdl") is registry.get("balldontlie")


def test_registering_same_factory_twice_is_idempotent() -> None:
    def factory():
        return "x"

    registry.register("test-idempotent-provider", factory)
    registry.register("test-idempotent-provider", factory)  # must not raise

    assert registry.get("test-idempotent-provider") is factory


def test_registering_different_factory_under_same_name_fails() -> None:
    registry.register("test-conflict-provider", lambda: "a")
    with pytest.raises(KeyError):
        registry.register("test-conflict-provider", lambda: "b")
