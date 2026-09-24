"""PHASE 3: explicit provider capability reporting.

`nflprops.providers.registry.capabilities()` is the production capability
system -- it reports which Protocol-defined capability groups a provider
actually satisfies via `isinstance` against the `runtime_checkable`
Protocols, not `hasattr()` probing.
"""

from __future__ import annotations

import httpx
from fake_provider import FakeProvider

from nflprops.providers import registry
from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.provider import BDLProvider

ALL_CAPABILITIES = {"reference", "schedule", "statistics", "availability", "market"}


def _bdl_factory(*_args, **_kwargs) -> BDLProvider:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    client = BDLClient(
        "https://example.test",
        "secret",
        client=httpx.Client(base_url="https://example.test", transport=transport),
    )
    return BDLProvider(client, require_real_spec=False)


def test_bdl_supports_every_capability_it_implements() -> None:
    registry.register("test-bdl-capabilities", _bdl_factory)
    caps = registry.capabilities("test-bdl-capabilities")
    # BDL implements the full protocol set today (blueprint §11: "preserve
    # working" -- do not weaken this).
    assert caps == ALL_CAPABILITIES


def test_fake_provider_supports_every_capability_it_implements() -> None:
    registry.register("test-fake-capabilities", FakeProvider)
    caps = registry.capabilities("test-fake-capabilities")
    assert caps == ALL_CAPABILITIES


def test_unsupported_capability_returns_false_for_a_partial_provider() -> None:
    """A provider implementing only a subset of the protocol must NOT be
    reported as supporting capabilities it does not implement."""

    class ReferenceOnlyProvider:
        name = "reference-only"

        def teams(self):
            return []

        def players(self):
            return []

        def active_players(self):
            return []

        def roster(self, team_id: str, season: int):
            return []

    registry.register("test-reference-only", ReferenceOnlyProvider)
    caps = registry.capabilities("test-reference-only")

    assert "reference" in caps
    assert "schedule" not in caps
    assert "statistics" not in caps
    assert "availability" not in caps
    assert "market" not in caps


def test_capability_check_requires_the_full_method_set_not_one_attribute() -> None:
    """The production capability system (`isinstance` against a
    `runtime_checkable` Protocol) requires every method a capability group
    declares -- unlike a single scattered `hasattr(provider, "roster")`
    check somewhere downstream, which would wrongly pass for a provider
    implementing only part of ReferenceDataProvider's four methods."""
    from nflprops.domain.protocols import ReferenceDataProvider

    class PartialReferenceProvider:
        def teams(self):
            return []

        def players(self):
            return []

        # active_players() and roster() deliberately missing.

    # A single hasattr() check on any one implemented method would say yes;
    # the real capability contract correctly says no.
    partial = PartialReferenceProvider()
    assert hasattr(partial, "teams")
    assert not isinstance(partial, ReferenceDataProvider)
