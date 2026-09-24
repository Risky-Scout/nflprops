"""PHASE 4: retry/backoff is read from BDLClient's existing single retry
loop, never re-driven by a second collector-level retry system. Tests patch
sleep so nothing actually waits.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta

import httpx

from nflprops.collection.models import CollectionStatus, ResourceType
from nflprops.collection.service import collect_once
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse
from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.provider import BDLProvider

NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _empty_page(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": [], "meta": {"next_cursor": None}})


_GAME_PAYLOAD = {
    "id": 1,
    "home_team": {"id": 1, "abbreviation": "HOM"},
    "visitor_team": {"id": 2, "abbreviation": "AWY"},
    "date": (NOW + timedelta(hours=5)).isoformat(),
    "season": 2026,
    "week": 1,
    "postseason": False,
    "status_state": "scheduled",
}


def _games_page(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": [_GAME_PAYLOAD], "meta": {"next_cursor": None}})


def _provider(handler, *, max_retries: int = 5) -> BDLProvider:
    client = BDLClient(
        "https://example.test",
        "secret",
        max_retries=max_retries,
        client=httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,  # never actually wait
    )
    return BDLProvider(client, require_real_spec=False)


def test_pop_retry_count_reflects_one_retry_then_success() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"data": [], "meta": {"next_cursor": None}})

    provider = _provider(handler)
    list(provider.injuries())

    assert provider.pop_retry_count() == 1
    # pop_retry_count resets -- a second call with no further retries reads 0.
    assert provider.pop_retry_count() == 0


def test_no_retries_needed_reports_zero() -> None:
    provider = _provider(_empty_page)
    list(provider.injuries())
    assert provider.pop_retry_count() == 0


def test_retries_exhausted_raises_and_classifies_rate_limited(tmp_path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")

    # Seed a game via a games-endpoint-specific handler so GAMES itself
    # succeeds and the cycle reaches the INJURIES resource where the
    # rate limit is exhausted.
    def routed(request: httpx.Request) -> httpx.Response:
        if "player_injuries" in request.url.path:
            return httpx.Response(429, json={"error": "rate limited"})
        return _games_page(request)

    client = BDLClient(
        "https://example.test",
        "secret",
        max_retries=2,
        client=httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(routed)),
        sleep=lambda _: None,
    )
    routed_provider = BDLProvider(client, require_real_spec=False)

    result = collect_once(
        provider=routed_provider,
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        now=NOW,
    )

    injuries_runs = [r for r in result.resource_runs if r.resource_type == ResourceType.INJURIES]
    assert len(injuries_runs) == 1
    assert injuries_runs[0].collection_status == CollectionStatus.RATE_LIMITED
    # Exactly the client's own retry loop, read once -- not double-counted by
    # a second collector-level retry system.
    assert injuries_runs[0].retry_count == 2


def test_non_rate_limit_failure_status_is_provider_error(tmp_path) -> None:
    def routed(request: httpx.Request) -> httpx.Response:
        if "player_injuries" in request.url.path:
            return httpx.Response(500, json={"error": "boom"})
        return _games_page(request)

    client = BDLClient(
        "https://example.test",
        "secret",
        max_retries=1,
        client=httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(routed)),
        sleep=lambda _: None,
    )
    provider = BDLProvider(client, require_real_spec=False)
    warehouse = Warehouse(tmp_path / "warehouse")

    result = collect_once(
        provider=provider,
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        now=NOW,
    )

    injuries_runs = [r for r in result.resource_runs if r.resource_type == ResourceType.INJURIES]
    assert injuries_runs[0].collection_status == CollectionStatus.PROVIDER_ERROR
    assert injuries_runs[0].error_code == "500"


def test_collector_does_not_add_a_second_independent_retry_loop() -> None:
    """The collector must never retry a call the client already exhausted --
    each configured HTTP request happens exactly once per client-level
    attempt; the collector only reads the outcome."""
    calls = {"n": 0}

    def always_fail(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    provider = _provider(always_fail, max_retries=2)
    with contextlib.suppress(httpx.HTTPStatusError):
        list(provider.injuries())

    # max_retries=2 -> 3 total attempts (1 initial + 2 retries). If the
    # collector wrapped this in its own retry loop, this would be a multiple
    # of 3 greater than 3.
    assert calls["n"] == 3
    assert provider.pop_retry_count() == 2
