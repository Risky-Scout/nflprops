"""Games, rosters and injuries are PIT-visible only from GENUINE receipt.

The BDL provider used to build the mapping context (`available_at` /
`ingested_at`) BEFORE the HTTP call -- for the lazily paginated `games` and
`injuries` before even the first page -- so a response received after a
checkpoint cutoff ``T`` could look known before ``T``. Invariant now:
system knowledge time >= successful response receipt time; provider
timestamps (game `date`, injury `date`) stay event metadata only.

RT1 request before T, response after T  -> not knowable at T
RT2 response before T                    -> knowable at T
RT3 slow / retried request               -> stamped after the successful receipt
RT4 provider timestamp predates receipt  -> availability not backdated
RT5 mapped row carries the genuine receipt as its PIT field
RT6 pagination                           -> no row visible before its page arrived
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from nflprops.data.warehouse import records_to_frame
from nflprops.features.asof import filter_pit
from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.provider import BDLProvider

PROVIDER_TIME = "2020-01-01T00:00:00Z"
TEAM_H = {"id": 1, "abbreviation": "HOM", "full_name": "Home"}
TEAM_A = {"id": 2, "abbreviation": "AWY", "full_name": "Away"}
GAME = {"id": 77, "home_team": TEAM_H, "visitor_team": TEAM_A, "week": 4,
        "date": PROVIDER_TIME, "season": 2026, "postseason": False}
ROSTER = {"player": {"id": 9, "first_name": "A", "last_name": "B"},
          "position": "WR", "depth": 1, "player_name": "A B", "injury_status": None}
INJURY = {"player": {"id": 9, "first_name": "A", "last_name": "B"},
          "status": "Questionable", "comment": "x", "date": PROVIDER_TIME}


class _Clock:
    """Marks taken by the fake server while it serves a response."""

    def __init__(self) -> None:
        self.first_attempt: datetime | None = None
        self.cutoff_mid_flight: datetime | None = None  # T: after request, before reply
        self.responses: list[datetime] = []


def _provider(pages: list[dict], clock: _Clock, *, fail_first: bool = False) -> BDLProvider:
    state = {"attempt": 0, "page": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempt"] += 1
        if clock.first_attempt is None:
            clock.first_attempt = datetime.now(UTC)
        if fail_first and state["attempt"] == 1:
            time.sleep(0.002)
            return httpx.Response(503, json={})  # transient: retried
        clock.cutoff_mid_flight = datetime.now(UTC)  # a checkpoint cutoff T in flight
        time.sleep(0.003)  # the response is still on its way at T
        page = pages[state["page"]]
        state["page"] += 1
        clock.responses.append(datetime.now(UTC))
        return httpx.Response(200, json=page)

    client = BDLClient(
        "https://example.test", "secret", sleep=lambda _s: time.sleep(0.002),
        client=httpx.Client(base_url="https://example.test",
                            transport=httpx.MockTransport(handler)),
    )
    return BDLProvider(client, require_real_spec=False)


def _single(row: dict) -> list[dict]:
    return [{"data": [row], "meta": {}}]


def _two_pages(row: dict, second: dict) -> list[dict]:
    return [{"data": [row], "meta": {"next_cursor": 2}}, {"data": [second], "meta": {}}]


RESOURCES: dict[str, tuple[Callable[[BDLProvider], list], list[dict]]] = {
    "games": (lambda p: list(p.games(seasons=[2026], weeks=[4])), _single(GAME)),
    "game": (lambda p: [p.game(77)], [{"data": GAME}]),
    "rosters": (lambda p: list(p.roster("1", 2026)), _single(ROSTER)),
    "injuries": (lambda p: list(p.injuries()), _single(INJURY)),
}


@pytest.fixture(params=sorted(RESOURCES))
def resource(request: pytest.FixtureRequest):
    return RESOURCES[request.param]


def test_rt1_response_after_t_is_not_knowable_at_t(resource) -> None:
    call, pages = resource
    clock = _Clock()
    rows = call(_provider(pages, clock))
    cutoff = clock.cutoff_mid_flight
    assert cutoff is not None and clock.first_attempt <= cutoff < clock.responses[-1]
    assert all(r.available_at > cutoff for r in rows)
    assert filter_pit(records_to_frame(rows), cutoff, strict=False).height == 0


def test_rt2_response_before_t_is_knowable_at_t(resource) -> None:
    call, pages = resource
    rows = call(_provider(pages, _Clock()))
    later_cutoff = datetime.now(UTC)
    assert filter_pit(records_to_frame(rows), later_cutoff, strict=False).height == len(rows)


def test_rt3_retried_request_is_stamped_after_the_successful_receipt(resource) -> None:
    call, pages = resource
    clock = _Clock()
    rows = call(_provider(pages, clock, fail_first=True))
    assert clock.first_attempt < clock.responses[-1]
    assert all(r.available_at >= clock.responses[-1] for r in rows)


def test_rt4_provider_timestamp_never_backdates_availability(resource) -> None:
    call, pages = resource
    rows = call(_provider(pages, _Clock()))
    provider_time = datetime(2020, 1, 1, tzinfo=UTC)
    for row in rows:
        assert row.available_at > provider_time
        assert row.available_at_is_estimated is False
        if row.event_time is not None:  # games / injuries keep it as metadata
            assert row.event_time == provider_time != row.available_at


def test_rt5_row_pit_field_is_the_genuine_receipt(resource) -> None:
    call, pages = resource
    clock = _Clock()
    rows = call(_provider(pages, clock))
    for row in rows:
        assert row.available_at == row.ingested_at
        assert row.available_at >= clock.responses[-1]
    frame = records_to_frame(rows)
    assert frame["available_at"].min() >= clock.responses[-1]


@pytest.mark.parametrize(
    ("name", "call", "pages"),
    [
        ("games", lambda p: list(p.games(seasons=[2026], weeks=[4])),
         _two_pages(GAME, {**GAME, "id": 78})),
        ("injuries", lambda p: list(p.injuries()),
         _two_pages(INJURY, {**INJURY, "player": {**INJURY["player"], "id": 10}})),
    ],
)
def test_rt6_no_paginated_row_is_visible_before_its_page_arrived(name, call, pages) -> None:
    clock = _Clock()
    rows = call(_provider(pages, clock))
    assert len(rows) == 2 and len(clock.responses) == 2
    first_page_received, last_page_received = clock.responses
    # Conservative semantics: every row carries one receipt taken after the
    # whole fetch -- never a time before any page (including its own) arrived.
    assert all(r.available_at >= last_page_received > first_page_received for r in rows)
    assert filter_pit(records_to_frame(rows), first_page_received, strict=False).height == 0
