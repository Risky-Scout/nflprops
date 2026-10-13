"""player_game_stats / team_game_stats are PIT-visible only from GENUINE
receipt -- the exact `available_at` Phase 10C3A filters on
(`state.build_player_states` / `build_team_states` ->
`features.asof.filter_pit(..., strict=False)`: `available_at <= as_of`).

Both endpoints are cursor-paginated and the client iterator is lazy; the
provider used to stamp the mapping context BEFORE the first page was even
requested. Now every page is received first and one conservative
post-fetch receipt is stamped. The provider's game date stays `event_time`
metadata only.

RT1 request before T, response after T  -> not knowable at T
RT2 response before T                    -> knowable at T
RT3 slow / retried request               -> stamped after the successful receipt
RT4 provider game date predates receipt  -> availability not backdated
RT5 the mapped row's PIT field (`available_at`) is the genuine receipt
RT6 pagination                           -> no row visible before its page arrived
RT7 as-of before receipt                 -> excluded by the 10C3A filter
RT8 as-of at/after receipt               -> included by the 10C3A filter
RT9 re-ingest                            -> never creates earlier visibility
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from nflprops.data.warehouse import Warehouse, records_to_frame
from nflprops.features.asof import filter_pit
from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.provider import BDLProvider

GAME_DATE = "2020-01-01T18:00:00Z"
TEAM_H = {"id": 1, "abbreviation": "HOM", "full_name": "Home"}
TEAM_A = {"id": 2, "abbreviation": "AWY", "full_name": "Away"}
GAME = {"id": 77, "home_team": TEAM_H, "visitor_team": TEAM_A, "week": 4,
        "date": GAME_DATE, "season": 2026, "postseason": False}
PLAYER = {"id": 9, "first_name": "A", "last_name": "B"}
PLAYER_STAT = {"player": PLAYER, "team": TEAM_H, "game": GAME,
               "receptions": 6, "receiving_yards": 71, "receiving_targets": 8}
TEAM_STAT = {"game": GAME, "team": TEAM_H, "home_away": "home", "first_downs": 21}


class _Clock:
    def __init__(self) -> None:
        self.first_attempt: datetime | None = None
        self.cutoff_mid_flight: datetime | None = None
        self.responses: list[datetime] = []


def _provider(pages: list[dict], clock: _Clock, *, fail_first: bool = False) -> BDLProvider:
    state = {"attempt": 0, "page": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempt"] += 1
        if clock.first_attempt is None:
            clock.first_attempt = datetime.now(UTC)
        if fail_first and state["attempt"] == 1:
            time.sleep(0.002)
            return httpx.Response(503, json={})
        clock.cutoff_mid_flight = datetime.now(UTC)  # cutoff T while the reply is in flight
        time.sleep(0.003)
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


Fetch = Callable[[BDLProvider], list]
RESOURCES: dict[str, tuple[Fetch, dict, str]] = {
    "player_game_stats": (
        lambda p: list(p.player_game_stats(seasons=[2026])), PLAYER_STAT, "player_game_stats"),
    "team_game_stats": (
        lambda p: list(p.team_game_stats(seasons=[2026])), TEAM_STAT, "team_game_stats"),
}
#: The production append keys (`pipelines.lean.LeanIngestor.ingest_season`).
APPEND_KEYS = {
    "player_game_stats": ["canonical_game_id", "canonical_player_id"],
    "team_game_stats": ["canonical_game_id", "canonical_team_id"],
}


@pytest.fixture(params=sorted(RESOURCES))
def resource(request: pytest.FixtureRequest):
    return RESOURCES[request.param]


def _one_page(row: dict) -> list[dict]:
    return [{"data": [row], "meta": {}}]


def _as_of_visible(rows: list, as_of: datetime) -> int:
    # Exactly the Phase-10C3A gate: filter_pit(strict=False) on available_at.
    return filter_pit(records_to_frame(rows), as_of, strict=False).height


def test_rt1_response_after_t_is_not_knowable_at_t(resource) -> None:
    fetch, row, _ = resource
    clock = _Clock()
    rows = fetch(_provider(_one_page(row), clock))
    cutoff = clock.cutoff_mid_flight
    assert clock.first_attempt <= cutoff < clock.responses[-1]
    assert _as_of_visible(rows, cutoff) == 0


def test_rt2_response_before_t_is_knowable_at_t(resource) -> None:
    fetch, row, _ = resource
    rows = fetch(_provider(_one_page(row), _Clock()))
    assert _as_of_visible(rows, datetime.now(UTC)) == len(rows) == 1


def test_rt3_retried_request_is_stamped_after_the_successful_receipt(resource) -> None:
    fetch, row, _ = resource
    clock = _Clock()
    rows = fetch(_provider(_one_page(row), clock, fail_first=True))
    assert clock.first_attempt < clock.responses[-1]
    assert all(r.available_at >= clock.responses[-1] for r in rows)


def test_rt4_provider_game_date_never_backdates_availability(resource) -> None:
    fetch, row, _ = resource
    rows = fetch(_provider(_one_page(row), _Clock()))
    game_date = datetime(2020, 1, 1, 18, tzinfo=UTC)
    for r in rows:
        assert r.event_time == game_date
        assert r.available_at > game_date
        assert r.available_at_is_estimated is False


def test_rt5_pit_field_is_the_genuine_receipt(resource) -> None:
    fetch, row, _ = resource
    clock = _Clock()
    rows = fetch(_provider(_one_page(row), clock))
    frame = records_to_frame(rows)
    assert (frame["available_at"] == frame["ingested_at"]).all()
    assert frame["available_at"].min() >= clock.responses[-1]


def test_rt6_no_paginated_row_is_visible_before_its_page_arrived(resource) -> None:
    fetch, row, _ = resource
    second = {**row, "game": {**GAME, "id": 78}}
    pages = [{"data": [row], "meta": {"next_cursor": 2}}, {"data": [second], "meta": {}}]
    clock = _Clock()
    rows = fetch(_provider(pages, clock))
    assert len(rows) == 2 and len(clock.responses) == 2
    first_page, last_page = clock.responses
    assert all(r.available_at >= last_page > first_page for r in rows)
    assert _as_of_visible(rows, first_page) == 0


def test_rt7_rt8_as_of_reads_follow_receipt_exactly(resource) -> None:
    fetch, row, _ = resource
    rows = fetch(_provider(_one_page(row), _Clock()))
    receipt = rows[0].available_at
    assert _as_of_visible(rows, receipt - timedelta(microseconds=1)) == 0  # RT7
    assert _as_of_visible(rows, receipt) == 1  # RT8 (at receipt)
    assert _as_of_visible(rows, receipt + timedelta(days=1)) == 1  # RT8 (after)


def test_rt9_reingest_never_creates_earlier_visibility(resource, tmp_path: Path) -> None:
    fetch, row, table = resource
    warehouse = Warehouse(tmp_path)
    first = fetch(_provider(_one_page(row), _Clock()))
    first_receipt = first[0].available_at
    warehouse.append(table, records_to_frame(first), key=APPEND_KEYS[table],
                     sort_by=["available_at"])
    again = fetch(_provider(_one_page(row), _Clock()))
    warehouse.append(table, records_to_frame(again), key=APPEND_KEYS[table],
                     sort_by=["available_at"])
    stored = warehouse.read(table)
    assert stored.height == 1
    assert stored["available_at"].min() >= first_receipt
    assert filter_pit(stored, first_receipt - timedelta(microseconds=1), strict=False).height == 0
