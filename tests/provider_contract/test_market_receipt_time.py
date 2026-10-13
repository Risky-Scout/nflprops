"""STEP 2D: a market quote's `collector_received_at` (its prediction-time
knowledge timestamp) is stamped only AFTER the provider response arrived --
never before the request, which would predate genuine receipt by the
request + retry latency and let a quote received after a checkpoint's
cutoff look as if it had been known before it.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import httpx
import pytest

from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.provider import BDLProvider

PROP = {
    "id": 5, "game_id": 77, "player_id": 9, "vendor": "draftkings",
    "prop_type": "receiving_yards", "line_value": "67.5",
    "market": {"type": "over_under", "over_odds": -115, "under_odds": -105},
    "updated_at": "2020-01-01T00:00:00Z",
}
OPENING_PROP = {**{k: v for k, v in PROP.items() if k != "updated_at"},
                "opened_at": "2020-01-01T00:00:00Z"}
ODDS = {
    "id": 42, "game_id": 77, "vendor": "fanduel", "spread_home_value": "-3.5",
    "spread_home_odds": -110, "spread_away_value": "3.5", "spread_away_odds": -110,
    "moneyline_home_odds": -180, "moneyline_away_odds": 155, "total_value": "47.5",
    "total_over_odds": -105, "total_under_odds": -115, "updated_at": "2020-01-01T00:00:00Z",
}
OPENING_ODDS = {**{k: v for k, v in ODDS.items() if k != "updated_at"},
                "opened_at": "2020-01-01T00:00:00Z"}


def _provider(payload: dict, responded: list[datetime]) -> BDLProvider:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        time.sleep(0.003)  # request latency
        if attempts["n"] == 1:
            return httpx.Response(503, json={})  # a retried transient failure
        responded.append(datetime.now(UTC))
        return httpx.Response(200, json={"data": [payload], "meta": {}})

    client = BDLClient(
        "https://example.test", "secret", sleep=lambda _s: time.sleep(0.003),
        client=httpx.Client(base_url="https://example.test",
                            transport=httpx.MockTransport(handler)),
    )
    return BDLProvider(client, require_real_spec=False)


@pytest.mark.parametrize(
    ("call", "payload"),
    [
        (lambda p: p.player_props("77"), PROP),
        (lambda p: p.opening_player_props("77"), OPENING_PROP),
        (lambda p: p.game_odds(season=2026, week=4), ODDS),
        (lambda p: p.opening_game_odds(season=2026, week=4), OPENING_ODDS),
    ],
    ids=["player_props", "opening_player_props", "game_odds", "opening_game_odds"],
)
def test_receipt_is_stamped_after_the_response_not_before_the_request(call, payload) -> None:
    responded: list[datetime] = []
    requested_at = datetime.now(UTC)
    rows = call(_provider(payload, responded))
    assert len(rows) == 1 and responded
    row = rows[0]
    # Never before the (retried) response actually arrived ...
    assert row.collector_received_at >= responded[-1] > requested_at
    # ... and the PIT availability is that same receipt, not a provider time.
    assert row.available_at == row.collector_received_at
    assert row.available_at > datetime(2020, 1, 2, tzinfo=UTC)
