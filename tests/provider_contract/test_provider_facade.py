"""Smoke-test provider facade without a live network."""
import httpx

from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.provider import BDLProvider


def test_provider_facade_teams():
    def handler(request):
        assert request.headers["Authorization"] == "secret"
        return httpx.Response(200, json={"data":[{
            "id":1, "conference":"AFC", "division":"West", "location":"X",
            "name":"Xs", "full_name":"X Xs", "abbreviation":"XXX"
        }]})
    transport = httpx.MockTransport(handler)
    hc = httpx.Client(base_url="https://example.test", transport=transport)
    client = BDLClient("https://example.test", "secret", client=hc)
    provider = BDLProvider(client, require_real_spec=False)
    teams = provider.teams()
    assert len(teams) == 1
    assert teams[0].provider_team_id == "1"


def test_provider_facade_translates_canonical_game_id():
    canonical_game_id = "canonical-game-test-id"

    def id_lookup(kind, value):
        assert kind == "game"
        assert value == canonical_game_id
        return 424242

    def handler(request):
        assert request.url.path == "/nfl/v1/odds/player_props"
        assert request.url.params["game_id"] == "424242"
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    hc = httpx.Client(base_url="https://example.test", transport=transport)
    client = BDLClient("https://example.test", "secret", client=hc)
    provider = BDLProvider(
        client,
        id_lookup=id_lookup,
        require_real_spec=False,
    )

    assert provider.player_props(canonical_game_id) == []


def test_provider_facade_translates_canonical_team_id():
    canonical_team_id = "canonical-team-test-id"

    def id_lookup(kind, value):
        assert kind == "team"
        assert value == canonical_team_id
        return 9

    def handler(request):
        assert request.url.path == "/nfl/v1/teams/9/roster"
        assert request.url.params["season"] == "2025"
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    hc = httpx.Client(base_url="https://example.test", transport=transport)
    client = BDLClient("https://example.test", "secret", client=hc)
    provider = BDLProvider(
        client,
        id_lookup=id_lookup,
        require_real_spec=False,
    )

    assert provider.roster(canonical_team_id, 2025) == []
