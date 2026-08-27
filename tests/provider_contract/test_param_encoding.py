"""Provider contract: exact live BDL array-query wire encoding.

The pinned OpenAPI document is not authoritative for several endpoint-local
array parameter spellings. These assertions encode behavior verified against
the live NFL API.
"""
import httpx

from nflprops.providers.bdl import endpoints
from nflprops.providers.bdl.client import BDLClient


def _client():
    return BDLClient(
        "https://example.test", "x", client=httpx.Client(base_url="https://example.test")
    )


def test_param_encoding():
    c = _client()
    assert c._encode_params(endpoints.PLAYERS, {"team_ids": [1, 2]}) == [
        ("team_ids[]", 1), ("team_ids[]", 2)
    ]
    assert c._encode_params(endpoints.TEAM_STATS, {"team_ids": [1, 2]}) == [
        ("team_ids[]", 1), ("team_ids[]", 2)
    ]
    assert c._encode_params(endpoints.PLAYER_PROPS, {"vendors": ["fanduel","draftkings"]}) == [
        ("vendors[]", "fanduel"), ("vendors[]", "draftkings")
    ]
    assert c._encode_params(
        endpoints.OPENING_PLAYER_PROPS,
        {"vendors": ["fanduel", "draftkings"]},
    ) == [
        ("vendors[]", "fanduel"),
        ("vendors[]", "draftkings"),
    ]
    assert c._encode_params(endpoints.GAME_ODDS, {"game_ids": [99, 100]}) == [
        ("game_ids[]", 99),
        ("game_ids[]", 100),
    ]
    assert c._encode_params(
        endpoints.OPENING_GAME_ODDS,
        {"game_ids": [99, 100]},
    ) == [
        ("game_ids[]", 99),
        ("game_ids[]", 100),
    ]
    assert c._encode_params(endpoints.DFS_SLATES, {"slate_ids": [10, 11], "providers": ["draftkings"]}) == [
        ("slate_ids[]", 10), ("slate_ids[]", 11), ("providers[]", "draftkings")
    ]
    assert c._encode_params(endpoints.DFS_DRAFTABLES, {"game_ids": [99], "positions": ["QB", "WR"]}) == [
        ("game_ids[]", 99), ("positions[]", "QB"), ("positions[]", "WR")
    ]
