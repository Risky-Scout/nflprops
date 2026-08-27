"""Provider contract: season_type is array for games and scalar for stats."""
import httpx
import pytest

from nflprops.providers.bdl import endpoints
from nflprops.providers.bdl.client import BDLClient


def test_season_type_shape():
    c = BDLClient("https://example.test", "x", client=httpx.Client(base_url="https://example.test"))
    assert c._encode_params(endpoints.GAMES, {"season_type": [2, 3]}) == [
        ("season_type", 2), ("season_type", 3)
    ]
    assert c._encode_params(endpoints.STATS, {"season_type": 2}) == [("season_type", 2)]
    with pytest.raises(TypeError):
        c._encode_params(endpoints.STATS, {"season_type": [2, 3]})
