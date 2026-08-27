"""Provider contract: cursor loop and non-paginated player-prop exception."""
import httpx
import pytest

from nflprops.providers.bdl import endpoints
from nflprops.providers.bdl.client import BDLClient


def test_pagination():
    calls = []
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(200, json={"data":[{"id":1}], "meta":{"next_cursor":123,"per_page":100}})
        assert cursor == "123"
        return httpx.Response(200, json={"data":[{"id":2}], "meta":{"next_cursor":None,"per_page":100}})

    h = httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(handler))
    c = BDLClient("https://example.test", "x", client=h, sleep=lambda _: None)
    assert list(c.paginated_get(endpoints.PLAYERS)) == [{"id":1},{"id":2}]
    assert len(calls) == 2
    with pytest.raises(ValueError):
        list(c.paginated_get(endpoints.PLAYER_PROPS, {"game_id": 1}))
