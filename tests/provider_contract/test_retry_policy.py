"""Provider contract: retry transient statuses; never retry 4xx contract/auth failures."""
import httpx
import pytest

from nflprops.providers.bdl.client import BDLClient


def test_retry_policy():
    count = {"n": 0}
    def transient(request):
        count["n"] += 1
        if count["n"] == 1:
            return httpx.Response(500, json={"error":"temporary"})
        return httpx.Response(200, json={"data":[]})
    c = BDLClient(
        "https://example.test", "x", max_retries=2,
        client=httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(transient)),
        sleep=lambda _: None,
    )
    assert c.get("/x") == {"data":[]}
    assert count["n"] == 2

    count2 = {"n": 0}
    def bad(request):
        count2["n"] += 1
        return httpx.Response(400, json={"error":"bad"})
    c2 = BDLClient(
        "https://example.test", "x", max_retries=5,
        client=httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(bad)),
        sleep=lambda _: None,
    )
    with pytest.raises(httpx.HTTPStatusError):
        c2.get("/x")
    assert count2["n"] == 1
