"""PHASE 4: collector_resource_runs is the authoritative source for
resource-level feed availability. Preserves the PHASE 2 PIT guarantee.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from nflprops.collection.models import CollectionStatus, ResourceType
from nflprops.collection.resource_availability import (
    deterministic_id,
    latest_resource_run_status,
    resource_feed_available_at,
)

AS_OF = datetime(2026, 9, 10, tzinfo=UTC)


def _run(
    *,
    resource_type: str = "INJURIES",
    collector_received_at: datetime | None,
    collection_status: str,
    provider: str = "balldontlie",
    row_count: int = 0,
    scope_type: str = "LEAGUE",
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "provider": [provider],
            "resource_type": [resource_type],
            "scope_type": [scope_type],
            "collector_received_at": [collector_received_at],
            "collection_status": [collection_status],
            "row_count": [row_count],
        }
    )


def test_successful_injuries_with_rows_is_available() -> None:
    runs = _run(collector_received_at=AS_OF - timedelta(hours=1), collection_status="SUCCESS", row_count=5)
    assert resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF) is True


def test_successful_injuries_zero_rows_is_available() -> None:
    """The core PHASE 2 -> PHASE 4 guarantee: zero rows from a successful
    collection is still a fully-observed, available reading."""
    runs = _run(collector_received_at=AS_OF - timedelta(hours=1), collection_status="SUCCESS", row_count=0)
    assert resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF) is True


def test_no_resource_run_at_all_is_unavailable() -> None:
    assert resource_feed_available_at(pl.DataFrame(), resource_type=ResourceType.INJURIES, as_of=AS_OF) is False


def test_failed_resource_run_is_unavailable() -> None:
    runs = _run(collector_received_at=AS_OF - timedelta(hours=1), collection_status="PROVIDER_ERROR")
    assert resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF) is False


def test_rate_limited_resource_run_is_unavailable() -> None:
    runs = _run(collector_received_at=AS_OF - timedelta(hours=1), collection_status="RATE_LIMITED")
    assert resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF) is False


def test_empty_response_is_unavailable() -> None:
    runs = _run(collector_received_at=AS_OF - timedelta(hours=1), collection_status="EMPTY_RESPONSE")
    assert resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF) is False


def test_partial_response_is_unavailable() -> None:
    runs = _run(collector_received_at=AS_OF - timedelta(hours=1), collection_status="PARTIAL_RESPONSE")
    assert resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF) is False


def test_unsupported_is_unavailable() -> None:
    runs = _run(collector_received_at=AS_OF - timedelta(hours=1), collection_status="UNSUPPORTED")
    assert resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF) is False


def test_future_successful_run_is_unavailable_to_earlier_as_of() -> None:
    """Preserves the PHASE 2 PIT guarantee: a later collection must never
    leak backward into an earlier as_of's availability read."""
    runs = _run(collector_received_at=AS_OF + timedelta(hours=1), collection_status="SUCCESS", row_count=3)
    assert resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF) is False


def test_market_not_posted_counts_as_checked_not_as_quote_available() -> None:
    """MARKET_NOT_POSTED proves the feed was checked -- it is available
    evidence for feed-availability purposes -- but it must never be read as
    "a quote exists." Quote existence is a separate question entirely,
    answered by the canonical snapshot tables, not this function."""
    runs = _run(
        resource_type="PLAYER_PROPS",
        scope_type="GAME",
        collector_received_at=AS_OF - timedelta(minutes=10),
        collection_status="MARKET_NOT_POSTED",
        row_count=0,
    )
    assert resource_feed_available_at(runs, resource_type=ResourceType.PLAYER_PROPS, as_of=AS_OF) is True


def test_provider_filter_excludes_other_providers() -> None:
    runs = _run(
        provider="fake", collector_received_at=AS_OF - timedelta(hours=1), collection_status="SUCCESS"
    )
    assert (
        resource_feed_available_at(
            runs, resource_type=ResourceType.INJURIES, as_of=AS_OF, provider="balldontlie"
        )
        is False
    )
    assert (
        resource_feed_available_at(
            runs, resource_type=ResourceType.INJURIES, as_of=AS_OF, provider="fake"
        )
        is True
    )


def test_provider_none_means_any_provider_counts() -> None:
    """Canonical pipeline code has no reason to know which provider is
    configured -- provider=None (the default) means any provider's
    successful collection counts."""
    runs = _run(
        provider="some-future-provider",
        collector_received_at=AS_OF - timedelta(hours=1),
        collection_status="SUCCESS",
    )
    assert resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF) is True


def test_different_resource_type_does_not_leak_availability() -> None:
    runs = _run(resource_type="GAME_ODDS", collector_received_at=AS_OF - timedelta(hours=1), collection_status="SUCCESS")
    assert resource_feed_available_at(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF) is False


def test_latest_resource_run_status() -> None:
    runs = pl.concat(
        [
            _run(collector_received_at=AS_OF - timedelta(hours=3), collection_status="PROVIDER_ERROR"),
            _run(collector_received_at=AS_OF - timedelta(hours=1), collection_status="SUCCESS"),
        ]
    )
    assert (
        latest_resource_run_status(runs, resource_type=ResourceType.INJURIES, as_of=AS_OF)
        == CollectionStatus.SUCCESS
    )


def test_latest_resource_run_status_none_when_empty() -> None:
    assert latest_resource_run_status(pl.DataFrame(), resource_type=ResourceType.INJURIES, as_of=AS_OF) is None


def test_deterministic_id_is_stable_sha256_not_builtin_hash() -> None:
    first = deterministic_id("a", "b", 1)
    second = deterministic_id("a", "b", 1)
    assert first == second
    assert len(first) == 64
    int(first, 16)
    assert deterministic_id("a", "b", 2) != first
