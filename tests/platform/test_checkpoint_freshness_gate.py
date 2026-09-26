"""BLOCK 3: official-checkpoint pre-cutoff evidence freshness gate.

`remote_execution_blocker` measures the age of each required feed's latest
successful pre-cutoff collection AT `scheduled_as_of` (never wall-clock
time) against the checkpoint's limit: age == limit is eligible, age > limit
is blocked, and a post-cutoff collection never rescues a stale feed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nflprops.collection.models import (
    CollectionStatus,
    ResourceRunResult,
    ResourceType,
    ScopeType,
)
from nflprops.collection.resource_availability import (
    deterministic_id,
    latest_feed_available_at,
    record_resource_run,
)
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.run_store import FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA
from nflprops.platform.checkpoint_prepare import (
    OFFICIAL_MAX_EVIDENCE_AGE,
    remote_execution_blocker,
)

CUTOFF = datetime(2026, 9, 27, 17, 0, tzinfo=UTC)
FEEDS = (
    ResourceType.GAMES,
    ResourceType.ROSTERS,
    ResourceType.INJURIES,
    ResourceType.GAME_ODDS,
    ResourceType.PLAYER_PROPS,
)
LIMITS = [
    ("T48H", timedelta(minutes=60)),
    ("T24H", timedelta(minutes=40)),
    ("T6H", timedelta(minutes=20)),
    ("T90M", timedelta(minutes=10)),
    ("T30M", timedelta(minutes=4)),
]


@pytest.fixture()
def warehouse(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "canonical", tmp_path / "nflprops.duckdb")


def _record(
    warehouse: Warehouse,
    feed: ResourceType,
    received_at: datetime,
    status: CollectionStatus = CollectionStatus.SUCCESS,
) -> None:
    record_resource_run(
        warehouse,
        ResourceRunResult(
            resource_run_id=deterministic_id(feed.value, received_at, status.value),
            collector_run_id=deterministic_id("run", received_at),
            provider="bdl",
            resource_type=feed,
            scope_type=ScopeType.WEEK,
            scope_json="{}",
            season=2026,
            week=3,
            started_at=received_at,
            collector_received_at=received_at,
            completed_at=received_at,
            collection_status=status,
            row_count=1,
            retry_count=0,
            error_code=None,
            error_detail=None,
            raw_payload_sha256=None,
            created_at=received_at,
        ),
    )


def _request(name: str) -> dict:
    return {"checkpoint_name": name, "scheduled_as_of": CUTOFF}


def test_limits_cover_exactly_the_official_checkpoints() -> None:
    assert dict(LIMITS) == OFFICIAL_MAX_EVIDENCE_AGE


@pytest.mark.parametrize(("name", "limit"), LIMITS)
def test_age_equal_to_limit_is_eligible(warehouse: Warehouse, name: str, limit: timedelta) -> None:
    for feed in FEEDS:
        _record(warehouse, feed, CUTOFF - limit)
    assert remote_execution_blocker(warehouse, _request(name)) is None


@pytest.mark.parametrize(("name", "limit"), LIMITS)
def test_age_over_limit_is_blocked_with_full_detail(
    warehouse: Warehouse, name: str, limit: timedelta
) -> None:
    stale_at = CUTOFF - limit - timedelta(seconds=1)
    for feed in FEEDS:
        _record(warehouse, feed, CUTOFF - limit if feed is not ResourceType.ROSTERS else stale_at)
    reason = remote_execution_blocker(warehouse, _request(name))
    assert reason is not None
    assert reason.startswith(f"{FAILURE_INSUFFICIENT_PRE_CUTOFF_PIT_DATA}: ")
    assert f"stale {name} evidence at scheduled_as_of {CUTOFF.isoformat()}" in reason
    assert f"ROSTERS latest successful pre-cutoff collection {stale_at.isoformat()}" in reason
    age_s = int((limit + timedelta(seconds=1)).total_seconds())
    limit_s = int(limit.total_seconds())
    assert f"age={age_s // 3600}h{age_s % 3600 // 60:02d}m{age_s % 60:02d}s" in reason
    assert f"max_allowed_age={limit_s // 3600}h{limit_s % 3600 // 60:02d}m00s" in reason
    for fresh in ("GAMES", "INJURIES", "GAME_ODDS", "PLAYER_PROPS"):
        assert f"{fresh} latest" not in reason  # only the stale feed is named


@pytest.mark.parametrize(("name", "limit"), LIMITS)
def test_post_cutoff_observation_never_rescues_a_stale_feed(
    warehouse: Warehouse, name: str, limit: timedelta
) -> None:
    for feed in FEEDS:
        _record(warehouse, feed, CUTOFF - limit - timedelta(minutes=1))
        _record(warehouse, feed, CUTOFF + timedelta(microseconds=1))
    assert latest_feed_available_at(
        warehouse.read("collector_resource_runs"), resource_type=ResourceType.GAMES, as_of=CUTOFF
    ) == CUTOFF - limit - timedelta(minutes=1)
    reason = remote_execution_blocker(warehouse, _request(name))
    assert reason is not None and "stale" in reason


def test_failed_collections_are_not_fresh_evidence(warehouse: Warehouse) -> None:
    for feed in FEEDS:
        _record(warehouse, feed, CUTOFF - timedelta(hours=17))
    _record(warehouse, ResourceType.GAME_ODDS, CUTOFF - timedelta(minutes=1),
            CollectionStatus.PROVIDER_ERROR)
    reason = remote_execution_blocker(warehouse, _request("T48H"))
    assert reason is not None and "GAME_ODDS latest successful" in reason


def test_missing_and_stale_are_both_reported(warehouse: Warehouse) -> None:
    for feed in FEEDS:
        if feed is not ResourceType.INJURIES:
            _record(warehouse, feed, CUTOFF - timedelta(hours=17))
    reason = remote_execution_blocker(warehouse, _request("T48H"))
    assert reason is not None
    assert f"no successful INJURIES collection at or before scheduled_as_of {CUTOFF.isoformat()}"\
        in reason
    assert "stale T48H evidence" in reason


def test_manual_is_unchanged_and_exempt(warehouse: Warehouse) -> None:
    assert remote_execution_blocker(warehouse, _request("MANUAL")) is None
    for feed in FEEDS:
        _record(warehouse, feed, CUTOFF - timedelta(days=3))
    assert remote_execution_blocker(warehouse, _request("MANUAL")) is None


def test_unknown_checkpoint_name_fails_closed(warehouse: Warehouse) -> None:
    for feed in FEEDS:
        _record(warehouse, feed, CUTOFF)
    reason = remote_execution_blocker(warehouse, _request("T12H"))
    assert reason is not None and "no evidence-freshness limit" in reason
