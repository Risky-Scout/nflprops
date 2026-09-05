"""§26 (collection dispatcher cadence), §46 (Prefect collection-flow
regression equivalence against `collect_once`), §15/no-BDL-dependency proof
via `FakeProvider` (any `FullProvider` works, not just BDL)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.collection.models import RUNS_TABLE
from nflprops.collection.service import collect_once
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse
from nflprops.orchestration.flows.collection import (
    collection_dispatch_flow,
    collection_due,
    collection_once_flow,
)

NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _seeded_provider() -> FakeProvider:
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    provider.seed_game(
        "g1",
        home_team_native_id="t1",
        visitor_team_native_id="t2",
        week=1,
        date=NOW + timedelta(hours=5),
    )
    return provider


def test_collection_due_with_no_prior_run_is_immediately_due(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    assert (
        collection_due(
            warehouse=warehouse,
            provider_name="fake",
            season=2026,
            week=1,
            now=NOW,
            config=Config(data={}),
        )
        is True
    )


def test_collection_due_respects_cadence_and_ignores_run_status(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    # Game kicks off in 5 hours -> h6_to_m90 cadence band -> 600s (default).
    games = pl.DataFrame(
        [
            {
                "canonical_game_id": "g1",
                "season": 2026,
                "week": 1,
                "date": NOW + timedelta(hours=5),
            }
        ]
    )
    warehouse.write("games", games)

    last_started = NOW - timedelta(seconds=100)
    runs = pl.DataFrame(
        [
            {
                "collector_run_id": "run-1",
                "provider": "fake",
                "season": 2026,
                "week": 1,
                "started_at": last_started,
                "completed_at": last_started,
                "status": "FAILED",  # a failing cycle still blocks a retry storm
            }
        ]
    )
    warehouse.write(RUNS_TABLE, runs)

    # Only 100s elapsed; the 90m-6h cadence band requires 300s -> not due yet.
    assert (
        collection_due(
            warehouse=warehouse,
            provider_name="fake",
            season=2026,
            week=1,
            now=NOW,
            config=Config(data={}),
        )
        is False
    )

    # 350s elapsed -> due, even though the last run FAILED.
    later = NOW + timedelta(seconds=250)
    assert (
        collection_due(
            warehouse=warehouse,
            provider_name="fake",
            season=2026,
            week=1,
            now=later,
            config=Config(data={}),
        )
        is True
    )


def test_collection_dispatch_flow_noop_when_not_due(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    provider = _seeded_provider()
    cfg = Config(data={})

    first = collection_dispatch_flow(
        provider=provider, season=2026, week=1, warehouse=warehouse, config=cfg, now=NOW
    )
    assert first is not None

    # Immediately again, same frozen now -> nothing elapsed -> not due -> no-op.
    second = collection_dispatch_flow(
        provider=provider, season=2026, week=1, warehouse=warehouse, config=cfg, now=NOW
    )
    assert second is None

    runs = warehouse.read(RUNS_TABLE)
    assert runs.height == 1


def test_collection_once_flow_matches_collect_once_directly(tmp_path: Path) -> None:
    # One shared, already-seeded provider instance: FakeProvider timestamps
    # (available_at/ingested_at) are stamped at seed time via a real
    # wall-clock read, so two *separately* seeded providers would diverge on
    # those columns even under a frozen `now` for the collection cycle
    # itself -- reusing one instance isolates the comparison to what
    # collect_once vs. collection_once_flow actually do differently (nothing).
    provider = _seeded_provider()
    cfg = Config(data={})

    warehouse_a = Warehouse(tmp_path / "a")
    warehouse_b = Warehouse(tmp_path / "b")

    direct = collect_once(
        provider=provider, season=2026, week=1, warehouse=warehouse_a, config=cfg, now=NOW
    )
    via_flow = collection_once_flow(
        provider=provider, season=2026, week=1, warehouse=warehouse_b, config=cfg, now=NOW
    )

    assert direct.status == via_flow.status
    assert direct.collector_run_id == via_flow.collector_run_id

    # NOTE: collect_once() itself reads the real wall clock (datetime.now(UTC))
    # for completed_at/created_at on collector_runs/collector_resource_runs,
    # despite its own docstring claiming full determinism under a frozen
    # `now` -- a pre-existing PHASE-4 property, not something this PHASE-5
    # wrapper changes or should paper over. Excluded here because it would
    # make ANY two calls (with or without Prefect) diverge, which is not
    # what §46 is testing; only the frozen-`now`-derived fields are compared.
    nondeterministic_columns = {"completed_at", "created_at", "collector_received_at"}
    for table in ("games", "collector_runs", "collector_resource_runs"):
        a = warehouse_a.read(table)
        b = warehouse_b.read(table)
        compare_cols = [c for c in a.columns if c not in nondeterministic_columns]
        assert a.select(compare_cols).equals(b.select(compare_cols)), (
            f"table {table!r} diverged between collect_once and collection_once_flow"
        )
