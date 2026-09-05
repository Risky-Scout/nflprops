"""PHASE 4: the foreground continuous collection loop.

Fully deterministic: fake clock, fake sleeper, no real waiting, no real
signals sent to the process (signal handler installation is disabled here so
tests can run off the main thread / in parallel without clobbering the
process's real signal handlers).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.collection.loop import run_collection_loop
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse

NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _provider_with_game(kickoff_delta: timedelta) -> FakeProvider:
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    provider.seed_game(
        "g1",
        home_team_native_id="t1",
        visitor_team_native_id="t2",
        week=1,
        date=NOW + kickoff_delta,
    )
    return provider


def test_loop_stops_after_max_iterations(tmp_path: Path) -> None:
    provider = _provider_with_game(timedelta(hours=5))
    warehouse = Warehouse(tmp_path / "warehouse")
    clock_calls = {"n": 0}
    sleep_calls: list[float] = []

    def fake_clock() -> datetime:
        clock_calls["n"] += 1
        return NOW

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    results = run_collection_loop(
        provider=provider,
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        clock=fake_clock,
        sleeper=fake_sleep,
        max_iterations=3,
        install_signal_handlers=False,
    )

    assert len(results) == 3
    # Sleeps between cycles only -- 2 sleeps for 3 cycles.
    assert len(sleep_calls) == 2
    assert all(s > 0 for s in sleep_calls)


def test_loop_never_actually_sleeps_in_tests(tmp_path: Path) -> None:
    """The sleeper is fully injectable -- a test asserting on call args,
    never real wall-clock time, proves this."""
    provider = _provider_with_game(timedelta(hours=5))
    warehouse = Warehouse(tmp_path / "warehouse")
    sleep_calls: list[float] = []

    import time

    start = time.monotonic()
    run_collection_loop(
        provider=provider,
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        clock=lambda: NOW,
        sleeper=sleep_calls.append,
        max_iterations=5,
        install_signal_handlers=False,
    )
    elapsed = time.monotonic() - start

    assert len(sleep_calls) == 4
    assert elapsed < 2.0  # would be minutes if sleeper were real


def test_loop_uses_result_cadence_as_next_sleep_interval(tmp_path: Path) -> None:
    provider = _provider_with_game(timedelta(minutes=20))  # -> 60s cadence
    warehouse = Warehouse(tmp_path / "warehouse")
    sleep_calls: list[float] = []

    run_collection_loop(
        provider=provider,
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        clock=lambda: NOW,
        sleeper=sleep_calls.append,
        max_iterations=2,
        install_signal_handlers=False,
    )

    assert sleep_calls == [60.0]


def test_loop_on_cycle_callback_fires_once_per_iteration(tmp_path: Path) -> None:
    provider = _provider_with_game(timedelta(hours=5))
    warehouse = Warehouse(tmp_path / "warehouse")
    seen = []

    run_collection_loop(
        provider=provider,
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        clock=lambda: NOW,
        sleeper=lambda _s: None,
        max_iterations=3,
        on_cycle=seen.append,
        install_signal_handlers=False,
    )

    assert len(seen) == 3


def test_loop_stops_cleanly_when_stop_is_requested_mid_run(tmp_path: Path) -> None:
    """Simulates a graceful shutdown request (what SIGINT/SIGTERM trigger in
    production) without actually sending a signal -- the loop must stop
    after completing its current cycle rather than mid-cycle."""
    provider = _provider_with_game(timedelta(hours=5))
    warehouse = Warehouse(tmp_path / "warehouse")

    call_count = {"n": 0}

    def clock_that_stops_after_two() -> datetime:
        call_count["n"] += 1
        return NOW

    results = run_collection_loop(
        provider=provider,
        season=2026,
        week=1,
        warehouse=warehouse,
        config=Config(data={}),
        clock=clock_that_stops_after_two,
        sleeper=lambda _s: None,
        max_iterations=None,  # unbounded -- only the stop request ends it
        install_signal_handlers=False,
        on_cycle=_StopAfter(2),
    )

    assert len(results) == 2


class _StopAfter:
    """Raises inside the on_cycle hook after N cycles to unwind the loop, as
    a stand-in for a real interrupt landing mid-loop."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.count = 0

    def __call__(self, _result: object) -> None:
        self.count += 1
        if self.count >= self.n:
            raise KeyboardInterrupt
