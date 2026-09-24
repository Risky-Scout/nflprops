"""Foreground continuous collection loop (PHASE 4).

A simple, testable wrapper around `collect_once()`: run a cycle, compute the
cadence for the next unstarted game, sleep, repeat. No daemonization, no
forking, no Prefect, no systemd unit -- Phase 5 orchestrates/wraps this
logic, it does not replace it.
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable
from datetime import UTC, datetime

from nflprops.collection.models import CollectorRunResult
from nflprops.collection.service import collect_once
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse
from nflprops.domain.protocols import FullProvider

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]


def _default_clock() -> datetime:
    return datetime.now(UTC)


def run_collection_loop(
    *,
    provider: FullProvider,
    season: int,
    week: int,
    warehouse: Warehouse,
    config: Config,
    clock: Clock = _default_clock,
    sleeper: Sleeper = time.sleep,
    max_iterations: int | None = None,
    on_cycle: Callable[[CollectorRunResult], None] | None = None,
    install_signal_handlers: bool = True,
) -> list[CollectorRunResult]:
    """Run `collect_once()` in a loop until interrupted (or `max_iterations`
    cycles complete, for tests).

    `clock`/`sleeper` are injectable so tests never actually wait and can
    drive a fully deterministic, frozen sequence of `now` values.
    `install_signal_handlers=False` lets tests / non-main-thread callers
    avoid `signal.signal()`, which only works on the main thread.
    """
    stop_requested = False

    def _request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        logger.info("collection loop received signal %s; stopping after this cycle", signum)
        stop_requested = True

    previous_handlers: dict[int, object] = {}
    if install_signal_handlers:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _request_stop)

    results: list[CollectorRunResult] = []
    try:
        iteration = 0
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break

            now = clock()
            result = collect_once(
                provider=provider,
                season=season,
                week=week,
                warehouse=warehouse,
                config=config,
                now=now,
            )
            results.append(result)
            if on_cycle is not None:
                on_cycle(result)

            iteration += 1
            if stop_requested:
                break
            if max_iterations is not None and iteration >= max_iterations:
                break

            interval = result.cadence_seconds or config.get_path(
                "collection.no_future_game_poll_seconds", 1800
            )
            sleeper(float(interval))
            if stop_requested:
                break
    except KeyboardInterrupt:
        logger.info("collection loop interrupted; stopping")
    finally:
        if install_signal_handlers:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

    return results
