"""§31/§32: retry classification is a plain function -- no real Prefect
retry execution, no sleeping."""

from __future__ import annotations

import pytest

from nflprops.backtest.leakage import LeakageError as BacktestLeakageError
from nflprops.backtest.leakage import LeakageFinding, LeakageRule
from nflprops.orchestration.tasks import (
    TRANSIENT_RETRIES,
    TRANSIENT_RETRY_DELAY_SECONDS,
    is_transient_failure,
)


def test_transient_retry_policy_constants() -> None:
    assert TRANSIENT_RETRIES == 2
    assert TRANSIENT_RETRY_DELAY_SECONDS == [15, 60]


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad input"),
        AssertionError("INV001 failed"),
        KeyError("missing"),
        TypeError("wrong type"),
        BacktestLeakageError(
            (LeakageFinding(LeakageRule.STATE_AFTER_ASOF, "detail"),)
        ),
    ],
)
def test_deterministic_failures_are_never_retried(exc: BaseException) -> None:
    assert is_transient_failure(exc) is False


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("db connection reset"),
        TimeoutError("upstream timeout"),
        OSError("network unreachable"),
    ],
)
def test_transient_infrastructure_failures_are_retried(exc: BaseException) -> None:
    assert is_transient_failure(exc) is True


def test_no_exception_is_not_transient() -> None:
    assert is_transient_failure(None) is False
