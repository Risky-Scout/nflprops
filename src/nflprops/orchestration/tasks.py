"""Prefect task wrappers and retry classification (PHASE 5, §31/§32).

Only transient orchestration/infrastructure failures may ever be retried:
temporary database or object-storage connectivity, a Prefect worker
interruption, or a transient provider/collection-infra error where the
underlying call is idempotent (the existing provider HTTP client retry
loop already owns HTTP-level retries -- this is not a second one).

Deterministic failures must never be blindly retried, because retrying
them reproduces the identical failure and only delays surfacing it:
a PIT leakage violation, a simulation invariant failure, invalid
configuration, an unknown canonical entity, or any other deterministic
input-validation failure.

`is_transient_failure` is a plain function (no Prefect dependency in its
logic) so the classification itself can be unit-tested without exercising
Prefect's retry/sleep machinery -- tests must not actually sleep (§32).
"""

from __future__ import annotations

from nflprops.backtest.leakage import LeakageError as BacktestLeakageError
from nflprops.errors import LeakageError as CoreLeakageError

# Exception types that represent a deterministic, input/model-derived
# failure -- retrying changes nothing. Never retry these.
NON_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    BacktestLeakageError,
    CoreLeakageError,
    AssertionError,
    ValueError,
    KeyError,
    TypeError,
)

# Default Prefect retry policy for genuinely transient orchestration-level
# failures (§32). Kept as module constants so tests can assert on them
# without constructing real delays.
TRANSIENT_RETRIES = 2
TRANSIENT_RETRY_DELAY_SECONDS = [15, 60]


def is_transient_failure(exc: BaseException | None) -> bool:
    """True iff `exc` should be retried under the Phase-5 retry policy.

    `None` (no exception / task did not fail) is treated as not-transient
    only in the trivial sense that there is nothing to retry.
    """
    if exc is None:
        return False
    return not isinstance(exc, NON_RETRYABLE_EXCEPTIONS)


def retry_condition_fn(task, task_run, state) -> bool:
    """Prefect `retry_condition_fn`: retry only transient failures (§31).

    Signature required by Prefect 3: `(task, task_run, state) -> bool`.
    `state.result(raise_on_failure=False)` returns the exception itself for
    a failed task run (verified against the installed Prefect version)
    rather than raising it.
    """
    exc = state.result(raise_on_failure=False)
    return is_transient_failure(exc if isinstance(exc, BaseException) else None)
