"""Official game-relative checkpoint definitions and timing rules (PHASE 5).

This module is deliberately free of any Prefect dependency: it is pure,
deterministic logic about *when* an official pregame prediction checkpoint
is due, and it must be testable at full speed with no orchestration runtime
involved.

Non-negotiable semantics (see docs/ORCHESTRATION_ARCHITECTURE.md):

    scheduled_as_of = kickoff_at - offset_seconds
    as_of           = scheduled_as_of

`scheduled_as_of` is the model's knowledge-time cutoff. It is NEVER the
wall-clock time a worker happens to start, a task happens to run, or a
database row happens to be written. Orchestration lateness (a worker that
starts late, but still before kickoff) must never leak later information
into an official checkpoint's forecast -- see `catch_up_due`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from nflprops.config import Config


class CheckpointName(str, Enum):  # noqa: UP042
    """Official checkpoint identities, plus MANUAL for ad-hoc diagnostic runs.

    MANUAL is not an official scheduled checkpoint: it is never produced by
    the checkpoint dispatcher and never satisfies an official checkpoint's
    due/claim logic. It exists only for `nflprops checkpoint run` (a human
    explicitly asking for a one-off prediction at an arbitrary as_of).
    """

    T48H = "T48H"
    T24H = "T24H"
    T6H = "T6H"
    T90M = "T90M"
    T30M = "T30M"
    MANUAL = "MANUAL"


# The five official, scheduled checkpoints -- in required descending-offset
# order. MANUAL is intentionally excluded: it has no fixed offset.
OFFICIAL_CHECKPOINTS: tuple[CheckpointName, ...] = (
    CheckpointName.T48H,
    CheckpointName.T24H,
    CheckpointName.T6H,
    CheckpointName.T90M,
    CheckpointName.T30M,
)

DEFAULT_OFFSET_SECONDS: dict[CheckpointName, int] = {
    CheckpointName.T48H: 172_800,
    CheckpointName.T24H: 86_400,
    CheckpointName.T6H: 21_600,
    CheckpointName.T90M: 5_400,
    CheckpointName.T30M: 1_800,
}


class CheckpointConfigError(ValueError):
    """Raised when official checkpoint configuration is invalid.

    Invalid production checkpoint configuration must fail at startup, not
    be silently corrected or ignored.
    """


@dataclass(frozen=True)
class CheckpointOffsets:
    """Validated offsets (seconds before kickoff) for the five official
    checkpoints. Construct via `CheckpointOffsets.from_mapping(...)` --
    never bypass validation by building the dataclass with unchecked data.
    """

    offset_seconds: dict[CheckpointName, int]

    @classmethod
    def from_mapping(cls, mapping: dict[str, int]) -> CheckpointOffsets:
        offsets: dict[CheckpointName, int] = {}
        for checkpoint in OFFICIAL_CHECKPOINTS:
            if checkpoint.value not in mapping:
                raise CheckpointConfigError(
                    f"missing offset_seconds for official checkpoint {checkpoint.value!r}"
                )
            offsets[checkpoint] = int(mapping[checkpoint.value])
        unexpected = set(mapping) - {c.value for c in OFFICIAL_CHECKPOINTS}
        if unexpected:
            raise CheckpointConfigError(
                f"unknown checkpoint name(s) in offset_seconds config: {sorted(unexpected)}"
            )
        instance = cls(offset_seconds=offsets)
        instance.validate()
        return instance

    def validate(self) -> None:
        for checkpoint in OFFICIAL_CHECKPOINTS:
            offset = self.offset_seconds[checkpoint]
            if offset <= 0:
                raise CheckpointConfigError(
                    f"checkpoint {checkpoint.value!r} offset_seconds must be > 0, got {offset}"
                )

        names = [c.value for c in OFFICIAL_CHECKPOINTS]
        if len(set(names)) != len(names):
            raise CheckpointConfigError("official checkpoint names must be unique")

        ordered = [self.offset_seconds[c] for c in OFFICIAL_CHECKPOINTS]
        if ordered != sorted(ordered, reverse=True) or len(set(ordered)) != len(ordered):
            raise CheckpointConfigError(
                "official checkpoint offsets must satisfy "
                "T48H > T24H > T6H > T90M > T30M "
                f"(got {dict(zip((c.value for c in OFFICIAL_CHECKPOINTS), ordered, strict=True))})"
            )

    def get(self, checkpoint: CheckpointName) -> int:
        if checkpoint not in self.offset_seconds:
            raise CheckpointConfigError(
                f"no offset configured for checkpoint {checkpoint.value!r}"
            )
        return self.offset_seconds[checkpoint]

    @classmethod
    def from_config(cls, cfg: Config) -> CheckpointOffsets:
        """Load from `[checkpoints.offset_seconds]`, falling back to the
        documented defaults for any checkpoint the config omits entirely.
        An explicitly-present-but-invalid config still fails validation."""
        configured = cfg.get_path("checkpoints.offset_seconds", {}) or {}
        mapping = {c.value: DEFAULT_OFFSET_SECONDS[c] for c in OFFICIAL_CHECKPOINTS}
        mapping.update(configured)
        return cls.from_mapping(mapping)


DEFAULT_CHECKPOINT_OFFSETS = CheckpointOffsets.from_mapping(
    {c.value: seconds for c, seconds in DEFAULT_OFFSET_SECONDS.items()}
)


@dataclass(frozen=True)
class CheckpointsRuntimeConfig:
    """`[checkpoints]` runtime toggles, distinct from the offsets table."""

    enabled: bool = True
    catch_up_before_kickoff: bool = True

    @classmethod
    def from_config(cls, cfg: Config) -> CheckpointsRuntimeConfig:
        return cls(
            enabled=bool(cfg.get_path("checkpoints.enabled", True)),
            catch_up_before_kickoff=bool(
                cfg.get_path("checkpoints.catch_up_before_kickoff", True)
            ),
        )


@dataclass(frozen=True)
class OrchestrationConfig:
    """`[orchestration]` settings."""

    enabled: bool = True
    prefect_work_pool: str = "nflprops-production"
    dispatcher_tick_seconds: int = 60

    @classmethod
    def from_config(cls, cfg: Config) -> OrchestrationConfig:
        return cls(
            enabled=bool(cfg.get_path("orchestration.enabled", True)),
            prefect_work_pool=str(
                cfg.get_path("orchestration.prefect_work_pool", "nflprops-production")
            ),
            dispatcher_tick_seconds=int(
                cfg.get_path("orchestration.dispatcher_tick_seconds", 60)
            ),
        )


def scheduled_as_of(
    *, kickoff_at: datetime, checkpoint: CheckpointName, offsets: CheckpointOffsets
) -> datetime:
    """The model knowledge-time cutoff for `checkpoint` given `kickoff_at`.

    scheduled_as_of = kickoff_at - offset_seconds. This is a pure function
    of (kickoff_at, checkpoint, offsets) -- it never reads the wall clock.
    """
    offset = offsets.get(checkpoint)
    return kickoff_at - timedelta(seconds=offset)


def is_past_kickoff(*, kickoff_at: datetime, now: datetime) -> bool:
    """True iff `now >= kickoff_at`: an official pregame checkpoint must
    never execute a new prediction once this is true."""
    return now >= kickoff_at


def checkpoint_due(
    *, scheduled_as_of_time: datetime, kickoff_at: datetime, now: datetime
) -> bool:
    """True iff an official checkpoint is currently due to be (or has
    become eligible to be, via catch-up) executed.

    Due iff scheduled_as_of <= now < kickoff_at. This function only answers
    the *time-window* question; it says nothing about whether the
    checkpoint has already been claimed/run -- callers must separately
    check `run_store` for an existing claim before executing.

    - now < scheduled_as_of  -> not due yet.
    - now >= kickoff_at      -> never due (too late; see `is_past_kickoff`
      / CHECKPOINT_MISSED handling in `run_store`).
    """
    return scheduled_as_of_time <= now < kickoff_at


class CheckpointAction(str, Enum):  # noqa: UP042
    """What an *unclaimed* official checkpoint should do right now."""

    NOT_DUE = "NOT_DUE"
    RUN = "RUN"
    MISSED = "MISSED"


def evaluate_checkpoint(
    *,
    scheduled_as_of_time: datetime,
    kickoff_at: datetime,
    now: datetime,
    catch_up_before_kickoff: bool = True,
    dispatcher_tick_seconds: int = 60,
) -> CheckpointAction:
    """Decide what an unclaimed official checkpoint should do right now
    (§12/§13/§14). Only the time-window/catch-up decision -- callers must
    separately confirm no claim already exists (`run_store.checkpoint_satisfied`)
    before treating `RUN` as license to execute.

    - now < scheduled_as_of                       -> NOT_DUE.
    - now >= kickoff_at                           -> MISSED: an official
      pregame forecast must never execute after kickoff (§12).
    - scheduled_as_of <= now < kickoff_at:
        - discovered within one dispatcher tick of scheduled_as_of -> RUN
          (the ordinary, on-time case).
        - discovered later than that (a worker recovering from an outage)
          and `catch_up_before_kickoff` -> RUN. The caller must still use
          the ORIGINAL `scheduled_as_of_time` as the model's `as_of`, never
          `now` -- this is what makes catch-up scientifically valid (§13).
        - discovered later than that and catch-up is disabled -> MISSED.
    """
    if is_past_kickoff(kickoff_at=kickoff_at, now=now):
        return CheckpointAction.MISSED
    if now < scheduled_as_of_time:
        return CheckpointAction.NOT_DUE
    late = (now - scheduled_as_of_time).total_seconds() > dispatcher_tick_seconds
    if late and not catch_up_before_kickoff:
        return CheckpointAction.MISSED
    return CheckpointAction.RUN


def all_scheduled_as_of(
    *, kickoff_at: datetime, offsets: CheckpointOffsets
) -> dict[CheckpointName, datetime]:
    """scheduled_as_of for every official checkpoint, given one kickoff."""
    return {
        checkpoint: scheduled_as_of(kickoff_at=kickoff_at, checkpoint=checkpoint, offsets=offsets)
        for checkpoint in OFFICIAL_CHECKPOINTS
    }
