"""Per-slot OPERATIONAL retry state for checkpoint preparation.

A preparation pass works on one slot (`checkpoint_prepare.
next_preparation_slot`). When its worker times out or fails, that slot --
not the whole queue -- is deferred until `next_retry_at`, so the next
eligible checkpoint can be prepared meanwhile instead of the same slot
heading the queue forever (2026-10-04: the earliest PREPARING slot timed out
at 180 s every pass and starved the other ten).

This is operational bookkeeping only. The deferred checkpoint stays exactly
as it was -- its run SCHEDULED, its request PREPARING or not yet claimed --
scientifically pending; a timeout is never a NOT_EXECUTABLE refusal and a
slot is never skipped permanently: once `next_retry_at` passes it is
eligible again. Eligible slots are taken in two deterministic tiers: slots
with no operational failure first, then slots eligible again after one --
each tier in the normal (cutoff) order -- so persistently failing slots at
the head of the queue can never starve the checkpoints behind them. The delay
grows exponentially with consecutive failures of that slot, capped, so a
slot that keeps failing costs one bounded attempt per cap interval (no hot
loop). A successful pass clears the slot's entry; entries for slots no
longer awaiting preparation (prepared, gated, missed) are pruned.

Stored by the runtime (the only writer) as one small JSON file under
`<runtime_root>/state/`, written atomically -- never in the warehouse, so
recording a deferral needs no writer lock and survives a restart.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from nflprops.platform.checkpoint_prepare import SlotKey

logger = logging.getLogger("nflprops.runtime")

RETRY_STATE_FILE = "checkpoint_prep_retry.json"
RETRY_STATE_SCHEMA = "nflprops.platform.checkpoint_prep_retry/v1"

#: First deferral after an operational failure -- the runtime's global
#: post-failure backoff (`runtime_loop.DEFAULT_CHECKPOINT_RETRY_SECONDS`), so
#: a lone failing slot is retried exactly as before -- doubling per
#: consecutive failure of the same slot up to the cap. Override with
#: NFLPROPS_CHECKPOINT_SLOT_RETRY_BASE_SECONDS / ..._MAX_SECONDS.
DEFAULT_SLOT_RETRY_BASE_SECONDS = 60.0
DEFAULT_SLOT_RETRY_MAX_SECONDS = 1800.0

FAILURE_TIMEOUT = "TIMEOUT"
FAILURE_FAILED = "FAILED"


def slot_retry_delay_seconds(retry_count: int, *, base: float, cap: float) -> float:
    """`base * 2**(retry_count - 1)`, at most `cap` (retry_count >= 1)."""
    if retry_count < 1:
        raise ValueError(f"retry_count must be >= 1, got {retry_count}")
    return float(min(cap, base * 2 ** min(retry_count - 1, 32)))


@dataclass(frozen=True)
class SlotRetry:
    game_id: str
    checkpoint_name: str
    scheduled_as_of: str
    retry_count: int
    last_failure_at: str
    last_timeout_at: str | None
    next_retry_at: str
    last_failure_kind: str
    last_failure_reason: str

    @property
    def key(self) -> SlotKey:
        return (self.game_id, self.checkpoint_name, self.scheduled_as_of)

    def deferred_at(self, now: datetime) -> bool:
        return now < datetime.fromisoformat(self.next_retry_at)


def _key_str(key: SlotKey) -> str:
    return "|".join(key)


@dataclass
class SlotRetryBook:
    path: Path
    entries: dict[str, SlotRetry]

    @classmethod
    def load(cls, path: Path) -> SlotRetryBook:
        """The stored book; an absent file is empty. An unreadable one is
        logged and treated as empty: deferral is an optimization, and with
        no deferrals every slot is simply eligible (the pre-deferral order)."""
        try:
            payload = json.loads(path.read_text())
            if payload.get("schema_version") != RETRY_STATE_SCHEMA:
                raise ValueError(f"unknown schema {payload.get('schema_version')!r}")
            entries = {
                _key_str(entry.key): entry
                for entry in (SlotRetry(**raw) for raw in payload.get("slots", []))
            }
        except FileNotFoundError:
            entries = {}
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "checkpoint slot retry state unreadable; starting empty",
                extra={"fields": {"path": str(path), "error": str(exc)[:300]}},
            )
            entries = {}
        return cls(path=path, entries=entries)

    def save(self) -> None:
        payload: dict[str, Any] = {
            "schema_version": RETRY_STATE_SCHEMA,
            "slots": [asdict(self.entries[k]) for k in sorted(self.entries)],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        tmp.replace(self.path)

    def get(self, key: SlotKey) -> SlotRetry | None:
        return self.entries.get(_key_str(key))

    def deferred(self, now: datetime) -> frozenset[SlotKey]:
        """Slots still inside their deferral at `now`."""
        return frozenset(e.key for e in self.entries.values() if e.deferred_at(now))

    def retried(self) -> frozenset[SlotKey]:
        """Every slot with an operational failure on record (deferred or
        eligible again)."""
        return frozenset(e.key for e in self.entries.values())

    def next_retry_at(self, now: datetime) -> datetime | None:
        """The earliest moment a currently deferred slot becomes eligible."""
        times = [
            datetime.fromisoformat(e.next_retry_at)
            for e in self.entries.values()
            if e.deferred_at(now)
        ]
        return min(times) if times else None

    def record_failure(
        self,
        key: SlotKey,
        *,
        at: datetime,
        timed_out: bool,
        reason: str,
        base_seconds: float,
        max_seconds: float,
    ) -> SlotRetry:
        previous = self.get(key)
        count = (previous.retry_count if previous else 0) + 1
        delay = slot_retry_delay_seconds(count, base=base_seconds, cap=max_seconds)
        entry = SlotRetry(
            game_id=key[0],
            checkpoint_name=key[1],
            scheduled_as_of=key[2],
            retry_count=count,
            last_failure_at=at.isoformat(),
            last_timeout_at=(
                at.isoformat() if timed_out else (previous.last_timeout_at if previous else None)
            ),
            next_retry_at=(at + timedelta(seconds=delay)).isoformat(),
            last_failure_kind=FAILURE_TIMEOUT if timed_out else FAILURE_FAILED,
            last_failure_reason=reason[:500],
        )
        self.entries[_key_str(key)] = entry
        return entry

    def clear(self, key: SlotKey) -> bool:
        return self.entries.pop(_key_str(key), None) is not None

    def prune(self, live: frozenset[SlotKey]) -> list[SlotKey]:
        """Drop entries for slots no longer awaiting preparation."""
        dropped = [e.key for e in self.entries.values() if e.key not in live]
        for key in dropped:
            del self.entries[_key_str(key)]
        return dropped

    def summary(self, now: datetime) -> list[dict[str, Any]]:
        """Status-file view: every entry, with whether it is deferred now."""
        return [
            {**asdict(self.entries[k]), "deferred": self.entries[k].deferred_at(now)}
            for k in sorted(self.entries)
        ]
