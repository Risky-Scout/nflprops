#!/usr/bin/env python3
"""BLOCK 2A: optional, safe utility to backfill compact `pmf_payload`
columns onto existing `player_prop_distributions` rows that only have the
legacy normalized `player_prop_distribution_outcomes` child rows.

Never required for correctness: `nflprops.orchestration.distribution_store.
read_distribution_pmf` already falls back to the legacy outcome rows for
any record with `pmf_payload IS NULL`. This utility exists purely to shrink
already-stored data by additively populating the compact payload -- it
NEVER deletes a legacy outcome row, and it never guesses: every candidate
is independently re-verified against its own legacy rows before anything
is written.

For each `player_prop_distributions` row with `pmf_payload IS NULL`:

  1. read its `player_prop_distribution_outcomes` child rows
  2. verify they sum to 1.0 within `NORMALIZATION_TOLERANCE` (source
     normalization)
  3. encode them with the compact codec
  4. decode the freshly-encoded payload back
  5. compare the decoded ``(outcome, probability)`` pairs to the ORIGINAL
     legacy rows -- exact equality required
  6. only if every check passes: write `pmf_codec_version` /
     `pmf_outcome_count` / `pmf_payload` / `pmf_payload_sha256` onto that
     one `player_prop_distributions` row (additive -- every other column,
     including the legacy outcome rows, is left untouched)

Any mismatch is reported, never silently repaired, and that one
distribution is skipped (nothing is written for it). `--dry-run` (the
default) reports what WOULD happen without writing anything; `--apply`
performs the writes.

This is a manual, offline, correctness-and-safety tool -- not a
performance-tuned bulk migration. It reads/writes through the ordinary
backend-agnostic `StorageBackend.append` path (the same one every other
persistence module in this package uses), not a bespoke bulk-UPDATE
fast path. Do not run it against a large production database without
first checking its walltime characteristics.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nflprops.distributions.pmf import NORMALIZATION_TOLERANCE
from nflprops.distributions.pmf_codec import (
    CODEC_VERSION,
    PMFCodecError,
    decode_pmf,
    encode_pmf,
    payload_sha256,
)
from nflprops.orchestration.distribution_store import (
    PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE,
    PLAYER_PROP_DISTRIBUTIONS_TABLE,
)


@dataclass(frozen=True)
class BackfillOutcome:
    distribution_id: str
    status: str  # "would_backfill" / "backfilled" / "skipped_no_legacy_rows" / "mismatch"
    detail: str = ""


def _load_backend(*, warehouse_root: Path | None, database_url: str | None):
    if (warehouse_root is None) == (database_url is None):
        raise SystemExit("exactly one of --warehouse-root or --database-url is required")
    if warehouse_root is not None:
        from nflprops.data.storage.duckdb import DuckDBStorageBackend

        return DuckDBStorageBackend(warehouse_root)
    from nflprops.data.storage.postgres import PostgresStorageBackend

    assert database_url is not None
    return PostgresStorageBackend(database_url)


def _candidates(backend, *, run_id: str | None) -> pl.DataFrame:
    if not backend.exists(PLAYER_PROP_DISTRIBUTIONS_TABLE):
        return pl.DataFrame()
    stored = backend.read(PLAYER_PROP_DISTRIBUTIONS_TABLE)
    if stored.is_empty():
        return stored
    out = stored
    if "pmf_payload" in out.columns:
        out = out.filter(pl.col("pmf_payload").is_null())
    if run_id is not None:
        out = out.filter(pl.col("run_id") == run_id)
    return out


def _legacy_outcomes_for(backend, distribution_key: int) -> pl.DataFrame:
    if not backend.exists(PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE):
        return pl.DataFrame()
    stored = backend.read(PLAYER_PROP_DISTRIBUTION_OUTCOMES_TABLE)
    if stored.is_empty() or "distribution_key" not in stored.columns:
        return pl.DataFrame()
    return stored.filter(pl.col("distribution_key") == distribution_key).sort("outcome")


def _verify_and_encode(
    legacy: pl.DataFrame,
) -> tuple[bytes, str] | tuple[None, str]:
    outcomes = tuple(int(x) for x in legacy["outcome"].to_list())
    probabilities = tuple(float(x) for x in legacy["p_raw"].to_list())

    total = sum(probabilities)
    if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
        return None, (
            f"legacy outcome rows sum to {total!r}, not 1.0 within "
            f"{NORMALIZATION_TOLERANCE} -- source normalization failed, not backfilled"
        )

    try:
        payload = encode_pmf(outcomes, probabilities)
        decoded = decode_pmf(payload)
    except PMFCodecError as exc:
        return None, f"codec rejected the legacy outcome rows: {exc}"

    if decoded.outcomes != outcomes or decoded.probabilities != probabilities:
        return None, "encode -> decode round trip did not match the legacy rows exactly"

    return payload, ""


def run_backfill(
    backend, *, run_id: str | None, apply: bool
) -> list[BackfillOutcome]:
    results: list[BackfillOutcome] = []
    candidates = _candidates(backend, run_id=run_id)
    if candidates.is_empty():
        return results

    to_write: list[dict[str, object]] = []
    for record in candidates.iter_rows(named=True):
        distribution_id = str(record["distribution_id"])
        distribution_key = int(record["distribution_key"])
        legacy = _legacy_outcomes_for(backend, distribution_key)
        if legacy.is_empty():
            results.append(
                BackfillOutcome(
                    distribution_id=distribution_id,
                    status="skipped_no_legacy_rows",
                    detail="no pmf_payload and no legacy outcome rows -- cannot backfill",
                )
            )
            continue

        payload, detail = _verify_and_encode(legacy)
        if payload is None:
            results.append(
                BackfillOutcome(
                    distribution_id=distribution_id, status="mismatch", detail=detail
                )
            )
            continue

        if not apply:
            results.append(
                BackfillOutcome(distribution_id=distribution_id, status="would_backfill")
            )
            continue

        updated = dict(record)
        updated["pmf_codec_version"] = CODEC_VERSION
        updated["pmf_outcome_count"] = len(legacy)
        updated["pmf_payload"] = payload
        updated["pmf_payload_sha256"] = payload_sha256(payload)
        to_write.append(updated)
        results.append(BackfillOutcome(distribution_id=distribution_id, status="backfilled"))

    if apply and to_write:
        backend.append(
            PLAYER_PROP_DISTRIBUTIONS_TABLE,
            pl.DataFrame(to_write),
            key=["distribution_key"],
            keep="last",
        )

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--warehouse-root", type=Path, help="local DuckDB/Parquet warehouse root")
    source.add_argument("--database-url", type=str, help="PostgreSQL DSN")
    parser.add_argument("--run-id", type=str, default=None, help="restrict to one run_id")
    parser.add_argument(
        "--apply", action="store_true", help="actually write (default: dry-run report only)"
    )
    args = parser.parse_args(argv)

    backend = _load_backend(warehouse_root=args.warehouse_root, database_url=args.database_url)
    results = run_backfill(backend, run_id=args.run_id, apply=args.apply)

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        if r.status == "mismatch":
            print(f"MISMATCH distribution_id={r.distribution_id}: {r.detail}")
        elif r.status == "skipped_no_legacy_rows":
            print(f"SKIP distribution_id={r.distribution_id}: {r.detail}")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] candidates={len(results)} " + " ".join(f"{k}={v}" for k, v in sorted(by_status.items())))

    return 1 if by_status.get("mismatch") else 0


if __name__ == "__main__":
    sys.exit(main())
