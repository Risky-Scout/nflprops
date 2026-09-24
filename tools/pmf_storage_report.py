#!/usr/bin/env python3
"""BLOCK 2A: MEASURED (from the existing lightweight Phase-7B test fixture,
never a new historical/production simulation) vs. PROJECTED row-count and
storage-size report comparing the legacy normalized
`player_prop_distribution_outcomes` representation (one row per
positive-mass outcome) against the BLOCK 2A compact `pmf_payload`
representation (one row per distribution).

Usage:
    python tools/pmf_storage_report.py [--n-draws 3000] [--json OUT.json]

Everything under MEASURED comes from actually building the PMF product
(`nflprops.distributions.build.build_player_prop_distributions`) against
`tests/projections/_projection_fixtures`'s small hand-built
`GameSimulationResult` (the same fixture `tests/distributions/test_pmf.py`
uses) and actually encoding every distribution with the compact codec.
Nothing here runs a new historical simulation, touches PostgreSQL, or
claims a PostgreSQL-measured byte size -- `player_prop_distribution_outcomes`
row *count* is exact (it is the literal row count this representation would
produce), but its on-disk PostgreSQL byte size is not measured here.

Everything under PROJECTED is explicit arithmetic on explicitly stated
assumptions (games/week, weeks/season) applied to the fixture's own
MEASURED per-game `E` (eligible player count) and per-distribution outcome
counts -- labeled as a projection, not a production fact. The fixture's `E`
is a small synthetic roster for test purposes, not a real NFL game's
eligible-player count; there is no real-`E` or historical rows/week figure
recorded anywhere in this repository to project from, so none is invented
here. Swap in a real measured `E` (or real rows/week) to re-run this
arithmetic against production-scale numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "projections"))

from _projection_fixtures import all_player_states, build_simulation  # noqa: E402

from nflprops.distributions import (
    ALL_PROP_TYPES,
    build_player_prop_distributions,
    encode_pmf,
)
from nflprops.projections import eligible_player_states

#: Explicit planning assumptions for the PROJECTED section (BLOCK 2A §9).
#: A regular NFL week has up to 16 games; the regular season is 18 weeks.
ASSUMED_GAMES_PER_WEEK = 16
ASSUMED_WEEKS_PER_SEASON = 18


def _percentile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[idx])


def measure(n_draws: int) -> dict[str, Any]:
    states = all_player_states()
    sim = build_simulation(n_draws=n_draws, player_states=states)
    eligible = eligible_player_states(sim, states)
    frame = build_player_prop_distributions(sim, player_states=states)

    per_distribution = frame.group_by(["player_id", "prop_type"]).len().rename(
        {"len": "outcome_count"}
    )
    outcome_counts = per_distribution["outcome_count"].to_list()

    payload_sizes: list[int] = []
    for (_player_id, _prop_type), group in frame.sort(
        ["player_id", "prop_type", "outcome"]
    ).group_by(["player_id", "prop_type"], maintain_order=True):
        outcomes = tuple(int(x) for x in group["outcome"].to_list())
        probabilities = tuple(float(x) for x in group["p_raw"].to_list())
        payload = encode_pmf(outcomes, probabilities)
        payload_sizes.append(len(payload))

    by_prop_type: dict[str, dict[str, float]] = {}
    for prop in ALL_PROP_TYPES:
        rows = per_distribution.filter(per_distribution["prop_type"] == prop.value)
        counts = rows["outcome_count"].to_list()
        if counts:
            by_prop_type[prop.value] = {
                "distributions": len(counts),
                "mean_outcome_count": statistics.fmean(counts),
                "total_outcome_rows": sum(counts),
            }

    n_distributions = len(eligible) * len(ALL_PROP_TYPES)
    n_legacy_outcome_rows = frame.height

    measured = {
        "n_draws": n_draws,
        "eligible_player_count_E": len(eligible),
        "distributions": n_distributions,
        "legacy_outcome_rows": n_legacy_outcome_rows,
        "outcome_count_per_distribution": {
            "mean": statistics.fmean(outcome_counts),
            "median": statistics.median(outcome_counts),
            "p90": _percentile(outcome_counts, 0.90),
            "p95": _percentile(outcome_counts, 0.95),
            "max": max(outcome_counts),
        },
        "by_prop_type": by_prop_type,
        "compact_payload_bytes": {
            "mean": statistics.fmean(payload_sizes),
            "median": statistics.median(payload_sizes),
            "p95": _percentile(payload_sizes, 0.95),
            "max": max(payload_sizes),
            "total": sum(payload_sizes),
        },
        "compact_distribution_rows": n_distributions,
        "row_count_reduction_ratio": (
            n_legacy_outcome_rows / n_distributions if n_distributions else None
        ),
    }
    return measured


def project(measured: dict[str, Any]) -> dict[str, Any]:
    games_in_fixture = 1  # the fixture builds exactly one GameSimulationResult
    legacy_rows_per_game = measured["legacy_outcome_rows"] / games_in_fixture
    compact_rows_per_game = measured["compact_distribution_rows"] / games_in_fixture

    legacy_rows_per_week = legacy_rows_per_game * ASSUMED_GAMES_PER_WEEK
    compact_rows_per_week = compact_rows_per_game * ASSUMED_GAMES_PER_WEEK

    return {
        "assumptions": {
            "games_per_week": ASSUMED_GAMES_PER_WEEK,
            "weeks_per_season": ASSUMED_WEEKS_PER_SEASON,
            "E_per_game": (
                f"the fixture's MEASURED E={measured['eligible_player_count_E']} "
                f"(a small synthetic test roster -- NOT a real NFL game's "
                f"eligible-player count; no real-production E is recorded "
                f"anywhere in this repository to project from instead)"
            ),
        },
        "legacy_rows_per_week": legacy_rows_per_week,
        "compact_rows_per_week": compact_rows_per_week,
        "legacy_rows_per_18_week_season": legacy_rows_per_week * ASSUMED_WEEKS_PER_SEASON,
        "compact_rows_per_18_week_season": compact_rows_per_week * ASSUMED_WEEKS_PER_SEASON,
    }


def render_human(measured: dict[str, Any], projected: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=== MEASURED (from tests/projections/_projection_fixtures) ===")
    lines.append(f"n_draws                         = {measured['n_draws']}")
    lines.append(f"eligible players (E)             = {measured['eligible_player_count_E']}")
    lines.append(f"distributions (E * 25)           = {measured['distributions']}")
    lines.append(f"legacy outcome rows               = {measured['legacy_outcome_rows']}")
    oc = measured["outcome_count_per_distribution"]
    lines.append(
        "outcomes/distribution: mean={mean:.2f} median={median:.1f} "
        "p90={p90:.1f} p95={p95:.1f} max={max}".format(**oc)
    )
    pb = measured["compact_payload_bytes"]
    lines.append(
        "compact payload bytes: mean={mean:.1f} median={median:.1f} "
        "p95={p95:.1f} max={max} total={total}".format(**pb)
    )
    lines.append(f"row-count reduction ratio (legacy/compact) = {measured['row_count_reduction_ratio']:.2f}x")
    lines.append("")
    lines.append("=== PROJECTED (explicit assumptions -- not a production fact) ===")
    for k, v in projected["assumptions"].items():
        lines.append(f"assumption[{k}] = {v}")
    lines.append(f"legacy rows/week                = {projected['legacy_rows_per_week']:.0f}")
    lines.append(f"compact rows/week               = {projected['compact_rows_per_week']:.0f}")
    lines.append(
        f"legacy rows/18-week season       = {projected['legacy_rows_per_18_week_season']:.0f}"
    )
    lines.append(
        f"compact rows/18-week season      = {projected['compact_rows_per_18_week_season']:.0f}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=3000)
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    measured = measure(args.n_draws)
    projected = project(measured)
    print(render_human(measured, projected))

    if args.json:
        args.json.write_text(
            json.dumps({"measured": measured, "projected": projected}, indent=2)
        )
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
