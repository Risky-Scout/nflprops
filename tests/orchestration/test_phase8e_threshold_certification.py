"""PHASE 8E: final certification of the canonical
`player_game_threshold_events` product.

This module does not re-implement any Phase-8 science. It runs the real
official checkpoint path (`game_checkpoint_flow`) end-to-end, reads the
*persisted* artifacts back, and certifies the locked product contract:

* the E x 131 completeness grid, from persisted rows;
* the projection player universe == the threshold player universe;
* exact hand-recomputed ``p_hit = count(vec >= T) / n_draws`` for a
  representative event in every canonical category (yards / TD milestone /
  FG milestone / first-half / offensive_tds);
* monotonicity across every complete persisted ladder;
* boundary probabilities 0.0 and 1.0 persist exactly (never clipped);
* the 23-series / 131-event / AT_LEAST catalog shape;
* the 5 binary Phase-7 events are never duplicated into this table.

Reuses the certified Phase-7D / Phase-8D checkpoint fixtures + helpers
(`_build_warehouse`, `_execute`, `_full_roster_warehouse`, ...).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_phase7d_checkpoint_projection_integration import (
    N_DRAWS,
    _build_warehouse,
    _execute,
    _projections,
)
from test_phase8d_checkpoint_threshold_integration import (
    CATALOG,
    EVENT_COUNT,
    _full_roster_warehouse,
    _thresholds,
)

import nflprops.pipelines.pregame as pregame_module
from nflprops.orchestration.flows import checkpoints as checkpoints_flow
from nflprops.orchestration.run_store import PredictionRunStatus, PublicationStatus
from nflprops.orchestration.threshold_event_store import (
    PLAYER_GAME_THRESHOLD_EVENTS_TABLE,
    compute_threshold_event_id,
)
from nflprops.projections.stats import REGISTRY

_REGISTRY_BY_NAME = {spec.name: spec for spec in REGISTRY}

# The 5 binary Phase-7 events whose canonical probability stays the
# Phase-7 projection mean -- they must NEVER get a Phase-8 threshold row
# (directive Phase 8E section 8).
_BINARY_PHASE7_EVENTS: frozenset[str] = frozenset(
    {"anytime_td", "anytime_td_1q", "anytime_td_1h", "anytime_td_2h", "first_td"}
)


def _source_vector(sim, player_id: str, stat_name: str) -> np.ndarray:
    """Recompute the Phase-8 source draw vector the exact way
    `nflprops.thresholds.build._source_vector` does: a registry extractor,
    or -- for `offensive_tds` -- the elementwise sum of its declared
    Phase-7 registry inputs. No simulation, no RNG."""
    derived = CATALOG.derived_stats.get(stat_name)
    if derived is None:
        return np.asarray(_REGISTRY_BY_NAME[stat_name].extract(sim, player_id))
    total: np.ndarray | None = None
    for component in derived.inputs:
        part = np.asarray(_REGISTRY_BY_NAME[component].extract(sim, player_id))
        total = part if total is None else total + part
    assert total is not None
    return total


def _run_with_captured_simulation(
    warehouse, monkeypatch: pytest.MonkeyPatch, *, run_id: str
):
    """Execute one real checkpoint and return the ONE GameSimulationResult
    object that the threshold builder consumed."""
    captured: dict[str, object] = {}
    real_thr = checkpoints_flow.build_player_game_threshold_events

    def _thr_spy(simulation, *, player_states, catalog=None):
        captured["sim"] = simulation
        return real_thr(simulation, player_states=player_states, catalog=catalog)

    real_price = pregame_module.price_current_markets

    def _price_spy(game, result, quotes, **kwargs):
        captured["price_result"] = result
        return real_price(game, result, quotes, **kwargs)

    monkeypatch.setattr(
        checkpoints_flow, "build_player_game_threshold_events", _thr_spy
    )
    monkeypatch.setattr(pregame_module, "price_current_markets", _price_spy)

    record = _execute(warehouse, run_id=run_id)
    assert record.status is PredictionRunStatus.SUCCESS
    return captured, record


# --------------------------------------------------------------------- catalog shape


def test_catalog_shape_is_23_series_131_events_at_least() -> None:
    assert CATALOG.version == "2026.1.0"
    assert CATALOG.event_type == "AT_LEAST"
    assert EVENT_COUNT == CATALOG.event_count == 131

    assert len(CATALOG.ladders) == 23
    derived = set(CATALOG.derived_stats)
    assert derived == {"offensive_tds"}
    registry_backed = [
        lad for lad in CATALOG.ladders if lad.stat_name not in derived
    ]
    assert len(registry_backed) == 22  # 22 Phase-7 registry-backed series
    assert len(derived) == 1  # + 1 catalog-derived series -> 23 total

    standard = [
        lad for lad in CATALOG.ladders
        if lad.classification == "STANDARD_THRESHOLD_ELIGIBLE"
    ]
    milestone = [
        lad for lad in CATALOG.ladders if lad.classification == "MILESTONE_ONLY"
    ]
    assert len(standard) == 16
    assert len(milestone) == 7  # 6 registry-backed + offensive_tds
    assert sum(len(lad.thresholds) for lad in standard) == 111
    assert sum(len(lad.thresholds) for lad in milestone) == 20

    # every registry-backed catalog series is a real Phase-7 registry stat
    for ladder in registry_backed:
        assert ladder.stat_name in _REGISTRY_BY_NAME

    # thresholds are ascending positive integers
    for ladder in CATALOG.ladders:
        assert list(ladder.thresholds) == sorted(ladder.thresholds)
        assert all(isinstance(t, int) and t >= 1 for t in ladder.thresholds)


def test_offensive_tds_ladder_is_exactly_two_and_three() -> None:
    off = next(
        lad for lad in CATALOG.ladders if lad.stat_name == "offensive_tds"
    )
    assert off.thresholds == (2, 3)  # 1+ is covered by the Phase-7 anytime_td binary
    derived = CATALOG.derived_stats["offensive_tds"]
    assert derived.inputs == ("receiving_tds", "rushing_tds")  # never passing_tds
    assert "passing_tds" not in derived.inputs


# --------------------------------------------------- end-to-end artifact certification


def test_end_to_end_artifact_grid_and_universe(tmp_path: Path) -> None:
    warehouse, expected_eligible = _full_roster_warehouse(tmp_path)
    record = _execute(warehouse, run_id="p8e-e2e")
    assert record.status is PredictionRunStatus.SUCCESS

    proj = _projections(warehouse, run_id="p8e-e2e")
    thr = _thresholds(warehouse, run_id="p8e-e2e")

    proj_players = set(proj["player_id"].to_list())
    thr_players = set(thr["player_id"].to_list())
    assert proj_players == expected_eligible
    assert thr_players == proj_players  # identical player universe

    e = len(proj_players)
    assert e == 9
    assert proj.height == e * 30 == 270
    assert thr.height == e * EVENT_COUNT == 9 * 131 == 1179

    # every threshold player carries exactly the 131 canonical catalog keys
    canonical_keys = {(s, "AT_LEAST", t) for s, t in CATALOG.iter_events()}
    assert len(canonical_keys) == 131
    for pid in thr_players:
        rows = thr.filter(pl.col("player_id") == pid)
        assert rows.height == 131
        got = set(
            zip(
                rows["stat_name"].to_list(),
                rows["event_type"].to_list(),
                rows["threshold"].to_list(),
                strict=True,
            )
        )
        assert got == canonical_keys

    # run-level scalars match the parent prediction run exactly
    assert thr["n_draws"].unique().to_list() == [N_DRAWS]
    assert thr["catalog_version"].unique().to_list() == [CATALOG.version]
    assert thr["event_type"].unique().to_list() == ["AT_LEAST"]
    assert thr["season"].unique().to_list() == [record.season]
    assert thr["week"].unique().to_list() == [record.week]
    assert thr["game_id"].unique().to_list() == [record.game_id]

    # deterministic id over (run_id | player_id | stat_name | event_type | threshold)
    for r in thr.iter_rows(named=True):
        assert r["threshold_event_id"] == compute_threshold_event_id(
            run_id="p8e-e2e",
            player_id=r["player_id"],
            stat_name=r["stat_name"],
            event_type=r["event_type"],
            threshold=r["threshold"],
        )

    # no downstream pricing columns leaked into the canonical model table
    for forbidden in (
        "p_miss", "american_odds", "decimal_odds", "p_push", "ev_per_unit",
        "vendor", "line_value", "fair_odds", "edge",
    ):
        assert forbidden not in thr.columns


# ----------------------------------------------------- exact numerical acceptance proofs


@pytest.mark.parametrize(
    "player_id, stat_name, threshold, category",
    [
        ("h:qb:starter", "passing_yards", 225, "passing-yards threshold"),
        ("a:rb:1", "rushing_yards", 40, "rushing-yards threshold"),
        (None, "receiving_yards", 50, "receiving-yards threshold"),
        ("h:qb:starter", "passing_tds", 2, "touchdown milestone"),
        ("h:k1:starter", "fg_made", 2, "field-goal milestone"),
        ("h:qb:starter", "passing_yards_1h", 125, "first-half threshold"),
        ("a:rb:1", "offensive_tds", 2, "offensive_tds milestone"),
    ],
)
def test_persisted_p_hit_is_exact_empirical_frequency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    player_id: str | None,
    stat_name: str,
    threshold: int,
    category: str,
) -> None:
    warehouse, _ = _full_roster_warehouse(tmp_path)
    captured, _ = _run_with_captured_simulation(
        warehouse, monkeypatch, run_id="p8e-num"
    )
    sim = captured["sim"]
    assert captured["price_result"] is sim  # pricing consumed the same object

    thr = _thresholds(warehouse, run_id="p8e-num")
    pid = player_id or thr["player_id"].sort().to_list()[0]

    vec = _source_vector(sim, pid, stat_name)
    assert vec.shape[0] == sim.n_draws == N_DRAWS
    expected = float(np.count_nonzero(vec >= threshold) / sim.n_draws)

    row = thr.filter(
        (pl.col("player_id") == pid)
        & (pl.col("stat_name") == stat_name)
        & (pl.col("threshold") == threshold)
    )
    assert row.height == 1, f"{category}: missing canonical row"
    assert row["p_hit"][0] == expected  # exact -- no tolerance, no rounding


def test_offensive_tds_excludes_passing_tds_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse, _ = _full_roster_warehouse(tmp_path)
    captured, _ = _run_with_captured_simulation(
        warehouse, monkeypatch, run_id="p8e-offtd"
    )
    sim = captured["sim"]
    thr = _thresholds(warehouse, run_id="p8e-offtd")

    for pid in thr["player_id"].unique().to_list():
        recv = np.asarray(_REGISTRY_BY_NAME["receiving_tds"].extract(sim, pid))
        rush = np.asarray(_REGISTRY_BY_NAME["rushing_tds"].extract(sim, pid))
        combined = recv + rush  # NOT + passing_tds
        for threshold in (2, 3):
            expected = float(np.count_nonzero(combined >= threshold) / sim.n_draws)
            row = thr.filter(
                (pl.col("player_id") == pid)
                & (pl.col("stat_name") == "offensive_tds")
                & (pl.col("threshold") == threshold)
            )
            assert row.height == 1
            assert row["p_hit"][0] == expected


# --------------------------------------------------------------------- monotonicity


def test_every_persisted_ladder_is_monotone_non_increasing(tmp_path: Path) -> None:
    warehouse, _ = _full_roster_warehouse(tmp_path)
    _execute(warehouse, run_id="p8e-mono")
    thr = _thresholds(warehouse, run_id="p8e-mono")

    checked = 0
    for (pid, stat), group in thr.group_by(
        ["player_id", "stat_name"], maintain_order=False
    ):
        ordered = group.sort("threshold")
        thresholds = ordered["threshold"].to_list()
        p_hits = ordered["p_hit"].to_list()
        assert thresholds == sorted(thresholds)
        for (t1, p1), (t2, p2) in zip(
            list(zip(thresholds, p_hits, strict=True))[:-1],
            list(zip(thresholds, p_hits, strict=True))[1:],
            strict=True,
        ):
            assert t1 < t2
            assert p1 >= p2, (
                f"{pid}/{stat}: p_hit rose from {p1} to {p2} "
                f"as threshold went {t1} -> {t2}"
            )
        checked += 1
    assert checked == 9 * 23  # every (eligible player, catalog series) ladder


# --------------------------------------------------------------- boundary probabilities


def test_zero_and_one_boundary_probabilities_persist_exactly(tmp_path: Path) -> None:
    warehouse, _ = _full_roster_warehouse(tmp_path)
    _execute(warehouse, run_id="p8e-bound")
    thr = _thresholds(warehouse, run_id="p8e-bound")

    p_hit = thr["p_hit"]
    # full [0, 1] containment, nothing clipped or repaired
    assert p_hit.min() == 0.0
    assert p_hit.max() == 1.0
    assert p_hit.is_finite().all()

    # every persisted p_hit is an exact empirical frequency k / n_draws --
    # a rational with denominator n_draws, stored with no rounding, no
    # clipping, no calibration. `k` recovers as a non-negative integer in
    # [0, n_draws], and re-dividing reproduces the stored value bit-for-bit.
    stored = p_hit.to_numpy()
    ks = np.round(stored * N_DRAWS)
    assert np.abs(stored * N_DRAWS - ks).max() < 1e-6  # k is integral
    assert ks.min() >= 0
    assert ks.max() <= N_DRAWS
    assert np.array_equal(ks / N_DRAWS, stored)  # k / n_draws round-trips exactly

    # p_hit == 0.0 boundary: a kicker's passing_yards ladder is an honest
    # all-zero distribution -- every one of its 10 canonical events is a
    # real row at exactly 0.0 (not omitted, not nudged).
    kicker_pass = thr.filter(
        (pl.col("player_id") == "a:k1:starter")
        & (pl.col("stat_name") == "passing_yards")
    )
    assert kicker_pass.height == 10
    assert kicker_pass["p_hit"].to_list() == [0.0] * 10

    # p_hit == 1.0 boundary: a lead back clears the low rungs of the
    # rush_attempts ladder on every draw -- stored as exactly 1.0.
    rb_floor = thr.filter(
        (pl.col("player_id") == "h:rb:1")
        & (pl.col("stat_name") == "rush_attempts")
        & (pl.col("threshold") == 5)
    )
    assert rb_floor.height == 1
    assert rb_floor["p_hit"][0] == 1.0


# ------------------------------------------------- binary Phase-7 non-duplication


def test_binary_phase7_events_have_no_threshold_rows(tmp_path: Path) -> None:
    warehouse, _ = _full_roster_warehouse(tmp_path)
    _execute(warehouse, run_id="p8e-binary")
    thr = warehouse.read(PLAYER_GAME_THRESHOLD_EVENTS_TABLE)

    present = set(thr["stat_name"].unique().to_list())
    assert present.isdisjoint(_BINARY_PHASE7_EVENTS)
    for event in _BINARY_PHASE7_EVENTS:
        assert thr.filter(pl.col("stat_name") == event).height == 0

    # the catalog itself never lists them either
    assert set(CATALOG.stat_names).isdisjoint(_BINARY_PHASE7_EVENTS)


# ------------------------------------------------------------ MODEL_ONLY / no-model


def test_model_only_certification_requires_both_complete_artifacts(
    tmp_path: Path,
) -> None:
    from datetime import timedelta

    from test_phase7d_checkpoint_projection_integration import AS_OF

    warehouse = _build_warehouse(
        tmp_path,
        quote_visible_at=AS_OF + timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(minutes=5),
    )
    record = _execute(warehouse, run_id="p8e-mo")
    assert record.status is PredictionRunStatus.SUCCESS
    assert record.publication_status is PublicationStatus.MODEL_ONLY

    proj = _projections(warehouse, run_id="p8e-mo")
    thr = _thresholds(warehouse, run_id="p8e-mo")
    e = proj["player_id"].n_unique()
    assert proj.height == e * 30
    assert thr.height == e * 131  # complete E x 131 threshold artifact
    assert warehouse.read("predictions").is_empty()  # zero executable prices, valid
