"""BLOCK 4 (Option A): official evidence is strict point-in-time.

* 2026+ never gets estimated (reconstructed) availability -- not from the
  lean historical backfill, not from a versioned append, not for a
  correction.
* Older-season estimates are RESEARCH_ONLY under LIVE_PIT: dropped from
  every official view, never live promotion evidence.
* Phase 10C3A (HISTORICAL_WALK_FORWARD, Gate 1): estimated completed-event
  rows are certified by event chronology -- a distinct class that only
  historical promotion accepts -- and `--evidence research` is never
  promotion evidence.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

from test_phase10c3a_runner import _build_two_season_warehouse

from nflprops.calibration.historical_runner import WarehouseTables, official_tables
from nflprops.calibration.phase10c3a_runner import (
    ConfigurationError,
    RunnerConfig,
    parse_args,
    run,
)
from nflprops.data.evidence_policy import (
    STRICT_PIT_FIRST_SEASON,
    EvidenceClass,
    EvidencePolicyError,
    classify_tables,
    estimated_availability_allowed,
    official_view,
    require_official_warehouse,
)
from nflprops.data.outcome_versions import OutcomeVersionError, append_outcome_versions
from nflprops.data.warehouse import Warehouse
from nflprops.pipelines.lean import LeanIngestor
from nflprops.platform import remote_training as rt

RECEIVED = datetime(2026, 10, 2, 15, 0, tzinfo=UTC)


def test_strict_pit_starts_in_2026() -> None:
    assert STRICT_PIT_FIRST_SEASON == 2026
    assert estimated_availability_allowed(2025)
    assert not estimated_availability_allowed(2026)
    assert not estimated_availability_allowed(2027)


def test_versioned_append_refuses_2026_estimates_even_when_allowed(tmp_path: Path) -> None:
    wh = Warehouse(tmp_path / "c")
    row = {"canonical_game_id": "g", "canonical_player_id": "p", "receiving_yards": 1,
           "available_at": RECEIVED, "ingested_at": RECEIVED, "available_at_is_estimated": True}
    with pytest.raises(OutcomeVersionError, match="strict-PIT"):
        append_outcome_versions(wh, "player_game_stats", pl.DataFrame([row]),
                                ingest_run_id="r", allow_estimated=True, season=2026)
    assert not wh.exists("player_game_stats")
    # The same row for a pre-2026 season is accepted -- and RESEARCH_ONLY.
    append_outcome_versions(wh, "player_game_stats", pl.DataFrame([row]),
                            ingest_run_id="r", allow_estimated=True, season=2025)
    evidence, counts = classify_tables({"player_game_stats": wh.read("player_game_stats")})
    assert evidence is EvidenceClass.RESEARCH_ONLY and counts == {"player_game_stats": 1}


class _SeasonProvider:
    """Minimal provider for `LeanIngestor.ingest_season` (regular season,
    no advanced/goat/pbp). Records carry the provider boundary's genuine
    receipt time, like `BDLProvider`."""

    def __init__(self, season: int) -> None:
        self.season = season

    def _pit(self, rid: str) -> dict[str, Any]:
        return {"available_at": RECEIVED, "ingested_at": RECEIVED,
                "available_at_is_estimated": False, "provider": "bdl",
                "provider_record_id": rid}

    def games(self, seasons=None, season_types=None, **_kw):
        kickoff = datetime(self.season, 9, 13, 17, tzinfo=UTC)
        return [{"canonical_game_id": "g1", "season": self.season, "week": 1,
                 "season_type": 2, "postseason": False, "status": "Final",
                 "status_state": "final", "date": kickoff, "home_team_score": 24,
                 "visitor_team_score": 17, "home_canonical_team_id": "h",
                 "visitor_canonical_team_id": "v", **self._pit("g1")}]

    def player_game_stats(self, seasons=None, season_type=None, **_kw):
        return [{"canonical_game_id": "g1", "canonical_player_id": "p1",
                 "canonical_team_id": "h", "receiving_yards": 50, "receiving_targets": 6,
                 **self._pit("g1:p1")}]

    def team_game_stats(self, seasons=None, season_type=None, **_kw):
        return [{"canonical_game_id": "g1", "canonical_team_id": t, "passing_attempts": 30,
                 "rushing_attempts": 25, **self._pit(f"g1:{t}")} for t in ("h", "v")]

    def player_season_stats(self, *_a, **_kw):
        return []


def _ingest(tmp_path: Path, season: int) -> Warehouse:
    wh = Warehouse(tmp_path / f"wh{season}")
    LeanIngestor(_SeasonProvider(season), wh).ingest_season(  # type: ignore[arg-type]
        season, include_postseason=False, include_advanced=False, historical_backfill=True
    )
    return wh


def test_lean_historical_backfill_never_estimates_2026(tmp_path: Path) -> None:
    wh = _ingest(tmp_path, 2026)
    for table in ("games", "player_game_stats", "team_game_stats"):
        frame = wh.read(table)
        assert not frame.is_empty(), table
        assert frame["available_at_is_estimated"].to_list() == [False] * frame.height, table
        assert set(frame["available_at"].to_list()) == {RECEIVED}, table
    ps = wh.read("player_game_stats")
    assert ps["first_seen_at"].to_list() == [RECEIVED]
    assert ps["provider_observed_at"].null_count() == ps.height
    evidence, _ = classify_tables({t: wh.read(t) for t in wh.tables()})
    assert evidence is EvidenceClass.OFFICIAL_PIT_FAITHFUL
    require_official_warehouse(wh, context="2026 ingest")  # does not raise


def test_lean_older_season_estimates_are_research_only(tmp_path: Path) -> None:
    wh = _ingest(tmp_path, 2025)
    ps = wh.read("player_game_stats")
    assert ps["available_at_is_estimated"].to_list() == [True]
    assert ps["available_at"][0] == datetime(2025, 9, 13, 17, tzinfo=UTC) + timedelta(hours=12)
    # first_seen_at stays genuine even when availability is a research estimate.
    assert ps["first_seen_at"].to_list() == [RECEIVED]
    evidence, counts = classify_tables({t: wh.read(t) for t in wh.tables()})
    assert evidence is EvidenceClass.RESEARCH_ONLY
    assert {"games", "player_game_stats", "team_game_stats"} <= set(counts)
    with pytest.raises(EvidencePolicyError, match="RESEARCH_ONLY"):
        require_official_warehouse(wh, context="official checkpoint")
    for table in ("games", "player_game_stats", "team_game_stats"):
        assert official_view(wh.read(table)).is_empty(), table


def test_official_tables_drop_every_estimated_row() -> None:
    mixed = pl.DataFrame({"x": [1, 2, 3], "available_at_is_estimated": [True, False, None]})
    unflagged = pl.DataFrame({"x": [1]})
    tables = WarehouseTables(
        games=mixed, player_stats=mixed, team_stats=mixed, players=unflagged,
        roster=mixed, injuries=mixed, injury_runs=unflagged, game_odds=unflagged,
        historical_positions=unflagged, historical_team_membership=unflagged,
    )
    official = official_tables(tables)
    for name, frame in official.as_mapping().items():
        if "available_at_is_estimated" in frame.columns:
            assert frame["x"].to_list() == [2, 3], name
    evidence, counts = classify_tables(official.as_mapping())
    assert evidence is EvidenceClass.OFFICIAL_PIT_FAITHFUL and counts == {}


# ------------------------------------------------------------ official training runner


def _config(root: Path, out: Path, *, evidence: str = "official") -> RunnerConfig:
    return RunnerConfig(
        data_root=root, output_dir=out, season_min=2023, season_max=2024, n_draws=60,
        mode="smoke", model_version="v1", regularization_lambda=0.01,
        max_fit_iterations=20, expect_data_manifest_sha256=None, evidence=evidence,
    )


def _mark_outcomes_estimated(wh: Warehouse) -> None:
    for table in ("player_game_stats", "team_game_stats"):
        wh.write(table, wh.read(table).with_columns(
            pl.lit(True).alias("available_at_is_estimated")
        ))


def test_runner_defaults_to_official_and_validates_evidence(tmp_path: Path) -> None:
    assert _config(tmp_path, tmp_path).evidence == "official"
    assert parse_args(["--data-root", "d", "--output-dir", "o"]).evidence == "official"
    assert parse_args(["--data-root", "d", "--output-dir", "o",
                       "--evidence", "research"]).evidence == "research"
    with pytest.raises(ConfigurationError, match="evidence"):
        _config(tmp_path, tmp_path, evidence="lenient")


def test_official_run_on_clean_data_is_historical_chronology_evidence(tmp_path: Path) -> None:
    wh = _build_two_season_warehouse(tmp_path)
    report = run(_config(wh.root, tmp_path / "out"))
    assert report["evidence_mode"] == "HISTORICAL_WALK_FORWARD"
    assert report["evidence_class"] == "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY"
    assert report["evidence_policy"]["estimated_rows_in_source"] == {}


def test_official_run_certifies_estimated_outcomes_by_event_chronology(tmp_path: Path) -> None:
    wh = _build_two_season_warehouse(tmp_path)
    _mark_outcomes_estimated(wh)
    # Estimated RECEIPT time on completed-event rows: certified by event
    # chronology (Gate 1), never LIVE_PIT evidence.
    report = run(_config(wh.root, tmp_path / "out"))
    assert report["evidence_class"] == "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY"
    assert report["evidence_class"] != "OFFICIAL_PIT_FAITHFUL"
    assert set(report["evidence_policy"]["estimated_rows_in_source"]) == {
        "player_game_stats", "team_game_stats"}


def test_research_run_uses_estimates_but_is_never_promotable(tmp_path: Path) -> None:
    wh = _build_two_season_warehouse(tmp_path)
    _mark_outcomes_estimated(wh)
    report = run(_config(wh.root, tmp_path / "out", evidence="research"))
    assert report["evidence_class"] == "RESEARCH_ONLY"
    assert report["evidence_policy"]["source_data_class"] == (
        "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY")
    assert set(report["evidence_policy"]["estimated_rows_in_source"]) == {
        "player_game_stats", "team_game_stats"}
    assert report["promotion_decision"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize(("evidence_mode", "evidence_class"), [
    (None, None),
    ("HISTORICAL_WALK_FORWARD", None),
    ("HISTORICAL_WALK_FORWARD", "RESEARCH_ONLY"),
    # Each mode accepts only its own certified class.
    ("HISTORICAL_WALK_FORWARD", "OFFICIAL_PIT_FAITHFUL"),
    ("LIVE_PIT", "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY"),
    (None, "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY"),
])
def test_remote_training_never_reports_non_official_evidence_eligible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_mode: str | None,
    evidence_class: str | None,
) -> None:
    from nflprops.calibration import phase10c3a_runner as science

    def fake_run(config: science.RunnerConfig) -> dict:
        result = {"model_version": config.model_version,
                  "promotion_decision": "ELIGIBLE_FOR_PROMOTION",
                  "registration": {"payload_sha256": "x"}}
        if evidence_class is not None:
            result["evidence_class"] = evidence_class
        if evidence_mode is not None:
            result["evidence_mode"] = evidence_mode
        return result

    monkeypatch.setattr(science, "run", fake_run)
    main = rt.resolve_science_entrypoint(rt.DEFAULT_SCIENCE_ENTRYPOINT)
    (tmp_path / "snap").mkdir()
    result = main(science_ref="a" * 40, data_manifest_sha256="b" * 64,
                  data_dir=tmp_path / "snap", n_draws=rt.PRODUCTION_N_DRAWS,
                  mode="production", promotion_evidence_eligible=True,
                  output_dir=tmp_path / "out")
    assert result["promotion_eligibility_result"]["eligible"] is False

    def official_run(config: science.RunnerConfig) -> dict:
        return {**fake_run(config), "evidence_mode": "HISTORICAL_WALK_FORWARD",
                "evidence_class": "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY"}

    monkeypatch.setattr(science, "run", official_run)
    main = rt.resolve_science_entrypoint(rt.DEFAULT_SCIENCE_ENTRYPOINT)  # binds `run` anew
    result = main(science_ref="a" * 40, data_manifest_sha256="b" * 64,
                  data_dir=tmp_path / "snap", n_draws=rt.PRODUCTION_N_DRAWS,
                  mode="production", promotion_evidence_eligible=True,
                  output_dir=tmp_path / "out2")
    assert result["promotion_eligibility_result"]["eligible"] is True
