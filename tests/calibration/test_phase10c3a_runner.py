"""PHASE 10C3A: lightweight smoke tests for the version-controlled runner/CLI
(`nflprops.calibration.phase10c3a_runner`) against a small synthetic
2-season warehouse -- never real historical data, never `n_draws=20000`.
This proves the CLI/report machinery (config lock, folds, skip accounting,
weight-health gate, coherence/reproducibility checks, non-promoting
registration, JSON report, exit codes) end to end without any heavy
compute.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_historical_runner import _rb_row, _team_row, _wr_row

from nflprops.calibration.phase10c3a_runner import (
    ConfigurationError,
    RunnerConfig,
    build_season_boundary_folds,
    main,
    parse_args,
    run,
)
from nflprops.data.warehouse import Warehouse

HOME_TEAM_ID = "p10c3a-cli:team:home"
AWAY_TEAM_ID = "p10c3a-cli:team:away"
HOME_WR_ID = "p10c3a-cli:player:home-wr"
HOME_RB_ID = "p10c3a-cli:player:home-rb"
AWAY_WR_ID = "p10c3a-cli:player:away-wr"

G1 = "p10c3a-cli:game:s2023-1"
G2 = "p10c3a-cli:game:s2023-2"
G3 = "p10c3a-cli:game:s2024-1"

G1_KICKOFF = datetime(2023, 9, 10, 17, 0, tzinfo=UTC)
G2_KICKOFF = datetime(2023, 9, 17, 17, 0, tzinfo=UTC)
G3_KICKOFF = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)


def _game_row(game_id: str, season: int, week: int, kickoff: datetime) -> dict:
    return {
        "canonical_game_id": game_id,
        "available_at": kickoff - timedelta(days=1),
        "date": kickoff,
        "season": season,
        "week": week,
        "status_state": "final",
        "postseason": False,
        "home_canonical_team_id": HOME_TEAM_ID,
        "visitor_canonical_team_id": AWAY_TEAM_ID,
    }


def _odds_row(game_id: str, kickoff: datetime) -> dict:
    return {
        "canonical_game_id": game_id,
        "vendor": "fakebook",
        "spread_home_value": -2.0,
        "total_value": 45.0,
        "available_at": kickoff - timedelta(days=1),
        "collector_received_at": kickoff - timedelta(days=1),
    }


def _build_two_season_warehouse(root: Path) -> Warehouse:
    warehouse = Warehouse(root / "wh")

    games = [
        _game_row(G1, 2023, 1, G1_KICKOFF),
        _game_row(G2, 2023, 2, G2_KICKOFF),
        _game_row(G3, 2024, 1, G3_KICKOFF),
    ]
    warehouse.write("games", pl.DataFrame(games))

    team_rows = []
    player_rows = []
    odds_rows = []
    for game_id, kickoff in ((G1, G1_KICKOFF), (G2, G2_KICKOFF), (G3, G3_KICKOFF)):
        available_at = kickoff + timedelta(hours=12)
        team_rows += [
            _team_row(game_id=game_id, team_id=HOME_TEAM_ID, available_at=available_at),
            _team_row(game_id=game_id, team_id=AWAY_TEAM_ID, available_at=available_at),
        ]
        player_rows += [
            _wr_row(game_id=game_id, team_id=HOME_TEAM_ID, player_id=HOME_WR_ID, available_at=available_at),
            _rb_row(game_id=game_id, team_id=HOME_TEAM_ID, player_id=HOME_RB_ID, available_at=available_at),
            _wr_row(game_id=game_id, team_id=AWAY_TEAM_ID, player_id=AWAY_WR_ID, available_at=available_at),
        ]
        odds_rows.append(_odds_row(game_id, kickoff))

    warehouse.write("team_game_stats", pl.DataFrame(team_rows))
    warehouse.write("player_game_stats", pl.DataFrame(player_rows))
    warehouse.write("game_odds_snapshots", pl.DataFrame(odds_rows))
    warehouse.write(
        "players",
        pl.DataFrame(
            [
                {"canonical_player_id": HOME_WR_ID, "position_group": "WR"},
                {"canonical_player_id": HOME_RB_ID, "position_group": "RB"},
                {"canonical_player_id": AWAY_WR_ID, "position_group": "WR"},
            ]
        ),
    )
    return warehouse


# ------------------------------------------------------------------ config lock


def test_production_mode_requires_exactly_20000_draws(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        RunnerConfig(
            data_root=tmp_path, output_dir=tmp_path, season_min=2022, season_max=2025,
            n_draws=2000, mode="production", model_version="v1",
            regularization_lambda=0.01, max_fit_iterations=50, expect_data_manifest_sha256=None,
        )


def test_smoke_mode_permits_reduced_draws(tmp_path: Path) -> None:
    config = RunnerConfig(
        data_root=tmp_path, output_dir=tmp_path, season_min=2022, season_max=2025,
        n_draws=50, mode="smoke", model_version="v1",
        regularization_lambda=0.01, max_fit_iterations=50, expect_data_manifest_sha256=None,
    )
    assert config.n_draws == 50


def test_cli_parsing_defaults_to_production_20000(tmp_path: Path) -> None:
    config = parse_args(["--data-root", str(tmp_path), "--output-dir", str(tmp_path), "--mode", "smoke", "--n-draws", "40"])
    assert config.n_draws == 40
    assert config.mode == "smoke"

    with pytest.raises(SystemExit):
        parse_args(["--data-root", str(tmp_path), "--output-dir", str(tmp_path), "--mode", "bogus"])


# ---------------------------------------------------------- fold construction


def test_build_season_boundary_folds_requires_two_seasons() -> None:
    from nflprops.calibration.challenger import LabeledGame, PropLabel
    from nflprops.domain.enums import PropType

    game = LabeledGame(
        game_id="only-one",
        simulation=None,  # not used by build_season_boundary_folds
        as_of=datetime(2024, 9, 1, tzinfo=UTC),
        outcome_available_at=datetime(2024, 9, 2, tzinfo=UTC),
        injury_data_available=False,
        labels=(PropLabel("p1", PropType.RECEIVING_YARDS, 50.0),),
    )
    from nflprops.calibration.phase10c3a_runner import InsufficientDataError

    with pytest.raises(InsufficientDataError):
        build_season_boundary_folds((game,), {"only-one": 2024})


# --------------------------------------------------------------- full smoke run


def test_full_smoke_run_produces_report_and_exit_zero(tmp_path: Path) -> None:
    warehouse = _build_two_season_warehouse(tmp_path)
    output_dir = tmp_path / "out"

    config = RunnerConfig(
        data_root=warehouse.root,
        output_dir=output_dir,
        season_min=2023,
        season_max=2024,
        n_draws=60,
        mode="smoke",
        model_version="smoke-v1",
        regularization_lambda=0.01,
        max_fit_iterations=20,
        expect_data_manifest_sha256=None,
    )
    report = run(config)

    assert report["promotion_decision"] == "INSUFFICIENT_EVIDENCE"
    assert report["mode"] == "smoke"
    assert report["n_draws"] == 60
    assert report["total_final_games"] == 3
    assert len(report["folds"]) == 1
    assert report["registration"]["champion_pointer_changed"] is False
    assert set(report["skip_accounting_by_season"].keys()) == {"2023", "2024"}

    for fold in report["folds"]:
        assert fold["weight_health"]["all_strictly_positive"] is True
        assert fold["weight_health"]["all_normalized_within_1e12"] is True

    assert report["real_game_coherence_check"]["passed"] is True
    assert report["reproducibility_check"]["passed"] is True

    # report is genuinely JSON-serializable (already written to disk by run())
    json.dumps(report)


def test_main_writes_json_report_file_and_returns_zero(tmp_path: Path) -> None:
    warehouse = _build_two_season_warehouse(tmp_path)
    output_dir = tmp_path / "cli-out"

    rc = main(
        [
            "--data-root", str(warehouse.root),
            "--output-dir", str(output_dir),
            "--season-min", "2023",
            "--season-max", "2024",
            "--n-draws", "60",
            "--mode", "smoke",
        ]
    )
    assert rc == 0
    report_path = output_dir / "phase10c3a_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["promotion_decision"] == "INSUFFICIENT_EVIDENCE"


def test_main_rejects_production_mode_with_reduced_draws(tmp_path: Path) -> None:
    rc = main(
        [
            "--data-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--n-draws", "500",
            "--mode", "production",
        ]
    )
    assert rc == 2
