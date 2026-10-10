"""Gate 1 science x current main: the seams the integration joined.

* Versioned outcomes (BLOCK 4 `as_known_at`) under the historical
  walk-forward clock: one version per outcome, chronology certification
  still required, legacy single-version frames unchanged.
* Phase 10C3A fails closed on absent or incompatible week-versioned
  identity tables (`historical_player_positions`,
  `historical_team_membership`).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nflprops.calibration.phase10c3a_runner import (
    ConfigurationError,
    _require_compatible_identity_tables,
)
from nflprops.data.outcome_versions import as_known_at
from nflprops.features.historical_evidence import (
    EVENT_CHRONOLOGY_COL,
    EVIDENCE_CLASS_COL,
    HistoricalChronologyError,
)
from nflprops.features.historical_positions import (
    HISTORICAL_POSITIONS_TABLE,
    NFLVERSE_WEEKLY_ROSTER_SOURCES,
    RESOLUTION_VERSION,
)
from nflprops.features.team_membership import (
    HISTORICAL_TEAM_MEMBERSHIP_TABLE,
    MEMBERSHIP_VERSION,
)

CERTIFIED = "CERTIFIED_HISTORICAL_EVENT_CHRONOLOGY"
AS_OF = datetime(2024, 10, 1, tzinfo=UTC)
PINNED_SHA = str(NFLVERSE_WEEKLY_ROSTER_SOURCES[2024]["sha256"])


def _versions(**overrides: object) -> pl.DataFrame:
    """Two versions of one outcome (original + later correction) and one
    single-version outcome, certified by event chronology before AS_OF."""
    event = AS_OF - timedelta(days=3)
    frame = pl.DataFrame({
        "canonical_game_id": ["g1", "g1", "g2"],
        "canonical_player_id": ["p1", "p1", "p1"],
        "receiving_yards": [50, 57, 30],
        "available_at": [AS_OF - timedelta(days=2), AS_OF - timedelta(days=1),
                         AS_OF - timedelta(days=9)],
        "available_at_is_estimated": [True, True, True],
        "first_seen_at": [AS_OF - timedelta(days=2), AS_OF - timedelta(days=1),
                          AS_OF - timedelta(days=9)],
        EVENT_CHRONOLOGY_COL: [event, event, event - timedelta(days=7)],
        EVIDENCE_CLASS_COL: [CERTIFIED] * 3,
    })
    return frame.with_columns(**{k: pl.lit(v) for k, v in overrides.items()})


def test_walk_forward_clock_selects_one_version_per_outcome() -> None:
    out = as_known_at(_versions(), "player_game_stats", AS_OF, strict=True,
                      time_col=EVENT_CHRONOLOGY_COL)
    assert out.sort("canonical_game_id")["receiving_yards"].to_list() == [57, 30]


def test_walk_forward_clock_never_admits_a_future_event() -> None:
    out = as_known_at(_versions(), "player_game_stats", AS_OF - timedelta(days=5),
                      strict=True, time_col=EVENT_CHRONOLOGY_COL)
    assert out["canonical_game_id"].to_list() == ["g2"]


def test_walk_forward_clock_still_requires_chronology_certification() -> None:
    with pytest.raises(HistoricalChronologyError):
        as_known_at(_versions(**{EVIDENCE_CLASS_COL: "RESEARCH_ONLY"}), "player_game_stats",
                    AS_OF, strict=True, time_col=EVENT_CHRONOLOGY_COL)


def test_live_clock_is_unchanged_and_drops_estimated_rows() -> None:
    assert as_known_at(_versions(), "player_game_stats", AS_OF, strict=True).is_empty()


def test_single_version_frame_is_returned_unchanged() -> None:
    frame = _versions().filter(pl.col("canonical_game_id") == "g2")
    out = as_known_at(frame, "player_game_stats", AS_OF, strict=True,
                      time_col=EVENT_CHRONOLOGY_COL)
    assert out.equals(frame)


# ----------------------------------------------- 10C3A identity-table gate


def _identity_tables(**overrides: pl.DataFrame) -> dict[str, pl.DataFrame]:
    positions = pl.DataFrame({
        "canonical_player_id": ["p1"], "season": [2024], "week": [1], "team": ["HOME"],
        "position_group": ["QB"], "conflict_status": ["NONE"],
        "resolution_version": [RESOLUTION_VERSION], "source_sha256": [PINNED_SHA],
    })
    membership = pl.DataFrame({
        "canonical_player_id": ["p1"], "season": [2024], "week": [1], "team": ["HOME"],
        "canonical_team_id": ["t1"], "is_member": [True],
        "membership_version": [MEMBERSHIP_VERSION], "source_sha256": [PINNED_SHA],
    })
    return {HISTORICAL_POSITIONS_TABLE: positions,
            HISTORICAL_TEAM_MEMBERSHIP_TABLE: membership, **overrides}


def test_pinned_identity_tables_pass_in_production() -> None:
    _require_compatible_identity_tables(_identity_tables(), production=True)


@pytest.mark.parametrize("table", [HISTORICAL_POSITIONS_TABLE, HISTORICAL_TEAM_MEMBERSHIP_TABLE])
def test_empty_or_schema_incompatible_identity_table_fails_closed(table: str) -> None:
    good = _identity_tables()
    for bad in (good[table].clear(), good[table].drop("canonical_player_id")):
        for production in (True, False):
            with pytest.raises(ConfigurationError, match=table):
                _require_compatible_identity_tables({**good, table: bad},
                                                    production=production)


@pytest.mark.parametrize("table", [HISTORICAL_POSITIONS_TABLE, HISTORICAL_TEAM_MEMBERSHIP_TABLE])
@pytest.mark.parametrize("tamper", ["unpinned_source", "no_source", "old_version", "no_version"])
def test_production_requires_pinned_sources_and_this_build_version(
    table: str, tamper: str
) -> None:
    good = _identity_tables()
    frame = good[table]
    version_col = ("resolution_version" if table == HISTORICAL_POSITIONS_TABLE
                   else "membership_version")
    bad = {
        "unpinned_source": frame.with_columns(source_sha256=pl.lit("0" * 64)),
        "no_source": frame.drop("source_sha256"),
        "old_version": frame.with_columns(pl.lit("v0").alias(version_col)),
        "no_version": frame.drop(version_col),
    }[tamper]
    with pytest.raises(ConfigurationError, match="pinned nflverse"):
        _require_compatible_identity_tables({**good, table: bad}, production=True)
    # Smoke runs (never promotable) may use synthetic identity tables.
    _require_compatible_identity_tables({**good, table: bad}, production=False)


# ------------------------------------- live calibration gate (remote checkpoint)


@pytest.mark.parametrize(("profile", "science"), [
    ("STRUCTURAL_CORE", "structural-core.v2-qb-roster-membership"),
    ("LIVE_ENHANCED", "live-enhanced.v2-qb-roster-membership"),
])
def test_remote_checkpoint_resolves_only_its_profiles_champion(
    tmp_path, monkeypatch: pytest.MonkeyPatch, profile: str, science: str
) -> None:
    """Main's remote-checkpoint calibration gate predates Gate 1: it must
    resolve a champion by the run's model profile and the profile-bound
    base model version, never by the bare model version."""
    from types import SimpleNamespace

    from nflprops.calibration import registry
    from nflprops.data.warehouse import Warehouse
    from nflprops.domain.model_profile import ModelProfile
    from nflprops.platform.remote_checkpoint import (
        CALIBRATION_NO_APPROVED_CHAMPION,
        evaluate_calibration_gate,
    )

    warehouse = Warehouse(tmp_path / "wh")
    warehouse.write("calibration_champions", pl.DataFrame({"x": [1]}))
    calls: list[dict] = []

    def resolve(_backend, **kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(registry, "resolve_calibration_champion", resolve)
    run = SimpleNamespace(checkpoint_name="T90M", model_version="2026.1.0")
    gate = evaluate_calibration_gate(warehouse, run, model_profile=ModelProfile(profile))
    assert gate["status"] == CALIBRATION_NO_APPROVED_CHAMPION and gate["approved"] is False
    assert [c["checkpoint_scope"] for c in calls] == ["T90M", "ALL_PREGAME_CHECKPOINTS"]
    for call in calls:
        assert call["model_profile"] == profile
        assert call["base_model_version"] == f"2026.1.0+{science}"
