"""§53: checkpoint offset calculation, exact boundaries, production config
validation, manual-checkpoint exclusion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nflprops.config import Config, load
from nflprops.orchestration.checkpoints import (
    DEFAULT_CHECKPOINT_OFFSETS,
    DEFAULT_OFFSET_SECONDS,
    OFFICIAL_CHECKPOINTS,
    CheckpointAction,
    CheckpointConfigError,
    CheckpointName,
    CheckpointOffsets,
    CheckpointsRuntimeConfig,
    OrchestrationConfig,
    all_scheduled_as_of,
    checkpoint_due,
    evaluate_checkpoint,
    is_past_kickoff,
    scheduled_as_of,
)

KICKOFF = datetime(2026, 9, 13, 20, 20, 0, tzinfo=UTC)


def test_scheduled_as_of_exact_offsets_for_every_official_checkpoint() -> None:
    expected = {
        CheckpointName.T48H: KICKOFF - timedelta(seconds=172_800),
        CheckpointName.T24H: KICKOFF - timedelta(seconds=86_400),
        CheckpointName.T6H: KICKOFF - timedelta(seconds=21_600),
        CheckpointName.T90M: KICKOFF - timedelta(seconds=5_400),
        CheckpointName.T30M: KICKOFF - timedelta(seconds=1_800),
    }
    all_times = all_scheduled_as_of(kickoff_at=KICKOFF, offsets=DEFAULT_CHECKPOINT_OFFSETS)
    assert all_times == expected
    for checkpoint, expected_time in expected.items():
        assert (
            scheduled_as_of(
                kickoff_at=KICKOFF, checkpoint=checkpoint, offsets=DEFAULT_CHECKPOINT_OFFSETS
            )
            == expected_time
        )


def test_default_offsets_match_documented_seconds() -> None:
    assert DEFAULT_OFFSET_SECONDS == {
        CheckpointName.T48H: 172_800,
        CheckpointName.T24H: 86_400,
        CheckpointName.T6H: 21_600,
        CheckpointName.T90M: 5_400,
        CheckpointName.T30M: 1_800,
    }


@pytest.mark.parametrize(
    "mapping",
    [
        {"T48H": 0, "T24H": 86_400, "T6H": 21_600, "T90M": 5_400, "T30M": 1_800},
        {"T48H": -100, "T24H": 86_400, "T6H": 21_600, "T90M": 5_400, "T30M": 1_800},
    ],
)
def test_from_mapping_rejects_non_positive_offsets(mapping: dict[str, int]) -> None:
    with pytest.raises(CheckpointConfigError):
        CheckpointOffsets.from_mapping(mapping)


def test_from_mapping_rejects_out_of_order_offsets() -> None:
    with pytest.raises(CheckpointConfigError):
        CheckpointOffsets.from_mapping(
            {"T48H": 100, "T24H": 200, "T6H": 50, "T90M": 20, "T30M": 10}
        )


def test_from_mapping_rejects_duplicate_offset_values() -> None:
    with pytest.raises(CheckpointConfigError):
        CheckpointOffsets.from_mapping(
            {"T48H": 100, "T24H": 100, "T6H": 50, "T90M": 20, "T30M": 10}
        )


def test_from_mapping_rejects_unknown_checkpoint_name() -> None:
    with pytest.raises(CheckpointConfigError, match="unknown checkpoint"):
        CheckpointOffsets.from_mapping(
            {
                "T48H": 172_800,
                "T24H": 86_400,
                "T6H": 21_600,
                "T90M": 5_400,
                "T30M": 1_800,
                "MANUAL": 900,
            }
        )


def test_from_config_rejects_bad_offsets_table() -> None:
    cfg = Config(
        data={
            "checkpoints": {
                "offset_seconds": {
                    "T48H": 100,
                    "T24H": 200,
                    "T6H": 50,
                    "T90M": 20,
                    "T30M": 10,
                }
            }
        }
    )
    with pytest.raises(CheckpointConfigError):
        CheckpointOffsets.from_config(cfg)


def test_from_config_falls_back_to_defaults_when_absent() -> None:
    cfg = Config(data={})
    offsets = CheckpointOffsets.from_config(cfg)
    assert offsets == DEFAULT_CHECKPOINT_OFFSETS


def test_production_base_toml_checkpoint_config_matches_documented_defaults() -> None:
    cfg = load()
    offsets = CheckpointOffsets.from_config(cfg)
    assert offsets.get(CheckpointName.T48H) == 172_800
    assert offsets.get(CheckpointName.T24H) == 86_400
    assert offsets.get(CheckpointName.T6H) == 21_600
    assert offsets.get(CheckpointName.T90M) == 5_400
    assert offsets.get(CheckpointName.T30M) == 1_800

    checkpoints_cfg = CheckpointsRuntimeConfig.from_config(cfg)
    assert checkpoints_cfg.enabled is True
    assert checkpoints_cfg.catch_up_before_kickoff is True

    orchestration_cfg = OrchestrationConfig.from_config(cfg)
    assert orchestration_cfg.enabled is True
    assert orchestration_cfg.prefect_work_pool == "nflprops-production"
    assert orchestration_cfg.dispatcher_tick_seconds == 60


def test_manual_is_excluded_from_official_checkpoints() -> None:
    assert CheckpointName.MANUAL not in OFFICIAL_CHECKPOINTS
    assert len(OFFICIAL_CHECKPOINTS) == 5


def test_is_past_kickoff_boundary() -> None:
    assert is_past_kickoff(kickoff_at=KICKOFF, now=KICKOFF) is True
    assert is_past_kickoff(kickoff_at=KICKOFF, now=KICKOFF - timedelta(seconds=1)) is False
    assert is_past_kickoff(kickoff_at=KICKOFF, now=KICKOFF + timedelta(seconds=1)) is True


def test_checkpoint_due_pure_time_window() -> None:
    sched = KICKOFF - timedelta(seconds=1_800)
    assert checkpoint_due(scheduled_as_of_time=sched, kickoff_at=KICKOFF, now=sched) is True
    assert (
        checkpoint_due(
            scheduled_as_of_time=sched, kickoff_at=KICKOFF, now=sched - timedelta(seconds=1)
        )
        is False
    )
    assert checkpoint_due(scheduled_as_of_time=sched, kickoff_at=KICKOFF, now=KICKOFF) is False


def test_evaluate_checkpoint_exact_boundaries() -> None:
    sched = KICKOFF - timedelta(seconds=1_800)

    assert (
        evaluate_checkpoint(scheduled_as_of_time=sched, kickoff_at=KICKOFF, now=sched - timedelta(seconds=1))
        is CheckpointAction.NOT_DUE
    )
    assert (
        evaluate_checkpoint(scheduled_as_of_time=sched, kickoff_at=KICKOFF, now=sched)
        is CheckpointAction.RUN
    )
    assert (
        evaluate_checkpoint(
            scheduled_as_of_time=sched, kickoff_at=KICKOFF, now=KICKOFF - timedelta(seconds=1)
        )
        is CheckpointAction.RUN
    )
    assert (
        evaluate_checkpoint(scheduled_as_of_time=sched, kickoff_at=KICKOFF, now=KICKOFF)
        is CheckpointAction.MISSED
    )


def test_evaluate_checkpoint_catch_up_disabled_marks_late_discovery_missed() -> None:
    sched = KICKOFF - timedelta(seconds=1_800)
    late_now = sched + timedelta(seconds=600)  # later than one dispatcher tick

    assert (
        evaluate_checkpoint(
            scheduled_as_of_time=sched,
            kickoff_at=KICKOFF,
            now=late_now,
            catch_up_before_kickoff=True,
            dispatcher_tick_seconds=60,
        )
        is CheckpointAction.RUN
    )
    assert (
        evaluate_checkpoint(
            scheduled_as_of_time=sched,
            kickoff_at=KICKOFF,
            now=late_now,
            catch_up_before_kickoff=False,
            dispatcher_tick_seconds=60,
        )
        is CheckpointAction.MISSED
    )


def test_evaluate_checkpoint_on_time_within_one_tick_always_runs_even_if_catch_up_disabled() -> None:
    sched = KICKOFF - timedelta(seconds=1_800)
    slightly_late = sched + timedelta(seconds=10)
    assert (
        evaluate_checkpoint(
            scheduled_as_of_time=sched,
            kickoff_at=KICKOFF,
            now=slightly_late,
            catch_up_before_kickoff=False,
            dispatcher_tick_seconds=60,
        )
        is CheckpointAction.RUN
    )
