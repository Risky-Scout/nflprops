"""IMPLEMENTATION_SPEC §66: feature.available_at <= prediction.as_of."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from nflprops.backtest.leakage import (
    LeakageError,
    PredictionLineage,
    assert_no_leakage,
)


def test_future_feature_is_rejected(
    as_of: datetime,
    clean_lineage_factory: Callable[[], PredictionLineage],
) -> None:
    lineage = replace(
        clean_lineage_factory(),
        feature_available_at=(
            as_of + timedelta(seconds=1),
        ),
    )

    with pytest.raises(LeakageError):
        assert_no_leakage(lineage)
