"""IMPLEMENTATION_SPEC §66: feature.available_at <= prediction.as_of."""

from dataclasses import replace
from datetime import timedelta

import pytest
from tests.leakage.test_all_leakage_rules import AS_OF, clean_lineage

from nflprops.backtest.leakage import LeakageError, assert_no_leakage


def test_future_feature_is_rejected() -> None:
    lineage = replace(
        clean_lineage(),
        feature_available_at=(AS_OF + timedelta(seconds=1),),
    )

    with pytest.raises(LeakageError):
        assert_no_leakage(lineage)
