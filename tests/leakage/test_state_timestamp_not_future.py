"""IMPLEMENTATION_SPEC §66: state timestamp <= prediction timestamp."""

from dataclasses import replace
from datetime import timedelta

import pytest
from tests.leakage.test_all_leakage_rules import AS_OF, clean_lineage

from nflprops.backtest.leakage import LeakageError, assert_no_leakage


def test_future_state_is_rejected() -> None:
    lineage = replace(
        clean_lineage(),
        state_as_of=AS_OF + timedelta(seconds=1),
    )

    with pytest.raises(LeakageError):
        assert_no_leakage(lineage)
