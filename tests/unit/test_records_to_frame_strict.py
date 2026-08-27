from datetime import datetime

import polars as pl
import pytest

from nflprops.data.warehouse import records_to_frame


def test_records_to_frame_uses_full_schema_inference_for_sparse_rows():
    rows = [{"id": i, "rare_value": None} for i in range(1500)]
    rows[-1]["rare_value"] = 1

    frame = records_to_frame(rows)

    assert frame.height == 1500
    assert frame.schema["rare_value"] == pl.Int64
    assert frame[-1, "rare_value"] == 1


def test_records_to_frame_accepts_sparse_same_type_values():
    rows = [
        {"id": 1, "value": 10},
        {"id": 2},
        {"id": 3, "value": None},
    ]

    frame = records_to_frame(rows)

    assert frame.to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": None},
        {"id": 3, "value": None},
    ]


def test_records_to_frame_does_not_hide_incompatible_datetime_string_mix():
    rows = [
        {"value": datetime(2025, 1, 1)},
        {"value": "2025-01-02"},
    ]

    with pytest.raises(pl.exceptions.ComputeError):
        records_to_frame(rows)
