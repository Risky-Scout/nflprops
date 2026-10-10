"""STEP 2D: quote-time / point-in-time market integrity certification.

Policy: a sportsbook quote is eligible for a prediction at ``T`` only if
this system genuinely RECEIVED that exact quote at or before ``T``
(``collector_received_at <= T``), with NO post-cutoff tolerance. Provider
timestamps (``opened_at``, ``provider_updated_at``) never backdate it. Per
sportsbook / player / prop the LATEST eligible receipt is used. Closing
quotes are retrospective CLV data only. Player-prop markets never enter the
football simulation.

Each test is one adversarial CASE from the Step-2D certification brief.
CASES 12/13 capture the exact simulator input and abort before any draw:
no simulation runs here.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestration"))

from _fixtures import TARGET_GAME_ID, build_pit_fixture_warehouse

from nflprops.domain.model_profile import ModelProfile
from nflprops.market.closing import select_closing_prop_quotes
from nflprops.market.consensus import latest_prop_quotes
from nflprops.market.timing import (
    MarketTimingError,
    quote_knowledge_time,
    quote_time_source,
)
from nflprops.pipelines import pregame

T = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
MIN = timedelta(minutes=1)


def _quote(
    received: datetime,
    *,
    vendor: str = "draftkings",
    line: float = 65.5,
    over: int = -110,
    under: int = -110,
    opened: datetime | None = None,
    provider_updated: datetime | None = None,
    player: str = "p1",
) -> dict:
    return {
        "canonical_game_id": "g1",
        "canonical_player_id": player,
        "prop_type": "receiving_yards",
        "vendor": vendor,
        "line_value": line,
        "market_type": "over_under",
        "over_odds": over,
        "under_odds": under,
        "milestone_odds": None,
        # The mapper copies receipt into available_at for live rows; a
        # pre-Step-2D backfill copied opened_at instead. Model the worst case.
        "available_at": opened or received,
        "opened_at": opened,
        "provider_updated_at": provider_updated,
        "collector_received_at": received,
    }


def _frame(*quotes: dict) -> pl.DataFrame:
    return pl.DataFrame(list(quotes))


def _select(frame: pl.DataFrame, as_of: datetime = T) -> pl.DataFrame:
    return latest_prop_quotes(frame, as_of=as_of, game_id="g1")


def _one(frame: pl.DataFrame) -> dict:
    assert frame.height == 1, frame
    return frame.row(0, named=True)


# ------------------------------------------------------------ selection


def test_case_01_quote_received_before_t_is_eligible() -> None:
    row = _one(_select(_frame(_quote(T - 10 * MIN))))
    assert row["collector_received_at"] == T - 10 * MIN
    assert quote_knowledge_time(row, market_mode="live") <= T


@pytest.mark.parametrize("market_mode", ["live", "opening"])
def test_case_02_opened_before_t_but_received_after_t_is_not_eligible(
    market_mode: str,
) -> None:
    quote = _quote(T + MIN, opened=T - 60 * MIN)
    assert _select(_frame(quote)).height == 0
    assert quote_knowledge_time(quote, market_mode=market_mode) == T + MIN
    assert quote_time_source(market_mode) == "collector_received_at"


def test_case_03_old_pre_t_quote_beats_newer_post_t_quote() -> None:
    row = _one(_select(_frame(_quote(T - 10 * MIN, line=60.5), _quote(T + MIN, line=70.5))))
    assert row["line_value"] == 60.5


def test_case_04_latest_receipt_at_or_before_t_wins_among_revisions() -> None:
    frame = _frame(
        _quote(T - 30 * MIN, line=61.5),
        _quote(T, line=63.5),  # received exactly at T: eligible (<=)
        _quote(T - 10 * MIN, line=62.5),
    )
    assert _one(_select(frame))["line_value"] == 63.5
    # ... and with no tolerance a receipt one microsecond late is excluded.
    late = _frame(_quote(T - 10 * MIN, line=62.5),
                  _quote(T + timedelta(microseconds=1), line=64.5))
    assert _one(_select(late))["line_value"] == 62.5


def test_case_05_book_specific_availability() -> None:
    frame = _frame(_quote(T - 5 * MIN, vendor="draftkings"), _quote(T + 5 * MIN, vendor="fanduel"))
    selected = _select(frame)
    assert selected["vendor"].to_list() == ["draftkings"]


def test_case_06_close_is_retrospective_clv_data_only() -> None:
    kickoff = T + 60 * MIN
    pre = _quote(T - 10 * MIN, line=60.5)
    close = _quote(kickoff - 2 * MIN, line=66.5, over=-130, under=110)
    frame = _frame(pre, close)
    # Prediction-time selection at T never sees the close ...
    assert _one(_select(frame))["line_value"] == 60.5
    # ... while retrospective closing selection (SPEC §59) does.
    closing = select_closing_prop_quotes(
        frame,
        pl.DataFrame({"canonical_game_id": ["g1"], "kickoff_at": [kickoff]}),
        close_buffer_seconds=0,
    )
    assert closing["closing_line"].to_list() == [66.5]


def test_case_07_provider_timestamps_never_backdate_availability() -> None:
    quote = _quote(T + 3 * MIN, opened=T - 2 * 60 * MIN, provider_updated=T - 30 * MIN)
    for mode in ("live", "opening"):
        assert quote_knowledge_time(quote, market_mode=mode) == T + 3 * MIN
    assert _select(_frame(quote)).height == 0
    # Fail closed when there is no genuine receipt at all.
    with pytest.raises(MarketTimingError, match="collector_received_at"):
        quote_knowledge_time({**quote, "collector_received_at": None}, market_mode="opening")


def test_case_08_later_revision_never_rewrites_what_was_known_at_t() -> None:
    original = _quote(T - 20 * MIN, line=58.5, over=-115, under=-105)
    revision = _quote(T + 20 * MIN, line=58.5, over=+105, under=-125)
    frame = _frame(original, revision)
    known_at_t = _one(_select(frame))
    assert (known_at_t["over_odds"], known_at_t["under_odds"]) == (-115, -105)
    assert _one(_select(frame, T + 30 * MIN))["over_odds"] == 105


def test_case_09_future_better_price_is_never_used() -> None:
    frame = _frame(_quote(T - 15 * MIN, over=-140, under=120),
                   _quote(T + MIN, over=+150, under=-170))
    assert _one(_select(frame))["over_odds"] == -140


def test_case_10_temporally_closer_future_quote_is_never_used() -> None:
    frame = _frame(_quote(T - 3 * 60 * MIN, line=55.5), _quote(T + timedelta(seconds=1), line=59.5))
    assert _one(_select(frame))["line_value"] == 55.5


def test_case_11_no_pit_valid_quote_means_book_unavailable() -> None:
    frame = _frame(_quote(T - 5 * MIN, vendor="draftkings"),
                   _quote(T + 5 * MIN, vendor="fanduel"),
                   _quote(T - 5 * MIN, vendor="draftkings", player="p2"))
    selected = _select(frame)
    fanduel = selected.filter(pl.col("vendor") == "fanduel")
    assert fanduel.height == 0  # never borrowed from another book or the future
    assert sorted(selected["canonical_player_id"].to_list()) == ["p1", "p2"]
    assert set(selected["vendor"].to_list()) == {"draftkings"}


def test_case_14_absent_close_is_missing_not_estimated() -> None:
    kickoff = T + 60 * MIN
    # The only quote arrives inside the close buffer -> no close exists.
    frame = _frame(_quote(kickoff - MIN))
    closing = select_closing_prop_quotes(
        frame,
        pl.DataFrame({"canonical_game_id": ["g1"], "kickoff_at": [kickoff]}),
        close_buffer_seconds=300,
    )
    assert closing.height == 0
    # Missing-close CLV semantics: tests/unit/test_clv.py::
    # test_missing_close_remains_explicitly_missing.


def test_case_15_replay_of_identical_pit_state_is_identical() -> None:
    quotes = [
        _quote(T - 30 * MIN, line=61.5), _quote(T - 10 * MIN, line=62.5),
        _quote(T + MIN, line=63.5), _quote(T - 5 * MIN, vendor="fanduel"),
        _quote(T - 7 * MIN, player="p2", vendor="fanduel"),
    ]
    baseline = _select(_frame(*quotes)).sort(["canonical_player_id", "vendor"])
    for permutation in (quotes[::-1], quotes[2:] + quotes[:2], quotes[1::2] + quotes[::2]):
        replay = _select(_frame(*permutation)).sort(["canonical_player_id", "vendor"])
        assert replay.to_dicts() == baseline.to_dicts()


# ------------------------------- market-independence of the football model


KICKOFF = datetime(2025, 9, 15, 17, 0, tzinfo=UTC)
AS_OF = KICKOFF - timedelta(minutes=30)


class _CapturedInputError(Exception):
    pass


def _simulator_input(warehouse, monkeypatch: pytest.MonkeyPatch, market_mode: str = "live"):
    """The exact `GameSimulationInput` the football simulator would get --
    captured and aborted before any draw (no simulation runs)."""
    captured = {}

    def _capture(sim_input, cfg):
        captured["input"] = sim_input
        raise _CapturedInputError

    monkeypatch.setattr(pregame, "simulate_game", _capture)
    with pytest.raises(_CapturedInputError):
        pregame.compute_game_prediction(
            warehouse, season=2025, week=2, game_id=TARGET_GAME_ID, as_of=AS_OF,
            n_draws=200, market_mode=market_mode,
            model_profile=ModelProfile.LIVE_ENHANCED,
        )
    return captured["input"]


def _fixture(tmp_path: Path):
    return build_pit_fixture_warehouse(
        tmp_path, kickoff_at=KICKOFF,
        quote_visible_at=AS_OF - timedelta(seconds=1),
        quote_hidden_at=AS_OF + timedelta(seconds=1),
    )


def test_case_12_removing_all_prop_market_data_leaves_the_football_input_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _fixture(tmp_path)
    props = warehouse.read("player_prop_snapshots")
    assert props.height > 0
    with_props = _simulator_input(warehouse, monkeypatch)
    warehouse.write("player_prop_snapshots", props.head(0))
    without_props = _simulator_input(warehouse, monkeypatch)
    assert with_props == without_props


def test_case_13_changing_prop_lines_and_prices_leaves_the_football_input_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _fixture(tmp_path)
    baseline = _simulator_input(warehouse, monkeypatch)
    props = warehouse.read("player_prop_snapshots")
    warehouse.write("player_prop_snapshots", props.with_columns(
        (pl.col("line_value").cast(pl.Float64) + 25.0).alias("line_value"),
        pl.lit(450).alias("over_odds"),
        pl.lit(-900).alias("under_odds"),
    ))
    assert _simulator_input(warehouse, monkeypatch) == baseline


def test_simulator_boundary_takes_no_player_prop_input() -> None:
    import inspect

    params = set(inspect.signature(pregame.simulate_game_for_prediction).parameters)
    assert not {p for p in params if "prop" in p or "quote" in p}
