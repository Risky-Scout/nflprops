"""PIT/data-history integrity of the two player-prop market tables.

`player_prop_snapshots` (PS1-PS6): one provider poll carries several
markets per player/prop/book -- milestone ladders and alternate lines (in
the local raw BDL receipts 878 of 1,351 live rows would have collapsed
under the old (game, player, prop, book, receipt) key; zero collide once
market_type + line_value are included). Every market and every genuine
receipt is preserved; exact re-ingest is idempotent; price is not identity.

`player_prop_openings` (PO1-PO5): one opening per logical market; the FIRST
genuinely received observation is immutable (the old key had no time
dimension and kept the LAST ingest, so a later backfill replaced history).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider

from nflprops.collection.service import collect_once
from nflprops.config import Config
from nflprops.data.warehouse import Warehouse
from nflprops.domain.enums import MarketType
from nflprops.domain.market_identity import (
    PLAYER_PROP_BOOK_KEY,
    PLAYER_PROP_MARKET_IDENTITY,
    PLAYER_PROP_SNAPSHOT_KEY,
)
from nflprops.market.consensus import latest_prop_quotes
from nflprops.paths import runtime_resource
from nflprops.pipelines.lean import append_prop_openings

KICKOFF = datetime(2026, 9, 13, 17, 0, tzinfo=UTC)
T1 = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
T2 = datetime(2026, 9, 10, 13, 0, tzinfo=UTC)
T3 = datetime(2026, 9, 10, 14, 0, tzinfo=UTC)


def _contract() -> dict:
    path = runtime_resource("contracts", "warehouse_tables.yml")
    return yaml.safe_load(Path(path).read_text())["tables"]["market"]


def test_contract_keys_are_the_code_identity() -> None:
    market = _contract()
    assert tuple(market["player_prop_snapshots"]["key"]) == PLAYER_PROP_SNAPSHOT_KEY
    assert tuple(market["player_prop_openings"]["key"]) == PLAYER_PROP_MARKET_IDENTITY
    assert market["player_prop_openings"]["write"] == "first_observation_wins"
    for key in (PLAYER_PROP_SNAPSHOT_KEY, PLAYER_PROP_MARKET_IDENTITY):
        assert not {"over_odds", "under_odds", "milestone_odds"} & set(key)  # price != identity


# ------------------------------------------------------- snapshots (collect_once)


def _provider(*props: dict) -> FakeProvider:
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    provider.seed_player("p1", first_name="Home", last_name="Player")
    provider.seed_game(
        "g1", home_team_native_id="t1", visitor_team_native_id="t2", week=1, date=KICKOFF
    )
    for spec in props:
        milestone = spec.pop("milestone_odds", None)
        prop = provider.seed_player_prop(game_native_id="g1", player_native_id="p1", **spec)
        if milestone is not None:
            provider._player_props[-1] = prop.model_copy(update={
                "market_type": MarketType.MILESTONE, "milestone_odds": milestone,
                "over_odds": None, "under_odds": None,
            })
    return provider


def _ou(line: str, at: datetime, *, vendor: str = "draftkings", over: int = -110) -> dict:
    return {"vendor": vendor, "prop_type": "receiving_yards", "line_value": Decimal(line),
            "over_odds": over, "under_odds": -110, "collector_received_at": at}


def _ms(line: str, at: datetime, odds: int, *, vendor: str = "draftkings") -> dict:
    return {"vendor": vendor, "prop_type": "receiving_yards", "line_value": Decimal(line),
            "collector_received_at": at, "milestone_odds": odds}


def _collect(warehouse: Warehouse, provider: FakeProvider, now: datetime) -> pl.DataFrame:
    collect_once(provider=provider, season=2026, week=1, warehouse=warehouse,
                 config=Config(data={}), now=now)
    return warehouse.read("player_prop_snapshots")


def _lines(frame: pl.DataFrame) -> list[tuple]:
    return sorted(
        (r["vendor"], r["market_type"], float(r["line_value"]), r["collector_received_at"])
        for r in frame.iter_rows(named=True)
    )


def test_ps1_two_lines_same_book_same_receipt_both_survive(tmp_path: Path) -> None:
    stored = _collect(Warehouse(tmp_path), _provider(_ou("55.5", T1), _ou("60.5", T1)), T1)
    assert _lines(stored) == [
        ("draftkings", "over_under", 55.5, T1), ("draftkings", "over_under", 60.5, T1)
    ]


def test_ps2_distinct_market_types_same_line_both_survive(tmp_path: Path) -> None:
    provider = _provider(_ou("60.5", T1), _ms("60.5", T1, 180), _ms("75.0", T1, 320),
                         _ms("100.0", T1, 900))
    stored = _collect(Warehouse(tmp_path), provider, T1)
    assert _lines(stored) == [
        ("draftkings", "milestone", 60.5, T1), ("draftkings", "milestone", 75.0, T1),
        ("draftkings", "milestone", 100.0, T1), ("draftkings", "over_under", 60.5, T1),
    ]


def test_ps3_exact_reingest_is_idempotent(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path)
    provider = _provider(_ou("55.5", T1), _ms("75.0", T1, 320))
    first = _collect(warehouse, provider, T1)
    again = _collect(warehouse, provider, T1)
    assert again.height == first.height == 2
    assert _lines(again) == _lines(first)


def test_ps4_later_revision_preserves_earlier_version(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path)
    _collect(warehouse, _provider(_ou("55.5", T1, over=-110)), T1)
    stored = _collect(warehouse, _provider(_ou("55.5", T2, over=-135)), T2)
    rows = sorted(stored.iter_rows(named=True), key=lambda r: r["collector_received_at"])
    assert [(r["collector_received_at"], r["over_odds"]) for r in rows] == [
        (T1, -110), (T2, -135)
    ]


def test_ps5_books_stay_separate(tmp_path: Path) -> None:
    stored = _collect(Warehouse(tmp_path),
                      _provider(_ou("55.5", T1), _ou("55.5", T1, vendor="fanduel")), T1)
    assert sorted(stored["vendor"].to_list()) == ["draftkings", "fanduel"]


def test_ps6_pit_replay_reproduces_what_was_known_at_t(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path)
    _collect(warehouse, _provider(_ou("55.5", T1, over=-110), _ms("75.0", T1, 320),
                                  _ms("100.0", T1, 900)), T1)
    # The later poll revises the main line and drops the 100+ rung.
    _collect(warehouse, _provider(_ou("57.5", T3, over=-120), _ms("75.0", T3, 300)), T3)
    stored = warehouse.read("player_prop_snapshots")

    at_t2 = latest_prop_quotes(stored, as_of=T2)
    assert _lines(at_t2) == [
        ("draftkings", "milestone", 75.0, T1), ("draftkings", "milestone", 100.0, T1),
        ("draftkings", "over_under", 55.5, T1),
    ]
    assert at_t2.filter(pl.col("market_type") == "over_under")["over_odds"].to_list() == [-110]
    # After the revision: exactly the later poll -- the withdrawn 100+ rung
    # and the superseded 55.5 line are not resurrected from the older poll.
    assert _lines(latest_prop_quotes(stored, as_of=T3)) == [
        ("draftkings", "milestone", 75.0, T3), ("draftkings", "over_under", 57.5, T3),
    ]
    # Deterministic: replay is independent of storage row order.
    shuffled = stored.sample(fraction=1.0, shuffle=True, seed=7)
    assert latest_prop_quotes(shuffled, as_of=T2).to_dicts() == at_t2.to_dicts()


# ------------------------------------------------------------- openings


def _opening(line: float, received: datetime, *, odds: int = -110,
             market_type: str = "over_under") -> dict:
    return {
        "canonical_game_id": "g1", "canonical_player_id": "p1",
        "prop_type": "receiving_yards", "vendor": "draftkings",
        "market_type": market_type, "line_value": line,
        "over_odds": odds, "under_odds": -110, "milestone_odds": None,
        "opened_at": datetime(2026, 9, 1, tzinfo=UTC), "is_opening": True,
        "available_at": received, "collector_received_at": received,
    }


def _openings(warehouse: Warehouse) -> list[tuple]:
    return sorted(
        (r["market_type"], r["line_value"], r["over_odds"], r["collector_received_at"])
        for r in warehouse.read("player_prop_openings").iter_rows(named=True)
    )


def test_po1_first_opening_observation_is_stored(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path)
    append_prop_openings(warehouse, [_opening(55.5, T1), _opening(60.5, T1)])
    assert _openings(warehouse) == [("over_under", 55.5, -110, T1),
                                    ("over_under", 60.5, -110, T1)]


def test_po2_exact_reingest_is_idempotent(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path)
    append_prop_openings(warehouse, [_opening(55.5, T1)])
    append_prop_openings(warehouse, [_opening(55.5, T1)])
    assert _openings(warehouse) == [("over_under", 55.5, -110, T1)]


def test_po3_later_observation_never_replaces_the_original(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path)
    append_prop_openings(warehouse, [_opening(55.5, T1, odds=-110)])
    # A later backfill re-observes the same market with a different price.
    append_prop_openings(warehouse, [_opening(55.5, T3, odds=-125)])
    assert _openings(warehouse) == [("over_under", 55.5, -110, T1)]
    # A genuinely new market observed later is added, not merged.
    append_prop_openings(warehouse, [_opening(80.0, T3, market_type="milestone")])
    assert _openings(warehouse) == [("milestone", 80.0, -110, T3),
                                    ("over_under", 55.5, -110, T1)]


def _known(warehouse: Warehouse, as_of: datetime) -> list[tuple]:
    frame = latest_prop_quotes(
        warehouse.read("player_prop_openings"), as_of=as_of,
        group_by=PLAYER_PROP_MARKET_IDENTITY,
    )
    return sorted((r["market_type"], r["line_value"], r["over_odds"])
                  for r in frame.iter_rows(named=True))


def test_po4_replay_before_a_later_observation_reproduces_the_original(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path)
    append_prop_openings(warehouse, [_opening(55.5, T1, odds=-110)])
    before = _known(warehouse, T2)
    append_prop_openings(warehouse, [_opening(55.5, T3, odds=-125),
                                     _opening(80.0, T3, market_type="milestone")])
    assert before == _known(warehouse, T2) == [("over_under", 55.5, -110)]


def test_po5_no_later_observation_alters_an_earlier_replay(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path)
    append_prop_openings(warehouse, [_opening(55.5, T1), _opening(70.0, T2, market_type="milestone")])
    snapshots = {t: _known(warehouse, t) for t in (T1 - timedelta(seconds=1), T1, T2, T3)}
    for later in (T3, T3 + timedelta(days=30)):
        append_prop_openings(warehouse, [_opening(55.5, later, odds=+999),
                                         _opening(70.0, later, odds=+999, market_type="milestone")])
        assert {t: _known(warehouse, t) for t in snapshots} == snapshots
    assert snapshots[T1 - timedelta(seconds=1)] == []
    # Openings are independent first observations: a market first observed
    # earlier stays known even after another market's later first receipt.
    assert snapshots[T2] == [("milestone", 70.0, -110), ("over_under", 55.5, -110)]


def test_live_selection_groups_by_book_openings_by_market() -> None:
    assert PLAYER_PROP_MARKET_IDENTITY[:4] == PLAYER_PROP_BOOK_KEY
