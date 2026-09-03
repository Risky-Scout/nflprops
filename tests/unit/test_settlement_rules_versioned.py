"""Acceptance tests for independently versioned settlement rules. SPEC §44."""

from pathlib import Path

import polars as pl
import pytest

from nflprops.market.rules import (
    SettlementRuleError,
    evaluate_actual_value,
    load_settlement_rules,
)
from nflprops.pipelines.settle import (
    settle_predictions,
)


def prediction(
    *,
    prop_type: str = "receiving_yards",
    vendor: str = "book",
    side: str = "OVER",
    line: float | None = 65.5,
    model_version: str = "model-A",
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "prediction_id": ["pred-1"],
            "game_id": ["g1"],
            "player_id": ["p1"],
            "prop_type": [prop_type],
            "vendor": [vendor],
            "side": [side],
            "line": [line],
            "american_odds": [-110],
            "model_version": [
                model_version
            ],
        }
    )


def stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_game_id": [
                "g1"
            ],
            "canonical_player_id": [
                "p1"
            ],
            "passing_attempts": [35],
            "passing_completions": [24],
            "passing_yards": [280],
            "passing_interceptions": [1],
            "rushing_attempts": [3],
            "rushing_yards": [14],
            "rushing_touchdowns": [1],
            "receptions": [6],
            "receiving_yards": [72],
            "receiving_touchdowns": [1],
            "passing_touchdowns": [2],
            "total_points": [6],
            "field_goals_made": [2],
            "long_rushing": [11],
            "long_reception": [31],
            "defensive_touchdowns": [4],
        }
    )


def test_settlement_rules_versioned() -> None:
    rules = load_settlement_rules()

    assert rules.version == "2026.1.0"

    rule = rules.rule_for(
        "receiving_yards",
        "book",
    )

    assert (
        rule.rule_id
        == "official_stat:receiving_yards"
    )

    settled = settle_predictions(
        prediction(),
        stats(),
        rules=rules,
    )

    assert settled.height == 1
    assert (
        settled[
            "settlement_rules_version"
        ][0]
        == "2026.1.0"
    )
    assert (
        settled[
            "settlement_rule_id"
        ][0]
        == "official_stat:receiving_yards"
    )
    assert (
        settled[
            "actual_value"
        ][0]
        == 72.0
    )


def test_settlement_version_is_independent_of_model_version() -> None:
    rules = load_settlement_rules()

    first = settle_predictions(
        prediction(
            model_version="model-A"
        ),
        stats(),
        rules=rules,
    )

    second = settle_predictions(
        prediction(
            model_version="totally-different-model"
        ),
        stats(),
        rules=rules,
    )

    assert (
        first[
            "settlement_rules_version"
        ][0]
        == second[
            "settlement_rules_version"
        ][0]
        == "2026.1.0"
    )

    assert (
        first[
            "settlement_rule_id"
        ][0]
        == second[
            "settlement_rule_id"
        ][0]
    )


def test_unknown_prop_rule_fails_closed() -> None:
    rules = load_settlement_rules()

    with pytest.raises(
        SettlementRuleError,
        match="no settlement rule registered",
    ):
        settle_predictions(
            prediction(
                prop_type="future_magic_prop"
            ),
            stats(),
            rules=rules,
        )


def test_anytime_td_definition_is_explicit_and_versioned() -> None:
    rules = load_settlement_rules()

    rule = rules.rule_for(
        "anytime_td",
        "book",
    )

    assert (
        rule.rule_id
        == "offensive_td:rushing+receiving"
    )

    value = evaluate_actual_value(
        rule,
        stats().row(
            0,
            named=True,
        ),
    )

    assert value == 2.0

    settled = settle_predictions(
        prediction(
            prop_type="anytime_td",
            side="HIT",
            line=None,
        ),
        stats(),
        rules=rules,
    )

    assert settled.height == 1
    assert settled["won"][0] is True
    assert (
        settled[
            "settlement_rule_id"
        ][0]
        == "offensive_td:rushing+receiving"
    )


def test_sum_rule_preserves_explicit_current_null_policy() -> None:
    rules = load_settlement_rules()

    rule = rules.rule_for(
        "rushing_receiving_yards",
        "book",
    )

    value = evaluate_actual_value(
        rule,
        {
            "rushing_yards": None,
            "receiving_yards": 70,
        },
    )

    assert value == 70.0

    assert (
        evaluate_actual_value(
            rule,
            {
                "rushing_yards": None,
                "receiving_yards": None,
            },
        )
        is None
    )


def test_rule_file_versions_must_match(
    tmp_path: Path,
) -> None:
    (
        tmp_path
        / "a.yml"
    ).write_text(
        """
version: "1"
rules:
  receiving_yards:
    default:
      rule_id: "a"
      kind: field
      fields: [receiving_yards]
""".lstrip()
    )

    (
        tmp_path
        / "b.yml"
    ).write_text(
        """
version: "2"
rules:
  rushing_yards:
    default:
      rule_id: "b"
      kind: field
      fields: [rushing_yards]
""".lstrip()
    )

    with pytest.raises(
        SettlementRuleError,
        match="conflicting versions",
    ):
        load_settlement_rules(
            tmp_path
        )


def test_prop_map_anytime_td_points_to_settlement_rule_layer() -> None:
    root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    prop_map = (
        root
        / "src"
        / "nflprops"
        / "resources"
        / "contracts"
        / "prop_map.yml"
    ).read_text()

    assert (
        "settlement_rule_ref: market/rules/anytime_td.yml"
        in prop_map
    )

    assert (
        root
        / "src"
        / "nflprops"
        / "resources"
        / "market"
        / "rules"
        / "anytime_td.yml"
    ).is_file()
