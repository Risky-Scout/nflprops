"""Canonical IDs must be stable forever. SPEC §9."""

from nflprops.domain.ids import (
    EntityKind,
    canonical_id,
    canonical_player_id,
    canonical_team_id,
)


def test_same_inputs_same_id():
    a = canonical_player_id("balldontlie", 490)
    b = canonical_player_id("balldontlie", 490)
    assert a == b


def test_int_and_str_provider_id_agree():
    assert canonical_player_id("balldontlie", 490) == canonical_player_id(
        "balldontlie", "490"
    )


def test_different_kinds_do_not_collide():
    assert canonical_player_id("balldontlie", 7) != canonical_team_id(
        "balldontlie", 7
    )


def test_different_providers_do_not_collide():
    assert canonical_player_id("balldontlie", 490) != canonical_player_id(
        "otherapi", 490
    )


def test_known_golden_value():
    """Pinned so an accidental namespace change is caught immediately.

    If this test fails, someone changed NAMESPACE or the key format, and every
    canonical ID ever minted is now unreproducible. Do not "update the expected
    value" — find out why it changed.
    """
    assert (
        canonical_id(EntityKind.PLAYER, "balldontlie", 490)
        == "050f1139-8249-5887-bc43-6473aa631f0e"
    )
