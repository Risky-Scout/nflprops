"""Canonical entity identity.

SPEC: docs/IMPLEMENTATION_SPEC.md §9
PHASE: 0

Rule: a canonical ID is minted ONCE from the tuple
    (entity_kind, first_seen_provider, first_seen_provider_id)
and is never regenerated. Cross-provider entity resolution writes additional
crosswalk rows; it never rewrites a canonical ID.

This is what makes replacing BALLDONTLIE a crosswalk change instead of a migration.
"""

from __future__ import annotations

import uuid
from enum import Enum

# Fixed namespace. NEVER change this value — every canonical ID ever minted depends
# on it. If it changes, all historical IDs become unreproducible.
NAMESPACE = uuid.UUID("6f0f2f2a-8f0c-5a3e-9c4b-1d7a2e5b8c31")


class EntityKind(str, Enum):  # noqa: UP042
    PLAYER = "player"
    TEAM = "team"
    GAME = "game"
    DFS_SLATE = "dfs_slate"


def canonical_id(kind: EntityKind, provider: str, provider_id: str | int) -> str:
    """Mint a deterministic canonical ID.

    Pure and deterministic: identical inputs produce an identical UUID on any
    machine, in any process, in any year.
    Tested by tests/unit/test_canonical_ids_stable.py.
    """
    key = f"{kind.value}|{provider}|{provider_id}"
    return str(uuid.uuid5(NAMESPACE, key))


def canonical_player_id(provider: str, provider_id: str | int) -> str:
    return canonical_id(EntityKind.PLAYER, provider, provider_id)


def canonical_team_id(provider: str, provider_id: str | int) -> str:
    return canonical_id(EntityKind.TEAM, provider, provider_id)


def canonical_game_id(provider: str, provider_id: str | int) -> str:
    return canonical_id(EntityKind.GAME, provider, provider_id)


def canonical_dfs_slate_id(provider: str, provider_id: str | int) -> str:
    return canonical_id(EntityKind.DFS_SLATE, provider, provider_id)
