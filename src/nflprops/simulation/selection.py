"""Deterministic starter selection.

`select_starter_index` is the *one* implementation of the simulator's
starter-selection rule. It was extracted verbatim from
`nflprops.simulation.game._starter_index` (PHASE 7B) so that the coherent
game simulation and the Phase-7 projection eligibility rule cannot drift
apart -- there is no second copy of this algorithm anywhere.

Semantics (unchanged from the pre-extraction simulator):

* QB : active candidate maximizing ``(qb_attempt_share, -(depth or 99))``
* K  : active candidate minimizing ``depth or 99``
* any other position : first active candidate, in the caller's order

Ties are broken by the order the candidates appear in ``players`` (Python's
``max``/``min`` keep the first extremal element), so callers that need a
reproducible pick must pass a canonically ordered sequence -- exactly what
`nflprops.simulation.game._ensure_players` already produces (players sorted
by ``player_id``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from nflprops.state.player import PlayerState


def select_starter_index(
    players: Sequence[PlayerState], position: str
) -> int | None:
    """Index into ``players`` of the selected starter for ``position``.

    Returns ``None`` when no active player plays ``position``. Behaviourally
    identical, element for element, to the simulator's original private
    ``_starter_index``.
    """
    candidates = [
        (i, p)
        for i, p in enumerate(players)
        if p.active and p.position_group == position
    ]
    if not candidates:
        return None
    if position == "QB":
        return max(
            candidates, key=lambda x: (x[1].qb_attempt_share, -(x[1].depth or 99))
        )[0]
    if position == "K":
        return min(candidates, key=lambda x: x[1].depth or 99)[0]
    return candidates[0][0]


def selected_starter_id(
    players: Sequence[PlayerState], position: str
) -> str | None:
    """``player_id`` of the selected starter for ``position``, or ``None``.

    A thin convenience over `select_starter_index` for callers (projection
    eligibility) that need the identity rather than a positional index.
    """
    index = select_starter_index(players, position)
    if index is None:
        return None
    return players[index].player_id
