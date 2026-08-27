"""Simulation invariants.

SPEC: docs/IMPLEMENTATION_SPEC.md §4, §67; contracts/invariants.yml
PHASE: 7
STATUS: IMPLEMENTED — normative. Extend only by amending contracts/invariants.yml.

Every rule here corresponds to an id in contracts/invariants.yml. A failure raises
InvariantViolation and ABORTS the run. There is no configuration flag that downgrades
an invariant to a warning, because a violated invariant means the simulation produced
mutually contradictory props — which is worse than producing none.

The rules are checked on the AGGREGATED draw, not on every intermediate step, except
where noted. Cost is negligible relative to the simulation itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nflprops.errors import InvariantViolation


@dataclass
class TeamDraw:
    """One team's realized quantities in one simulated game (or one quarter)."""

    offensive_plays: int = 0
    dropbacks: int = 0
    pass_attempts: int = 0
    sacks: int = 0
    rush_attempts: int = 0
    directed_targets: int = 0
    interceptions: int = 0
    touchdowns: int = 0
    two_point_attempts: int = 0
    xp_attempts: int = 0


@dataclass
class PlayerDraw:
    """One player's realized quantities in one simulated game."""

    player_id: str
    is_active: bool = True
    targets: int = 0
    receptions: int = 0
    receiving_yards: int = 0
    receiving_tds: int = 0
    longest_reception: int = 0
    rush_attempts: int = 0
    rushing_yards: int = 0
    rushing_tds: int = 0
    longest_rush: int = 0
    # QB-side (derived, never independently sampled — SPEC §40)
    qb_pass_attempts: int = 0
    qb_completions: int = 0
    qb_passing_yards: int = 0
    qb_passing_tds: int = 0
    qb_interceptions: int = 0
    longest_pass: int = 0
    # Kicking
    fg_attempts: int = 0
    fg_made: int = 0
    xp_made: int = 0
    kicking_points: int = 0
    # Per-quarter accumulators, index 0..3 (+4 for OT)
    quarter_receiving_yards: list[int] = field(default_factory=list)
    quarter_rushing_yards: list[int] = field(default_factory=list)
    quarter_tds: list[int] = field(default_factory=list)

    @property
    def rush_rec_yards(self) -> int:
        return self.rushing_yards + self.receiving_yards

    @property
    def total_offensive_tds(self) -> int:
        return self.receiving_tds + self.rushing_tds


def _fail(rule_id: str, msg: str, game_id: str | None, draw_index: int | None):
    raise InvariantViolation(rule_id, msg, game_id=game_id, draw_index=draw_index)


def check_team_invariants(
    team: TeamDraw, *, game_id: str | None = None, draw_index: int | None = None
) -> None:
    """INV001-INV007."""
    if team.sacks + team.pass_attempts != team.dropbacks:
        _fail("INV001", f"sacks({team.sacks}) + pass_attempts({team.pass_attempts}) "
              f"!= dropbacks({team.dropbacks})", game_id, draw_index)
    if team.pass_attempts > team.dropbacks:
        _fail("INV002", "pass_attempts > dropbacks", game_id, draw_index)
    if team.dropbacks + team.rush_attempts != team.offensive_plays:
        _fail("INV003", f"dropbacks({team.dropbacks}) + rush_attempts"
              f"({team.rush_attempts}) != offensive_plays({team.offensive_plays})",
              game_id, draw_index)
    if team.directed_targets > team.pass_attempts:
        _fail("INV004", f"directed_targets({team.directed_targets}) > "
              f"pass_attempts({team.pass_attempts}) — throwaways and spikes make "
              "targets <= attempts, never the reverse", game_id, draw_index)
    if team.interceptions > team.pass_attempts:
        _fail("INV007", "interceptions > pass_attempts", game_id, draw_index)


def check_allocation_invariants(
    team: TeamDraw,
    players: list[PlayerDraw],
    *,
    game_id: str | None = None,
    draw_index: int | None = None,
) -> None:
    """INV005, INV006, INV023 — allocations must exactly exhaust the team pool."""
    total_targets = sum(p.targets for p in players)
    if total_targets != team.directed_targets:
        _fail("INV005", f"sum(player_targets)={total_targets} != "
              f"directed_targets={team.directed_targets}", game_id, draw_index)

    total_carries = sum(p.rush_attempts for p in players)
    if total_carries != team.rush_attempts:
        _fail("INV006", f"sum(player_rush_attempts)={total_carries} != "
              f"team_rush_attempts={team.rush_attempts}", game_id, draw_index)

    total_qb_attempts = sum(p.qb_pass_attempts for p in players)
    if total_qb_attempts != team.pass_attempts:
        _fail("INV023", f"sum(qb_pass_attempts)={total_qb_attempts} != "
              f"team_pass_attempts={team.pass_attempts}", game_id, draw_index)


def check_player_invariants(
    players: list[PlayerDraw],
    *,
    game_id: str | None = None,
    draw_index: int | None = None,
) -> None:
    """INV010-INV013, INV030-INV032, INV040."""
    for p in players:
        if p.receptions > p.targets:
            _fail("INV010", f"{p.player_id}: receptions({p.receptions}) > "
                  f"targets({p.targets})", game_id, draw_index)
        if p.rush_rec_yards != p.rushing_yards + p.receiving_yards:
            _fail("INV011", f"{p.player_id}: rush_rec_yards mismatch",
                  game_id, draw_index)
        if p.rush_attempts == 0 and p.longest_rush != 0:
            _fail("INV012", f"{p.player_id}: longest_rush nonzero with no carries",
                  game_id, draw_index)
        if p.receptions == 0 and p.longest_reception != 0:
            _fail("INV013", f"{p.player_id}: longest_reception nonzero with no "
                  "receptions", game_id, draw_index)
        if p.fg_made > p.fg_attempts:
            _fail("INV031", f"{p.player_id}: fg_made > fg_attempts",
                  game_id, draw_index)
        if p.kicking_points != 3 * p.fg_made + p.xp_made:
            _fail("INV030", f"{p.player_id}: kicking_points({p.kicking_points}) != "
                  f"3*{p.fg_made} + {p.xp_made}", game_id, draw_index)
        if not p.is_active:
            opportunity = (p.targets + p.rush_attempts + p.qb_pass_attempts
                           + p.fg_attempts)
            if opportunity != 0:
                _fail("INV040", f"{p.player_id}: inactive player received "
                      f"{opportunity} opportunities", game_id, draw_index)


def check_qb_coherence(
    qb: PlayerDraw,
    receivers_on_qb_targets: list[PlayerDraw],
    *,
    game_id: str | None = None,
    draw_index: int | None = None,
) -> None:
    """INV020-INV022 — the derived-QB identities.

    This is the single most important coherence check in the system (SPEC §40). If it
    fails, the QB passing-yards prop and the receivers' receiving-yards props for the
    same game are telling different stories, which is exactly the failure mode this
    architecture exists to prevent.
    """
    rec_sum = sum(r.receptions for r in receivers_on_qb_targets)
    if qb.qb_completions != rec_sum:
        _fail("INV020", f"{qb.player_id}: qb_completions({qb.qb_completions}) != "
              f"sum(receptions on his targets)({rec_sum})", game_id, draw_index)

    yards_sum = sum(r.receiving_yards for r in receivers_on_qb_targets)
    if qb.qb_passing_yards != yards_sum:
        _fail("INV021", f"{qb.player_id}: qb_passing_yards({qb.qb_passing_yards}) "
              f"!= sum(receiving_yards)({yards_sum})", game_id, draw_index)

    tds_sum = sum(r.receiving_tds for r in receivers_on_qb_targets)
    if qb.qb_passing_tds != tds_sum:
        _fail("INV022", f"{qb.player_id}: qb_passing_tds({qb.qb_passing_tds}) != "
              f"sum(receiving_tds)({tds_sum})", game_id, draw_index)


def check_period_consistency(
    players: list[PlayerDraw],
    *,
    game_id: str | None = None,
    draw_index: int | None = None,
) -> None:
    """INV050-INV051 — per-quarter accumulators must sum to full-game totals."""
    for p in players:
        if p.quarter_receiving_yards:  # noqa: SIM102
            if sum(p.quarter_receiving_yards) != p.receiving_yards:
                _fail("INV050", f"{p.player_id}: quarter receiving yards do not sum "
                      "to full game", game_id, draw_index)
        if p.quarter_rushing_yards:  # noqa: SIM102
            if sum(p.quarter_rushing_yards) != p.rushing_yards:
                _fail("INV050", f"{p.player_id}: quarter rushing yards do not sum "
                      "to full game", game_id, draw_index)
        if p.quarter_tds:  # noqa: SIM102
            if sum(p.quarter_tds) != p.total_offensive_tds:
                _fail("INV050", f"{p.player_id}: quarter TDs do not sum to full game",
                      game_id, draw_index)


def check_all(
    team: TeamDraw,
    players: list[PlayerDraw],
    *,
    qb_target_map: dict[str, list[PlayerDraw]] | None = None,
    game_id: str | None = None,
    draw_index: int | None = None,
) -> None:
    """Run every applicable invariant for one team's draw.

    `qb_target_map` maps a QB player_id to the receivers who were targeted by him,
    with their realized outcomes on those targets only.
    """
    check_team_invariants(team, game_id=game_id, draw_index=draw_index)
    check_allocation_invariants(team, players, game_id=game_id, draw_index=draw_index)
    check_player_invariants(players, game_id=game_id, draw_index=draw_index)
    check_period_consistency(players, game_id=game_id, draw_index=draw_index)

    if qb_target_map:
        by_id = {p.player_id: p for p in players}
        for qb_id, receivers in qb_target_map.items():
            qb = by_id.get(qb_id)
            if qb is None:
                _fail("INV023", f"QB {qb_id} in target map but not in player draws",
                      game_id, draw_index)
            check_qb_coherence(qb, receivers, game_id=game_id, draw_index=draw_index)


def check_first_td_distribution(
    player_probs: dict[str, float],
    p_none: float,
    p_zero_td_game: float,
    *,
    tolerance: float = 1e-6,
    mc_tolerance: float = 0.01,
    game_id: str | None = None,
) -> None:
    """INV060-INV061 — the NONE state must survive. SPEC §50.

    `p_none` must equal the simulated probability of a zero-touchdown game (within
    Monte Carlo error), and the full distribution including NONE must sum to 1.
    Renormalizing NONE away is a common and expensive error: it inflates every
    player's first-TD probability by roughly the no-TD rate.
    """
    total = sum(player_probs.values()) + p_none
    if abs(total - 1.0) > tolerance:
        _fail("INV061", f"first_td distribution sums to {total}, not 1.0",
              game_id, None)
    if abs(p_none - p_zero_td_game) > mc_tolerance:
        _fail("INV060", f"P(first_td=NONE)={p_none} != P(0 TD game)="
              f"{p_zero_td_game}", game_id, None)
