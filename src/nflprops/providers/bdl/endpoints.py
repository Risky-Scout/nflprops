"""BDL endpoint path constants.

SPEC: docs/IMPLEMENTATION_SPEC.md §10; contracts/bdl_endpoints.yml
PHASE: 1
STATUS: IMPLEMENTED.

THIS IS THE ONLY FILE IN THE PACKAGE THAT MAY CONTAIN A BDL ENDPOINT STRING.
Enforced by tests/unit/test_import_boundaries.py, which greps src/ for "nfl/v1".

Do not add a path here that is absent from contracts/bdl_endpoints.yml.
"""

from __future__ import annotations

# --- reference data ---------------------------------------------------------
TEAMS = "/nfl/v1/teams"
TEAM = "/nfl/v1/teams/{id}"
ROSTER = "/nfl/v1/teams/{id}/roster"          # GOAT tier, 2025+ only
PLAYERS = "/nfl/v1/players"
ACTIVE_PLAYERS = "/nfl/v1/players/active"
PLAYER = "/nfl/v1/players/{id}"

# --- schedule ---------------------------------------------------------------
GAMES = "/nfl/v1/games"
GAME = "/nfl/v1/games/{id}"

# --- outcomes ---------------------------------------------------------------
STATS = "/nfl/v1/stats"                        # PRIMARY GROUND TRUTH
SEASON_STATS = "/nfl/v1/season_stats"          # QA/priors only — never a PIT feature
TEAM_STATS = "/nfl/v1/team_stats"              # live API uses bracketed array params
TEAM_SEASON_STATS = "/nfl/v1/team_season_stats"
STANDINGS = "/nfl/v1/standings"

# --- advanced ---------------------------------------------------------------
ADVANCED_PASSING = "/nfl/v1/advanced_stats/passing"
ADVANCED_RUSHING = "/nfl/v1/advanced_stats/rushing"
ADVANCED_RECEIVING = "/nfl/v1/advanced_stats/receiving"

# --- availability -----------------------------------------------------------
INJURIES = "/nfl/v1/player_injuries"

# --- play-by-play -----------------------------------------------------------
PLAYS = "/nfl/v1/plays"

# --- market -----------------------------------------------------------------
GAME_ODDS = "/nfl/v1/odds"
OPENING_GAME_ODDS = "/nfl/v1/odds/opening"                  # GOAT tier
PLAYER_PROPS = "/nfl/v1/odds/player_props"                  # live only, no pagination
OPENING_PLAYER_PROPS = "/nfl/v1/odds/player_props/opening"   # GOAT tier

# --- DFS (optional, gated off by default) -----------------------------------
DFS_SLATES = "/nfl/v1/dfs/slates"
DFS_SLATE = "/nfl/v1/dfs/slates/{id}"
DFS_DRAFTABLES = "/nfl/v1/dfs/draftables"


#: Endpoints that do NOT use the standard cursor pagination loop. SPEC §11.
NON_PAGINATED: frozenset[str] = frozenset({
    TEAMS, TEAM, ROSTER, PLAYER, GAME, STANDINGS,
    PLAYER_PROPS, OPENING_PLAYER_PROPS, DFS_SLATE,
})

# Live BDL wire-name mapping for array query parameters.
# Some endpoint-local arrays are spelled without [] in the pinned OpenAPI document,
# but the live NFL API requires the bracketed forms recorded below.
# Values use form/explode semantics: one key/value pair per item.
ARRAY_WIRE_NAMES: dict[str, dict[str, str]] = {
    PLAYERS: {"team_ids": "team_ids[]", "player_ids": "player_ids[]"},
    ACTIVE_PLAYERS: {"team_ids": "team_ids[]", "player_ids": "player_ids[]"},
    GAMES: {
        "dates": "dates[]", "team_ids": "team_ids[]", "seasons": "seasons[]",
        "season_type": "season_type", "weeks": "weeks[]",
    },
    STATS: {
        "player_ids": "player_ids[]", "game_ids": "game_ids[]",
        "seasons": "seasons[]",
    },
    INJURIES: {"team_ids": "team_ids[]", "player_ids": "player_ids[]"},
    TEAM_STATS: {"team_ids": "team_ids[]", "seasons": "seasons[]", "game_ids": "game_ids[]"},
    TEAM_SEASON_STATS: {"team_ids": "team_ids[]"},
    GAME_ODDS: {"game_ids": "game_ids[]"},
    OPENING_GAME_ODDS: {"game_ids": "game_ids[]"},
    PLAYER_PROPS: {"vendors": "vendors[]"},
    OPENING_PLAYER_PROPS: {"vendors": "vendors[]"},
    DFS_SLATES: {
        "slate_ids": "slate_ids[]",
        "providers": "providers[]",
        "formats": "formats[]",
        "scopes": "scopes[]",
        "seasons": "seasons[]",
        "weeks": "weeks[]",
    },
    DFS_DRAFTABLES: {
        "slate_ids": "slate_ids[]",
        "game_ids": "game_ids[]",
        "player_ids": "player_ids[]",
        "team_ids": "team_ids[]",
        "positions": "positions[]",
    },
}

#: Query names that are arrays even when a caller supplies one scalar.
ARRAY_PARAM_NAMES: dict[str, frozenset[str]] = {
    path: frozenset(mapping) for path, mapping in ARRAY_WIRE_NAMES.items()
}

#: Endpoints whose array parameters are UNBRACKETED. SPEC §12.
UNBRACKETED_ARRAY_PARAMS: frozenset[str] = frozenset()

#: Endpoints where `season_type` is an ARRAY rather than a scalar. SPEC §12.
SEASON_TYPE_IS_ARRAY: frozenset[str] = frozenset({GAMES})

#: Endpoints requiring GOAT-tier access. Surfacing this early gives a clear error
#: instead of an opaque 403 halfway through a bootstrap.
GOAT_TIER: frozenset[str] = frozenset({
    ROSTER, OPENING_GAME_ODDS, OPENING_PLAYER_PROPS,
    DFS_SLATES, DFS_SLATE, DFS_DRAFTABLES,
})

ALL_ENDPOINTS: frozenset[str] = frozenset({
    TEAMS, TEAM, ROSTER, PLAYERS, ACTIVE_PLAYERS, PLAYER,
    GAMES, GAME, STATS, SEASON_STATS, TEAM_STATS, TEAM_SEASON_STATS, STANDINGS,
    ADVANCED_PASSING, ADVANCED_RUSHING, ADVANCED_RECEIVING,
    INJURIES, PLAYS,
    GAME_ODDS, OPENING_GAME_ODDS, PLAYER_PROPS, OPENING_PLAYER_PROPS,
    DFS_SLATES, DFS_SLATE, DFS_DRAFTABLES,
})
