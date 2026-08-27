# BDL Field Inventory — every datapoint, where it goes, what it can't do

**Provenance warning.** These field lists were transcribed from a human review of the
live BALLDONTLIE NFL OpenAPI 3.1.0 document. They are encoded machine-readably in
`contracts/bdl_endpoints.yml` and mirrored in `src/nflprops/domain/models.py`.

**They are not yet machine-verified.** Run this before trusting any of it:

```bash
nflprops provider pin bdl --url <bdl-openapi-url>
nflprops provider verify bdl --strict-fields
```

---

## Endpoint criticality

| Resource | Endpoint | Role in this system |
|---|---|---|
| Teams | `GET /nfl/v1/teams` | Core reference |
| Team | `GET /nfl/v1/teams/{id}` | Core reference |
| Roster / depth chart | `GET /nfl/v1/teams/{id}/roster` | **GOAT tier, 2025+ only.** Role features |
| Players | `GET /nfl/v1/players` | Core reference |
| Active players | `GET /nfl/v1/players/active` | Availability baseline |
| Player | `GET /nfl/v1/players/{id}` | Core reference |
| Games | `GET /nfl/v1/games` | Schedule + quarter/OT scores |
| Game | `GET /nfl/v1/games/{id}` | Schedule |
| **Player game stats** | `GET /nfl/v1/stats` | **PRIMARY GROUND TRUTH** |
| Player season stats | `GET /nfl/v1/season_stats` | QA / priors only — never a PIT feature |
| Advanced passing | `GET /nfl/v1/advanced_stats/passing` | QB latent state |
| Advanced rushing | `GET /nfl/v1/advanced_stats/rushing` | RB latent state |
| Advanced receiving | `GET /nfl/v1/advanced_stats/receiving` | WR/TE latent state |
| Team game stats | `GET /nfl/v1/team_stats` | Team + opponent state (by reversal) |
| Team season stats | `GET /nfl/v1/team_season_stats` | QA only. **No opponent fields** |
| Injuries | `GET /nfl/v1/player_injuries` | Availability. Append-only snapshots |
| Standings | `GET /nfl/v1/standings` | Store, but not a primary prop predictor |
| Play-by-play | `GET /nfl/v1/plays` | Period splits, first TD, longest pass |
| Current game odds | `GET /nfl/v1/odds` | Market environment |
| Opening game odds | `GET /nfl/v1/odds/opening` | **GOAT tier.** Line movement |
| Current player props | `GET /nfl/v1/odds/player_props` | **Live only, not retained upstream** |
| Opening player props | `GET /nfl/v1/odds/player_props/opening` | **GOAT tier, limited coverage** |
| DFS slates / draftables | `GET /nfl/v1/dfs/*` | **GOAT tier.** Optional, gated off |

---

## Where each field group is consumed

### `/nfl/v1/stats` — the ground truth

Every property is retained in raw and silver storage. Do **not** drop defensive or
special-teams columns because v1 focuses on offensive props: the return and defensive
TD fields (`kick_return_touchdowns`, `punt_return_touchdowns`,
`interception_touchdowns`, `fumbles_touchdowns`) are needed for the `anytime_td`
rare-event component under settlement rules that include them, and re-ingesting a
dropped column later means re-ingesting everything.

| Group | Feeds |
|---|---|
| Passing | QB completion/INT/sack state, prop labels |
| Rushing | rush share, rush efficiency, TD rush share, prop labels |
| Receiving | target share, catch probability, receiving efficiency, TD share |
| Fumbles | turnover state; `fumbles_touchdowns` → anytime TD component |
| Defense | opponent state; `interception_touchdowns` → anytime TD component |
| Kick/punt returns | anytime TD component under inclusive settlement rules |
| Kicking | FG make state, XP make state, kicking-points labels |
| Punting | field-position context (v2) |

### Advanced passing → QB latent state

`completion_percentage_above_expectation` (CPOE), `expected_completion_percentage`,
`avg_intended_air_yards` (aDOT), `avg_completed_air_yards`, `avg_time_to_throw`,
`aggressiveness`, `avg_air_yards_to_sticks`, `interceptions` → the catch model, the
interception model, and the sack model.

### Advanced rushing → RB latent state

`rush_attempts` (workload), `expected_rush_yards` (run expectation),
`rush_yards_over_expected_per_att` (individual value),
`percent_attempts_gte_eight_defenders` (box environment), `avg_time_to_los`
(decision speed) → the rush-gain mean model.

### Advanced receiving → WR/TE latent state

`percent_share_of_intended_air_yards` (air-yard share), `avg_intended_air_yards`
(aDOT), `catch_percentage`, `avg_separation`, `avg_cushion`, `avg_yac`,
`avg_expected_yac`, `avg_yac_above_expectation` → the catch model and the
receiving-gain mean model.

### Team game stats → team and opponent state

Derived quantities:

```
offensive_plays   = passing_attempts + rushing_attempts + sacks
dropbacks         = passing_attempts + sacks
sack_rate         = sacks / dropbacks
pass_attempt_rate = passing_attempts / offensive_plays
third_down_rate   = third_down_conversions / third_down_attempts
```

**Opponent features come from pairing the opponent's row in the same game.** Not from
`team_season_stats` — that schema has no opponent or defensive fields despite what
the endpoint description claims.

---

## What BDL does NOT give you

These must never be referenced as BDL fields. They are listed in the model card as
known gaps.

| Absent | Consequence | Path forward |
|---|---|---|
| `player_id` per play | Tier-3 prop labels need text parsing + reconciliation | PBP parser gated on `pbp_quality == HIGH` |
| EPA | No efficiency-context feature in v1 | Compute locally **after** yard-line semantics are validated |
| Route participation | Noisier target-share priors | Air-yard share partially substitutes |
| Snap % | Weaker role inference | Usage-derived role proxies |
| Structured weather | Outdoor/wind games mispriced | Optional provider behind the same protocol pattern |
| Structured practice participation | Weaker availability model | Injury status + comment-change features only |
| Pinnacle player props | No sharp benchmark | Retail consensus is the benchmark — say so |
| Per-attempt FG distance | Kicking props approximate | PBP-derived distance after parser validation |

---

## Traps that will bite you

1. **`/nfl/v1/team_stats` uses unbracketed array params** (`team_ids`, `seasons`,
   `game_ids`) while most endpoints use `team_ids[]`.
2. **`season_type` is an array on `/nfl/v1/games` and a scalar on `/nfl/v1/stats`.**
3. **`/nfl/v1/odds/player_props` returns everything for a game in one response** and
   does not use the cursor loop. Calling `paginated_get` on it is a bug.
4. **`NFLOpeningPlayerProp` marks `updated_at` required but defines `opened_at`.**
   Accept either at the adapter; expose only `opened_at` canonically.
5. **Yard-line orientation is undefined.** `start_yard_line`/`end_yard_line` are
   integers with no documented convention. Do not assume 0–100 offense-oriented.
   Validate empirically first (`tests/provider_contract/test_bdl_yardline_semantics.py`).
6. **Line and odds values arrive as strings.** Parse to `Decimal`. A line that goes
   through binary float becomes 67.49999999999999 and stops joining.
7. **Several rate-like `NFLStats` fields are typed integer in the spec** even though
   they are statistically continuous. Preserve raw, canonicalize to sensible numeric
   types — do not propagate the OpenAPI primitive through the model.
8. **`possession_time` is a clock string.** Normalize to integer seconds, retain raw.
9. **Preseason is unavailable** on several resources. Do not silently treat an empty
   result as "no games played."
10. **Live player props are not retained upstream.** This is not a trap so much as
    the single most consequential fact in the whole integration: if you do not
    collect, that history does not exist.
