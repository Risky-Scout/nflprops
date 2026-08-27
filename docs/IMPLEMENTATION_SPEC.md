# NFL Player Prop Prediction System — Master Implementation Specification

**Version:** 2026.1.0
**Status:** NORMATIVE BUILD CONTRACT
**Target season:** 2026–2027 NFL season
**Primary data provider:** BALLDONTLIE NFL API (OpenAPI 3.1.0, API v1.0.0)

---

## 0. How to use this document

This specification is the **build contract**. It is paired with a code skeleton in
`src/nflprops/`. Every skeleton module carries a `SPEC:` docstring tag pointing back
to a section number here.

**Intended workflow:**

1. Hand an implementing agent (Claude Code) exactly one phase document from
   `docs/phases/PHASE_NN_*.md`.
2. That phase document names the exact files to fill in, the acceptance tests that
   must pass, and the definition of done.
3. The agent implements only that phase. It does not touch later phases.
4. You run `make verify-phase-NN`. If green, the phase is done. If red, it is not.

**Three companion machine-readable contracts govern the build and are enforced by CI:**

| File | Governs |
|---|---|
| `contracts/bdl_endpoints.yml` | every BDL endpoint, parameter, and field |
| `contracts/feature_registry.yml` | every feature that may reach a model |
| `contracts/prop_map.yml` | every prop and how it is derived from the simulation |
| `contracts/invariants.yml` | every assertion enforced per simulated game |
| `contracts/warehouse_tables.yml` | every canonical table and its keys |

An implementing agent may not add a feature, endpoint, prop, or table that is absent
from these contracts. To add one, amend the contract **and** this spec first.

---

## 0.1 OUTSTANDING BLOCKER — read before Phase 1

The pinned provider spec file `specs/providers/bdl/nfl.yml` is a **placeholder**.
The field inventories in `contracts/bdl_endpoints.yml` were transcribed from a human
review of the live BDL OpenAPI document; they have **not** been machine-verified.

**Phase 1 cannot be declared complete until:**

```bash
# 1. Download the live spec and pin it
nflprops provider pin bdl --url <bdl-openapi-url>

# 2. Verify the contract matches the spec, field by field
nflprops provider verify bdl
```

`verify` fails the build on any missing endpoint, missing field, extra field,
type mismatch, enum mismatch, or parameter-name mismatch. Do not paper over
failures by editing the contract to match a wrong assumption — investigate each one.

---

## PART I — NON-NEGOTIABLE ARCHITECTURAL RULES

These are the rules that make the system survivable. Violating any one of them
turns the package into an unmaintainable one-season research script.

### 1. Provider independence

No model, feature, state, simulation, calibration, backtest, or explanation module
may import BALLDONTLIE-specific code. The only modules permitted to know that
endpoints like `/nfl/v1/stats` exist are under `src/nflprops/providers/bdl/`.

Dependency direction is one-way:

```
BDL JSON -> BDL raw schemas -> BDL mapper -> CANONICAL DOMAIN OBJECTS
         -> warehouse -> features -> states -> models -> simulator
```

**Forbidden, and enforced by `tests/unit/test_import_boundaries.py`:**

```python
# models/passing.py
import requests                                    # NO
from nflprops.providers.bdl import BDLClient       # NO
```

Import rules (enforced):

| Module | May import |
|---|---|
| `domain` | nothing provider-specific |
| `providers.*` | `domain` |
| `data` | `domain` |
| `features` | `domain`, `data` |
| `state` | `domain`, `data`, `features` |
| `models` | `domain`, `features`, `state` |
| `simulation` | `domain`, `models` |
| `market` | `domain` |
| `calibration` | `domain` |
| `backtest` | `domain`, `market`, `calibration` |

Forbidden edges: `simulation -> providers.*`, `models -> providers.*`,
`features -> providers.*`.

This single rule is what makes replacing the data provider a two-day job instead of
a rewrite.

### 2. Point-in-time correctness

Every time-varying record carries:

```
event_time          when the real-world event occurred
available_at        earliest timestamp at which this could legitimately be known
ingested_at         when we wrote it
provider
provider_record_id
```

A prediction stamped `as_of = T` may consume only records where `available_at <= T`.

Applies to: injuries, rosters, depth charts, game odds, player props, game results,
advanced stats, state snapshots, and calibration data.

`available_at` is **not** `ingested_at`. If you backfill 2024 injuries in 2026,
`ingested_at` is 2026 but `available_at` must be reconstructed as the true
publication time, and where it cannot be reconstructed the row must be marked
`available_at_is_estimated = true` and excluded from strict-leakage training.

This is the single highest-value discipline in the entire system. Backtests that
look extraordinary are almost always leaking here.

### 3. Determinism

Given identical (code commit, configuration, raw API files, spec SHA, training
cutoff, model artifacts, simulation seed), the system reproduces identical
prediction bytes. Tested by `tests/determinism/test_byte_reproducibility.py`.

Never use Python's built-in `hash()` for seeding — it is salted per process.

### 4. Simulation coherence

Every prop for a game comes from **one** simulation of that game. There is no
separate "receiving yards model" and "receptions model" that can disagree. The
required identities are in `contracts/invariants.yml` and are asserted on every
simulated game. Any failure aborts the run.

The most important ones:

```
team_sacks + team_pass_attempts        == team_dropbacks
sum(player_targets)                    == directed_targets  <= team_pass_attempts
sum(player_receptions)                 == sum over QBs of qb_completions
qb_passing_yards                       == sum(receiving_yards on that QB's completions)
qb_passing_tds                         == sum(receiving_tds  on that QB's completions)
rush_rec_yards                         == rushing_yards + receiving_yards
kicking_points                         == 3*fg_made + xp_made
inactive_player_opportunities          == 0
```

The simulator creates these relationships. Independent regressions do not.

### 5. No architecture changes without a spec change

The implementing agent MAY: reorganize internals, add tests, vectorize, improve
numerical stability, add logging, add type annotations.

The implementing agent MAY NOT independently: change the causal hierarchy, hardcode
BDL inside model modules, remove point-in-time controls, use random train/test
splits, use season-ending data in historical weeks, model correlated props
independently, skip market snapshot history, drop simulation invariants, assume
undocumented BDL fields, or add endpoints absent from the pinned spec.

---

## PART II — THE PROVIDER BOUNDARY

### 6. Critical BDL facts the implementation must respect

**The play schema contains no player IDs and no EPA.**

Actual `NFLPlay` fields: `id, game, type_slug, type_abbreviation, type_text, text,
short_text, away_score, home_score, scoring_play, period, clock_display, team,
start_yard_line, start_down, start_distance, end_yard_line, end_down, end_distance,
stat_yardage, home_win_probability, wallclock`.

Consequences:

- **EPA must never be referenced as a BDL field.** v1 does not need EPA. It may be
  computed locally later, only after yard-line orientation semantics are empirically
  validated.
- **Player-level quarter/half outcomes are not structured.** Full-game player stats
  are. Therefore historical labels for `passing_yards_1h`, `passing_tds_1h`,
  `receiving_yards_1h`, `rushing_yards_1h`, `anytime_td_1q`, `anytime_td_1h`,
  `anytime_td_2h`, `fg_made_1h`, `first_td`, and `longest_pass` require a validated
  play-description parser. See §22–24.

**Yard-line orientation is undefined in the spec.** `start_yard_line` and
`end_yard_line` are integers but the published schema does not define whether they
are offense-oriented 0–100, absolute field position, or something else. Do not build
red-zone or EPA logic on them until
`tests/provider_contract/test_bdl_yardline_semantics.py` passes.

**Pinnacle is not a BDL player-prop vendor.** The enum is `draftkings, fanduel,
caesars, betmgm, fanatics, betrivers`. The phrases "Pinnacle-caliber model" and
"benchmarked against Pinnacle" are therefore not interchangeable. A future
market-maker feed implements the same `MarketProvider` interface.

**Live player props are not retained upstream.** BDL documents player props as live
and real-time, with no historical snapshot storage, and returns all props for a game
in a single response rather than paginating. **You must run your own collector from
day one** or CLV and market-timing analysis are permanently impossible for the
2026 season. This is the single most time-sensitive item in the whole build:
every week you do not collect is a week of history you cannot recover.

**Roster/depth-chart data begins in 2025 and requires GOAT tier.** Historical role
features before 2025 must be reconstructed from usage, not depth charts. Mark them
`depth_chart_position__is_missing = 1` rather than imputing.

**Opening player props have limited coverage** (most recently completed season plus
ongoing seasons where available) and require GOAT tier.

### 7. Spec pinning

Place the exact OpenAPI document used at `specs/providers/bdl/nfl.yml`. At build time
compute `sha256(spec_bytes)`. Every model manifest contains:

```json
{
  "provider": "balldontlie",
  "provider_spec_version": "1.0.0",
  "provider_spec_sha256": "...",
  "provider_spec_captured_at": "..."
}
```

A nightly job (`nflprops provider drift bdl`) refetches the spec, compares SHAs, and
on difference: stores the new spec, generates a structural diff, runs provider
contract tests, alerts — and **does not silently upgrade production**.

### 8. Capability interfaces, not one god-object

```python
class ReferenceDataProvider(Protocol):
    def teams(self) -> Sequence[Team]: ...
    def players(self) -> Sequence[Player]: ...
    def active_players(self) -> Sequence[Player]: ...
    def roster(self, team_id: str, season: int) -> Sequence[RosterEntry]: ...

class ScheduleProvider(Protocol):
    def games(self, seasons=None, weeks=None, team_ids=None) -> Sequence[Game]: ...

class StatisticsProvider(Protocol):
    def player_game_stats(self, ...) -> Sequence[PlayerGameStat]: ...
    def team_game_stats(self, ...) -> Sequence[TeamGameStat]: ...
    def advanced_passing(self, ...) -> Sequence[AdvancedPassing]: ...
    def advanced_rushing(self, ...) -> Sequence[AdvancedRushing]: ...
    def advanced_receiving(self, ...) -> Sequence[AdvancedReceiving]: ...
    def plays(self, game_id: str) -> Sequence[Play]: ...

class AvailabilityProvider(Protocol):
    def injuries(self, ...) -> Sequence[Injury]: ...

class MarketProvider(Protocol):
    def game_odds(self, ...) -> Sequence[GameOdds]: ...
    def opening_game_odds(self, ...) -> Sequence[GameOdds]: ...
    def player_props(self, game_id: str, ...) -> Sequence[PlayerProp]: ...
    def opening_player_props(self, game_id: str, ...) -> Sequence[PlayerProp]: ...
```

BDL implements these. A replacement provider implements exactly these.

### 9. Canonical IDs are not provider IDs

Never let a BDL player ID become the permanent model player ID.

```
canonical_players / canonical_teams / canonical_games
provider_player_ids / provider_team_ids / provider_game_ids   (crosswalks)
```

Example crosswalk row:

```
canonical_player_id = 8f35...
provider            = balldontlie
provider_player_id  = 490
```

Every feature, state, training, simulation, and prediction table keys on
`canonical_*` IDs. If a new API calls the same player `823771`, only the crosswalk
changes.

**Canonical ID generation rule (normative):** a canonical ID is a UUIDv5 over a
namespace and the tuple `(entity_kind, first_seen_provider, first_seen_provider_id)`.
It is minted once and never regenerated. Entity resolution across providers writes
additional crosswalk rows; it never rewrites the canonical ID.

### 10. Endpoint inventory and auth

The full inventory, parameters, fields, tiers, and quirks live in
`contracts/bdl_endpoints.yml`. Endpoint path constants live in exactly one file:
`src/nflprops/providers/bdl/endpoints.py`. No endpoint string appears anywhere else.

Auth: `Authorization: <raw api key>` header (no `Bearer` prefix), plus
`Accept: application/json`. Config:

```toml
[provider.bdl]
base_url = "https://api.balldontlie.io"
api_key_env = "BDL_API_KEY"
timeout_seconds = 30
max_retries = 5
per_page = 100
```

API keys never appear in git, run manifests, logs, raw response metadata, or model
artifacts. The raw-store writer redacts the `Authorization` header before persisting
request metadata; `tests/unit/test_no_secret_leakage.py` enforces this.

### 11. Pagination

Cursor-based, `per_page` default 25, max 100, `meta.next_cursor` terminates.

```python
def paginate(fetch_page):
    cursor = None
    while True:
        response = fetch_page(cursor=cursor)
        yield from response.data
        cursor = response.meta.next_cursor
        if cursor is None:
            break
```

Only provider code paginates. **Exception:** `/nfl/v1/odds/player_props` returns all
props for a game in one response and does not use the cursor loop. This is encoded in
`contracts/bdl_endpoints.yml` under `pagination.exceptions` and must be handled
explicitly in the client, not by accident.

### 12. Known provider quirks (all handled in `providers/bdl/quirks.py`, nowhere else)

| Quirk | Handling |
|---|---|
| `/nfl/v1/team_stats` uses unbracketed `team_ids`, `seasons`, `game_ids` while other endpoints use `team_ids[]` | per-endpoint param encoder driven by `array_param` flag in the contract |
| `season_type` is an **array** on `/nfl/v1/games` but a **scalar** on `/nfl/v1/stats` | per-endpoint param type from the contract |
| `NFLOpeningPlayerProp` marks `updated_at` required but defines `opened_at` | `opened_at = payload.get("opened_at") or payload.get("updated_at")`; canonical layer exposes only `opened_at` |
| Line/odds values arrive as strings | parse to `Decimal` at ingestion, never binary float |
| `possession_time` is a clock string | normalize to integer seconds, retain raw |
| `height`/`weight` are strings | normalize to inches/pounds, retain raw |
| Several rate-like `NFLStats` fields are typed integer in the spec | preserve raw, canonicalize to numeric types suitable for statistics |
| DFS `game_id`, `player_id`, `team_id` may be null | nullable in raw schema, filtered at mapper |

Provider weirdness ends at the mapper. It never enters downstream code.

### 13. Raw storage is immutable

Every HTTP response is written before any transformation, under
`data/raw/bdl/<endpoint>/`, with a sidecar:

```json
{
  "provider": "balldontlie",
  "endpoint": "/nfl/v1/stats",
  "request_params": {},
  "requested_at": "...",
  "received_at": "...",
  "http_status": 200,
  "spec_sha256": "...",
  "response_sha256": "..."
}
```

Training reads raw snapshots via a dataset manifest. **Training never hits a live
API.** This is what makes reproduction possible a year later.

### 14. Schema policy: permissive at the boundary, strict in the core

Raw provider schemas allow extra fields and tolerate the documented inconsistencies.
Canonical domain models are strict, typed, normalized, and versioned. The adapter
tolerates additive upstream changes and fails loudly on incompatible ones.

### 15. Retry policy

Retry on `408, 429, 500, 502, 503, 504` and on transport failures, with exponential
backoff and bounded jitter. Do **not** retry `400, 401, 403, 404`. On `429`, honor
`Retry-After` if present.

---

## PART III — DATA LAYER

### 16. Warehouse

Parquet files queried through DuckDB. No database server dependency.

```
data/
├── raw/              immutable provider responses
├── bronze/           1:1 typed parquet, provider IDs intact
├── silver/           canonical IDs, normalized, deduplicated
├── features/
├── states/
├── market_snapshots/
├── artifacts/
├── predictions/
├── backtests/
└── manifests/
```

Full table inventory and keys: `contracts/warehouse_tables.yml`.

### 17. Snapshot discipline

Roster and injury data are **append-only**. Saturday's injury row never overwrites
Monday's. Role change over time is predictive signal; overwriting destroys it.

Snapshot row shape:

```
canonical_team_id, canonical_player_id, season, position, depth,
injury_status, available_at, ingested_at, raw_record_hash
```

Deduplicate on `raw_record_hash` so repeated polls of unchanged data do not bloat
storage, but retain the first and last `available_at` for each distinct state.

### 18. Coverage report, not assumed coverage

Do not hardcode any claimed historical game or play counts. Build
`src/nflprops/pipelines/coverage_report.py`, which empirically discovers: earliest
game, latest game, games per season, stats rows, advanced-stat weeks, PBP games,
opening-odds coverage, opening-prop coverage, and roster coverage.

The coverage report output is a required input to deciding the training window. It is
stored as an artifact and referenced in the training manifest.

### 19. Data quality gates

`src/nflprops/data/quality.py` runs after each ingestion and emits a report:

- Games marked `final` with zero player stat rows
- Player stat rows whose team does not appear in that game
- Team stat rows without a paired opponent row
- Sum of player targets exceeding team pass attempts (real-data violation of a
  simulator invariant — investigate before modeling)
- Negative or impossible values in count fields
- Duplicate `(game, player)` stat rows
- Advanced-stat weeks with no matching game
- Injury rows for players not on any roster snapshot

Every gate has a severity: `INFO`, `WARN`, `BLOCK`. `BLOCK` halts the pipeline.
Nothing is auto-corrected silently; corrections are explicit, versioned, and logged.

---

## PART IV — PLAY-BY-PLAY SUBSYSTEM

### 20. Why it exists

Tier-3 props (halves, quarters, first TD, longest pass) have no structured historical
label in BDL. They require reconstructing player attribution from play text.

### 21. Parser design (`src/nflprops/pbp/parser.py`)

1. Build active player aliases from canonical rosters for that game's teams
   (`pbp/aliases.py`): full name, "F.Last", "First Last Jr.", hyphen and apostrophe
   variants, suffix stripping, punctuation normalization.
2. Read `type_slug`, `type_abbreviation`, `type_text` to classify the play family
   before parsing free text.
3. Parse `text` for participant roles.
4. Attach canonical IDs; ambiguity across two players with the same alias on the same
   team is a parse failure, not a coin flip.
5. Assign `parser_confidence` in [0,1].

Derived per-play fields:

```
canonical_game_id, offense_team_id, defense_team_id,
quarter, clock_seconds, game_seconds_remaining,
home_score_before, away_score_before, score_differential,
is_scoring_play,
play_family in {run, pass, sack, field_goal, punt, kickoff, penalty, kneel, spike, other},
is_offensive_play, is_red_zone_candidate, is_goal_to_go_candidate,
parsed_passer_id, parsed_rusher_id, parsed_receiver_id, parsed_kicker_id,
parser_confidence
```

`is_red_zone_candidate` and `is_goal_to_go_candidate` are **gated** on the yard-line
semantics test passing. Until it does, they emit null with `is_missing = 1`.

### 22. Reconciliation (`src/nflprops/pbp/reconcile.py`)

For each game and player, compare parser-reconstructed totals against structured
`/nfl/v1/stats`:

pass attempts, completions, pass yards, rush attempts, rush yards, receptions,
receiving yards, rushing TDs, receiving TDs, FG made.

Compute a per-game reconciliation score and assign:

```
pbp_quality in {HIGH, MEDIUM, LOW, FAIL}
```

Thresholds live in config (`[pbp] minimum_reconciliation_score = 0.99`), not in code.

**Gate:** period-specific training, first-TD training, and longest-pass training use
only `pbp_quality == HIGH` games. Low-confidence games are excluded from those label
sets and the exclusion is recorded in the training manifest with counts, so the
effective sample size for tier-3 props is always visible.

Never silently trust regex output.

---

## PART V — FEATURES AND STATES

### 23. Feature discipline

Every feature has a registry entry in `contracts/feature_registry.yml` with: name,
definition, source fields, `available_at` rule, lag, null behavior, consumers, and
owner version. A feature without an entry cannot reach a model — CI enforces this.

Every feature materializes as two columns: `<name>` and `<name>__is_missing`.
**Blanket zero-filling is prohibited.** Null policies:

| Situation | Policy |
|---|---|
| No career data for the metric | positional prior |
| No advanced metric available | population posterior |
| No injury row | explicit configured default (`features.injury.missing_row_means`) |
| No market quote | feature unavailable; consumer branches |

The simulator must still run with every optional feature absent. This is tested.

### 24. Feature families

`game`, `team`, `opponent`, `player_state`, `role`, `advanced`, `market`.

Opponent features are constructed by **reversing team-game rows within a game**, not
by reading `/nfl/v1/team_season_stats` — that schema exposes no opponent or defensive
fields despite its description.

### 25. Empirical-Bayes state system

Every major metric maintains: population prior, player posterior, effective sample
size, posterior uncertainty, `last_updated`.

Update:

```
theta_t = w_t * x_t + (1 - w_t) * theta_prior_t
w_t     = n_t / (n_t + k)
```

Temporal transition between observations:

```
theta_prior_{t+1} = lambda * theta_t + (1 - lambda) * theta_population
```

`k` (shrinkage strength) and `lambda` (persistence) are **learned per metric** by
maximizing out-of-sample predictive likelihood on a walk-forward grid, not guessed.
The fitted values are stored in the model artifact and printed in the training report.

Posterior variance is tracked, not just the mean, because it drives allocation
concentration (§34) and uncertainty reporting.

### 26. Role state moves faster than skill state

This separation is fundamental and is the highest-leverage modeling idea in the
system.

```
WR receiving-skill lambda   -> HIGH persistence (skill changes slowly)
WR target-share    lambda   -> LOWER persistence (role changes fast)
```

When a WR1 goes down and the WR2 is promoted, his **opportunity** should move
immediately while his **catch rate and YAC ability** should not suddenly change.
Systems that model "projected receiving yards" as one blob get this wrong and it is
exactly where the market is beatable.

Enforced by a test: `tests/unit/test_role_faster_than_skill.py` asserts
`lambda_role < lambda_skill` for every position group after fitting.

### 27. States maintained

**QB:** pass_attempt_share, completion_probability, int_probability, sack_probability,
aDOT, CPOE, yards_per_completion, pass_td_rate, rush_share, rush_efficiency

**RB:** rush_share, target_share, catch_probability, rush_efficiency,
receiving_efficiency, td_rush_share, td_receiving_share

**WR/TE:** target_share, air_yard_share, aDOT, catch_probability,
receiving_efficiency, YAC, YACOE, td_receiving_share

**Kicker:** fg_attempt_rate, fg_make_probability, xp_make_probability

**Team:** offensive plays, opponent plays allowed, pass tendency, rush tendency, sack
rate, sacks allowed, yards/play, pass efficiency, rush efficiency, third-down
conversion, turnover rate, points-per-drive proxy, offensive TD rate, defensive pass
allowance, defensive rush allowance, pace, directed-target rate.

### 28. Rookie and early-season priors

BDL supplies age, experience, position, height, weight, college, depth chart, team.
Build BDL-only positional priors conditioned on (position, depth, experience).
External draft capital and college production would improve these materially, but
must arrive through an optional provider implementing the provider protocols — never
as a hardcoded lookup table.

Early-season is where the market is softest and where an unbalanced prior does the
most damage. Weeks 1–4 performance is a separate promotion gate (§53).

---

## PART VI — THE SIMULATOR

### 29. Central contract

```
Plays -> Run/Pass Mix -> Player Opportunities -> Player Efficiency
      -> Player Results -> Props
```

One simulation per game per `as_of`. Every prop derives from it.

### 30. Overall algorithm

```
shared game environment (correlated pace/scoring shocks)
    -> availability scenario draw
    -> for q in Q1..Q4 [and OT]:
         team plays_q
         dropbacks_q / rush_attempts_q          (game-script aware)
         sacks_q -> pass_attempts_q
         QB attempt allocation
         directed_targets_q
         player target allocation (Dirichlet-multinomial)
         completion / INT / incompletion per target
         receiving gains per completion
         rush allocation (Dirichlet-multinomial)
         rush gains per carry
         scoring opportunities -> TD events, FG attempts
         XP / 2PT
         update score differential  ->  feeds next quarter
    -> aggregate full game
    -> derive every prop
```

Quarter-by-quarter (rather than drive-by-drive) buys real game-script feedback at a
fraction of the complexity. A trailing team passing more, which suppresses RB carries
and inflates target volume, emerges **within the same simulated game**.

### 31. Stage A — shared game environment

Draw correlated latent shocks shared by both teams:

```
Z_pace  ~ N(0, 1)
Z_score ~ N(0, 1)
```

These capture the empirical fact that one unusually fast or high-scoring game lifts
opportunity for both offenses. Loadings on `Z_pace` and `Z_score` are fitted, not
assumed, and are the primary source of **cross-player correlation within a game** —
which is what makes the joint distribution (and therefore SGP pricing, §46) honest.

### 32. Stage B — offensive plays

Target quantity:

```
team_official_plays = pass_attempts + sacks + rush_attempts
```

Mean model: regularized count regression on team pace state, opponent pace state,
market total, spread, home/away, rest, week, QB availability. Dispersion is
calibrated against empirical residuals — **do not assume Poisson**; NFL play counts
are overdispersed relative to Poisson.

Quarter counts are drawn sequentially with a fitted quarter-share profile that
depends on score differential (trailing teams run more plays in Q4).

### 33. Stages C, D — dropbacks, run/pass mix, sacks

```
dropbacks = pass_attempts + sacks

dropbacks_q ~ Binomial(plays_q, p_dropback_q)
logit(p_dropback_q) = f(team pass tendency, opponent, pregame spread,
                        current score differential, quarter, time remaining)

rush_attempts_q = plays_q - dropbacks_q

sacks_q ~ Binomial(dropbacks_q, p_sack)
p_sack = g(QB sack state, team sacks allowed, opponent sack generation, score context)

pass_attempts_q = dropbacks_q - sacks_q
```

The score-differential term in `p_dropback_q` is what creates game script. It must be
fitted on realized in-game score states from team-game data, not hand-tuned.

### 34. Stage E — QB attempt allocation

If one active starter: he receives nearly all attempts, but an `OTHER_QB` category
retains a nonzero share fitted from historical in-game replacement and exit
frequency. If starting status is uncertain, the availability scenario (§42) resolves
it before allocation. **No QB receives attempts while inactive** (INV041).

### 35. Stage F — directed target pool

Do **not** force `targets == pass_attempts`. Throwaways, spikes, and batted balls are
attempts without a directed receiver.

```
directed_targets ~ Binomial(pass_attempts, p_directed)
p_directed estimated as  sum(player targets) / team pass attempts
```

By construction `directed_targets <= pass_attempts` (INV004).

### 36. Stage G — receiver target allocation

Dirichlet-multinomial over active receiving categories (WRs, TEs, RBs, OTHER):

```
(T_1, ..., T_n) ~ DirichletMultinomial(N_targets; alpha_1, ..., alpha_n)
alpha_i = kappa * s_i
```

where `s_i` is the posterior target share and `kappa` is the concentration, driven by
posterior uncertainty — **higher uncertainty gives lower kappa gives fatter share
variance**. This is why the state system must track variance and not just means.

Three properties this buys: targets sum exactly to the team total; players genuinely
compete for a fixed pool; role uncertainty naturally widens the outcome distribution
instead of requiring an arbitrary variance fudge.

### 37. Injury redistribution

If a player is inactive, his share is zero — but **do not proportionally redistribute
his share across everyone else**. That is the classic error and it systematically
misprices the backup who actually inherits the role.

Learned redistribution sequence:

```
removed share
  -> depth-chart replacement bonus   (who is next at that position)
  -> same-position redistribution
  -> cross-position redistribution   (WR out often lifts TE/RB differently)
  -> OTHER bucket
  -> normalize
```

Replacement effects are fit from historical absence events using
`(position, depth, team, prior replacement history)`. Where roster depth is missing
(pre-2025), fall back to usage-derived depth ordering and mark it.

### 38. Stage H — pass outcomes

Each directed target draws one of {completion, interception, incompletion}.

Catch probability uses: receiver catch posterior, aDOT, QB CPOE, QB completion state,
opponent pass efficiency allowed, separation, cushion.

Interception probability uses: QB INT state, opponent INT generation, game script,
aDOT, aggressiveness.

Because outcomes are drawn per target, `receptions <= targets`,
`interceptions <= attempts`, and `completions <= attempts` hold automatically. No
post-hoc clipping is ever required — and if you find yourself clipping, the
architecture has been violated.

### 39. Stage I — receiving yards

For each completion:

```
Y_catch = mu_hat(x) + epsilon
```

`mu_hat` uses player receiving-efficiency state, aDOT, YAC, YACOE, QB, opponent.
`epsilon` is drawn from an **empirical residual pool stratified by (position, aDOT
bucket, catch-depth bucket)**.

Empirical residuals rather than a fitted positive distribution, because completed NFL
receptions produce negative yardage and explosive heavy tails that a Gamma cannot
represent. Residual pools are frozen in the model artifact and versioned.

### 40. QB passing statistics are derived, never modeled separately

```
completions_j    = sum over j's targets of catch events
passing_yards_j  = sum over j's completions of receiving gain
passing_tds_j    = sum over j's completions that were TD events
```

This is one of the two or three most important rules in the entire specification. It
is what makes a QB passing-yards prop and every receiver's receiving-yards prop
mutually consistent, and it is what a naive multi-model shop gets wrong.

### 41. Stages J, K — rushing

```
(C_1, ..., C_n) ~ DirichletMultinomial(N_rush; alpha_1, ..., alpha_n)
```

over RBs, QB, WRs, FB, OTHER. QB scrambles and kneels are officially rushing attempts
and are absorbed into the QB/OTHER rushing states until structured play-call data
supports finer treatment. **Kneels matter**: a leading team's Q4 kneels are real
negative-yardage carries and inflate no one's prop. The QB rushing state must be
conditioned on score state so the kneel effect appears in the right games.

Rush gains: `mu_hat(x) + epsilon` with features player rush-efficiency posterior,
RYOE, RYOE/att, expected rush yards, eight-defender rate, time-to-LOS, opponent rush
state, score context. Empirical residuals include negative runs, zero-yard carries,
ordinary gains, and explosive carries.

### 42. Availability scenarios (mixture, not point estimate)

Before simulating, draw an availability scenario for questionable players:

```
P(final prop) = sum over scenarios s of  P(s) * P(prop | s)
```

`P(s)` comes from a model of injury status, hours since status change, comment
change, practice signal (absent in BDL — flagged as a known gap), and days since last
game. Scenarios are drawn per simulation batch, so the reported prop distribution is
a genuine mixture, and the reported uncertainty widens correctly for a questionable
player rather than pretending we know.

`tests/invariants/test_scenario_mixture.py` verifies the mixture identity holds to
Monte Carlo error.

### 43. Stage L — scoring, touchdowns, and the two-point problem

Do **not** independently predict player TD props.

1. Simulate **team scoring opportunities** per quarter from quarter yardage produced,
   plays, market implied points, and opponent scoring defense state.
2. Convert opportunities into TD / FG-attempt / no-score outcomes.
3. Split team TDs into passing vs rushing TDs via dynamic team/QB scoring states.
4. **Assign** each passing TD to one of the already-simulated completions, weighted by
   posterior TD-receiving shares. Assign each rushing TD to one of the simulated
   carries, weighted by TD-rush shares.

Because TDs are assigned to existing events, `qb.passing_tds == sum(receiving_tds by
that QB)` holds by construction, and a receiver cannot score a TD on a game where he
caught nothing.

**Two-point conversions.** After each TD the offense attempts XP or 2PT. The choice is
score-state dependent. Model:

```
P(go_for_two | score_differential, quarter, time) fitted from history
XP_attempts = TDs - two_point_attempts
XP_made ~ Binomial(XP_attempts, p_xp_kicker)
```

The source blueprint's `XPAttempts ~= TeamTD` is an approximation that breaks
`kicking_points` in exactly the high-leverage late-game spots where the market is
soft. Model it explicitly.

### 44. Return and defensive touchdowns

`anytime_td` settlement rules vary by book on whether return/defensive TDs count.
BDL exposes `kick_return_touchdowns`, `punt_return_touchdowns`,
`interception_touchdowns`, `fumbles_touchdowns`, so a pooled rare-event component is
estimable.

**Market settlement rules are versioned separately from model rules**, in
`src/nflprops/market/rules/`. The simulator produces both `offensive_tds` and
`all_tds`; the settlement rule selects which one prices a given vendor's market. Never
bake a settlement assumption into simulator code.

### 45. Stage M — field goals and kicking

Team FG attempts derive from the scoring-opportunity layer (§43), not an independent
model. For kicker k:

```
fg_made ~ Binomial(fg_attempts, p_fg_k)
kicking_points = 3 * fg_made + xp_made
```

`p_fg_k` should be distance-aware. BDL does not expose per-attempt distance
structurally; `long_field_goal_made` and PBP text are the only sources. **v1 uses a
distance-marginal make rate with an opportunity-quality adjustment, and flags
distance-aware kicking as a v2 item gated on PBP parser validation.** Say this out
loud in the model card rather than implying more precision than exists.

### 46. Stage N — score feedback and joint outputs

At quarter end, passing TDs, rushing TDs, FGs, XPs, 2PTs, and rare non-offensive
scores update the simulated score. The next quarter's pass tendency reads the new
differential.

Because all players in a game share one simulation, the system exports a **joint**
draw matrix, not just marginals. `simulation_player_results` retains per-draw player
outcomes (subject to a configurable retention policy on draw count), enabling:

- honest same-game-parlay pricing
- correlation diagnostics against realized outcomes
- conditional props ("receiving yards given the team scores 27+")

Never price a correlated parlay by multiplying marginals.

### 47. Overtime

BDL `NFLGame` exposes `home_team_ot` and `visitor_team_ot`. Full-game props settle
including OT. The quarter loop therefore runs Q1–Q4 and then, with probability
`P(tie at end of regulation)` implied by the simulated score, an OT period with its
own truncated play model and sudden-death termination rule.

Ignoring OT biases full-game yardage and TD props downward by a small but systematic
amount, concentrated in exactly the close games where lines are tightest. It is
cheap to include and it is included.

### 48. Longest props

```
longest_rush       = max(rush gain events),           0 if no carries
longest_reception  = max(reception gain events),      0 if no receptions
longest_pass       = max(completed pass gains by QB), 0 if no completions
```

Extreme-event behavior then depends correctly on opportunity count: a back with 20
carries has a genuinely fatter max than one with 8. No separate max-value regressor.

### 49. Period props

```
1H = Q1 + Q2      2H = Q3 + Q4      1Q = Q1
```

All derived from the same simulation. Historical **labels** for these still require
`pbp_quality == HIGH` (§22).

### 50. First TD

Every simulated TD event carries `(quarter, event_order, player, td_type)`. The
earliest event identifies `first_td_player`. If the game has no TD,
`first_td_player = NONE`.

**The NONE state must remain in the probability space.** Renormalizing it away is a
common and expensive error: it inflates every player's first-TD probability by
roughly the no-TD rate.

### 51. Deterministic RNG

```python
def deterministic_seed(*parts: str) -> int:
    key = "|".join(parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")

def make_rng(model_version, game_id, as_of) -> np.random.Generator:
    return np.random.default_rng(deterministic_seed(model_version, game_id, as_of))
```

Never `hash()`. Same run, same stream.

**Common random numbers:** when comparing two model versions or two lines on the same
game, reuse the same seed so differences reflect the model change, not Monte Carlo
noise.

### 52. Simulation count and Monte Carlo error

```toml
[simulation]
min_draws = 20000
max_draws = 100000
batch_size = 5000
probability_se_target = 0.0025
```

Stop when `SE = sqrt(p(1-p)/N)` is below target for the props in play. Do not use a
round 10,000 because a paper did.

**Note the asymmetry:** a `first_td` longshot at p ≈ 0.03 and a `receptions` line at
p ≈ 0.5 need very different N for the same *relative* precision. The stopping rule
targets absolute SE by default with a configurable relative-SE floor for longshots.

### 53. Dispersion calibration (do not skip this)

Structural simulators are almost always **under-dispersed** — the simulated
distribution is too narrow because latent sources of variance are missing. This
shows up as: good mean predictions, poor tail probabilities, and systematically
losing on overs at high lines.

Required diagnostic: **PIT / rank histograms** per prop family. For each historical
prediction, compute the simulated CDF at the realized value. If the model is
correctly dispersed, those values are uniform. A U-shape means under-dispersion.

Remedy, in preference order:
1. Find the missing structural variance source (usually role/availability uncertainty).
2. Widen residual pools or Dirichlet concentration where justified.
3. Only as a last resort, apply a fitted variance-inflation factor per prop family,
   stored in the artifact and reported in the model card as a known crutch.

---

## PART VII — MARKET SYSTEM

### 54. Two probabilities, never confused

| Name | Inputs | Question it answers |
|---|---|---|
| `p_fundamental` | everything except the target player-prop price | Does our football model independently beat the prop market? |
| `p_final` | fundamental + game market + prop market + line movement + cross-book consensus | What is the best probability we can produce? |

Game-level odds (spread/total) are permitted in `p_fundamental`; the **target prop's
own price is not**, nor is any other book's price for that same prop. Both are stored
on every prediction. Reporting `p_final` beating the market when `p_final` consumed
the market is self-deception.

### 55. Odds conversion

```
a < 0:  q = (-a) / (-a + 100)
a > 0:  q = 100 / (a + 100)
```

Paired over/under proportional devig:

```
p_over  = q_over  / (q_over + q_under)
p_under = q_under / (q_over + q_under)
```

Proportional devig is the transparent baseline. Power and Shin methods are
**challengers only**, promoted only if historical testing proves improvement on
held-out data.

### 56. Devigging one-sided markets (a real problem, handled explicitly)

`first_td` and milestone-style `anytime_td` quotes are often one-sided: there is no
paired "under" price. Pairwise devig is impossible.

Normative handling:

- **`first_td`:** collect the vendor's full slate of first-TD prices for the game,
  including any NONE/no-TD quote. Sum implied probabilities to get the book's
  overround, then normalize across the whole player set **plus the NONE state**. If
  the vendor does not quote NONE, estimate it from the simulated no-TD probability
  and mark the devig as `partial`.
- **`anytime_td` milestone:** if only one side is quoted, estimate overround from the
  same vendor's paired markets on the same game and apply it. Mark the resulting
  fair probability `devig_method = "borrowed_overround"` so downstream analysis can
  segment on it.
- **Never** silently treat a raw implied probability as fair. Every prediction row
  carries `devig_method` and `devig_confidence`.

### 57. Push probability

Do not discard pushes. For integer line L:

```
P_over  = P(X > L)
P_under = P(X < L)
P_push  = P(X = L)
EV_over(decimal d) = P_over * (d - 1) - P_under      # push contributes zero
```

Matters for receptions, attempts, touchdowns, interceptions, and FGs — all of which
see integer lines. Because the simulator produces integer-valued draws, `P_push` is
read directly off the empirical distribution rather than approximated from a
continuous density. This is a real edge over shops that model yardage continuously
and then fudge the push.

### 58. Market snapshot collector

BDL retains no live prop history, so **the collector is the moat**. Snapshot:

```
provider, vendor, canonical_game_id, canonical_player_id, prop_type,
line, price(s), market_type, provider_updated_at, collector_received_at,
minutes_to_start
```

Retained checkpoints: `OPEN, T-48H, T-24H, T-12H, T-6H, T-3H, T-1H, T-30M, T-10M,
CLOSE`. Poll frequency increases as kickoff approaches. Do not store only "latest."

The collector must be running **before** week 1 of the 2026 season. Treat it as the
first thing deployed and the last thing turned off.

### 59. Closing-line definition

Frozen in configuration:

```toml
[market]
close_buffer_seconds = 60
```

The closing quote is the latest valid quote received at least `close_buffer_seconds`
before scheduled kickoff. Never hand-pick whichever closing quote flatters a
backtest. Games with no quote inside the window are marked `closing_line_missing` and
excluded from CLV aggregates rather than filled.

### 60. CLV and EV

Track: CLV in probability terms and in cents, expected value at bet price, realized
ROI, drawdown, bet count, and per-prop-family and per-vendor breakdowns.

**Prop availability bias warning:** props that disappear from the board are not
missing at random — they are often the ones the model liked. Backtests must record
whether a prop was still quoted at close and report edge conditional on availability,
or the CLV number is flattering nonsense.

---

## PART VIII — TRAINING, CALIBRATION, EVALUATION

### 61. Splitting

**Absolutely no random train/test split.** Expanding-window walk-forward only:

```
train through T -> predict T+1 -> freeze -> observe T+1 -> add T+1 -> predict T+2
```

Every feature computed at the prediction timestamp. Every state snapshot at the
prediction timestamp.

### 62. Model inventory for v1

Deliberately limited: empirical Bayes, regularized GLMs, empirical residual
distributions, Dirichlet-multinomial allocations, Monte Carlo, isotonic/logistic
calibration.

Optional challengers: HistGradientBoosting, LightGBM, as a **residual layer** on top
of the structural GLM:

```
prediction = structural_GLM + residual_ML
```

Not production requirements in v1: EKF, UKF, CMP everywhere, ZAGA everywhere, QRF
ensembles, neural networks, transformers. They are challenger research. They do not
dictate the baseline unless walk-forward evidence promotes them.

Interpretability stays anchored in the structural component, which is also what makes
§65 explanations possible.

### 63. Calibration

Out-of-fold predictions only. Never fit a calibrator to in-sample fitted
probabilities.

Hierarchy with fallback: `prop family -> position -> global`. Small samples shrink
toward the broader group. Methods: logistic, beta, isotonic — chosen per family by
out-of-sample log loss.

### 64. Evaluation metrics

Distributional: MAE, median absolute error, CRPS, WIS, quantile coverage, PIT
uniformity.
Binary: log loss, Brier, calibration error, reliability curves.
Versus market: model log loss vs devigged market log loss; model Brier vs market
Brier — **at the same information timestamp**.
Trading: CLV, EV, ROI, drawdown, bet count.

### 65. Promotion gates

A challenger does not reach production because ROI improved. All of the following:

- better aggregate log loss
- better or neutral Brier
- no material calibration degradation
- better or neutral CRPS/WIS
- PIT histogram no worse
- stable across seasons
- stable in weeks 1–4 **and** weeks 14+ separately
- zero leakage test failures
- all simulation invariants pass
- reproducibility test passes

### 66. Leakage tests (CI must fail on any)

- `feature.available_at > prediction.as_of`
- result from the predicted game appears in a feature row
- a future week's game appears in a historical aggregation
- closing odds appear in an opening-time prediction
- an injury update after `as_of` appears in a feature row
- a season aggregate contains games after `as_of`
- a calibration fold overlaps its training target
- a state timestamp exceeds the prediction timestamp
- the target prop's own price appears in `p_fundamental` inputs

### 67. Reproducibility test

Run simulation A. Delete derived outputs. Rebuild from raw + manifest. Run
simulation B. Assert byte-equivalent probabilities under deterministic
serialization.

---

## PART IX — OUTPUTS

### 68. Prediction schema

```json
{
  "prediction_id": "...",
  "model_version": "2026.1.0",
  "as_of": "...",
  "canonical_game_id": "...",
  "canonical_player_id": "...",
  "prop_type": "receiving_yards",
  "vendor": "fanduel",
  "line": "67.5",
  "market_type": "over_under",
  "over_odds": -115,
  "under_odds": -110,

  "mean": 71.8, "median": 69.7,
  "p05": 17.0, "p10": 28.0, "p25": 47.0, "p50": 69.7,
  "p75": 93.0, "p90": 121.0, "p95": 138.0,

  "p_over_fundamental": 0.574,
  "p_over_calibrated": 0.561,
  "p_over_final": 0.548,
  "p_under": 0.452,
  "p_push": 0.0,

  "market_fair_over": 0.512,
  "devig_method": "proportional",
  "devig_confidence": "full",

  "edge": 0.036,
  "expected_value": 0.049,

  "n_draws": 40000,
  "mc_standard_error": 0.0025,
  "availability_scenario_entropy": 0.14,

  "feature_snapshot_id": "...",
  "state_snapshot_id": "...",
  "artifact_manifest_sha256": "..."
}
```

### 69. Explanation schema

Additive, in the units of the projection:

```
Baseline player projection          64.2
Team-volume effect                  +3.1
Player-role effect                  +6.7
QB effect                           +2.4
Opponent effect                     -2.8
Game-market environment             +1.9
Injury redistribution               +4.0
-----------------------------------------
Fundamental median                  79.5

Probability calibration             -1.7 pp
Market-aware adjustment             -2.4 pp
```

Attribution is computed by ablating each effect group through the simulator with
common random numbers, so the decomposition is causal within the model rather than a
post-hoc SHAP narrative. Components must sum to the total within a documented
tolerance; `tests/unit/test_explanation_additivity.py` enforces it.

### 70. Model artifact

```
artifacts/models/{model_version}/
├── models/
├── states/
├── calibrators/
├── residual_pools/
├── feature_schema.json
├── config.toml
├── training_manifest.json
├── metrics.json
├── model_card.md
└── artifact_manifest.json
```

```json
{
  "model_version": "...", "git_commit": "...", "python_version": "...",
  "lockfile_sha256": "...", "spec_sha256": "...", "config_sha256": "...",
  "training_data_sha256": "...", "training_cutoff": "...",
  "random_seed": "...", "metrics": {}
}
```

The `model_card.md` states known limitations plainly: no distance-aware FG model, no
snap counts, no route participation, no weather, no practice participation, tier-3
props gated on PBP quality, roster depth unavailable before 2025.

### 71. Configuration holds all mutable behavior

No magic numbers in source. Example:

```toml
[model]
version = "2026.1.0"

[state]
target_share_prior_games = 8
rush_share_prior_games = 8
efficiency_prior_games = 20

[simulation]
min_draws = 20000
max_draws = 100000
probability_se_target = 0.0025
include_overtime = true

[market]
close_buffer_seconds = 60
default_devig = "proportional"

[pbp]
minimum_reconciliation_score = 0.99

[features.dfs]
enabled = false
```

Values ultimately come from validation, not taste. Any constant that a reviewer might
want to question belongs here.

---

## PART X — OPERATIONS

### 72. Pipelines

**Bootstrap:** fetch teams, historical players, games, game stats, team stats,
advanced stats, PBP, available historical openings; build entity crosswalk; validate;
freeze manifest.

**Weekly:** settle last week; ingest final stats; update states; fetch next-week
games; snapshot roster; snapshot injuries; fetch game odds; build features; simulate
baseline.

**Continuous pregame:** snapshot props; snapshot game odds; snapshot injuries;
snapshot rosters; rerun affected games; publish a new prediction version.

**Inactive window:** when availability changes materially, the change propagates
through role shares to opportunity allocation to the whole game simulation to every
related prop. **No manual patching of individual prop numbers, ever.** If a number
looks wrong, the input or the model is wrong.

### 73. CLI

```
nflprops provider pin bdl --url <url>
nflprops provider verify bdl
nflprops provider drift bdl

nflprops ingest bootstrap --provider bdl
nflprops ingest season --season 2025
nflprops ingest week --season 2026 --week 1

nflprops snapshot injuries | rosters | odds | props

nflprops pbp parse --season 2025
nflprops pbp reconcile --season 2025

nflprops features build --as-of <ts>
nflprops state update --as-of <ts>

nflprops train --cutoff <ts>
nflprops backtest --seasons 2023,2024,2025

nflprops predict --season 2026 --week 1 --as-of <ts>
nflprops settle  --season 2026 --week 1
nflprops report  --season 2026 --week 1
nflprops reproduce --run-id <id>
nflprops coverage
```

### 74. Definition of success

Not one profitable season. Prospectively:

```
LogLoss_model < LogLoss_market   and/or   Brier_model < Brier_market
```

with acceptable calibration and positive CLV over a sufficient sample, **at the same
information timestamp**.

The objective, stated precisely: *produce better-calibrated probabilities than the
sportsbook market at the same information timestamp.*

### 75. What will actually determine whether this beats the market

Not exotic distributions. In descending order of expected value:

1. Point-in-time data discipline
2. Your own historical odds collection (nobody can sell you 2026's missed weeks)
3. Role-change detection and injury redistribution
4. Opportunity forecasting and QB-change handling
5. Target/carry competition modeling
6. Game-script feedback
7. Early-season priors
8. Probability calibration and dispersion
9. Market timing

Engineering quality and NFL structure create durable edge. Model exotica does not.

---

## PART XI — KNOWN GAPS AND HONEST LIMITATIONS

These belong in the model card and must not be quietly dropped.

| Gap | Impact | Mitigation / v2 path |
|---|---|---|
| No snap counts | Role inference weaker | Usage-derived role proxies; optional provider |
| No route participation | Target-share priors noisier | Air-yard share partially substitutes |
| No weather | Outdoor/wind games mispriced | Optional weather provider behind `MarketProvider`-style protocol |
| No practice participation | Availability model weaker | Injury status + comment-change features only |
| No per-attempt FG distance | Kicking props approximate | PBP-derived distance after parser validation |
| No player IDs in PBP | Tier-3 labels depend on text parsing | Reconciliation gate; HIGH-quality games only |
| No EPA | No efficiency-context features | Compute locally after yard-line validation |
| Roster/depth 2025+ only | Pre-2025 role features missing | `is_missing` flags, not imputation |
| Opening props limited coverage | Shorter market-timing history | Own collector from 2026 week 1 |
| No Pinnacle in prop vendors | No sharp benchmark | Treat retail consensus as the benchmark; state it |

---

## APPENDIX A — Build sequence

Phases are documented in `docs/phases/`. Build strictly in order.

| Phase | Name |
|---|---|
| 0 | Foundation: package, config, logging, manifests, canonical IDs |
| 1 | BDL adapter: auth, client, pagination, raw schemas, endpoints, mappers, contract tests |
| 2 | Immutable data layer: raw store, Parquet, DuckDB, entity resolution, quality, coverage |
| 3 | PBP system: classifier, parser, quarter splits, reconciliation, quality flags |
| 4 | Point-in-time features: team, opponent, player, role, market |
| 5 | Dynamic states: player EB, team EB, roles, uncertainty |
| 6 | Structural models: plays, dropbacks, sacks, targets, carries, catch, INT, efficiency, TD, kicking |
| 7 | Coherent simulator: Q1–Q4 + OT, game script, allocation, results, prop derivation |
| 8 | Calibration: OOF probabilities, family calibrators, fallback hierarchy, dispersion |
| 9 | Market system: odds conversion, devig, consensus, snapshots, CLV, EV |
| 10 | Walk-forward backtesting: leakage tests, market benchmark, family diagnostics |
| 11 | Explanation and reporting |
| 12 | Prospective 2026 execution |

**Recommended first hand-off: Phase 0 + Phase 1 together.** The quality of every later
phase depends on getting the provider boundary correct and provider-independent
first. But start the market snapshot collector (Phase 9's collector piece) as soon as
Phase 1 lands — collected history is the one thing that cannot be backfilled.

## APPENDIX B — Final model statement

```
Game Environment
  -> Availability Scenario
  -> Quarter-by-Quarter Team Plays
  -> Dropbacks / Rush Attempts
  -> Sacks / Pass Attempts / Directed Targets
  -> QB + Target + Carry Allocation
  -> Completion / INT / Rush / Receiving Efficiency
  -> Scoring Opportunities -> TD + FG + XP/2PT Events
  -> Overtime if tied
  -> Coherent Player Game Outcomes (joint, not marginal)
  -> Every BDL Player Prop Derived From One Simulation
  -> Calibration -> Dispersion Check -> Market Comparison -> EV -> CLV
```
