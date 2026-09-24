# Simulation / Pricing Architecture (Phase 6)

Phase 6 decouples the coherent per-game football simulation from
current-sportsbook-market pricing:

```
GAME/PLAYER STATE
      |
ONE COHERENT GAME SIMULATION      (simulate_game_for_prediction)
      |
JOINT PLAYER/TEAM DRAWS           (GameSimulationResult, unchanged type)
      |
  +----------------------------+
  | downstream derivations     |
  +----------------------------+
  | current sportsbook pricing | (price_current_markets)
  | Phase-7 projections        | (later)
  | Phase-8 thresholds         | (later)
  +----------------------------+
```

This is architectural, not scientific: Phase 6 changed no simulation
math, no probability calculations, no fair-odds/EV formulas. A pre- vs
post-refactor regression comparison against a deterministic fixture
produced byte-identical output across every column. See "Regression
result" below.

## Why this was a smaller change than it looked

A read-only audit (§1 of the Phase-6 directive) found the architecture
was already very close to this target:

- `simulate_game(sim_input, config)` (`nflprops.simulation.game`) already
  took **no player-prop parameter of any kind** — no vendor, line, price,
  quote ID, or `player_prop_snapshot`. Its only market-derived inputs were
  `implied_points`/`team_spread` on `TeamSimulationInput`, computed from
  the **game-level** odds consensus (spread/total) — a legitimate,
  pre-existing model input, architecturally distinct from player props.
- It was already called **exactly once per game**, before any
  quote-pricing loop.
- `GameSimulationResult.player_draws`/`.team_draws` were already
  long-format, draw-aligned Polars tables (one row per
  `(player_id, draw_id)` / `(team_id, draw_id)`) — a genuine per-draw
  vector representation, not a summary.
- `nflprops.simulation.props.summarize_prop`/`prop_values` already read
  *only* from `result.player_draws`/`result.first_td_player` — no RNG, no
  resampling — and already implemented one canonical, closed
  `prop_type -> distribution` mapping (see "Prop-type taxonomy" below).
- The player universe passed into simulation was already every
  `PlayerState` for the two teams playing, built entirely from
  roster/stat history — never filtered by whether a sportsbook quote
  existed for that player.
- `simulation_player_results` was already written from
  `result.real_player_draws()` unconditionally per game (gated only by
  `retain_joint_draws > 0`, a persistence-volume config, not quote
  presence) — already sportsbook-independent.

What Phase 6 actually did: **formalized these already-true properties
into an explicit, testable, named boundary**, and closed one real gap it
found along the way (below).

## The boundary

- `nflprops.pipelines.pregame.simulate_game_for_prediction(...)` — builds
  one `GameSimulationInput` from state + the game-level market consensus,
  calls `simulate_game()` exactly once, validates draw alignment, and
  returns a `PreparedGameSimulation`. **No player-prop-shaped parameter
  exists on this function's signature, and none is read from anywhere
  inside it.** Returns `None` only when the pre-existing
  expansion/new-team skip condition applies (unchanged behavior).
- `nflprops.market.current_pricing.price_current_markets(...)` — maps
  every currently-supported quote for one game to the already-computed
  `GameSimulationResult`. **Never calls `simulate_game`/
  `simulate_game_for_prediction`, never constructs an RNG, never
  resamples.** Every value comes from `summarize_prop` reading
  `result.player_draws`/`.first_td_player`.
- `nflprops.pipelines.pregame.predict_week()`/`predict_game()` —
  orchestrate the two layers per game, in that order, exactly as before.
  External behavior (arguments, return shape, persistence) is unchanged.

```python
prepared = simulate_game_for_prediction(
    game=game, team_states=team_states, player_states=player_states,
    game_odds=game_odds, as_of=as_of, model_version=model_version,
    market_mode=market_mode, simulation_config=simulation_config, n_draws=n_draws,
)
if prepared is None:
    continue  # unchanged expansion/new-team skip

prediction_rows.extend(
    price_current_markets(
        prepared.game, prepared.result, latest_quotes,
        season=season, week=week, as_of=as_of, state_context=state_context,
        roster=roster, injuries=injuries,
        game_market_available_at=prepared.game_market_available_at,
        market_mode=market_mode, max_confidence_tier=max_confidence_tier,
    )
)
```

## `PreparedGameSimulation`

A thin bundle (`nflprops.pipelines.pregame.PreparedGameSimulation`) —
**not a second simulation-result type**. It wraps the single canonical
`GameSimulationResult` (unchanged) alongside the minimal prediction-run
context `price_current_markets` needs: the game's own row (for
provenance), the game-level market knowledge timestamp, and an optional
`simulation_input_sha256` fingerprint (see below). `GameSimulationResult`
itself was not modified.

## Player universe: quote-independent by construction

`simulate_game_for_prediction` builds `home_players`/`away_players` from
`player_states.values()` filtered only by `team_id` — every player with
built state for that team, independent of quote existence. A player with
positive modeled opportunity (non-zero `target_share`/`rush_share`/
`qb_attempt_share` from `PlayerState`, i.e. pre-outcome role — not
whether any *finite* simulation draw happened to realize a non-zero
value) remains in the simulation regardless of whether any sportsbook
posted a prop for them. Whether that player is ultimately *priced*
depends only on whether `price_current_markets` finds a quote for them —
a completely separate, later decision.

Required test (`tests/simulation_pricing/test_boundary_basics.py::test_unquoted_nonzero_opportunity_player_retained_in_simulation`):
a receiving player (quoted) and a rushing player (never quoted) both
appear in `GameSimulationResult`; priced output contains only the quoted
one.

## Quote independence: proven on the full result, not aggregates

`tests/simulation_pricing/test_quote_independence.py` compares the
**complete** simulation result — player universe, every `player_draws`/
`team_draws` row (sorted by canonical key), `first_td_player`, and the
`simulation_input_sha256` fingerprint — across: zero quotes, one quote,
many quotes, reordered quotes, materially different lines/prices, and
different vendors. All produce bit-identical simulation output. Pricing
outputs (obviously) differ; the simulation never does.

## Prop-type taxonomy (audit table, §D)

Every `nflprops.domain.enums.PropType` member (25 total) is present in
`PROP_CONFIDENCE_TIER` (`nflprops.simulation.props`) *and* has an
explicit branch in `prop_values()` mapping it to a `player_draws`/
`first_td_player` column or derived combination. The taxonomy is closed:
no `PropType` exists that isn't both tiered and mapped, and nothing
outside `PropType` is ever priced (`prop_confidence_tier` returns `None`
for anything that doesn't parse as a `PropType`). No prop type required a
new distribution, a new model, or a fallback — **no `STOP AND REPORT`
condition (§H) was triggered.**

| prop_type | pre-Phase-6 priced | post-Phase-6 mapping | preserved |
|---|---|---|---|
| passing_attempts | YES | `player_draws.passing_attempts` | YES |
| passing_completions | YES | `player_draws.passing_completions` | YES |
| passing_yards | YES | `player_draws.passing_yards` | YES |
| interceptions | YES | `player_draws.interceptions` | YES |
| rushing_attempts | YES | `player_draws.rush_attempts` | YES |
| rushing_yards | YES | `player_draws.rushing_yards` | YES |
| receptions | YES | `player_draws.receptions` | YES |
| receiving_yards | YES | `player_draws.receiving_yards` | YES |
| rushing_receiving_yards | YES | `player_draws.rushing_receiving_yards` | YES |
| passing_tds | YES | `player_draws.passing_tds` | YES |
| anytime_td | YES | `receiving_tds + rushing_tds` | YES |
| kicking_points | YES | `player_draws.kicking_points` | YES |
| fg_made | YES | `player_draws.fg_made` | YES |
| longest_rush | YES | `player_draws.longest_rush` | YES |
| longest_reception | YES | `player_draws.longest_reception` | YES |
| passing_yards_1h | YES | `q1+q2 passing_yards` | YES |
| passing_tds_1h | YES | `q1+q2 passing_tds` | YES |
| receiving_yards_1h | YES | `q1+q2 receiving_yards` | YES |
| rushing_yards_1h | YES | `q1+q2 rushing_yards` | YES |
| fg_made_1h | YES | `q1+q2 fg_made` | YES |
| anytime_td_1h | YES | `q1+q2 (receiving_tds+rushing_tds)` | YES |
| anytime_td_2h | YES | `q3+q4(+OT) (receiving_tds+rushing_tds)` | YES |
| anytime_td_1q | YES | `q1 (receiving_tds+rushing_tds)` | YES |
| first_td | YES | `result.first_td_player == player_id` | YES |
| longest_pass | YES | `player_draws.longest_pass` | YES |

## Joint-draw alignment

`player_draws`/`team_draws` are single Polars tables where every
player's/team's block has exactly `result.n_draws` rows indexed by
`draw_id`. Reading multiple stat columns for the same `player_id`
(filter-then-select) is guaranteed aligned because they come from the
same filtered frame — this is a structural property of the storage
format, not a runtime coincidence. `nflprops.simulation.results` adds a
read-only accessor API (`player_ids`, `player_distribution`, `team_ids`,
`team_distribution`) and an explicit `validate_draw_alignment()` check
(called defensively inside `simulate_game_for_prediction`) that fails
loudly — never truncates, pads, or silently broadcasts — if any entity's
row count doesn't equal `n_draws`.

Required tests: `test_shared_distribution.py` (two lines share one
vector; OVER/UNDER share one vector; multiple props for one player read
aligned draw indices from one frame) and
`test_draw_length_validation.py` (hand-constructed malformed results
raise `ValueError`).

## Zero-quote games

A game with valid football state and zero posted player-prop quotes
still produces a full `GameSimulationResult` via
`simulate_game_for_prediction` — pricing is never a precondition for
simulation. `predict_week` on such a game returns an empty predictions
frame, but the simulation itself succeeded and (if `retain_joint_draws >
0`) is still persisted to `simulation_player_results`.

## RNG / determinism

Unchanged: `child_rng(model_version, game_id, as_of_str, salt)`
(`nflprops.simulation.rng`). Never incorporates `run_id`, vendor, prop
type, line, quote count, or quote ordering — those never existed in the
seed and Phase 6 added nothing to it. Same game + same `model_version` +
same `as_of` + same state/config -> same coherent simulation, proven
directly by the quote-independence tests and by the official-checkpoint
catch-up test (`tests/orchestration/test_phase6_checkpoint_simulation_boundary.py`):
a checkpoint executed late still simulates with `as_of=scheduled_as_of`,
never the actual (later) execution time, and a quote that becomes
available only after `scheduled_as_of` never enters that checkpoint's
priced output.

## `simulation_input_sha256` (§18, optional, added)

No existing fingerprint proved specifically the *football-simulation*
input set independent of market data (the Phase-5
`data_manifest_sha256` deliberately *does* include market quotes, since
it fingerprints the official run's full PIT input set — see below). So
`simulate_game_for_prediction` computes a lightweight
`simulation_input_sha256` over `dataclasses.asdict(sim_input)` (the exact
`GameSimulationInput` content: game_id, both teams' state/opponent-state/
players, implied points/spread, `as_of`, `model_version`) via
`nflprops.domain.hashing.hash_payload` (SHA-256, never `hash()`). It is
in-memory only — no new database column or migration (§41).

**Bug found and fixed while testing this**: the naive
`dataclasses.asdict(sim_input)` payload is sensitive to
`player_states.values()` dict iteration order, which is not itself part
of the reproducibility contract — `simulate_game`'s own
`_ensure_players()` already re-sorts players by `player_id` internally
for exactly this reason (see its docstring). The actual simulation output
(`player_draws`/`team_draws`) was already fully order-invariant thanks to
that existing sort; the *new* fingerprint was not, until
`simulate_game_for_prediction` was changed to sort
`home`/`away.players` by `player_id` before hashing, matching the same
canonicalization. This was a bug in the new Phase-6 addition, not a
pre-existing production issue — no simulation output was ever affected.

## Phase-5 data manifest: unchanged in scope, still correct

`prediction_runs.data_manifest_sha256` (Phase 5, corrected in the
"Complete checkpoint manifest and concurrent claim guarantees" commit)
continues to include selected game-odds and player-prop-quote rows —
because the *official run's output* (current sportsbook pricing) genuinely
depends on them, even though the *coherent simulation* does not. Phase 6
did not touch `nflprops.orchestration.manifest` and did not remove
player-prop quotes from that manifest. The two concepts are deliberately
different:

- **Run data manifest** (`data_manifest_sha256`): what PIT data produced
  the *official output* (simulation + pricing together) — includes
  quotes/odds.
- **Simulation input fingerprint** (`simulation_input_sha256`): what
  drove the *football simulation specifically* — excludes them, because
  `GameSimulationInput` never contained them in the first place.

## `retained_joint_draws` semantics

Unchanged from Phase 5: `n_draws` is the number of simulation draws
generated (`SimulationConfig.n_draws`); `retained_joint_draws` is how many
coherent draw indices are actually persisted to
`simulation_player_results` for that run (`min(retain_joint_draws,
result.n_draws)`, config-driven, default `0`). When
`retained_joint_draws == n_draws`, every draw was persisted; both are
in-memory-then-Parquet numbers for one run, not a claim about how many
seasons of history are retained — that policy question is explicitly
deferred (§30).

## What Phase 6 deliberately did not do

- No new distributions, player-usage model, TD model, role model,
  correlation, calibration, priors, or fitted parameters (§7). The one
  behavioral fix was the `simulation_input_sha256` ordering bug above —
  a Phase-6-introduced defect, not a correction to pre-existing model
  math.
- No `player_game_projections` (Phase 7), no threshold/milestone pricing
  or `player_threshold_prices` (Phase 8), no push-aware odds-math rewrite
  (Phase 9), no consensus/best-price/confidence/Kelly logic.
- No database migration. `prediction_runs`'s schema is unchanged;
  `simulation_input_sha256` is in-memory only.

## Regression result

A pre-refactor baseline was captured from the Phase-5-approved commit
(`a3cd6de`, `git show a3cd6de:src/nflprops/pipelines/pregame.py`, loaded
via `importlib.util.spec_from_file_location` into an isolated module) run
against the existing deterministic fixture
(`tests/orchestration/_fixtures.py::build_pit_fixture_warehouse`) and
compared to the post-refactor `predict_week` against an independent copy
of the same fixture. **Every column present in both outputs — not a
curated subset — compared exactly equal** after sorting by canonical key
(`polars.DataFrame.equals`). The two captured rows are hardcoded as a
permanent regression guard in
`tests/simulation_pricing/test_pricing_regression_baseline.py`.
