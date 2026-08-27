# Lean Market-Superior Build Notes — 2026-08-20

## Build intent

This pass intentionally does **not** add infrastructure services. The executable path
is local Python + immutable JSON + Parquet + DuckDB, followed immediately by dynamic
football states and one coherent game simulator.

The target is not "complexity." The target is a model that can be run every NFL week,
audited, reproduced, and compared to the same-timestamp sportsbook market.

## Implemented in this pass

### Minimal data layer
- Immutable content-addressed raw API response store with SHA-256 metadata.
- Canonical provider-independent records and crosswalks.
- Parquet tables with local DuckDB views; no database server.
- Append-only injury, roster, odds and player-prop snapshot design.
- Conservative historical result availability timestamps to prevent final-score leakage.
- Core data-quality blockers.

### BDL live/historical workflow
- Season backfill for games, player stats, team stats and advanced weekly stats.
- GOAT-gated historical opening game odds.
- GOAT-gated historical opening player props for provider-supported recent coverage.
- Current-week refresh of schedule, active players, game odds, injuries, rosters and
  live player props.
- Live prop collector fails loudly instead of silently losing irreplaceable market history.

### Dynamic football states
- Team offense and defense are learned from game rows; defense is created by reversing
  opponent rows rather than relying on undocumented season-defense fields.
- Team state uses recency decay plus empirical-Bayes population shrinkage.
- Player opportunity/role state reacts quickly (35-day baseline half-life).
- Player efficiency/skill state moves slowly (180-day baseline half-life).
- Team environment uses a 120-day baseline half-life.
- All three speeds live in TOML and are challenger-tuned only by expanding-window tests.
- Role uncertainty is carried into Dirichlet opportunity allocations.

### Coherent simulation
One simulation produces:
plays → dropbacks/rushes → sacks/pass attempts → directed targets → player target/carry
allocations → completions/interceptions → receiving/rushing yards → TD/FG/XP → score
feedback → next quarter → optional overtime.

Hard identities include:
- sacks + pass attempts = dropbacks
- targets never exceed pass attempts
- player targets exhaust directed team targets
- player rush attempts exhaust team rush attempts
- receptions <= targets
- QB completions = receptions thrown by that QB
- QB passing yards = receiving yards thrown by that QB
- QB passing TDs = receiving TDs thrown by that QB
- rush+receiving yards = rush yards + receiving yards

The simulator uses deterministic named RNG substreams. First-TD selection was
vectorized so 20k–100k draws do not perform one dataframe filter per simulation.

### Prop pricing
All 25 BDL prop identifiers remain derivable from the same joint draw matrix.

Default publication is limited to confidence tiers 1–2. Tier 3 markets stay
research-derivable but are not published until the PBP label/reconciliation gate is
validated. This is deliberate: market superiority is more important than board coverage.

### Market comparison
- Target player-prop lines do **not** enter the fundamental football simulation.
- Game spread/total may anchor team environment.
- Two-sided market probabilities use transparent proportional de-vig as baseline.
- One-sided milestone prices are marked as unbenchmarked rather than given fake fair probabilities.
- Predictions store raw model probability separately from calibrated probability.
- Calibrated probability stays null until a genuine out-of-fold calibrator is fitted.
- Same-timestamp Brier/log-loss comparison and a multi-metric promotion gate are implemented.
- A model cannot be promoted only because realized ROI was positive.

### Weekly operator path

After pinning the exact BDL spec and installing dependencies:

```bash
nflprops ingest bootstrap
nflprops ingest season --season 2024
nflprops ingest season --season 2025

# live board:
nflprops run --season 2026 --week 1

# exact reproducible cut:
nflprops ingest week --season 2026 --week 1
nflprops predict --season 2026 --week 1 \
  --as-of 2026-09-10T16:00:00-04:00

# after final stats:
nflprops settle --season 2026 --week 1
nflprops report --season 2026 --week 1
```

## Deliberately not claimed

This package is now a substantially implemented **lean executable candidate model**.
It is **not yet empirically proven market-superior**, because that claim requires
actual BDL-authorized historical data, same-timestamp market rows and an expanding-window
evaluation run. The code contains the scoring/promotion rules so that the label
"market superior" must be earned, not declared.

The sandbox also cannot install missing Polars/DuckDB/PyArrow packages or download the
authoritative BDL OpenAPI bytes due external DNS restrictions. Therefore:
- source compilation was checked;
- dependency-light allocation/market-superiority tests were executed;
- the existing repository suite was executed;
- the full local Parquet/DuckDB/simulator path must be runtime-tested after dependencies
  are installed in the target environment.

## Required before first live BDL run

```bash
nflprops provider pin bdl \
  --url https://www.balldontlie.io/openapi/nfl.yml

nflprops provider verify bdl --strict-fields

uv lock
pytest
```

Do not bypass the provider-spec gate.

## Next model work — not infrastructure

The next work should be restricted to improving forecasting performance:
1. execute historical ingestion with the user's actual BDL tier;
2. run expanding-window forecasts against available opening player props;
3. fit state/simulator challenger parameters from pre-cutoff data only;
4. add out-of-fold probability calibration;
5. promote changes only when they improve same-timestamp log loss/Brier without
   unacceptable calibration degradation;
6. start/maintain the 2026 live prop snapshot collector for prospective CLV.

No additional database services, cloud feature stores, queues or distributed systems
are required for this model.
