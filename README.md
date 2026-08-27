# nflprops

**An NFL player prop prediction system where every prop for a game is derived from
one coherent simulation of that game.**

Built against the BALLDONTLIE NFL API (OpenAPI 3.1.0), for the 2026–2027 season.


> **Repository policy:** keep this project private. Runtime data, market snapshots,
> predictions, fitted artifacts, API keys, and `.env` files are excluded from Git.
>
> **2026 publication status:** the coherent baseline can be exercised after real BDL
> data is pinned/ingested, but it is **not approved to publish production edges** until
> the gates in [`docs/READINESS_2026.md`](docs/READINESS_2026.md) pass.


---

## The one idea

Most prop models are a stack of independent regressors: one for receiving yards, one
for receptions, one for passing yards. They disagree with each other, and the
disagreements are invisible until someone asks why the QB is projected for 260 yards
while his receivers sum to 291.

This system simulates the game once and reads every prop off the same draws:

```
Game Environment
  → Availability Scenario
  → Quarter-by-Quarter Team Plays
  → Dropbacks / Rush Attempts
  → Sacks / Pass Attempts / Directed Targets
  → QB + Target + Carry Allocation
  → Completion / INT / Rush / Receiving Efficiency
  → Scoring Opportunities → TD + FG + XP/2PT Events
  → Overtime if tied
  → Coherent Player Game Outcomes (joint, not marginal)
  → Every prop derived from one simulation
  → Calibration → Dispersion Check → Market Comparison → EV → CLV
```

QB passing yards are not modeled. They are the **sum of the receiving gains on that
QB's completions**. Same for completions and passing TDs. That identity is asserted
on every simulated game and a violation aborts the run.

---

## GitHub setup

See [`docs/GITHUB_SETUP.md`](docs/GITHUB_SETUP.md) for the first private push and secrets policy.

## Repository layout

```
docs/
  IMPLEMENTATION_SPEC.md    the normative build contract (read this first)
  phases/PHASE_00..12       one hand-off document per build stage
  BDL_FIELD_INVENTORY.md    every BDL datapoint, where it goes, and what it can't do
  AGENT_HANDOFF.md          how to drive this with an implementing agent

contracts/                  machine-readable, CI-enforced
  bdl_endpoints.yml         every endpoint, param, field, tier, and quirk
  feature_registry.yml      every feature that may reach a model
  prop_map.yml              every prop and how it derives from the simulation
  invariants.yml            every assertion enforced per simulated game
  warehouse_tables.yml      every canonical table and its keys

specs/providers/bdl/nfl.yml exact OpenAPI bytes after `nflprops provider pin bdl` (placeholder in this sandbox artifact)

src/nflprops/               the package skeleton
tests/                      provider/foundation tests + visible future acceptance criteria
tools/                      contract and spec verification
```

---

## Status

| Layer | State |
|---|---|
| Blueprint (`docs/IMPLEMENTATION_SPEC.md`) | Complete |
| Phase documents 0–12 | Complete |
| Machine-readable contracts | Complete, CI-validated |
| Skeleton: every module, docstring, spec reference | Complete |
| Phase 0–1 provider foundation | Implemented substantially; exact real-spec pin + authorized fixtures still required |
| Lean local data layer | Implemented: immutable raw JSON → canonical Parquet → DuckDB views |
| Point-in-time layer | Implemented baseline: append-only snapshots + conservative historical result availability |
| Dynamic team/player states | Implemented baseline: fast role decay, slow skill decay, EB population shrinkage |
| Joint game simulator | Implemented lean production baseline: Q1→Q4(+OT), shared plays/mix/opportunities/results |
| Prop derivation | All 25 BDL prop types derivable; Tier 3 gated from default publication |
| Market comparison | Same-timestamp no-vig pricing, EV, settlement, Brier/log-loss comparison implemented |
| Calibration / full walk-forward fitting | Framework present; OOF calibrator and complete promotion backtest still require historical/local data |

```bash
pytest                          # current sandbox run: 53 passed, 46 skipped, 0 failed
python tools/validate_contracts.py
```

The remaining skipped tests are visible acceptance criteria for still-unfinished/gated work, plus exact-spec and BDL yard-line verification. They are
skipped rather than deleted so the bar stays visible from day one.

---

## Open blocker

`specs/providers/bdl/nfl.yml` is still an **intentional placeholder in this sandbox-built artifact**. The provider contract was corrected against a live review of the official specification, but exact-byte machine verification requires pinning the authoritative document in a networked development environment.

Phase 1 is not complete until:

```bash
nflprops provider pin bdl --url https://www.balldontlie.io/openapi/nfl.yml
nflprops provider verify bdl --strict-fields
```

`verify` fails on any missing endpoint, missing field, extra field, type mismatch,
enum mismatch, or parameter-name mismatch. Until it passes, treat every field name
in this repository as a claim rather than a fact.

---

## Time-sensitive: start the collector early

BDL documents live player props as real-time and **retains no historical snapshots**.
Every week you do not collect is a week of market history that cannot be bought,
backfilled, or recovered — which means no CLV analysis and no market-timing research
for those weeks, permanently.

Deploy `nflprops snapshot props` as soon as Phase 1 lands, well before you have a
model worth betting. The collector is the moat.

---

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env          # add BDL_API_KEY
make verify-contracts
make test
```

Then hand `docs/phases/PHASE_00_foundation.md` to your implementing agent.

---

## How to build it

One phase at a time, in order. Each phase document names the exact files, the
contracts it must obey, the acceptance tests, and what is explicitly out of scope.

```bash
make verify-phase-00   # ... through verify-phase-10
```

A phase is done when its gate is green. Not when it looks done.

Recommended first hand-off: **Phase 0 + Phase 1 together.** The quality of every
later phase depends on getting the provider boundary correct and provider-independent
first.

---


## Lean build: actual weekly use

The repository is deliberately a local-file application, not an infrastructure
platform. After the authoritative BDL spec is pinned and dependencies are installed:

```bash
# One-time identity + historical foundation
nflprops ingest bootstrap
nflprops ingest season --season 2024
nflprops ingest season --season 2025

# Current week: one command refreshes BDL and immediately prices the board.
# Run it again whenever injuries/markets materially change.
nflprops run --season 2026 --week 1

# Or separate refresh from an exact reproducible as-of prediction:
nflprops ingest week --season 2026 --week 1
nflprops predict --season 2026 --week 1 --as-of 2026-09-10T16:00:00-04:00

# After games
nflprops settle --season 2026 --week 1
nflprops report --season 2026 --week 1
```

The fundamental simulator may use game spread/total as team-environment information,
but it never consumes the target player's own prop line to create its football
distribution. The prop quote is introduced only after the game is simulated, so
model-vs-market evaluation remains meaningful.


## Design rules that are not negotiable

1. **Provider independence.** Nothing under `models/`, `features/`, `simulation/`,
   `state/`, `calibration/`, or `backtest/` may import provider code. Replacing
   BALLDONTLIE means writing one new adapter directory, not a rewrite.
2. **Point-in-time correctness.** A prediction stamped `as_of = T` sees only records
   with `available_at <= T`. Backtests that look extraordinary are almost always
   leaking here.
3. **Determinism.** Same commit, config, raw files, spec SHA, cutoff, artifacts, and
   seed → identical prediction bytes. `hash()` is banned; blake2b seeds everything.
4. **Simulation coherence.** The identities in `contracts/invariants.yml` are asserted
   on every simulated game. A failure aborts the run. There is no warn-and-continue.
5. **No architecture changes without a spec change.** Amend the contract and the spec
   first, then the code.

---

## Known limitations (stated up front, kept in the model card)

| Gap | Consequence |
|---|---|
| No snap counts or route participation | Weaker role inference; usage proxies substitute |
| No weather | Outdoor/wind games mispriced |
| No practice participation | Weaker availability model |
| No per-attempt FG distance in the schema | v1 kicking is distance-marginal, not distance-aware |
| No player IDs or EPA in the play schema | Half/quarter, first-TD, and longest-pass labels depend on validated text parsing |
| Roster/depth chart from 2025 only, GOAT tier | Pre-2025 role features are missing by design, not imputed |
| Pinnacle is not a BDL prop vendor | Retail consensus is the benchmark. Say so; do not imply a sharp benchmark |

The system is designed so these are visible rather than quietly papered over.

---

## Definition of success

Not one profitable season. Prospectively, at the **same information timestamp**:

```
LogLoss_model < LogLoss_market   and/or   Brier_model < Brier_market
```

with acceptable calibration and positive CLV across a sufficient sample.

Stated precisely: *produce better-calibrated probabilities than the sportsbook market
at the same information timestamp.*

---

## License

See `LICENSE`.
