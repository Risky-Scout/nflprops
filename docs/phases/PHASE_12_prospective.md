# Phase 12 — Prospective 2026 execution

## Objective
Freeze the architecture and run the season honestly. This phase is operational
discipline, not new modeling.

## Spec sections
§72, §74, §75.

## Files to implement
```
src/nflprops/pipelines/pregame.py
src/nflprops/pipelines/settle.py
scripts/                              # scheduling / cron entrypoints
.github/workflows/                    # CI + scheduled collector health checks
```

## Requirements
1. Architecture is frozen. Changes require a spec version bump and a full re-run of
   the promotion gates.
2. Weekly loop: settle -> ingest -> update states -> fetch schedule -> snapshot roster
   -> snapshot injuries -> fetch odds -> build features -> simulate -> publish.
3. Continuous pregame loop: snapshot props/odds/injuries/rosters -> rerun affected
   games -> publish a new prediction version. Versions are appended, never overwritten.
4. **No manual patching of individual prop numbers, ever.** If a number looks wrong,
   the input or the model is wrong. Fix the cause.
5. Prospective scoring is recorded weekly against the §74 definition of success, and
   published unedited — including the bad weeks.
6. Collector health is monitored: missing checkpoints raise an alert, because silent
   collector failure destroys history that cannot be recovered.

## Definition of done
- [ ] Full weekly cycle runs unattended on fixture + live data
- [ ] Prospective scorecard published weekly
- [ ] Collector health alerting in place
- [ ] Model card current, with every known gap from spec Part XI still listed
