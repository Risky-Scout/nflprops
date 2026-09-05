"""Production orchestration (PHASE 5).

Prefect 3 orchestrates the existing Phase-4 collector and the existing
pregame prediction pipeline against official, game-relative checkpoints. It
never redefines model knowledge time, never re-implements collection or
prediction math, and never leaks orchestration lateness into a prediction.

Core model modules (`simulation`, `state`, `backtest`, `market`,
`collection`) do not depend on Prefect. Only this package, and specifically
`orchestration.tasks` / `orchestration.flows.*` / `orchestration.deployments`,
may import it. `orchestration.checkpoints` and `orchestration.run_store` are
deliberately Prefect-free pure logic, so they can be tested (and reasoned
about) without a Prefect runtime.

See `docs/ORCHESTRATION_ARCHITECTURE.md`.
"""
