"""Platform automation: GitHub-controlled remote compute for NFL training.

This package is deliberately science-free. It contains no simulation math,
calibration algorithm, PMF semantics, pricing math, or promotion thresholds
-- those remain exclusively in `nflprops.calibration`, `nflprops.simulation`,
`nflprops.distributions`, `nflprops.projections`, `nflprops.thresholds`, and
`nflprops.market.current_pricing`. This package only builds the operational
plumbing that lets GitHub-controlled remote compute invoke that science at
an explicit, verified git SHA, using verified immutable data, and never
promotes a challenger on its own authority.

Modules:
    remote_training  -- workflow input validation, the production draw-count
                         lock, git-SHA verification, and Science entry-point
                         resolution/invocation for the remote training runner.
    data_snapshot     -- SHA-256-verified training-data manifest + ephemeral
                         object-store download for the remote runner.
    health            -- read-only, best-effort infrastructure health
                         reporting (not a publication/readiness gate).

See docs/PLATFORM_AUTOMATION.md for the full architecture and the current
EXTERNAL_PROVISIONING_STILL_REQUIRED list.
"""

from __future__ import annotations
