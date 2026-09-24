"""Generalized point-in-time collection framework (PHASE 4).

``collector_runs`` records one overall collection cycle; ``collector_resource_runs``
records the result of each individual resource fetch within that cycle. The
resource-level table is authoritative for feed availability -- a single
aggregate ``collector_runs.status`` is never sufficient evidence, since
injuries may succeed with zero rows while odds fails in the same cycle.

See ``docs/COLLECTION_ARCHITECTURE.md``.
"""
