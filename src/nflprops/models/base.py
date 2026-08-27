"""ComponentModel interface.

SPEC: docs/IMPLEMENTATION_SPEC.md §62
PHASE: 6
STATUS: SKELETON
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ComponentModel(Protocol):
    """Every structural model implements this. PHASE 6.

    fit(X, y)            -> None
    predict(X)           -> conditional mean / probability
    sample(X, rng, n)    -> draws, used by the simulator
    to_artifact(path)    -> serialize including any residual pool
    from_artifact(path)  -> load
    diagnostics()        -> out-of-sample fit statistics for the training report

    `sample` is separate from `predict` on purpose: the simulator needs DRAWS, and a
    model that can only produce a mean cannot participate in a coherent simulation.
    """

    name: str
    version: str
