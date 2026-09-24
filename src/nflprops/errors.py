"""Exception hierarchy.

SPEC: docs/IMPLEMENTATION_SPEC.md §0 (build contract), Phase 0

Design rule: every failure mode that should stop a run has its own exception type so
callers can never accidentally swallow a serious one with a bare `except Exception`.
Nothing in this package warns-and-continues past a correctness failure.
"""


class NflpropsError(Exception):
    """Base for every error raised by this package."""


# ---------------------------------------------------------------- configuration
class ConfigError(NflpropsError):
    """Configuration is missing, malformed, or contains unknown keys."""


# ------------------------------------------------------------------- provider
class ProviderError(NflpropsError):
    """Any failure at the provider boundary."""


class ProviderAuthError(ProviderError):
    """401/403. Never retried."""


class ProviderRateLimitError(ProviderError):
    """429. Retried with backoff, honoring Retry-After when present."""


class ProviderTransportError(ProviderError):
    """Network-level failure. Retried."""


class ProviderSchemaError(ProviderError):
    """Response did not match the permissive raw schema in an incompatible way.

    Additive upstream changes (new fields) must NOT raise this. Incompatible ones
    (removed field, changed type, unknown enum in a modeled position) must.
    """


# ----------------------------------------------------------------- contracts
class ContractViolation(NflpropsError):  # noqa: N818
    """A machine-readable contract under contracts/ was violated.

    Examples: a feature not in feature_registry.yml reached a model; an endpoint
    not in bdl_endpoints.yml was called; a prop not in prop_map.yml was priced.
    """


class SpecCoverageError(ContractViolation):
    """contracts/bdl_endpoints.yml disagrees with the pinned OpenAPI spec."""


# --------------------------------------------------------------- data quality
class DataQualityError(NflpropsError):
    """A BLOCK-severity data quality gate fired. The pipeline halts."""


class EntityResolutionError(NflpropsError):
    """Ambiguous or contradictory identity mapping. Never auto-resolved."""


# -------------------------------------------------------------------- leakage
class LeakageError(NflpropsError):
    """Information from after `as_of` reached a prediction. Always fatal."""


# ------------------------------------------------------------------ modelling
class InvariantViolation(NflpropsError):  # noqa: N818
    """A simulation invariant from contracts/invariants.yml failed.

    Carries the rule id, game, and draw index. ALWAYS aborts the run — there is no
    configuration that downgrades this to a warning.
    """

    def __init__(self, rule_id: str, message: str, *, game_id: str | None = None,
                 draw_index: int | None = None) -> None:
        self.rule_id = rule_id
        self.game_id = game_id
        self.draw_index = draw_index
        detail = f"[{rule_id}]"
        if game_id is not None:
            detail += f" game={game_id}"
        if draw_index is not None:
            detail += f" draw={draw_index}"
        super().__init__(f"{detail} {message}")


class ConvergenceError(NflpropsError):
    """Monte Carlo did not reach the target standard error within max_draws."""


class ProjectionError(NflpropsError):
    """A player-game projection could not be built from a coherent simulation.

    Raised by `nflprops.projections` when a registry stat has no draw vector
    for an eligible player, a vector's length disagrees with `n_draws`, or a
    vector contains a non-finite value. Never downgraded: a projection is
    published whole or not at all. An existing coherent all-zero vector is
    valid and is NOT an error.
    """


class NotFittedError(NflpropsError):
    """A component model was asked to predict before being fitted."""
