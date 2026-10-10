"""Model profiles: which inputs the fundamental football model consumes.

* `STRUCTURAL_CORE` -- only inputs with certified historical/live
  equivalence: completed prior-slate player/team game stats, schedule
  identity, player position group. No game market, no injury flags, no
  roster/depth enhancement. Phase 10C3A validates exactly this model
  historically, and it runs live as the identical model.
* `LIVE_ENHANCED` -- the same simulator plus genuinely available live game
  odds, injury status and roster/depth (LIVE_PIT receipt rules). A separate
  scientific model: it needs its own prospective validation/calibration.

The profile is a configuration leaf (`model.profile`), so it is part of
`config_sha256` and therefore of every run identity derived from it.

`PROFILE_SCIENCE_VERSIONS` versions each profile's effective science
contract. A calibrator binds to it through `profile_base_model_version`
(its `base_model_version`, part of the compatibility digest), so an
artifact fitted on an earlier contract never resolves for a later one.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

MODEL_PROFILE_CONFIG_PATH = "model.profile"


class ModelProfile(StrEnum):
    STRUCTURAL_CORE = "STRUCTURAL_CORE"
    LIVE_ENHANCED = "LIVE_ENHANCED"


class ModelProfileError(ValueError):
    """A model profile is missing, unknown, or mismatched."""


#: v2: QB candidates are the target team's structural roster members and
#: `qb_attempt_share` is team-relative (nflprops.features.team_membership).
PROFILE_SCIENCE_VERSIONS: dict[ModelProfile, str] = {
    ModelProfile.STRUCTURAL_CORE: "structural-core.v2-qb-roster-membership",
    ModelProfile.LIVE_ENHANCED: "live-enhanced.v2-qb-roster-membership",
}


def profile_base_model_version(model_version: str, profile: object) -> str:
    """`model_version` qualified by the profile's science contract."""
    return f"{model_version}+{PROFILE_SCIENCE_VERSIONS[parse_model_profile(profile)]}"


class _ConfigLike(Protocol):
    def get_path(self, path: str, default: object = None) -> object: ...


def parse_model_profile(value: object) -> ModelProfile:
    """Fail closed on a missing or unknown profile -- never a default."""
    if value is None or str(value) == "":
        raise ModelProfileError("model_profile is missing")
    try:
        return ModelProfile(str(value))
    except ValueError:
        raise ModelProfileError(
            f"unknown model_profile {value!r}; expected one of {[p.value for p in ModelProfile]}"
        ) from None


def resolve_model_profile(cfg: _ConfigLike) -> ModelProfile:
    """The configured `model.profile`. Fails closed when absent."""
    return parse_model_profile(cfg.get_path(MODEL_PROFILE_CONFIG_PATH, None))


def require_profile_match(*, prediction_profile: object, calibrator_profile: object) -> ModelProfile:
    """A calibrator applies only to predictions of its own profile."""
    prediction = parse_model_profile(prediction_profile)
    calibrator = parse_model_profile(calibrator_profile)
    if prediction is not calibrator:
        raise ModelProfileError(
            f"calibrator fitted on {calibrator.value} cannot be applied to "
            f"{prediction.value} predictions"
        )
    return prediction
