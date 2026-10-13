"""Live application of the certified joint-game calibrator to an official
checkpoint (production wiring; no calibration math is defined here).

Resolution (`resolve_live_calibrator`) runs BEFORE the checkpoint
simulation and fails closed, with a distinct status, unless every one of
these holds:

1. the checkpoint has a calibration scope and the snapshot carries a
   calibration registry;
2. an approved, promoted, compatible champion resolves for the run's OWN
   model profile (`resolve_calibration_champion`; a champion that resolves
   only for another profile is `PROFILE_MISMATCH` -- never applied);
3. the champion is one of the certified production calibrators pinned in
   `contracts/production_calibrators/certified.json`, with every identity
   field equal to the pin (`NOT_CERTIFIED` otherwise);
4. the run's profile-qualified model version equals the certified profile
   id (`SCIENCE_MISMATCH`), and the executor's shipped configuration
   reproduces the certified model / simulation configuration hashes
   exactly (`CONFIG_MISMATCH`);
5. the payload exists (`PAYLOAD_MISSING`) and its bytes verify against the
   registered SHA-256 / byte count and parse as the pinned algorithm and
   feature contract (`PAYLOAD_CORRUPT`).

Only then is the calibrator handed to the flow, which reweights the SAME
20k draws once (`calibrated_draw_weights` -- the certified
`softmax_weights(theta, compute_draw_features(result))`) and prices every
already-priced raw quote's identical event under those weights
(`calibrate_priced_rows`). The raw canonical artifacts are never modified:
the calibrated product is a separate table keyed by the raw
`prediction_id`, so raw and calibrated output stay separately identifiable.
There is no fallback to another calibrator, scope or profile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from nflprops.calibration.entropy_tilting import (
    ALGORITHM_FAMILY,
    ALGORITHM_VERSION,
    softmax_weights,
)
from nflprops.calibration.joint_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    compute_draw_features,
)
from nflprops.calibration.payload import (
    CalibrationPayloadError,
    deserialize_calibration_payload,
)
from nflprops.calibration.payload_store import (
    CalibrationPayloadMissingError,
    CorruptCalibrationPayloadError,
    PayloadObjectStore,
    get_calibration_payload,
)
from nflprops.domain.enums import PropType
from nflprops.domain.model_profile import (
    ModelProfile,
    ModelProfileError,
    profile_base_model_version,
    require_profile_match,
)
from nflprops.market.odds import (
    american_to_decimal,
    expected_value,
    fair_american_odds,
    fair_decimal_odds,
)
from nflprops.paths import runtime_resource
from nflprops.platform.remote_checkpoint import (
    CALIBRATION_CONTRACT_VERSION,
    CALIBRATION_SCOPE_TYPE,
    PROP_CONTRACT_VERSION,
    SIMULATION_CONFIG_VERSION,
)
from nflprops.simulation.props import AT_LEAST_HIT_PROPS, prop_values

if TYPE_CHECKING:
    from nflprops.config import Config
    from nflprops.data.warehouse import Warehouse
    from nflprops.orchestration.run_store import PredictionRunRecord
    from nflprops.simulation.game import GameSimulationResult

#: Pre-simulation resolution succeeded; APPLIED only once the flow has
#: persisted the calibrated product.
CALIBRATION_RESOLVED = "RESOLVED"
CALIBRATION_APPLIED = "APPLIED"
CALIBRATION_REGISTRY_ABSENT = "REGISTRY_ABSENT"
CALIBRATION_SCOPE_NOT_APPLICABLE = "SCOPE_NOT_APPLICABLE"
CALIBRATION_NO_APPROVED_CHAMPION = "NO_APPROVED_CHAMPION"
CALIBRATION_PROFILE_MISMATCH = "PROFILE_MISMATCH"
CALIBRATION_NOT_CERTIFIED = "NOT_CERTIFIED"
CALIBRATION_SCIENCE_MISMATCH = "SCIENCE_MISMATCH"
CALIBRATION_CONFIG_MISMATCH = "CONFIG_MISMATCH"
CALIBRATION_PAYLOAD_MISSING = "PAYLOAD_MISSING"
CALIBRATION_PAYLOAD_CORRUPT = "PAYLOAD_CORRUPT"
CALIBRATION_APPLICATION_FAILED = "APPLICATION_FAILED"

CERTIFIED_CALIBRATORS_SCHEMA = "nflprops.certified_production_calibrators/v1"
CALIBRATED_PRICES_TABLE = "player_prop_calibrated_prices"
CALIBRATION_APPLICATIONS_TABLE = "player_prop_calibration_applications"

#: Identity fields a resolved registry artifact must share with its pin.
_PINNED_ARTIFACT_FIELDS: dict[str, str] = {
    "object_uri": "object_uri",
    "payload_sha256": "payload_sha256",
    "payload_byte_count": "payload_byte_count",
    "scientific_content_sha256": "scientific_content_sha256",
    "model_profile": "model_profile",
    "base_model_version": "model_profile_id",
    "scope_type": "scope_type",
    "simulation_config_version": "simulation_config_version",
    "feature_contract_version": "feature_contract_version",
    "prop_contract_version": "prop_contract_version",
    "calibration_contract_version": "calibration_contract_version",
}

_HIT_SIDE = "HIT"


class LiveCalibrationError(ValueError):
    """The resolved calibrator could not be applied to this simulation."""


# ------------------------------------------------------------------ pins


def certified_calibrators_dir() -> Path:
    return runtime_resource("contracts", "production_calibrators")


@lru_cache(maxsize=4)
def _load_pins(path: str) -> dict[str, dict[str, Any]]:
    document = json.loads(Path(path).read_text())
    if document.get("schema_version") != CERTIFIED_CALIBRATORS_SCHEMA:
        raise LiveCalibrationError(
            f"certified calibrator registry schema {document.get('schema_version')!r} "
            f"is not {CERTIFIED_CALIBRATORS_SCHEMA!r}"
        )
    pins: dict[str, dict[str, Any]] = {}
    for entry in document["calibrators"]:
        cid = str(entry["production_calibrator_id"])
        if cid in pins:
            raise LiveCalibrationError(f"duplicate certified calibrator {cid}")
        pins[cid] = dict(entry)
    return pins


def load_certified_calibrators(directory: Path | None = None) -> dict[str, dict[str, Any]]:
    """`production_calibrator_id` -> pinned certified identity."""
    root = directory if directory is not None else certified_calibrators_dir()
    return _load_pins(str(root / "certified.json"))


class PackagedPayloadStore:
    """Read-only payload store over the certified payloads shipped with
    the release (`contracts/production_calibrators/payloads/`). Identical
    bytes on Wizard and on the GitHub executor; every read is verified
    against the registry by `get_calibration_payload`."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root if root is not None else certified_calibrators_dir() / "payloads"

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if self._root.resolve() not in path.parents:
            raise CalibrationPayloadMissingError(f"payload key {key!r} escapes the store")
        return path

    def put_bytes(self, key: str, data: bytes) -> None:
        raise PermissionError("the certified payload store is read-only")

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).is_file()
        except CalibrationPayloadMissingError:
            return False


# ------------------------------------------------------------ resolution


@dataclass(frozen=True)
class LiveCalibrator:
    """A verified, applicable calibrator: everything application needs."""

    calibration_artifact_id: str
    payload_sha256: str
    model_profile: str
    model_profile_id: str
    checkpoint_scope: str
    algorithm: str
    theta: tuple[float, ...]
    certification_run_id: str
    fit_run_id: str

    def identity(self) -> dict[str, Any]:
        return {
            "calibration_artifact_id": self.calibration_artifact_id,
            "payload_sha256": self.payload_sha256,
            "model_profile": self.model_profile,
            "model_profile_id": self.model_profile_id,
            "checkpoint_scope": self.checkpoint_scope,
            "algorithm": self.algorithm,
            "certification_run_id": self.certification_run_id,
            "fit_run_id": self.fit_run_id,
        }


@dataclass(frozen=True)
class CalibrationResolution:
    status: str
    detail: str
    base: dict[str, Any]
    calibrator: LiveCalibrator | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.status == CALIBRATION_RESOLVED and self.calibrator is not None

    def as_dict(self) -> dict[str, Any]:
        out = {**self.base, **self.extra, "status": self.status, "approved": False,
               "detail": self.detail}
        if self.calibrator is not None:
            out.update(self.calibrator.identity())
        return out


def production_config_identity(config: Config, pin: dict[str, Any]) -> dict[str, str]:
    """The certified evidence's model / simulation configuration hashes,
    recomputed from the executor's SHIPPED configuration."""
    from nflprops.calibration.phase10c3a_runner import science_config_identity
    from nflprops.pipelines.pregame import (
        simulation_config_from_app_config,
        state_configs_from_app_config,
    )
    from nflprops.platform.remote_training import PRODUCTION_N_DRAWS

    player_cfg, team_cfg = state_configs_from_app_config(config)
    return science_config_identity(
        model_version=str(pin["evidence_model_version"]),
        regularization_lambda=float(pin["regularization_lambda"]),
        player_state_config=player_cfg,
        team_state_config=team_cfg,
        simulation_config=simulation_config_from_app_config(
            config, n_draws=PRODUCTION_N_DRAWS
        ),
    )


def _resolve_champion(
    warehouse: Warehouse, *, checkpoint_name: str, profile: str, base_model_version: str
) -> Any:
    from nflprops.calibration.contract import load_calibration_registry_contract
    from nflprops.calibration.registry import resolve_calibration_champion
    from nflprops.orchestration.dispatch_plan import as_run_store_backend

    contract = load_calibration_registry_contract()
    for scope in (checkpoint_name, "ALL_PREGAME_CHECKPOINTS"):
        if scope not in contract.checkpoint_scopes:
            continue
        artifact = resolve_calibration_champion(
            as_run_store_backend(warehouse),
            scope_type=CALIBRATION_SCOPE_TYPE,
            checkpoint_scope=scope,
            model_profile=profile,
            base_model_version=base_model_version,
            simulation_config_version=SIMULATION_CONFIG_VERSION,
            feature_contract_version=FEATURE_CONTRACT_VERSION,
            prop_contract_version=PROP_CONTRACT_VERSION,
            calibration_contract_version=CALIBRATION_CONTRACT_VERSION,
            contract=contract,
        )
        if artifact is not None:
            return artifact
    return None


def resolve_live_calibrator(
    warehouse: Warehouse,
    run: PredictionRunRecord,
    *,
    model_profile: ModelProfile,
    config: Config,
    payload_store: PayloadObjectStore | None = None,
    certified: dict[str, dict[str, Any]] | None = None,
) -> CalibrationResolution:
    """Fail-closed pre-simulation resolution (module docstring, 1-5)."""
    from nflprops.calibration.contract import load_calibration_registry_contract

    base: dict[str, Any] = {
        "scope_type": CALIBRATION_SCOPE_TYPE,
        "model_profile": model_profile.value,
        "base_model_version": profile_base_model_version(run.model_version, model_profile),
        "simulation_config_version": SIMULATION_CONFIG_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "prop_contract_version": PROP_CONTRACT_VERSION,
        "calibration_contract_version": CALIBRATION_CONTRACT_VERSION,
    }

    def refuse(status: str, detail: str, **extra: Any) -> CalibrationResolution:
        return CalibrationResolution(status=status, detail=detail, base=base, extra=extra)

    if run.checkpoint_name not in load_calibration_registry_contract().checkpoint_scopes:
        return refuse(CALIBRATION_SCOPE_NOT_APPLICABLE,
                      f"checkpoint {run.checkpoint_name!r} has no calibration scope")
    if not warehouse.exists("calibration_champions"):
        return refuse(CALIBRATION_REGISTRY_ABSENT, "no calibration registry in the snapshot")

    artifact = _resolve_champion(
        warehouse, checkpoint_name=run.checkpoint_name, profile=model_profile.value,
        base_model_version=base["base_model_version"],
    )
    if artifact is None:
        for other in ModelProfile:
            if other is model_profile:
                continue
            foreign = _resolve_champion(
                warehouse, checkpoint_name=run.checkpoint_name, profile=other.value,
                base_model_version=profile_base_model_version(run.model_version, other),
            )
            if foreign is not None:
                return refuse(
                    CALIBRATION_PROFILE_MISMATCH,
                    f"the only approved champion was fitted on {other.value} predictions; "
                    f"it is never applied to {model_profile.value}",
                    calibration_artifact_id=foreign.calibration_artifact_id,
                )
        return refuse(CALIBRATION_NO_APPROVED_CHAMPION,
                      "no approved, promoted, compatible champion resolves")

    try:
        require_profile_match(prediction_profile=model_profile.value,
                              calibrator_profile=artifact.model_profile)
    except ModelProfileError as exc:
        return refuse(CALIBRATION_PROFILE_MISMATCH, str(exc),
                      calibration_artifact_id=artifact.calibration_artifact_id)

    pins = certified if certified is not None else load_certified_calibrators()
    pin = pins.get(artifact.calibration_artifact_id)
    if pin is None:
        return refuse(CALIBRATION_NOT_CERTIFIED,
                      "the resolved champion is not a certified production calibrator",
                      calibration_artifact_id=artifact.calibration_artifact_id)
    for attr, key in _PINNED_ARTIFACT_FIELDS.items():
        if getattr(artifact, attr) != pin[key]:
            return refuse(
                CALIBRATION_NOT_CERTIFIED,
                f"registry {attr}={getattr(artifact, attr)!r} differs from the certified "
                f"pin {pin[key]!r}",
                calibration_artifact_id=artifact.calibration_artifact_id,
            )
    if artifact.checkpoint_scope not in (pin["checkpoint_scope"],):
        return refuse(CALIBRATION_NOT_CERTIFIED,
                      f"champion scope {artifact.checkpoint_scope!r} is not the certified "
                      f"{pin['checkpoint_scope']!r}",
                      calibration_artifact_id=artifact.calibration_artifact_id)

    if base["base_model_version"] != pin["model_profile_id"]:
        return refuse(
            CALIBRATION_SCIENCE_MISMATCH,
            f"run science {base['base_model_version']!r} is not the certified "
            f"{pin['model_profile_id']!r}",
            calibration_artifact_id=artifact.calibration_artifact_id,
        )
    observed = production_config_identity(config, pin)
    expected = {"model_config_sha256": pin["evidence_model_config_sha256"],
                "sim_config_sha256": pin["evidence_sim_config_sha256"]}
    if observed != expected:
        return refuse(
            CALIBRATION_CONFIG_MISMATCH,
            f"shipped configuration identity {observed} is not the certified {expected}",
            calibration_artifact_id=artifact.calibration_artifact_id,
        )

    store = payload_store if payload_store is not None else PackagedPayloadStore()
    try:
        data = get_calibration_payload(
            store, key=artifact.object_uri, expected_sha256=artifact.payload_sha256,
            expected_byte_count=artifact.payload_byte_count,
        )
    except CalibrationPayloadMissingError as exc:
        return refuse(CALIBRATION_PAYLOAD_MISSING, str(exc),
                      calibration_artifact_id=artifact.calibration_artifact_id)
    except (CorruptCalibrationPayloadError, OSError) as exc:
        return refuse(CALIBRATION_PAYLOAD_CORRUPT, str(exc),
                      calibration_artifact_id=artifact.calibration_artifact_id)
    try:
        payload = deserialize_calibration_payload(data)
    except CalibrationPayloadError as exc:
        return refuse(CALIBRATION_PAYLOAD_CORRUPT, f"payload does not parse: {exc}",
                      calibration_artifact_id=artifact.calibration_artifact_id)
    algorithm = f"{payload.algorithm_family}/{payload.algorithm_version}"
    if (
        algorithm != pin["algorithm"]
        or algorithm != f"{ALGORITHM_FAMILY}/{ALGORITHM_VERSION}"
        or payload.feature_contract_version != FEATURE_CONTRACT_VERSION
        or not np.all(np.isfinite(np.asarray(payload.theta, dtype=np.float64)))
    ):
        return refuse(CALIBRATION_PAYLOAD_CORRUPT,
                      f"payload algorithm {algorithm!r} / feature contract "
                      f"{payload.feature_contract_version!r} is not the certified one",
                      calibration_artifact_id=artifact.calibration_artifact_id)

    calibrator = LiveCalibrator(
        calibration_artifact_id=artifact.calibration_artifact_id,
        payload_sha256=artifact.payload_sha256,
        model_profile=artifact.model_profile,
        model_profile_id=artifact.base_model_version,
        checkpoint_scope=artifact.checkpoint_scope,
        algorithm=algorithm,
        theta=tuple(float(t) for t in payload.theta),
        certification_run_id=str(pin["certification_run_id"]),
        fit_run_id=str(pin["fit_run_id"]),
    )
    return CalibrationResolution(
        status=CALIBRATION_RESOLVED, detail="certified champion verified", base=base,
        calibrator=calibrator, extra={"config_identity": observed},
    )


# ----------------------------------------------------------- application


def calibrated_draw_weights(
    result: GameSimulationResult, calibrator: LiveCalibrator, *, model_profile: ModelProfile
) -> np.ndarray:
    """The certified per-draw weights for THIS simulation; strict health."""
    require_profile_match(prediction_profile=model_profile.value,
                          calibrator_profile=calibrator.model_profile)
    if profile_base_model_version(result.model_version, model_profile) != (
        calibrator.model_profile_id
    ):
        raise LiveCalibrationError(
            f"simulation science {result.model_version!r} is not the calibrator's "
            f"{calibrator.model_profile_id!r}"
        )
    weights = softmax_weights(np.asarray(calibrator.theta, dtype=np.float64),
                              compute_draw_features(result))
    if not (
        weights.shape == (result.n_draws,)
        and np.all(np.isfinite(weights))
        and np.all(weights > 0)
        and abs(float(weights.sum()) - 1.0) <= 1e-12
    ):
        raise LiveCalibrationError(f"{result.game_id}: calibrated weights fail strict health")
    return weights


def calibrated_event_probability(
    values: np.ndarray, weights: np.ndarray, *, prop: PropType, side: str, line: float | None
) -> tuple[float, float] | None:
    """(win, push) probability of the identical priced event under
    `weights` -- the same event definitions as the raw pricer."""
    if side == _HIT_SIDE:
        if prop not in AT_LEAST_HIT_PROPS:
            return None
        return float(weights[values >= 1].sum()), 0.0
    if line is None:
        return None
    push = float(weights[values == line].sum())
    mask = values > line if side == "OVER" else values < line
    return float(weights[mask].sum()), push


def calibrate_priced_rows(
    result: GameSimulationResult,
    weights: np.ndarray,
    priced: list[dict[str, Any]],
    *,
    run_id: str,
    season: int,
    week: int,
    calibrator: LiveCalibrator,
) -> list[dict[str, Any]]:
    """One calibrated row per raw priced row (same `prediction_id`, same
    book quote, same event). Quote prices are read only to define the
    event and the downstream EV; nothing here feeds back into the model."""
    by_player: dict[str, np.ndarray] = {}
    out: list[dict[str, Any]] = []
    for row in priced:
        prop = PropType(str(row["prop_type"]))
        player_id = str(row["player_id"])
        key = f"{player_id}|{prop.value}"
        if key not in by_player:
            by_player[key] = prop_values(result, player_id, prop).astype(float)
        values = by_player[key]
        line = None if row.get("line") is None else float(row["line"])
        event = calibrated_event_probability(
            values, weights, prop=prop, side=str(row["side"]), line=line
        )
        if event is None:
            raise LiveCalibrationError(
                f"priced row {row['prediction_id']} has no calibrated event definition"
            )
        p_win, p_push = event
        odds = int(row["american_odds"])
        p_market = row.get("p_market_fair")
        out.append({
            "prediction_id": str(row["prediction_id"]),
            "run_id": run_id,
            "season": season,
            "week": week,
            "game_id": str(row["game_id"]),
            "player_id": player_id,
            "prop_type": prop.value,
            "market_type": str(row["market_type"]),
            "vendor": str(row["vendor"]),
            "side": str(row["side"]),
            "line": line,
            "american_odds": odds,
            "p_model_raw": float(row["p_model_raw"]),
            "p_model_calibrated": p_win,
            "p_push_calibrated": p_push,
            "model_fair_american_calibrated": fair_american_odds(p_win, p_push),
            "model_fair_decimal_calibrated": fair_decimal_odds(p_win, p_push),
            "p_market_fair": None if p_market is None else float(p_market),
            "edge_calibrated": None if p_market is None else p_win - float(p_market),
            "ev_per_unit_calibrated": expected_value(p_win, american_to_decimal(odds), p_push),
            "model_mean_calibrated": float(values @ weights),
            "model_mean_raw": float(values.mean()),
            "quote_available_at": row["quote_available_at"],
            "as_of": row["as_of"],
            "n_draws": int(result.n_draws),
            "model_version": result.model_version,
            "calibration_artifact_id": calibrator.calibration_artifact_id,
            "calibration_status": CALIBRATION_APPLIED,
        })
    return out


def application_record(
    *,
    run_id: str,
    status: str,
    calibrator: LiveCalibrator,
    weights: np.ndarray | None,
    priced_rows: int,
    calibrated_rows: int,
    detail: str | None,
    created_at: Any,
) -> pl.DataFrame:
    """The one per-run row persisting calibration identity and status."""
    ess = None if weights is None else float(1.0 / np.square(weights).sum())
    return pl.DataFrame([{
        "run_id": run_id,
        "calibration_status": status,
        "calibration_artifact_id": calibrator.calibration_artifact_id,
        "payload_sha256": calibrator.payload_sha256,
        "model_profile": calibrator.model_profile,
        "model_profile_id": calibrator.model_profile_id,
        "checkpoint_scope": calibrator.checkpoint_scope,
        "algorithm": calibrator.algorithm,
        "certification_run_id": calibrator.certification_run_id,
        "fit_run_id": calibrator.fit_run_id,
        "effective_sample_size": ess,
        "max_weight": None if weights is None else float(weights.max()),
        "priced_rows": priced_rows,
        "calibrated_rows": calibrated_rows,
        "detail": detail,
        "created_at": created_at,
    }], schema_overrides={"effective_sample_size": pl.Float64, "max_weight": pl.Float64,
                          "detail": pl.String})


def _classify(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:500]}"


@dataclass(frozen=True)
class LiveCalibrationStep:
    """The checkpoint flow's `post_pricing_step` for a resolved certified
    calibrator: reweight the SAME draws once, persist the calibrated
    product and the per-run application record. Never raises and never
    touches a raw artifact; any failure is persisted as APPLICATION_FAILED
    (which blocks PUBLIC_READY)."""

    calibrator: LiveCalibrator

    def __call__(
        self, ctx: Any, simulation: GameSimulationResult, priced: list[dict[str, Any]]
    ) -> tuple[str, str | None, int]:
        from datetime import UTC, datetime

        weights = None
        try:
            weights = calibrated_draw_weights(
                simulation, self.calibrator, model_profile=ctx.model_profile
            )
            rows = calibrate_priced_rows(
                simulation, weights, priced, run_id=ctx.run_id, season=ctx.season,
                week=ctx.week, calibrator=self.calibrator,
            )
            if rows:
                ctx.warehouse.append(
                    CALIBRATED_PRICES_TABLE, pl.DataFrame(rows, infer_schema_length=None),
                    key=["prediction_id"], keep="first",
                )
            status, detail, count = CALIBRATION_APPLIED, None, len(rows)
        except Exception as exc:
            status, detail, count = CALIBRATION_APPLICATION_FAILED, _classify(exc), 0
        try:
            ctx.warehouse.append(
                CALIBRATION_APPLICATIONS_TABLE,
                application_record(
                    run_id=ctx.run_id, status=status, calibrator=self.calibrator,
                    weights=weights, priced_rows=len(priced), calibrated_rows=count,
                    detail=detail, created_at=datetime.now(UTC),
                ),
                key=["run_id"], keep="first",
            )
        except Exception as exc:
            return CALIBRATION_APPLICATION_FAILED, _classify(exc), 0
        return status, detail, count
