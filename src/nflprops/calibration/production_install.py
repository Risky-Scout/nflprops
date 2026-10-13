"""Install the certified production calibrator into the LIVE registry and
promote it to the STRUCTURAL_CORE champion (Wizard runtime owner only;
`wizard_runtime calibrator-promote`, ops `calibrator-promote`).

The only calibrator this can ever install is a pin in
`contracts/production_calibrators/certified.json`. Fail-closed sequence:

1. the shipped registry rows (the immutable `registry_warehouse/` of the
   certifying GitHub fit run) re-hash to the pinned SHA-256s, and the
   shipped payload re-hashes to the pinned payload SHA / byte count;
2. the rows describe exactly the pinned artifact (id, payload, profile,
   profile id, scope, contract versions) and one validation that passes
   every promotion gate;
3. under the writer lock: `register_calibration_artifact` (recomputes the
   artifact's scientific content hash; an identical retry is a no-op), the
   certified validation row is inserted unless already identical, then
   `approve_calibration_artifact` + `promote_calibration_champion` -- the
   registry's own gates;
4. the live registry then resolves THIS artifact for STRUCTURAL_CORE and
   no champion for LIVE_ENHANCED.

Dry-run (steps 1-2 only) unless `apply`. Idempotent: an already-promoted
champion is left exactly as it is.
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.calibration.joint_feature_contract import FEATURE_CONTRACT_VERSION
from nflprops.calibration.live_application import (
    PackagedPayloadStore,
    certified_calibrators_dir,
    load_certified_calibrators,
)
from nflprops.calibration.payload_store import get_calibration_payload
from nflprops.data.warehouse import Warehouse
from nflprops.domain.model_profile import ModelProfile, profile_base_model_version
from nflprops.errors import NflpropsError
from nflprops.platform.remote_checkpoint import (
    CALIBRATION_CONTRACT_VERSION,
    CALIBRATION_SCOPE_TYPE,
    PROP_CONTRACT_VERSION,
    SIMULATION_CONFIG_VERSION,
)

REGISTRY_TABLES = (
    "calibration_artifacts",
    "calibration_validations",
    "calibration_lifecycle_events",
)
REASON_CODE = "CERTIFIED_PRODUCTION_LAUNCH"


class CalibratorInstallError(NflpropsError):
    """The certified calibrator cannot be installed; nothing was written."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_certified_bundle(calibrator_id: str) -> dict[str, Any]:
    """Steps 1-2: read-only verification of the shipped certified evidence.
    Returns the loaded artifact / validation and the pin."""
    from nflprops.calibration.contract import load_calibration_registry_contract
    from nflprops.orchestration.calibration_store import (
        load_artifact,
        load_validations,
    )
    from nflprops.orchestration.dispatch_plan import as_run_store_backend

    pin = load_certified_calibrators().get(calibrator_id)
    if pin is None:
        raise CalibratorInstallError(f"{calibrator_id} is not a certified production calibrator")
    rows_pin = pin.get("registry_rows") or {}
    root = certified_calibrators_dir() / "registry"
    for table in REGISTRY_TABLES:
        path = root / f"{table}.parquet"
        if not path.is_file() or _sha(path) != rows_pin.get(f"{table}.parquet"):
            raise CalibratorInstallError(f"shipped {table} rows do not match the certified pin")
    get_calibration_payload(
        PackagedPayloadStore(), key=pin["object_uri"], expected_sha256=pin["payload_sha256"],
        expected_byte_count=int(pin["payload_byte_count"]),
    )

    with tempfile.TemporaryDirectory() as scratch:
        staging = Warehouse(Path(scratch) / "registry")
        for table in REGISTRY_TABLES:
            staging.write(table, pl.read_parquet(root / f"{table}.parquet"))
        backend = as_run_store_backend(staging)
        artifact = load_artifact(backend, calibrator_id)
        validations = load_validations(backend, calibrator_id)
    if artifact is None or len(validations) != 1:
        raise CalibratorInstallError("certified rows must hold exactly the artifact + one validation")
    validation = validations[0]
    expected = {
        "calibration_artifact_id": calibrator_id,
        "payload_sha256": pin["payload_sha256"],
        "payload_byte_count": int(pin["payload_byte_count"]),
        "object_uri": pin["object_uri"],
        "scientific_content_sha256": pin["scientific_content_sha256"],
        "model_profile": ModelProfile.STRUCTURAL_CORE.value,
        "base_model_version": pin["model_profile_id"],
        "scope_type": CALIBRATION_SCOPE_TYPE,
        "checkpoint_scope": pin["checkpoint_scope"],
        "simulation_config_version": SIMULATION_CONFIG_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "prop_contract_version": PROP_CONTRACT_VERSION,
        "calibration_contract_version": CALIBRATION_CONTRACT_VERSION,
    }
    for attr, value in expected.items():
        if getattr(artifact, attr) != value:
            raise CalibratorInstallError(
                f"certified artifact {attr}={getattr(artifact, attr)!r} is not {value!r}")
    gates = load_calibration_registry_contract().promotion_required_gates
    failing = [g for g in gates if not getattr(validation, g)]
    if failing:
        raise CalibratorInstallError(f"certified validation fails promotion gates {failing}")
    return {"pin": pin, "artifact": artifact, "validation": validation}


def install_and_promote(
    warehouse: Warehouse,
    *,
    calibrator_id: str,
    lock_path: Path,
    now: datetime,
    apply: bool,
    model_version: str,
) -> dict[str, Any]:
    from nflprops.calibration.registry import (
        approve_calibration_artifact,
        promote_calibration_champion,
        register_calibration_artifact,
        resolve_calibration_champion,
    )
    from nflprops.orchestration.calibration_store import (
        insert_validation,
        load_validation,
    )
    from nflprops.orchestration.dispatch_plan import as_run_store_backend
    from nflprops.platform.writer_lock import WriterLock

    verified = verify_certified_bundle(calibrator_id)
    artifact, validation, pin = verified["artifact"], verified["validation"], verified["pin"]
    if profile_base_model_version(model_version, ModelProfile.STRUCTURAL_CORE) != (
        pin["model_profile_id"]
    ):
        raise CalibratorInstallError(
            f"shipped model.version {model_version!r} is not the certified science")

    def resolve(backend: Any, profile: ModelProfile) -> str | None:
        found = resolve_calibration_champion(
            backend, scope_type=CALIBRATION_SCOPE_TYPE, checkpoint_scope=pin["checkpoint_scope"],
            model_profile=profile.value,
            base_model_version=profile_base_model_version(model_version, profile),
            simulation_config_version=SIMULATION_CONFIG_VERSION,
            feature_contract_version=FEATURE_CONTRACT_VERSION,
            prop_contract_version=PROP_CONTRACT_VERSION,
            calibration_contract_version=CALIBRATION_CONTRACT_VERSION,
            payload_store=PackagedPayloadStore(),
        )
        return None if found is None else found.calibration_artifact_id

    summary: dict[str, Any] = {
        "calibrator_id": calibrator_id, "payload_sha256": pin["payload_sha256"],
        "model_profile": artifact.model_profile, "model_profile_id": artifact.base_model_version,
        "validation_id": validation.validation_id, "certification_run_id":
        pin["certification_run_id"], "fit_run_id": pin["fit_run_id"],
        "verified": True, "applied": False,
    }
    backend = as_run_store_backend(warehouse)
    summary["champion_before"] = resolve(backend, ModelProfile.STRUCTURAL_CORE)
    if not apply:
        return {**summary, "dry_run": True}

    with WriterLock(lock_path, timeout_seconds=60.0):
        backend = as_run_store_backend(warehouse)
        if resolve(backend, ModelProfile.STRUCTURAL_CORE) != calibrator_id:
            register_calibration_artifact(backend, artifact)
            stored = load_validation(backend, validation.validation_id)
            if stored is None:
                insert_validation(backend, validation)
            elif stored != validation:
                raise CalibratorInstallError("a different validation row already holds this id")
            approve_calibration_artifact(
                backend, calibration_artifact_id=calibrator_id,
                validation_id=validation.validation_id, reason_code=REASON_CODE, created_at=now)
            promote_calibration_champion(
                backend, calibration_artifact_id=calibrator_id,
                validation_id=validation.validation_id, reason_code=REASON_CODE, created_at=now)
            summary["applied"] = True
        champion = resolve(backend, ModelProfile.STRUCTURAL_CORE)
        enhanced = resolve(backend, ModelProfile.LIVE_ENHANCED)
    if champion != calibrator_id or enhanced is not None:
        raise CalibratorInstallError(
            f"post-promotion resolution failed: STRUCTURAL_CORE={champion} "
            f"LIVE_ENHANCED={enhanced}")
    return {**summary, "champion_after": champion, "live_enhanced_champion": enhanced,
            "champion_compatibility": "PASS"}
