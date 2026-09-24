"""Remote training platform harness (PLATFORM AUTOMATION).

This module is the only thing the `remote-training.yml` GitHub Actions
workflow invokes directly (`python -m nflprops.platform.remote_training run
...`). It validates the workflow's inputs, enforces the production
draw-count lock, verifies the checked-out git SHA matches the explicitly
requested `science_ref`, downloads and SHA-256-verifies the pinned training
data snapshot, and resolves/invokes the version-controlled Science entry
point -- never inventing a fallback, never promoting a challenger, and never
silently continuing on a failure.

`science_ref` is a git SHA in *this* repository: Platform and Science are
two branches of the same codebase, developed independently and consumed
through an explicit interface (this module), never copied or rewritten.
Until Platform and Science are integrated onto the same commit, checking
out a Science-only `science_ref` will not contain this module at all, and a
checkout of a Platform-only ref will not contain the Science entry point --
either way, `resolve_science_entrypoint` fails closed with a clear message
rather than pretending success. See docs/PLATFORM_AUTOMATION.md.
"""

from __future__ import annotations

import importlib
import json
import re
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

import typer

from nflprops.errors import NflpropsError

# Production is locked to this exact draw count (§3/§13 of the platform
# brief). Smoke/development mode may use fewer draws for a fast contract
# check, but MAX_SMOKE_N_DRAWS is deliberately far below PRODUCTION_N_DRAWS
# so smoke evidence can never be mistaken for -- or substituted as --
# production promotion evidence.
PRODUCTION_N_DRAWS = 20_000
MAX_SMOKE_N_DRAWS = 2_000

DEFAULT_SCIENCE_ENTRYPOINT = "nflprops.calibration.phase10c3a_runner"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class RemoteTrainingError(NflpropsError):
    """Base class for remote-training platform failures."""


class RemoteTrainingConfigError(RemoteTrainingError):
    """Raised when workflow inputs are missing or invalid. Fails closed --
    never falls back to an implicit/default science_ref or manifest."""


class ScienceRefMismatchError(RemoteTrainingError):
    """Raised when the actual checked-out git SHA does not equal the
    explicitly requested `science_ref`. Never trains against an
    unverified checkout."""


class ScienceEntrypointNotAvailableError(RemoteTrainingError):
    """Raised when the requested Science entry point is not importable (or
    has no `main`) at the checked-out `science_ref`. This is a legitimate,
    expected failure until Platform and Science are integrated -- see the
    module docstring -- and must fail the GitHub job, never continue with a
    fallback or partial result."""


class TrainingMode(str, Enum):  # noqa: UP042
    PRODUCTION = "production"
    SMOKE = "smoke"


def _require_sha(value: str | None, *, label: str, pattern: re.Pattern[str]) -> str:
    if not value or not value.strip():
        raise RemoteTrainingConfigError(f"{label} is required and was not provided")
    candidate = value.strip().lower()
    if not pattern.match(candidate):
        raise RemoteTrainingConfigError(
            f"{label} must be an explicit, full-length hex SHA; got {value!r}"
        )
    return candidate


@dataclass(frozen=True)
class RemoteTrainingRequest:
    """A validated, immutable remote-training invocation.

    Construct only via `from_inputs` -- never build this directly from
    unchecked workflow input strings.
    """

    science_ref: str
    data_manifest_sha256: str
    mode: TrainingMode
    entry_point: str = DEFAULT_SCIENCE_ENTRYPOINT
    requested_n_draws: int | None = None

    @classmethod
    def from_inputs(
        cls,
        *,
        science_ref: str | None,
        data_manifest_sha256: str | None,
        mode: str | None,
        entry_point: str | None = None,
        requested_n_draws: int | None = None,
    ) -> RemoteTrainingRequest:
        validated_science_ref = _require_sha(
            science_ref, label="science_ref", pattern=_SHA_RE
        )
        validated_manifest_sha = _require_sha(
            data_manifest_sha256, label="data_manifest_sha256", pattern=_MANIFEST_SHA_RE
        )
        if not mode or not mode.strip():
            raise RemoteTrainingConfigError("mode is required and was not provided")
        try:
            mode_enum = TrainingMode(mode.strip().lower())
        except ValueError:
            valid = [m.value for m in TrainingMode]
            raise RemoteTrainingConfigError(
                f"mode must be one of {valid}, got {mode!r}"
            ) from None
        if requested_n_draws is not None and requested_n_draws <= 0:
            raise RemoteTrainingConfigError(
                f"requested_n_draws must be positive, got {requested_n_draws}"
            )
        return cls(
            science_ref=validated_science_ref,
            data_manifest_sha256=validated_manifest_sha,
            mode=mode_enum,
            entry_point=(entry_point or DEFAULT_SCIENCE_ENTRYPOINT).strip(),
            requested_n_draws=requested_n_draws,
        )

    @property
    def n_draws(self) -> int:
        """The locked draw count for this request.

        Production mode ALWAYS returns exactly `PRODUCTION_N_DRAWS`,
        regardless of any requested value -- there is no input that can
        override this. Smoke mode is capped at `MAX_SMOKE_N_DRAWS`, strictly
        below production, so a smoke run structurally cannot masquerade as a
        production run by requesting a high draw count.
        """
        if self.mode is TrainingMode.PRODUCTION:
            return PRODUCTION_N_DRAWS
        requested = (
            self.requested_n_draws
            if self.requested_n_draws is not None
            else MAX_SMOKE_N_DRAWS
        )
        if requested > MAX_SMOKE_N_DRAWS:
            raise RemoteTrainingConfigError(
                f"smoke mode n_draws must be <= {MAX_SMOKE_N_DRAWS} "
                f"(production promotion evidence requires mode=production), got {requested}"
            )
        return requested

    @property
    def promotion_evidence_eligible(self) -> bool:
        """Whether this run's output may ever be used as promotion
        evidence. Structurally False for every non-production mode --
        this is checked independently of whatever the Science entry point
        itself reports, so a smoke run can never produce promotion evidence
        even if the entry point's own result claims otherwise."""
        return self.mode is TrainingMode.PRODUCTION


def verify_checked_out_sha(
    expected_sha: str,
    *,
    actual_sha_provider: Callable[[], str] | None = None,
    repo_root: Path | None = None,
) -> str:
    """Verify the actual checked-out HEAD SHA equals `expected_sha` exactly.

    `actual_sha_provider` defaults to `git rev-parse HEAD` in `repo_root`
    (or the current directory); tests inject a stub instead of requiring a
    real git checkout.
    """
    if actual_sha_provider is None:

        def actual_sha_provider() -> str:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

    actual = actual_sha_provider().strip().lower()
    expected = expected_sha.strip().lower()
    if actual != expected:
        raise ScienceRefMismatchError(
            f"checked-out HEAD ({actual}) does not match the requested "
            f"science_ref ({expected}) -- refusing to train against an "
            "unverified checkout"
        )
    return actual


# The Phase 10C3A CLI's own `parse_args` defaults (see
# nflprops/calibration/phase10c3a_runner.py::parse_args) -- mirrored here so
# a structured-runner invocation from this module is configured identically
# to `python -m nflprops.calibration.phase10c3a_runner` with only
# --data-root/--output-dir/--n-draws/--mode/--expect-data-manifest-sha256
# supplied. Argument-mapping glue only; no science default is invented here.
_STRUCTURED_RUNNER_SEASON_MIN = 2022
_STRUCTURED_RUNNER_SEASON_MAX = 2025
_STRUCTURED_RUNNER_MODEL_VERSION = "phase10c3a-real-run-v1"
_STRUCTURED_RUNNER_REGULARIZATION_LAMBDA = 0.01
_STRUCTURED_RUNNER_MAX_FIT_ITERATIONS = 200


def _adapt_structured_runner(
    module: Any, runner_config_cls: Any, structured_run: Callable[[Any], dict[str, Any]]
) -> Callable[..., dict[str, Any]]:
    """Wrap the Phase 10C3A `RunnerConfig` + `run(config) -> dict` interface
    (the version-controlled Science entry point) in this module's plain
    kwargs entrypoint contract. Pure argument mapping: constructs the same
    `RunnerConfig` the Science CLI would build from equivalent flags and
    calls the unmodified `run()`; no science calculation is reimplemented or
    altered here.
    """

    def adapter(
        *,
        science_ref: str,
        data_manifest_sha256: str | None = None,
        data_dir: Path | None = None,
        n_draws: int | None = None,
        mode: str | None = None,
        promotion_evidence_eligible: bool | None = None,
        output_dir: Path | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if data_dir is None:
            raise RemoteTrainingConfigError(
                "structured Science runner requires a prepared data_dir "
                "(the verified downloaded data snapshot) to use as "
                "--data-root"
            )
        resolved_output_dir = (
            Path(output_dir) if output_dir is not None else Path(data_dir) / "phase10c3a-output"
        )
        config = runner_config_cls(
            data_root=Path(data_dir),
            output_dir=resolved_output_dir,
            season_min=_STRUCTURED_RUNNER_SEASON_MIN,
            season_max=_STRUCTURED_RUNNER_SEASON_MAX,
            n_draws=n_draws,
            mode=mode,
            model_version=_STRUCTURED_RUNNER_MODEL_VERSION,
            regularization_lambda=_STRUCTURED_RUNNER_REGULARIZATION_LAMBDA,
            max_fit_iterations=_STRUCTURED_RUNNER_MAX_FIT_ITERATIONS,
            expect_data_manifest_sha256=data_manifest_sha256,
        )
        result = structured_run(config)

        # Same filename/location the Science CLI's own `main()` writes to
        # (`config.output_dir / "phase10c3a_report.json"`), so the
        # machine-readable report lands in the same place whether the
        # runner is invoked via CLI or via this structured call.
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        report_path = resolved_output_dir / "phase10c3a_report.json"
        report_path.write_text(json.dumps(result, indent=2, default=str))

        registration = result.get("registration") or {}
        return {
            "model_version": result.get("model_version"),
            "challenger_payload_hash": registration.get("payload_sha256"),
            "validation_result": {
                "real_game_coherence_check": result.get("real_game_coherence_check"),
                "reproducibility_check": result.get("reproducibility_check"),
            },
            "promotion_eligibility_result": {
                "eligible": result.get("promotion_decision") == "ELIGIBLE_FOR_PROMOTION",
                "decision": result.get("promotion_decision"),
                "overall_promotion_gate": result.get("overall_promotion_gate"),
            },
            "report_path": str(report_path),
        }

    return adapter


def resolve_science_entrypoint(entry_point: str) -> Callable[..., dict[str, Any]]:
    """Import `entry_point` and return a callable matching this module's
    entrypoint contract: `main(*, science_ref, data_manifest_sha256,
    data_dir, n_draws, mode, promotion_evidence_eligible, ...) -> dict`.

    Two supported Science module shapes:

    1. The version-controlled Phase 10C3A structured interface --
       `RunnerConfig` + `run(config) -> dict` -- adapted into this module's
       kwargs contract by `_adapt_structured_runner`. This is the shape
       `nflprops.calibration.phase10c3a_runner` actually exposes; its own
       `main(argv) -> int` is a separate, CLI-only convenience wrapper and
       is never called from here.
    2. A module that already exposes `main(**kwargs) -> dict` directly,
       matching this module's contract natively (e.g. a future Science
       entry point, or a test double).

    Fails closed with `ScienceEntrypointNotAvailableError` -- never returns
    a stand-in, never silently skips training. This is the only place this
    module is allowed to import Science code, and it does so purely by
    name/string, never a hardcoded `from nflprops.calibration import ...`,
    so this module compiles and is testable even when the named entry point
    does not exist on this checkout.
    """
    try:
        module = importlib.import_module(entry_point)
    except ImportError as exc:
        raise ScienceEntrypointNotAvailableError(
            f"Science entry point {entry_point!r} is not importable on this "
            "checkout. This is expected until Platform and Science are "
            "integrated onto the same commit -- see "
            "docs/PLATFORM_AUTOMATION.md. Failing the job rather than "
            "pretending training happened."
        ) from exc

    runner_config_cls = getattr(module, "RunnerConfig", None)
    structured_run = getattr(module, "run", None)
    if runner_config_cls is not None and callable(structured_run):
        return _adapt_structured_runner(module, runner_config_cls, structured_run)

    main = getattr(module, "main", None)
    if main is None or not callable(main):
        raise ScienceEntrypointNotAvailableError(
            f"Science entry point {entry_point!r} has no callable `main()`"
        )
    return cast("Callable[..., dict[str, Any]]", main)


@dataclass
class RemoteTrainingRunReport:
    """The machine-readable run report published as a workflow artifact
    (§8/§15 of the platform brief). Written to disk unconditionally --
    including on failure -- so the workflow's `upload-artifact` step (which
    runs with `if: always()`) always has something to publish."""

    repository: str
    science_sha: str
    workflow_sha: str
    data_manifest_sha256: str
    mode: str
    n_draws: int
    runner_identity: str
    start_timestamp: str
    promotion_evidence_eligible: bool
    end_timestamp: str | None = None
    exit_status: str = "RUNNING"
    model_version: str | None = None
    config_version: str | None = None
    challenger_payload_hash: str | None = None
    validation_result: dict[str, Any] | None = None
    promotion_eligibility_result: dict[str, Any] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "science_sha": self.science_sha,
            "workflow_sha": self.workflow_sha,
            "data_manifest_sha256": self.data_manifest_sha256,
            "mode": self.mode,
            "n_draws": self.n_draws,
            "runner_identity": self.runner_identity,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "exit_status": self.exit_status,
            "promotion_evidence_eligible": self.promotion_evidence_eligible,
            "model_version": self.model_version,
            "config_version": self.config_version,
            "challenger_payload_hash": self.challenger_payload_hash,
            "validation_result": self.validation_result,
            "promotion_eligibility_result": self.promotion_eligibility_result,
            "error": self.error,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)
        )


def _runner_identity() -> str:
    return socket.gethostname()


# NOTE for reviewers and future maintainers: this module must never import
# or call `nflprops.calibration.registry.promote_calibration_champion` (or
# any other champion-promotion function). Champion promotion stays a
# separate, explicit, human-gated action -- see docs/PLATFORM_AUTOMATION.md
# and tests/platform/test_remote_training.py::test_module_never_references_promotion.
def execute_remote_training(
    request: RemoteTrainingRequest,
    *,
    repository: str,
    workflow_sha: str,
    report_path: Path,
    actual_sha_provider: Callable[[], str] | None = None,
    entrypoint_resolver: Callable[
        [str], Callable[..., dict[str, Any]]
    ] = resolve_science_entrypoint,
    prepare_data: Callable[[], Path] | None = None,
    output_dir: Path | None = None,
) -> RemoteTrainingRunReport:
    """Run one remote-training invocation end to end and always write a
    report. Raises on any failure (after writing the failure report) so the
    caller's process exits non-zero -- see `run` below."""
    report = RemoteTrainingRunReport(
        repository=repository,
        science_sha=request.science_ref,
        workflow_sha=workflow_sha,
        data_manifest_sha256=request.data_manifest_sha256,
        mode=request.mode.value,
        n_draws=request.n_draws,
        runner_identity=_runner_identity(),
        start_timestamp=datetime.now(UTC).isoformat(),
        promotion_evidence_eligible=request.promotion_evidence_eligible,
    )
    try:
        verify_checked_out_sha(
            request.science_ref, actual_sha_provider=actual_sha_provider
        )

        data_dir = prepare_data() if prepare_data is not None else None

        main = entrypoint_resolver(request.entry_point)
        result = main(
            science_ref=request.science_ref,
            data_manifest_sha256=request.data_manifest_sha256,
            data_dir=data_dir,
            n_draws=request.n_draws,
            mode=request.mode.value,
            promotion_evidence_eligible=request.promotion_evidence_eligible,
            output_dir=output_dir,
        )
        if not isinstance(result, dict):
            raise RemoteTrainingError(
                f"Science entry point {request.entry_point!r} must return a dict, "
                f"got {type(result).__name__}"
            )

        report.model_version = result.get("model_version")
        report.config_version = result.get("config_version")
        report.challenger_payload_hash = result.get("challenger_payload_hash")
        report.validation_result = result.get("validation_result")
        # Even if the entry point's own result claims eligibility, the
        # report never reports eligibility beyond what this request
        # structurally allows (§ promotion_evidence_eligible above).
        promotion_result = result.get("promotion_eligibility_result")
        if promotion_result is not None and not request.promotion_evidence_eligible:
            promotion_result = {
                **promotion_result,
                "eligible": False,
                "reason": "mode is not production; promotion evidence is structurally ineligible",
            }
        report.promotion_eligibility_result = promotion_result
        report.exit_status = "SUCCEEDED"
        return report
    except Exception as exc:
        report.exit_status = "FAILED"
        report.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report.end_timestamp = datetime.now(UTC).isoformat()
        report.write(report_path)


app = typer.Typer(
    name="remote-training",
    help="Remote training platform harness invoked by .github/workflows/remote-training.yml.",
    no_args_is_help=True,
)


@app.command("run")
def run(
    science_ref: str = typer.Option(
        ..., help="Explicit 40-character git SHA to check out and run."
    ),
    data_manifest_sha256: str = typer.Option(
        ..., help="Explicit SHA-256 of the pinned training data manifest."
    ),
    mode: str = typer.Option(..., help="production or smoke."),
    entry_point: str = typer.Option(
        DEFAULT_SCIENCE_ENTRYPOINT,
        help="Dotted module path of the Science entry point to invoke.",
    ),
    smoke_n_draws: int | None = typer.Option(
        None,
        help=f"Draw count for smoke mode only (<= {MAX_SMOKE_N_DRAWS}). Ignored in production mode.",
    ),
    report_path: str = typer.Option(
        "remote_training_report.json", help="Where to write the run report JSON."
    ),
    repository: str = typer.Option(
        ..., help="owner/repo, from the GitHub Actions context."
    ),
    workflow_sha: str = typer.Option(
        ..., help="The commit SHA the workflow file itself ran from."
    ),
    data_dir: str | None = typer.Option(
        None, help="Ephemeral workspace already populated by `prepare-data`, if any."
    ),
    output_dir: str | None = typer.Option(
        None,
        help=(
            "Directory for the Science entry point's own machine-readable "
            "report + logs (its --output-dir, for a structured Science "
            "runner). Defaults to a subdirectory of data_dir."
        ),
    ),
) -> None:
    """Validate inputs, verify the checkout, and invoke the Science entry
    point. Exits non-zero on any failure -- never silently continues."""
    try:
        request = RemoteTrainingRequest.from_inputs(
            science_ref=science_ref,
            data_manifest_sha256=data_manifest_sha256,
            mode=mode,
            entry_point=entry_point,
            requested_n_draws=smoke_n_draws,
        )
    except RemoteTrainingConfigError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc

    resolved_report_path = Path(report_path)
    resolved_data_dir = Path(data_dir) if data_dir is not None else None
    resolved_output_dir = Path(output_dir) if output_dir is not None else None
    try:
        execute_remote_training(
            request,
            repository=repository,
            workflow_sha=workflow_sha,
            report_path=resolved_report_path,
            prepare_data=(lambda: resolved_data_dir)
            if resolved_data_dir is not None
            else None,
            output_dir=resolved_output_dir,
        )
    except RemoteTrainingError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"SUCCEEDED: report written to {resolved_report_path}")


@app.command("prepare-data")
def prepare_data_command(
    manifest_object_key: str = typer.Option(
        ..., help="Object-store key of the manifest.json describing the snapshot."
    ),
    data_manifest_sha256: str = typer.Option(
        ..., help="Expected SHA-256 of the manifest (must match the dispatch input)."
    ),
    workspace: str = typer.Option(
        ..., help="Ephemeral local directory to download the verified snapshot into."
    ),
) -> None:
    """Download the manifest, verify its identity and every object's
    SHA-256, and populate `workspace`. Never mutates the source objects.
    Exits non-zero -- and leaves nothing claimed as verified -- on any
    mismatch."""
    from nflprops.data.storage.settings import (
        StorageSettings,
        build_object_store_client,
    )
    from nflprops.platform.data_snapshot import (
        DataSnapshotError,
        DataSnapshotManifest,
        download_and_verify_snapshot,
        verify_manifest_identity,
    )

    try:
        settings = StorageSettings.from_env()
        client = build_object_store_client(settings)
        manifest_bytes = client.get_bytes(manifest_object_key)
        manifest = DataSnapshotManifest.from_dict(json.loads(manifest_bytes))
        verify_manifest_identity(manifest, expected_sha256=data_manifest_sha256)
        written = download_and_verify_snapshot(
            client, manifest, dest_dir=Path(workspace)
        )
    except DataSnapshotError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"SUCCEEDED: {len(written)} object(s) verified into {workspace}")


@app.command("cleanup-data")
def cleanup_data_command(
    workspace: str = typer.Option(..., help="The ephemeral directory to remove."),
) -> None:
    """Remove the ephemeral download workspace. Always run this, even on
    failure -- the workflow calls it with `if: always()`."""
    from nflprops.platform.data_snapshot import cleanup_workspace

    cleanup_workspace(Path(workspace))
    typer.echo(f"SUCCEEDED: removed {workspace}")


if __name__ == "__main__":
    app()
