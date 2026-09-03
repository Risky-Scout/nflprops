"""Real historical SPEC §67 reproduction builder.

This script performs one genuine historical prediction checkpoint. It is invoked
twice by nflprops.backtest.reproduce; the first derived output is deleted before
the second invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from nflprops.config import load
from nflprops.pipelines.lean import open_warehouse
from nflprops.pipelines.pregame import (
    predict_week,
    simulation_config_from_app_config,
    state_configs_from_app_config,
)


EXPECTED_DATA_MANIFEST_SHA256 = (
    "1c823c1884eb7d27e35b9e13d6607b7f"
    "b373c0d35baf1ad3e1ec751383f083af"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text()
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object: {path}"
        )

    return payload


def safe_path(
    root: Path,
    relative: str,
) -> Path:
    root = root.resolve()
    candidate = (
        root / relative
    ).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"path escapes verified root: {relative}"
        ) from exc

    return candidate


def verify_hash_entries(
    manifest: dict[str, Any],
    *,
    root: Path,
) -> None:
    files = manifest.get("files")

    if not isinstance(files, list):
        raise ValueError(
            "hash manifest files must be a list"
        )

    for raw in files:
        if not isinstance(raw, dict):
            raise ValueError(
                "hash manifest entry must be an object"
            )

        relative = raw.get("path")
        expected = raw.get("sha256")

        if not isinstance(relative, str):
            raise ValueError(
                "hash manifest path must be a string"
            )

        if not isinstance(expected, str):
            raise ValueError(
                "hash manifest sha256 must be a string"
            )

        path = safe_path(
            root,
            relative,
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"verified input missing: {relative}"
            )

        actual = sha256_file(path)

        if actual != expected:
            raise ValueError(
                "verified input hash mismatch: "
                f"{relative}"
            )


def verify_source_and_config(
    run_dir: Path,
    repo_root: Path,
) -> None:
    source_manifest = load_object(
        run_dir / "source_manifest.json"
    )

    verify_hash_entries(
        source_manifest,
        root=repo_root,
    )

    config_manifest = load_object(
        run_dir / "config_manifest.json"
    )

    verify_hash_entries(
        config_manifest,
        root=repo_root,
    )

    checkpoint_sha = config_manifest.get(
        "checkpoint_sha256"
    )

    if not isinstance(
        checkpoint_sha,
        str,
    ):
        raise ValueError(
            "config manifest checkpoint_sha256 missing"
        )

    if (
        sha256_file(
            run_dir / "checkpoint.json"
        )
        != checkpoint_sha
    ):
        raise ValueError(
            "checkpoint hash mismatch"
        )


def verify_canonical_snapshot(
    run_dir: Path,
    repo_root: Path,
) -> None:
    manifest_path = (
        run_dir
        / "data_manifest.json"
    )

    if (
        sha256_file(manifest_path)
        != EXPECTED_DATA_MANIFEST_SHA256
    ):
        raise ValueError(
            "data manifest hash mismatch"
        )

    manifest = load_object(
        manifest_path
    )

    files = manifest.get("files")

    if not isinstance(files, list):
        raise ValueError(
            "data manifest files must be a list"
        )

    canonical = (
        repo_root
        / "data"
        / "canonical"
    ).resolve()

    expected_paths: set[str] = set()

    for raw in files:
        if not isinstance(raw, dict):
            raise ValueError(
                "data manifest entry must be an object"
            )

        relative = raw.get("path")
        expected_sha = raw.get("sha256")
        expected_size = raw.get("size")

        if not isinstance(relative, str):
            raise ValueError(
                "data path must be a string"
            )

        if not isinstance(expected_sha, str):
            raise ValueError(
                "data sha256 must be a string"
            )

        if not isinstance(expected_size, int):
            raise ValueError(
                "data size must be an integer"
            )

        path = safe_path(
            canonical,
            relative,
        )

        expected_paths.add(relative)

        if not path.is_file():
            raise FileNotFoundError(
                f"canonical input missing: {relative}"
            )

        if path.stat().st_size != expected_size:
            raise ValueError(
                f"canonical input size mismatch: {relative}"
            )

        if sha256_file(path) != expected_sha:
            raise ValueError(
                f"canonical input hash mismatch: {relative}"
            )

    actual_paths = {
        path.relative_to(
            canonical
        ).as_posix()
        for path in canonical.rglob("*")
        if path.is_file()
    }

    if actual_paths != expected_paths:
        missing = sorted(
            expected_paths - actual_paths
        )
        extra = sorted(
            actual_paths - expected_paths
        )

        raise ValueError(
            "canonical snapshot file-set mismatch; "
            f"missing={missing} extra={extra}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    parser.add_argument(
        "--experiment-manifest",
        required=True,
    )

    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    run_dir = Path(
        args.run_dir
    ).resolve()

    experiment_manifest = Path(
        args.experiment_manifest
    ).resolve()

    if not experiment_manifest.is_file():
        raise FileNotFoundError(
            experiment_manifest
        )

    verify_source_and_config(
        run_dir,
        repo_root,
    )

    verify_canonical_snapshot(
        run_dir,
        repo_root,
    )

    checkpoint = load_object(
        run_dir / "checkpoint.json"
    )

    db_path = (
        repo_root
        / "data"
        / "nflprops.duckdb"
    )

    if db_path.exists():
        db_path.unlink()

    cfg = load()

    player_state_config, team_state_config = (
        state_configs_from_app_config(
            cfg
        )
    )

    n_draws = int(
        checkpoint["n_draws"]
    )

    as_of = __import__(
        "datetime"
    ).datetime.fromisoformat(
        str(
            checkpoint["as_of"]
        )
    )

    game_ids_raw = checkpoint[
        "game_ids"
    ]

    if not isinstance(
        game_ids_raw,
        list,
    ):
        raise ValueError(
            "checkpoint game_ids must be a list"
        )

    game_ids = {
        str(value)
        for value in game_ids_raw
    }

    predictions = predict_week(
        open_warehouse(cfg),
        season=int(
            checkpoint["season"]
        ),
        week=int(
            checkpoint["week"]
        ),
        as_of=as_of,
        model_version=str(
            checkpoint["model_version"]
        ),
        n_draws=n_draws,
        retain_joint_draws=0,
        simulation_config=(
            simulation_config_from_app_config(
                cfg,
                n_draws=n_draws,
            )
        ),
        player_state_config=(
            player_state_config
        ),
        team_state_config=(
            team_state_config
        ),
        max_confidence_tier=int(
            checkpoint[
                "max_confidence_tier"
            ]
        ),
        market_mode="opening",
        persist=False,
        game_ids=game_ids,
    )

    if predictions.is_empty():
        raise RuntimeError(
            "genuine historical checkpoint produced zero predictions"
        )

    required = {
        "prediction_id",
        "p_model_raw",
        "p_push",
        "p_market_fair",
    }

    missing = sorted(
        required
        - set(
            predictions.columns
        )
    )

    if missing:
        raise RuntimeError(
            "historical prediction output missing "
            + ", ".join(missing)
        )

    output = (
        predictions.select(
            "prediction_id",
            "p_model_raw",
            "p_push",
            "p_market_fair",
        )
        .sort(
            "prediction_id"
        )
    )

    if (
        output["prediction_id"]
        .n_unique()
        != output.height
    ):
        raise RuntimeError(
            "historical prediction_id values are not unique"
        )

    output_path = (
        run_dir
        / "derived"
        / "probabilities.parquet"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_parquet(
        output_path
    )

    print(
        "HISTORICAL_BUILD=PASS "
        f"rows={output.height} "
        f"checkpoint={checkpoint['checkpoint_id']} "
        f"draws={n_draws}"
    )


if __name__ == "__main__":
    main()
