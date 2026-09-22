"""PHASE 10C1 regression locks: the calibration artifact REGISTRY must not
have touched certified Phase-9 pricing, Phase-10B raw PMFs, live checkpoint
sequencing, or the (still-unwired) scalar calibrators. No calibration
mathematics, no draw weights, no calibrated PMF, no live
`p_model_calibrated` population were added by this phase.

Full re-execution of the Phase-7/8/9/10B test suites (proving those
products themselves still behave identically) is covered by the ordinary
full-suite run, not duplicated here -- this file only locks the specific
NEW boundary claims Phase 10C1 makes about not touching those systems.
"""

from __future__ import annotations

import ast
import inspect

import pytest


def test_scalar_calibrator_still_supports_exactly_the_original_three_methods() -> None:
    """`ProbabilityCalibrator` (PHASE 8) is untouched: still exactly
    logistic/beta/isotonic, no new algorithm was added by PHASE 10C1."""
    from nflprops.calibration.calibrators import ProbabilityCalibrator

    for method in ("logistic", "beta", "isotonic"):
        ProbabilityCalibrator(method)
    with pytest.raises(ValueError):
        ProbabilityCalibrator("entropy_tilt")


def test_oof_calibrate_is_still_only_called_from_backtest_historical() -> None:
    """`prequential_oof_calibrate` (PHASE 8) must still never be imported
    by any live prediction/checkpoint path -- only by the offline backtest
    fold adapter, exactly as the PHASE 10C-A audit found."""
    import nflprops.backtest.historical as historical_module
    import nflprops.orchestration.flows.checkpoints as checkpoints_module
    import nflprops.pipelines.pregame as pregame_module

    assert "prequential_oof_calibrate" in inspect.getsource(historical_module)
    assert "prequential_oof_calibrate" not in inspect.getsource(checkpoints_module)
    assert "prequential_oof_calibrate" not in inspect.getsource(pregame_module)


def test_current_pricing_still_hardcodes_p_model_calibrated_none() -> None:
    """PHASE 10C1 added no live calibration math and populated no live
    `p_model_calibrated` value -- `_price_quote` must still hardcode it to
    `None` with the same PHASE 9B comment marking it deliberately blank."""
    import nflprops.market.current_pricing as current_pricing_module

    source = inspect.getsource(current_pricing_module)
    assert '"p_model_calibrated": None' in source
    assert "Deliberately blank until an OOF calibrator is fitted" in source


def test_checkpoints_module_has_no_calibration_registry_import_or_reference() -> None:
    """PHASE 10C1 does not modify live checkpoint sequencing: the
    checkpoint flow module must not import or reference the calibration
    registry/artifact/store modules at all."""
    import nflprops.orchestration.flows.checkpoints as checkpoints_module

    tree = ast.parse(inspect.getsource(checkpoints_module))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    calibration_registry_imports = {
        m for m in imported_modules if "calibration" in m
    }
    assert calibration_registry_imports == set(), (
        f"checkpoints.py must not import calibration modules yet, found: "
        f"{calibration_registry_imports}"
    )


def test_checkpoints_module_source_contains_no_calibration_champion_reference() -> None:
    import nflprops.orchestration.flows.checkpoints as checkpoints_module

    source = inspect.getsource(checkpoints_module)
    for forbidden in (
        "resolve_calibration_champion",
        "calibration_artifact",
        "calibrated_pmf",
        "draw_weight",
    ):
        assert forbidden not in source, f"unexpected calibration reference: {forbidden!r}"


def test_distribution_store_module_has_no_calibrated_probability_column() -> None:
    """PHASE 10B raw PMF outcomes must remain immutable and uncalibrated:
    no `p_calibrated`-style column was added to the distribution store."""
    import nflprops.orchestration.distribution_store as distribution_store_module

    source = inspect.getsource(distribution_store_module)
    assert "p_calibrated" not in source
    assert "calibrated" not in source.lower()


def test_pricing_store_module_still_has_the_same_scientific_field_count() -> None:
    """PHASE 9C's certified SCIENTIFIC_FIELDS tuple is untouched by PHASE
    10C1 -- exact same length and exact same membership as certified in
    Phase 9E."""
    from nflprops.orchestration.pricing_store import SCIENTIFIC_FIELDS

    assert len(SCIENTIFIC_FIELDS) == 39
    assert "p_model_calibrated" in SCIENTIFIC_FIELDS


def test_migration_head_extends_the_linear_chain_through_0008() -> None:
    """PHASE 10C1 locked this as "0008 extends 0007, never replaces it";
    BLOCK 2A's migration 0009 (compact PMF payload columns) extends that
    same linear chain the same way -- so the head has moved forward to
    0009, but 0006 -> 0007 -> 0008 must still all be reachable ancestors,
    never rewritten or forked."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    heads = subprocess.run(
        [sys.executable, "-m", "alembic", "heads"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=30,
    )
    assert "0009_compact_pmf_payload" in heads.stdout, heads.stdout + heads.stderr

    history = subprocess.run(
        [sys.executable, "-m", "alembic", "history"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=30,
    )
    for revision in (
        "0006_player_prop_pricing",
        "0007_player_prop_distributions",
        "0008_calibration_registry",
        "0009_compact_pmf_payload",
    ):
        assert revision in history.stdout, history.stdout + history.stderr
