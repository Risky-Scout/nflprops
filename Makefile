.PHONY: help install lint format typecheck test verify-contracts pin-bdl-spec verify-bdl-spec release-gate clean \
        verify-phase-00 verify-phase-01 verify-phase-02 verify-phase-03 \
        verify-phase-04 verify-phase-05 verify-phase-06 verify-phase-07 \
        verify-phase-08 verify-phase-09 verify-phase-10 collector

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:  ## Install package + dev dependencies
	pip install -e ".[dev]"

lint:  ## Ruff lint
	ruff check src tests

format:  ## Ruff format
	ruff format src tests

typecheck:  ## mypy strict
	mypy

test:  ## Full test suite
	pytest

verify-contracts:  ## Validate every YAML contract parses and cross-references
	python tools/validate_contracts.py

sync-resources:  ## Mirror runtime configs/contracts/specs into installable package
	python tools/sync_runtime_resources.py

check-resources:  ## Fail if package runtime resources differ from repository sources
	python tools/sync_runtime_resources.py --check


pin-bdl-spec:  ## Pin the exact official BDL NFL OpenAPI bytes + SHA lock
	nflprops provider pin bdl --url https://www.balldontlie.io/openapi/nfl.yml

verify-bdl-spec:  ## Verify BDL contract against the exact pinned spec
	nflprops provider verify bdl --strict-fields

release-gate:  ## Final release only: fail if ANY test is skipped or failed
	@set -o pipefail; pytest -ra | tee .release-pytest.log
	@! grep -q "SKIPPED" .release-pytest.log || (echo "FAIL: skipped acceptance tests remain" && rm -f .release-pytest.log && exit 1)
	@rm -f .release-pytest.log

# --- per-phase gates. A phase is DONE only when its gate is green. ------------
verify-phase-00:  ## Phase 0 gate
	pytest tests/unit -q && ruff check src && mypy

verify-phase-01:  ## Phase 1 gate — INCLUDES the real spec verification
	nflprops provider verify bdl --strict-fields
	pytest tests/provider_contract tests/unit -q
	@echo "--- checking endpoint-string containment ---"
	@! grep -rn "nfl/v1" src/ --include=*.py | grep -v "providers/bdl/endpoints.py" \
	  || (echo "FAIL: endpoint string outside endpoints.py" && exit 1)
	@echo "--- checking provider leakage into model layers ---"
	@! grep -rn "providers.bdl" src/nflprops/models src/nflprops/features \
	  src/nflprops/simulation src/nflprops/state src/nflprops/calibration \
	  || (echo "FAIL: BDL imported into a provider-independent layer" && exit 1)

verify-phase-02:  ## Phase 2 gate
	pytest tests/unit tests/leakage -q -k "raw_store or pit or entity or snapshot or quality"

verify-phase-03:  ## Phase 3 gate
	pytest tests/provider_contract/test_bdl_yardline_semantics.py tests/unit -q -k "alias or play_family or reconcil or tier3"

verify-phase-04:  ## Phase 4 gate
	pytest tests/leakage -q

verify-phase-05:  ## Phase 5 gate
	pytest tests/unit -q -k "eb_ or role_faster or state_"

verify-phase-06:  ## Phase 6 gate
	pytest tests/unit -q -k "component_model or passing_yards_regressor or overdispersion or residual_pool or kappa or two_point"

verify-phase-07:  ## Phase 7 gate — invariants + determinism
	pytest tests/invariants tests/determinism -q

verify-phase-08:  ## Phase 8 gate
	pytest tests/leakage tests/unit -q -k "calibration or pit_histogram"

verify-phase-09:  ## Phase 9 gate
	pytest tests/unit -q -k "odds or devig or push or closing or settlement"

verify-phase-10:  ## Phase 10 gate — the full bar
	pytest -q

collector:  ## Run the market snapshot collector (START THIS EARLY - see SPEC §58)
	nflprops snapshot props

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
