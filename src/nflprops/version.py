"""Package and model versioning.

SPEC: docs/IMPLEMENTATION_SPEC.md §70 (model artifact), §71 (configuration)

`__version__` is the code version. `MODEL_VERSION` is the trained-artifact version
and is what appears in prediction rows and RNG seeds. They move independently: a
bugfix release may not retrain, and a retrain may not change code.
"""

__version__ = "2026.1.1"

# Overridden by configs/models/*.toml at runtime. This is the fallback default.
MODEL_VERSION = "2026.1.0"

# Bump when the FORMAT of a prediction row changes (consumers must be told).
PREDICTION_SCHEMA_VERSION = "1"
