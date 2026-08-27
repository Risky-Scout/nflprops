"""Built-in hash() is salted per process and must never seed anything. SPEC §51.

AST-based: a docstring that discusses hash() is fine; a CALL to hash() is not.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "nflprops"
CHECKED_LAYERS = ["simulation", "state", "models", "calibration"]


def test_no_builtin_hash_call():
    offenders = []
    for layer in CHECKED_LAYERS:
        layer_dir = SRC / layer
        if not layer_dir.exists():
            continue
        for path in layer_dir.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "hash"
                ):
                    offenders.append(f"{layer}/{path.name}:{node.lineno}")
    assert not offenders, (
        "built-in hash() called — it is salted per process (PYTHONHASHSEED) and "
        "destroys reproducibility. Use simulation.rng.deterministic_seed instead.\n"
        + "\n".join(offenders)
    )
