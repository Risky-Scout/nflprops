"""Provider independence. SPEC §1.

This is the rule that makes replacing the data provider a two-day job instead of a
rewrite, so it gets a test that runs on every commit.

These checks are AST-based, not grep-based. Grepping source text produces false
positives on docstrings that legitimately DISCUSS the boundary (this file, for one),
which trains people to ignore the test. Parsing the AST checks what the code
actually does.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "nflprops"

PROVIDER_INDEPENDENT_LAYERS = [
    "models", "features", "simulation", "state", "calibration", "backtest", "explain",
]

HTTP_LIBS = {"requests", "httpx", "urllib3", "aiohttp", "urllib"}


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids() of string constants that serve as docstrings."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant
            ) and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _string_literals(path: Path):
    """Yield (lineno, value) for every non-docstring string literal."""
    tree = ast.parse(path.read_text())
    docs = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):  # noqa: SIM102
            if id(node) not in docs:
                yield node.lineno, node.value


def _imported_modules(path: Path):
    """Yield (lineno, dotted_module) for every import in the file."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):  # noqa: SIM102
            if node.module:
                yield node.lineno, node.module


def test_endpoint_strings_only_in_endpoints_module():
    """A BDL path may appear in exactly one file. SPEC §10."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "endpoints.py":
            continue
        for lineno, value in _string_literals(path):
            if "nfl/v1" in value:
                offenders.append(f"{path.relative_to(SRC)}:{lineno}")
    assert not offenders, (
        "BDL endpoint string outside providers/bdl/endpoints.py: "
        + ", ".join(offenders)
    )


def test_no_provider_imports_in_model_layers():
    """models/features/simulation/state must never import a provider. SPEC §1."""
    offenders = []
    for layer in PROVIDER_INDEPENDENT_LAYERS:
        layer_dir = SRC / layer
        if not layer_dir.exists():
            continue
        for path in layer_dir.rglob("*.py"):
            for lineno, module in _imported_modules(path):
                if "providers" in module.split("."):
                    offenders.append(f"{layer}/{path.name}:{lineno}: {module}")
    assert not offenders, (
        "provider import inside a provider-independent layer:\n" + "\n".join(offenders)
    )


def test_domain_imports_nothing_provider_specific():
    offenders = []
    for path in (SRC / "domain").rglob("*.py"):
        for lineno, module in _imported_modules(path):
            if "providers" in module.split("."):
                offenders.append(f"domain/{path.name}:{lineno}: {module}")
    assert not offenders, "\n".join(offenders)


def test_no_http_client_outside_providers():
    """Only the provider transport layer may speak HTTP."""
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("providers/"):
            continue
        for lineno, module in _imported_modules(path):
            if module.split(".")[0] in HTTP_LIBS:
                offenders.append(f"{rel}:{lineno}: {module}")
    assert not offenders, (
        "HTTP client imported outside providers/: " + "; ".join(offenders)
    )
