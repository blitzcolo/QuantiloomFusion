"""AST-level boundary checks for the quantiloom_fusion package.

The layering rules are documented in docs/work-guide.md; check_tree() walks
every module under the package and enforces them mechanically so a violation
fails the suite instead of slipping through review.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "src" / "quantiloom_fusion"

ROOT = "<root>"

# Internal targets each subpackage may import. The importer itself is always
# allowed; the bare root package is allowed everywhere because importing any
# submodule runs the root __init__ anyway.
ALLOWED_INTERNAL = {
    "core": frozenset(),
    "datagen": frozenset({"core"}),
    "dataset": frozenset({"core"}),
    "geometry": frozenset({"core", "dataset"}),
    "signal": frozenset({"core", "dataset"}),
    "registration": frozenset({"core", "dataset", "geometry", "signal"}),
    "fusion": frozenset({"core", "dataset", "geometry", "signal", "registration"}),
    "evaluation": frozenset({"core", "dataset", "geometry", "signal", "registration", "fusion"}),
    "experiments": frozenset({"core", "dataset", "geometry", "signal", "registration", "fusion", "evaluation"}),
    ROOT: frozenset(),
}

# core, datagen and the root __init__ must stay standard-library-only at module
# level so datagen runs on a bare Python; function-level imports are exempt.
STDLIB_ONLY = frozenset({"core", "datagen", ROOT})
SUBPROCESS_MODULE = "datagen/renderer.py"
EXR_MODULE = "dataset/exr.py"
EXR_NAMES = frozenset({"OpenEXR", "Imath"})
TORCH_NAMES = frozenset({"torch", "torchvision"})
PATH_CTORS = frozenset({"Path", "PurePath", "PureWindowsPath", "PosixPath", "WindowsPath"})
DRIVE_LETTER = re.compile(r"^[A-Za-z]:[\\/]")


def _resolve_from(root_name: str, package_parts: list[str], level: int, module: str | None) -> str:
    """Resolve an ImportFrom to a dotted base name."""
    if level == 0:
        return module or ""
    keep = len(package_parts) - (level - 1)
    if keep < 0:
        return ""
    base = ".".join([root_name, *package_parts[:keep]])
    return f"{base}.{module}" if module else base


class _ImportCollector(ast.NodeVisitor):
    """Records (dotted_name, lineno, module_level) for every import."""

    def __init__(self, root_name: str, package_parts: list[str]) -> None:
        self.root_name = root_name
        self.package_parts = package_parts
        self.depth = 0
        self.records: list[tuple[str, int, bool]] = []

    def _enter_scope(self, node: ast.AST) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_scope(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._enter_scope(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.records.append((alias.name, node.lineno, self.depth == 0))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = _resolve_from(self.root_name, self.package_parts, node.level, node.module)
        if not base:
            return
        self.records.append((base, node.lineno, self.depth == 0))
        for alias in node.names:
            if alias.name != "*":
                self.records.append((f"{base}.{alias.name}", node.lineno, self.depth == 0))


def _declares_torch_optional(tree: ast.Module) -> bool:
    """True if the module assigns __optional_deps__ a literal containing 'torch'."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__optional_deps__" for t in targets):
            continue
        if isinstance(value, (ast.Tuple, ast.List)):
            return any(isinstance(e, ast.Constant) and e.value == "torch" for e in value.elts)
    return False


def _drive_letter_violations(tree: ast.Module, rel: str) -> list[tuple[str, int, str]]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if name not in PATH_CTORS:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str) and DRIVE_LETTER.match(arg0.value):
            out.append((rel, node.lineno, f"drive-letter path literal {arg0.value!r}; go through native_path()"))
    return out


def check_tree(pkg_root: Path) -> list[str]:
    """Return sorted 'relpath:lineno: message' violations for the package tree."""
    root_name = pkg_root.name
    violations: list[tuple[str, int, str]] = []
    for path in sorted(pkg_root.rglob("*.py")):
        rel = path.relative_to(pkg_root).as_posix()
        parts = list(path.relative_to(pkg_root).with_suffix("").parts)
        if parts[-1] == "__init__":
            module_parts, package_parts = parts[:-1], parts[:-1]
        else:
            module_parts, package_parts = parts, parts[:-1]
        subpackage = module_parts[0] if module_parts else ROOT

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        collector = _ImportCollector(root_name, package_parts)
        collector.visit(tree)
        torch_declared = _declares_torch_optional(tree)
        if subpackage not in ALLOWED_INTERNAL:
            violations.append((rel, 1, f"unknown subpackage {subpackage}; add it to the layering table first"))
        allowed = ALLOWED_INTERNAL.get(subpackage, frozenset()) | {subpackage}

        for dotted, lineno, module_level in collector.records:
            top = dotted.split(".", 1)[0]
            if top == root_name:
                target = dotted.split(".")[1] if "." in dotted else ROOT
                if target != ROOT and target not in allowed:
                    violations.append((rel, lineno, f"{subpackage} must not import {dotted}"))
            elif (
                subpackage in STDLIB_ONLY
                and module_level
                and top not in sys.stdlib_module_names
                and top != "__future__"
            ):
                violations.append((rel, lineno, f"{subpackage} is stdlib-only at module level; got {top}"))
            if top == "subprocess" and rel != SUBPROCESS_MODULE:
                violations.append((rel, lineno, f"subprocess may only be imported by {SUBPROCESS_MODULE}"))
            if top in EXR_NAMES and rel != EXR_MODULE:
                violations.append((rel, lineno, f"{top} may only be imported by {EXR_MODULE}"))
            if top in TORCH_NAMES and not torch_declared:
                violations.append((rel, lineno, "torch may only be imported by modules declaring __optional_deps__"))

        violations.extend(_drive_letter_violations(tree, rel))

    return [f"{rel}:{lineno}: {msg}" for rel, lineno, msg in sorted(set(violations))]


def _make_tree(base: Path, files: dict[str, str]) -> Path:
    """Write a fake quantiloom_fusion tree under base and return its root."""
    root = base / "quantiloom_fusion"
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    return root


def test_package_respects_boundaries():
    violations = check_tree(PKG_ROOT)
    assert violations == [], "\n".join(violations)


def test_absolute_datagen_import_flagged(tmp_path):
    root = _make_tree(tmp_path, {"fusion/__init__.py": "import quantiloom_fusion.datagen\n"})
    violations = check_tree(root)
    assert any(v.startswith("fusion/__init__.py:1:") and "quantiloom_fusion.datagen" in v for v in violations)


def test_relative_datagen_import_flagged(tmp_path):
    root = _make_tree(tmp_path, {"fusion/tools.py": "from .. import datagen\n"})
    violations = check_tree(root)
    assert any(v.startswith("fusion/tools.py:1:") and "quantiloom_fusion.datagen" in v for v in violations)


def test_geometry_importing_fusion_flagged(tmp_path):
    bad = _make_tree(tmp_path / "bad", {"geometry/g.py": "import quantiloom_fusion.fusion\n"})
    violations = check_tree(bad)
    assert any(v.startswith("geometry/g.py:1:") and "quantiloom_fusion.fusion" in v for v in violations)
    ok = _make_tree(tmp_path / "ok", {"geometry/g.py": "import quantiloom_fusion.dataset\n"})
    assert check_tree(ok) == []


def test_core_stdlib_only_at_module_level(tmp_path):
    bad = _make_tree(tmp_path / "bad", {"core/util.py": "import numpy\n"})
    violations = check_tree(bad)
    assert any(v.startswith("core/util.py:1:") and "numpy" in v for v in violations)
    ok = _make_tree(tmp_path / "ok", {"core/util.py": "def f():\n    import numpy\n    return numpy\n"})
    assert check_tree(ok) == []


def test_root_init_importing_subpackage_flagged(tmp_path):
    root = _make_tree(tmp_path, {
        "__init__.py": "from . import core\n",
        "core/__init__.py": '"""core."""\n',
    })
    violations = check_tree(root)
    assert any(v.startswith("__init__.py:1:") and "quantiloom_fusion.core" in v for v in violations)


def test_unknown_subpackage_flagged(tmp_path):
    root = _make_tree(tmp_path, {"helpers.py": '"""stray."""\n'})
    assert check_tree(root) == ["helpers.py:1: unknown subpackage helpers; add it to the layering table first"]


def test_subprocess_restricted_to_renderer(tmp_path):
    bad = _make_tree(tmp_path / "bad", {"datagen/plan.py": "from subprocess import run\n"})
    violations = check_tree(bad)
    assert violations == ["datagen/plan.py:1: subprocess may only be imported by datagen/renderer.py"]
    ok = _make_tree(tmp_path / "ok", {"datagen/renderer.py": "import subprocess\n"})
    assert check_tree(ok) == []


def test_openexr_restricted_to_dataset_exr(tmp_path):
    bad = _make_tree(tmp_path / "bad", {"datagen/audit.py": "def f():\n    import OpenEXR\n"})
    violations = check_tree(bad)
    assert any(v.startswith("datagen/audit.py:2:") and "OpenEXR" in v for v in violations)
    ok = _make_tree(tmp_path / "ok", {"dataset/exr.py": "import Imath\nimport OpenEXR\n"})
    assert check_tree(ok) == []


def test_torch_requires_optional_deps_marker(tmp_path):
    bad = _make_tree(tmp_path / "bad", {"fusion/net.py": "def f():\n    import torch\n"})
    violations = check_tree(bad)
    assert any(v.startswith("fusion/net.py:2:") and "torch" in v for v in violations)
    ok = _make_tree(tmp_path / "ok", {"fusion/net.py": '__optional_deps__ = ("torch",)\nimport torch\n'})
    assert check_tree(ok) == []


def test_drive_letter_path_literal_flagged(tmp_path):
    bad = _make_tree(tmp_path / "bad", {"core/paths.py": 'from pathlib import Path\nP = Path("H:/Quantiloom-dev")\n'})
    violations = check_tree(bad)
    assert any(v.startswith("core/paths.py:2:") and "drive-letter" in v for v in violations)
    ok = _make_tree(tmp_path / "ok", {"core/paths.py": 'from pathlib import Path\nP = Path("relative/x")\n'})
    assert check_tree(ok) == []
