"""Recipe fingerprint over the datagen import closure.

The dataset recipe must notice a change in any module the generator actually
imports -- and only those, so consumer-side edits do not invalidate a resume.
"""

import ast
import hashlib
from pathlib import Path

from ..core.hashing import fingerprint

PKG_ROOT = Path(__file__).resolve().parents[1]  # .../src/quantiloom_fusion


def _module_file(name: str, pkg_root: Path) -> Path | None:
    """Map a dotted module name to a file under pkg_root, or None."""
    parts = name.split(".")
    if parts[0] != pkg_root.name:
        return None
    if len(parts) == 1:
        return pkg_root / "__init__.py"
    candidate = pkg_root.joinpath(*parts[1:]).with_suffix(".py")
    if candidate.is_file():
        return candidate
    init = pkg_root.joinpath(*parts[1:], "__init__.py")
    if init.is_file():
        return init
    return None


def _internal_imports(path: Path, pkg_root: Path) -> set[str]:
    """Dotted names of internal modules imported anywhere in a file."""
    rel = path.relative_to(pkg_root).with_suffix("")
    package_parts = [pkg_root.name, *rel.parts[:-1]]
    names = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - node.level + 1]
            else:
                base = []
            if node.module:
                base = [*base, *node.module.split(".")]
            names.add(".".join(base))
            names.update(".".join([*base, alias.name]) for alias in node.names)
    return {n for n in names if n.split(".")[0] == pkg_root.name}


def generator_files(pkg_root: Path = PKG_ROOT) -> list[dict]:
    """[{'path': 'quantiloom_fusion/<rel>.py', 'sha256': <hex>}] sorted by path."""
    audit_root = pkg_root / "datagen" / "audit"
    seen = {
        p
        for p in (pkg_root / "datagen").rglob("*.py")
        if not p.is_relative_to(audit_root)
    }
    todo = list(seen)

    def add(path: Path) -> None:
        if path not in seen and not path.is_relative_to(audit_root):
            seen.add(path)
            todo.append(path)

    while todo:
        path = todo.pop()
        # Importing a module executes the __init__.py of every enclosing
        # package up to the distribution root, so they gate too.
        parent = path.relative_to(pkg_root).parent
        while True:
            init = pkg_root / parent / "__init__.py"
            if init.is_file():
                add(init)
            if str(parent) == ".":
                break
            parent = parent.parent
        for name in _internal_imports(path, pkg_root):
            target = _module_file(name, pkg_root)
            if target is not None:
                add(target)
    files = []
    for path in seen:
        data = path.read_bytes().replace(b"\r\n", b"\n")
        files.append(
            {
                "path": f"{pkg_root.name}/{path.relative_to(pkg_root).as_posix()}",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return sorted(files, key=lambda f: f["path"])


def generator_fingerprint(files: list[dict]) -> str:
    return fingerprint(files)
