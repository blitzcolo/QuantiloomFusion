"""The generator fingerprint covers exactly the datagen import closure."""

import shutil
from pathlib import Path

from quantiloom_fusion.datagen.fingerprint import (
    generator_files,
    generator_fingerprint,
)

REPO = Path(__file__).resolve().parents[1]


def _copy_pkg(tmp_path):
    root = tmp_path / "quantiloom_fusion"
    shutil.copytree(
        REPO / "src" / "quantiloom_fusion",
        root,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return root


def _paths(files):
    return {f["path"] for f in files}


def test_closure_contents(tmp_path):
    paths = _paths(generator_files(_copy_pkg(tmp_path)))
    assert {
        "quantiloom_fusion/__init__.py",
        "quantiloom_fusion/core/__init__.py",
        "quantiloom_fusion/core/paths.py",
        "quantiloom_fusion/datagen/planner.py",
        "quantiloom_fusion/datagen/fingerprint.py",
    } <= paths
    assert "quantiloom_fusion/dataset/types.py" not in paths
    assert "quantiloom_fusion/core/bands.py" not in paths


def test_included_edit_changes_fingerprint(tmp_path):
    root = _copy_pkg(tmp_path)
    before = generator_fingerprint(generator_files(root))
    with (root / "core" / "paths.py").open("a", encoding="utf-8") as f:
        f.write("\n# comment\n")
    assert generator_fingerprint(generator_files(root)) != before


def test_consumer_and_unreached_edits_do_not_change_fingerprint(tmp_path):
    root = _copy_pkg(tmp_path)
    before = generator_fingerprint(generator_files(root))
    for rel in ("dataset/types.py", "core/bands.py"):
        with (root / rel).open("a", encoding="utf-8") as f:
            f.write("\n# comment\n")
    assert generator_fingerprint(generator_files(root)) == before


def test_crlf_normalisation(tmp_path):
    root = _copy_pkg(tmp_path)
    before = generator_fingerprint(generator_files(root))
    target = root / "core" / "paths.py"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    assert generator_fingerprint(generator_files(root)) == before


def test_audit_directory_excluded(tmp_path):
    root = _copy_pkg(tmp_path)
    before = generator_fingerprint(generator_files(root))
    audit = root / "datagen" / "audit"
    audit.mkdir()
    (audit / "__init__.py").write_text('"""Audits."""\n', encoding="utf-8")
    (audit / "closure.py").write_text("x = 1\n", encoding="utf-8")
    assert generator_fingerprint(generator_files(root)) == before
    with (root / "datagen" / "solar.py").open("a", encoding="utf-8") as f:
        f.write("\ndef _audit_ref():\n    from .audit import closure\n")
    assert "quantiloom_fusion/datagen/audit/closure.py" not in _paths(
        generator_files(root)
    )


def test_function_level_imports_are_followed(tmp_path):
    root = _copy_pkg(tmp_path)
    (root / "core" / "newmod.py").write_text("THING = 1\n", encoding="utf-8")
    with (root / "datagen" / "solar.py").open("a", encoding="utf-8") as f:
        f.write("\ndef _new_ref():\n    from ..core import newmod\n")
    assert "quantiloom_fusion/core/newmod.py" in _paths(generator_files(root))
