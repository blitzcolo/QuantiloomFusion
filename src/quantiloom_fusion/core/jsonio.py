"""Atomic JSON writes (LF, fsync, os.replace) and strict JSON loading."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def load_json(path: Path):
    def unique(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError(f"duplicate JSON key: {k}")
            result[k] = v
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
