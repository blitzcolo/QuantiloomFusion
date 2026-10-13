"""Summarize a result store: means over samples per (label, condition).

Grouping key is (variant label, illumination condition, value name).  Each
value in a row's values dict is its own series, so a registration-EPE row
yields epe_mean, epe_median, ... lines.  NaN values never enter a mean;
``n`` counts finite values only.  An "all" row per (label, metric) pools
every condition.  The CSV is written atomically.
"""

from __future__ import annotations

import csv
import math
import os
import uuid
from pathlib import Path

from ..core.jsonio import load_json
from ..core.paths import native_path
from ..dataset.reader import ThreeBandDataset
from ..evaluation.base import ResultStore
from .config import load_config


def _labels(out_dir: Path) -> dict[str, str]:
    """fingerprint -> label, from variants.json (fusion + EPE entries)."""
    path = out_dir / "variants.json"
    if not path.exists():
        return {}
    return {entry["fingerprint"]: label
            for label, entry in load_json(path).items()}


def summarize(out_dir: Path, dataset: ThreeBandDataset | None = None) -> Path:
    """Write <out_dir>/summary.csv; returns its path."""
    out_dir = Path(out_dir)
    labels = _labels(out_dir)
    conditions = {}
    if dataset is not None:
        conditions = {s.sample_id: s.condition.value for s in dataset}

    groups: dict[tuple[str, str, str], list[float]] = {}
    with ResultStore(out_dir / "results.jsonl") as store:
        for key, values in store.rows():
            label = labels.get(key.method_fingerprint, key.method_fingerprint)
            condition = conditions.get(key.sample_id, "unknown")
            for name, value in values.items():
                for cond in (condition, "all"):
                    groups.setdefault((label, cond, name), []).append(value)

    rows = []
    for (label, condition, name), vals in sorted(groups.items()):
        finite = [v for v in vals if math.isfinite(v)]
        n = len(finite)
        mean = sum(finite) / n if n else float("nan")
        var = (sum((v - mean) ** 2 for v in finite) / (n - 1)) if n > 1 else 0.0
        rows.append((label, condition, name, n,
                     f"{mean:.6g}" if n else "", f"{math.sqrt(var):.6g}" if n else ""))

    target = out_dir / "summary.csv"
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["label", "condition", "metric", "n", "mean", "std"])
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, target)
    return target


def summarize_config(config_path: Path) -> Path:
    """The `summarize` subcommand: resolve out_dir + dataset from the config."""
    cfg = load_config(config_path)
    from .driver import repo_root  # local: driver imports summary too

    out_dir = repo_root() / "build" / "experiments" / cfg.name
    dataset = ThreeBandDataset(native_path(cfg.dataset))
    return summarize(out_dir, dataset)
