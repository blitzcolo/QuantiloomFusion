"""The experiment driver: sample-major, incremental, resumable.

For every sample it computes the ResultKeys (variant x metric, plus the
registration-EPE rows) that are not yet in the store; a sample with nothing
pending is skipped without loading a single image.  Band inputs are
normalized once per (sample, normalizer), warps estimated once per
(sample, registrar, band) — the oracle is expensive — and one fuse() call
is shared by the grey and colour output variants of the same spec.

Rows are appended and fsynced one at a time: a crash loses at most the
current row.  A failed (sample, variant) is recorded in errors.jsonl and
retried on the next run; it never aborts the rest.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from ..core.bands import Band
from ..core.jsonio import atomic_json
from ..core.paths import native_path
from ..dataset.reader import ThreeBandDataset
from ..dataset.types import Sample
from ..evaluation.base import ResultKey, ResultStore
from ..evaluation.metrics import (
    AverageGradient,
    Entropy,
    MsSsim,
    Qabf,
    SCD,
    SpatialFrequency,
    StandardDeviation,
)
from ..fusion.align import colourize
from ..fusion.base import FusionInput, FusionOutput
from ..fusion.registry import create
from ..geometry.warp import apply_warp
from ..registration import (
    InfinityHomographyRegistrar,
    OracleRegistrar,
    ResizeRegistrar,
    registration_error,
)
from ..signal.colour import luminance
from ..signal.normalize import normalized_input
from .config import ExperimentConfig, load_config
from .summary import summarize
from .variants import expand, fingerprint, ir_normalizer, vis_normalizer

_METRIC_CLASSES = {
    "EN": Entropy,
    "SD": StandardDeviation,
    "SF": SpatialFrequency,
    "AG": AverageGradient,
    "SCD": SCD,
    "Qabf": Qabf,
    "MS_SSIM": MsSsim,
}

# Source-reference metrics are undefined on a single source; single-band
# variants record NaN for them rather than failing every run.
_SOURCE_REF_METRICS = (SCD, Qabf, MsSsim)

REGISTRATION_EPE_METRIC = "registration_epe"
REGISTRATION_EPE_VERSION = 1


def repo_root() -> Path:
    """The directory containing pyproject.toml above this package."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"no pyproject.toml found above {here}")


def _registrar(name: str, dataset: ThreeBandDataset):
    if name == "oracle":
        return OracleRegistrar(dataset)
    if name == "resize":
        return ResizeRegistrar()
    if name == "infinity_homography":
        return InfinityHomographyRegistrar()
    raise ValueError(f"unknown registrar {name!r}")


def _epe_fingerprint(registration: str, band: Band, reference: Band) -> str:
    return fingerprint({"registration": registration, "band": str(band),
                        "reference": str(reference)})


class _ErrorLog:
    """errors.jsonl: one LF-terminated ASCII line per failed (sample, variant)."""

    def __init__(self, path: Path):
        self._file = path.open("ab")
        self.count = 0

    def write(self, dataset_id: str, sample_id: str, fingerprint_: str,
              label: str, err: Exception) -> None:
        rec = {"dataset_id": dataset_id, "sample_id": sample_id,
               "fingerprint": fingerprint_, "label": label, "error": repr(err)}
        line = json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n"
        self._file.write(line.encode("ascii"))
        self._file.flush()
        os.fsync(self._file.fileno())
        self.count += 1
        print(f"  error {sample_id} {label}: {err!r}")

    def close(self) -> None:
        self._file.close()


class _SampleWork:
    """Per-sample caches: normalized inputs, warps, oracle warps."""

    def __init__(self, dataset: ThreeBandDataset, sample: Sample, reference: Band):
        self.dataset = dataset
        self.sample = sample
        self.reference = reference
        self.images: dict = {}
        self.warps: dict = {}
        self.registrars: dict = {}

    def normalized(self, band: Band, kind: str) -> np.ndarray:
        key = (band, kind)
        if key not in self.images:
            obs = self.sample.observations[band]
            nrm = vis_normalizer(kind) if band == Band.VIS else ir_normalizer(kind)
            self.images[key] = normalized_input(self.dataset, obs, nrm)
        return self.images[key]

    def grey_input(self, band: Band, kind: str) -> np.ndarray:
        img = self.normalized(band, kind)
        return luminance(img) if img.ndim == 3 else img

    def warp(self, registration: str, band: Band):
        key = (registration, band)
        if key not in self.warps:
            if registration not in self.registrars:
                self.registrars[registration] = _registrar(registration, self.dataset)
            # None, None: the registered registrars do not read the images.
            self.warps[key] = self.registrars[registration].estimate(
                self.sample.observations[self.reference],
                self.sample.observations[band], None, None)
        return self.warps[key]


def _result_key(dataset_id: str, sample_id: str, variant, metric) -> ResultKey:
    return ResultKey(dataset_id, sample_id, variant.method_fingerprint,
                     metric.name, metric.version)


def _variant_pending(dataset_id: str, sample_id: str, v, metrics, store) -> bool:
    return any(not store.has(_result_key(dataset_id, sample_id, v, m))
               for m in metrics)


def _fuse_and_score(dataset, work: _SampleWork, cfg: ExperimentConfig,
                    variants, group_key, metrics, store, errors: _ErrorLog) -> int:
    """One fuse() call shared by the group's grey and colour variants."""
    band_set, vis_in, ir_in, reg, fusion_name, params = group_key
    sid = work.sample.sample_id
    ref = cfg.reference
    try:
        images = {}
        geometry = {}
        for b in band_set:
            images[b] = work.grey_input(b, vis_in if b == Band.VIS else ir_in)
            geometry[b] = work.sample.observations[b].geometry
        warps = None if reg == "none" else {
            b: work.warp(reg, b) for b in band_set if b != ref}
        x = FusionInput(reference=ref, images=images, geometry=geometry, warps=warps)
        out = create(fusion_name, **params).fuse(x)
        fused_y = luminance(out.image) if out.image.ndim == 3 else np.asarray(out.image)
    except Exception as e:
        for v in variants:
            errors.write(dataset.dataset_id, sid, v.method_fingerprint, v.label, e)
        return 0

    appended = 0
    for v in variants:
        try:
            if v.output == "colour":
                if Band.VIS not in band_set:
                    raise ValueError("colour output needs the vis band in the set")
                rgb = work.normalized(Band.VIS, vis_in)
                if ref != Band.VIS:
                    rgb = apply_warp(rgb, work.warp(reg, Band.VIS), "linear")
                    rgb = np.where(np.isfinite(rgb), rgb, 0.0)
                image = colourize(fused_y, rgb)
            else:
                image = fused_y
            vout = FusionOutput(image, out.extras)
            for m in metrics:
                key = _result_key(dataset.dataset_id, sid, v, m)
                if not store.has(key):
                    if len(images) < 2 and isinstance(m, _SOURCE_REF_METRICS):
                        store.append(key, {m.name: float("nan")})
                    else:
                        store.append(key, m(vout, x, work.sample))
                    appended += 1
        except Exception as e:
            errors.write(dataset.dataset_id, sid, v.method_fingerprint, v.label, e)
    return appended


def run(config_path: Path, *, limit: int | None = None,
        samples: list[str] | None = None, out_root: Path | None = None) -> int:
    """Run the experiment described by config_path; 0 = clean, 1 = errors."""
    cfg = load_config(config_path)
    dataset = ThreeBandDataset(native_path(cfg.dataset))
    out_dir = (out_root or repo_root() / "build" / "experiments") / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    config_copy = out_dir / "config.toml"
    src_bytes = Path(config_path).read_bytes()
    if config_copy.exists():
        if config_copy.read_bytes() != src_bytes:
            raise ValueError(
                f"{config_copy} differs from {config_path}; a changed config "
                "needs a new experiment name")
    else:
        config_copy.write_bytes(src_bytes)

    variants = expand(cfg)
    variant_index = {v.label: {"fingerprint": v.method_fingerprint, "spec": v.spec}
                     for v in variants}
    # Registration-EPE rows also need a label -> fingerprint -> spec entry.
    for band_set in cfg.band_sets:
        for band in band_set:
            if band == cfg.reference:
                continue
            for reg in cfg.registration:
                if reg in ("none", "oracle"):
                    continue
                label = f"reg={reg}|band={band}"
                spec = {"registration": reg, "band": str(band),
                        "reference": str(cfg.reference)}
                variant_index[label] = {
                    "fingerprint": _epe_fingerprint(reg, band, cfg.reference),
                    "spec": spec}
    atomic_json(out_dir / "variants.json", variant_index)

    metrics = [_METRIC_CLASSES[m]() for m in cfg.metrics]
    sample_ids = dataset.sample_ids()
    if samples is not None:
        unknown = set(samples) - set(sample_ids)
        if unknown:
            raise ValueError(f"--samples not in dataset: {sorted(unknown)}")
        sample_ids = tuple(s for s in sample_ids if s in set(samples))
    if limit is not None:
        sample_ids = sample_ids[:limit]

    t0 = time.monotonic()
    errors = _ErrorLog(out_dir / "errors.jsonl")
    try:
        with ResultStore(out_dir / "results.jsonl") as store:
            for i, sid in enumerate(sample_ids, 1):
                pending_variants = [
                    v for v in variants
                    if _variant_pending(dataset.dataset_id, sid, v, metrics, store)]
                pairs = {(v.registration, b) for v in variants
                         for b in v.band_set if b != cfg.reference}
                epe_pending = [
                    (reg, b) for reg, b in sorted(pairs, key=lambda p: (p[0], str(p[1])))
                    if reg not in ("none", "oracle")
                    and not store.has(ResultKey(
                        dataset.dataset_id, sid,
                        _epe_fingerprint(reg, b, cfg.reference),
                        REGISTRATION_EPE_METRIC, REGISTRATION_EPE_VERSION))]
                if not pending_variants and not epe_pending:
                    print(f"sample {i}/{len(sample_ids)} {sid}: nothing pending, skipped")
                    continue
                work = _SampleWork(dataset, dataset.sample(sid), cfg.reference)

                groups: dict = {}
                for v in pending_variants:
                    key = (v.band_set, v.vis_input, v.ir_input, v.registration,
                           v.fusion_name,
                           json.dumps(v.fusion_params, sort_keys=True))
                    groups.setdefault(key, []).append(v)
                done = 0
                for (band_set, vis_in, ir_in, reg, name, params_json), group in groups.items():
                    done += _fuse_and_score(
                        dataset, work, cfg, group,
                        (band_set, vis_in, ir_in, reg, name, json.loads(params_json)),
                        metrics, store, errors)

                for reg, band in epe_pending:
                    key = ResultKey(dataset.dataset_id, sid,
                                    _epe_fingerprint(reg, band, cfg.reference),
                                    REGISTRATION_EPE_METRIC, REGISTRATION_EPE_VERSION)
                    try:
                        store.append(key, registration_error(
                            work.warp(reg, band), work.warp("oracle", band)))
                    except Exception as e:
                        errors.write(dataset.dataset_id, sid,
                                     _epe_fingerprint(reg, band, cfg.reference),
                                     f"reg={reg}|band={band}", e)
                finished = sum(1 for v in pending_variants
                               if not _variant_pending(dataset.dataset_id, sid, v,
                                                       metrics, store))
                print(f"sample {i}/{len(sample_ids)} {sid}: variants "
                      f"{finished}/{len(pending_variants)} done, rows +{done}, "
                      f"errors {errors.count}, elapsed {time.monotonic() - t0:.1f}s")
    finally:
        errors.close()
    summarize(out_dir, dataset)
    return 1 if errors.count else 0
