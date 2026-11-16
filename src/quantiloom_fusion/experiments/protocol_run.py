"""The chapter-6 protocol runner: resumable, sample-major, one result store.

Four items, each switched on in the config's ``items`` list:

* ``matrix``: registration (oracle, epipolar sweep, infinity homography,
  resize) x the two masked pyramids x the layer groups of
  ``evaluation.layer_report``; samples with co-view references also get both
  truth-referenced modes.
* ``sensitivity``: perturbed-oracle sweeps (``evaluation.protocol``) and the
  no-reference inflation experiment, on samples with co-view references.
* ``registration_study``: EPE against the oracle of the infinity homography,
  the epipolar sweep per similarity and the free-search control per
  similarity and grid step, on the correspondable and the edge layer.
* ``complementarity``: ``evaluation.complementarity`` on the whole dataset,
  written once as a Markdown report.
* ``band_ablation``: on samples with co-view references, every band set of the config key
  ``band_sets`` (lists starting with ``"vis"``; ``["vis"]`` alone means the VIS input is the
  fused image) fused with the oracle warps by ``laplacian_pyramid_masked`` and scored in the
  FUSION truth mode against one common reference: the baseline rule on the clean co-view images
  of *all* configured bands (``evaluation.protocol.evaluate_band_subset``).  Layer groups are the
  full configuration's, so every band set is scored on the same pixels; the clean-source metrics
  use every clean band.  The spec names the band set and the reference's band set.
* ``zeroshot``: the public deep models named in the config key ``zeroshot``
  (``fusion.zeroshot``, author weights) on the raw unregistered VIS + LWIR
  pair of every sample with co-view references -- native resolutions, the
  model's own alignment, never our warps -- scored by
  ``evaluation.protocol.evaluate_zeroshot`` in the FUSION truth mode.  The
  models take the VIS display image and LWIR through the protocol's fixed
  temperature window, so the truth reference is built from the clean co-view
  images in that same domain, and an anchor (``laplacian_pyramid_masked`` on
  the oracle warps, same domain and reference) is scored beside them.  The
  spec names the model, its clone commit and the SHA-256 of every weight file,
  so changed weights are new rows.  A model whose weights or torch are missing
  gets an error line in ``errors.jsonl`` per sample (retried next run) and
  makes the run exit 1; it is never skipped silently.

Rows go to ``build/experiments/<name>/results.jsonl`` (``evaluation.base.
ResultStore``): one row per (sample, variant, layer group) for the layered
items, keyed by the variant spec's fingerprint, so an interrupted run resumes
with the same command and a finished sample is skipped without loading it.
``variants.json`` maps each fingerprint to its spec; ``protocol_summary.csv``
and ``protocol_summary.md`` are rewritten at the end.

Noise for ``noise_aware_pyramid`` is built per sample from the camera config
of each band (``signal.noise``): VIS per mosaic site from its raw electrons
with the VIS processing chain for the level gains; LWIR from NETD plus the
fixed pattern left after NUC; MWIR per pixel from its raw electrons, converted
to K through dS/dT when windowed in Kelvin, and resampled onto the reference
grid through each registration's own warp (as a variance). Registration
estimates that read images (epipolar sweep, free search) get the inputs the
sweep's gates were tuned on: VIS raw linear with a percentile scale, IR
corrected signal with a percentile window.

CLI: ``python -m quantiloom_fusion.experiments protocol CONFIG [--limit N]
[--samples a,b]``.
"""

from __future__ import annotations

import csv
import math
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.bands import BANDS, Band
from ..core.jsonio import atomic_json, load_json
from ..core.paths import native_path
from ..dataset.reader import ThreeBandDataset
from ..evaluation.base import ResultKey, ResultStore
from ..evaluation.complementarity import AnalysisSettings, analyse_dataset
from ..evaluation.complementarity_report import render_markdown
from ..evaluation.layer_report import REPORT_GROUPS
from ..evaluation.protocol import (
    PerturbationKind,
    Sweep,
    epe_columns,
    evaluate_band_subset,
    evaluate_point,
    evaluate_zeroshot,
    inflation_experiment,
    is_calibrated,
    sensitivity_curve,
)
from ..evaluation.scene import EvalScene, InputSpec, load_scene
from ..evaluation.truth_ref import BASELINE_RULE, TruthMode
from ..fusion.registry import create
from ..fusion.zeroshot_assets import ZEROSHOT_MODELS, model_identity
from ..geometry.layers import Layer, layer_map
from ..geometry.warp import Warp, apply_warp
from ..registration import (
    EpipolarSweepRegistrar,
    FreeSearchRegistrar,
    InfinityHomographyRegistrar,
    ResizeRegistrar,
    Similarity,
    SweepParams,
    registration_error,
)
from ..signal.colour import luminance
from ..signal.noise import band_sigma, electrons_from_raw, estimate_dsdt, sample_noise_config, vis_luminance_chain
from ..signal.normalize import IrLinear, IrTemperature, VisDisplay, VisRawLinear, normalized_input
from .driver import _ErrorLog
from .variants import fingerprint, ir_normalizer, vis_normalizer

ITEMS = ("matrix", "sensitivity", "registration_study", "complementarity", "zeroshot", "band_ablation")
REGISTRATIONS = ("oracle", "epipolar_sweep", "infinity_homography", "resize")
OPERATORS = ("laplacian_pyramid_masked", "noise_aware_pyramid")
PROTOCOL_VERSION = 1

_TOP = {"name", "dataset", "bands", "vis_input", "ir_input", "items", "registration", "operators",
        "levels", "kappa", "fpn", "similarity", "z_min_m", "n_hypotheses", "sensitivity", "registration_study",
        "complementarity", "zeroshot", "band_sets"}
_SENS = {"bias", "field", "corr_len_px", "seed", "inflation", "operators", "truth_modes"}
_STUDY = {"similarities", "free_steps"}
_COMP = {"view_group_blocks"}


@dataclass(frozen=True)
class SensitivitySpec:
    bias: tuple[float, ...] = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)
    field: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    corr_len_px: tuple[float, ...] = (8.0,)  # smooth-field correlation lengths; one sweep each
    seed: int = 0
    inflation: bool = True
    operators: tuple[str, ...] | None = None  # subset of the config's operators; None = all
    truth_modes: tuple[str, ...] | None = None  # subset of TruthMode values; None = all


@dataclass(frozen=True)
class StudySpec:
    similarities: tuple[str, ...] = tuple(str(s) for s in Similarity)
    free_steps: tuple[float | str, ...] = ("sweep", 1.0)


@dataclass(frozen=True)
class ProtocolConfig:
    name: str
    dataset: str
    bands: tuple[Band, ...]
    vis_input: str
    ir_input: str
    items: tuple[str, ...]
    registration: tuple[str, ...]
    operators: tuple[str, ...]
    levels: int = 4
    kappa: float = 2.5
    fpn: bool = True
    similarity: str = "grad_ncc"
    z_min_m: float = 3.0
    n_hypotheses: int = 16
    sensitivity: SensitivitySpec = field(default_factory=SensitivitySpec)
    registration_study: StudySpec = field(default_factory=StudySpec)
    view_group_blocks: int | None = None
    zeroshot: tuple[str, ...] = ()  # registered zero-shot model names (item "zeroshot")
    band_sets: tuple[tuple[Band, ...], ...] = ()  # item "band_ablation": subsets of bands, each from "vis"

    @property
    def reference(self) -> Band:
        return self.bands[0]

    def sweep_params(self, similarity: str) -> SweepParams:
        """Epipolar sweep / free-search parameters: the inverse-depth range [0, 1/z_min_m] split into
        ``n_hypotheses``; a longer baseline or a nearer z_min needs more hypotheses to keep the
        per-hypothesis displacement step (f * b / z_min / (n - 1)) at a fraction of a pixel."""
        return SweepParams(similarity=Similarity(similarity), z_min_m=self.z_min_m, n_hypotheses=self.n_hypotheses)


def _bad(key: str, msg: str) -> ValueError:
    return ValueError(f"{key}: {msg}")


def _check(table: dict, allowed: set[str], where: str) -> None:
    for key in table:
        if key not in allowed:
            raise _bad(f"{where}{key}", "unknown config key")


def _choices(data: dict, key: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value or any(v not in allowed for v in value):
        raise _bad(key, f"expected a non-empty list from {list(allowed)}, got {value!r}")
    return tuple(value)


def _numbers(value, key: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise _bad(key, f"expected a non-empty list of numbers, got {value!r}")
    return tuple(float(v) for v in value)


def _corr_lengths(value) -> tuple[float, ...]:
    """``sensitivity.corr_len_px``: one positive number or a non-empty list of them."""
    values = _numbers(value if isinstance(value, list) else [value], "sensitivity.corr_len_px")
    if any(v <= 0 for v in values):
        raise _bad("sensitivity.corr_len_px", f"expected positive numbers, got {value!r}")
    return values


def load_protocol_config(path: Path) -> ProtocolConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    _check(data, _TOP, "")
    for key in ("name", "dataset"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise _bad(key, f"expected a non-empty string, got {data.get(key)!r}")
    bands = tuple(Band(b) for b in _choices(data, "bands", tuple(str(b) for b in BANDS)))
    if bands[0] != Band.VIS or len(set(bands)) != len(bands) or len(bands) < 2:
        raise _bad("bands", "expected distinct bands starting with the reference 'vis' and at least one IR band")
    vis_input = data.get("vis_input", "raw_adc")
    if vis_input not in ("raw_adc",):
        raise _bad("vis_input", "the noise model and the truth references need 'raw_adc' (a fixed linear window)")
    ir_input = data.get("ir_input", "temperature")
    if ir_input not in ("temperature",):
        raise _bad("ir_input", "the truth references need 'temperature' (a fixed Kelvin window)")
    sens = data.get("sensitivity", {})
    study = data.get("registration_study", {})
    comp = data.get("complementarity", {})
    _check(sens, _SENS, "sensitivity.")
    _check(study, _STUDY, "registration_study.")
    _check(comp, _COMP, "complementarity.")
    steps = study.get("free_steps", list(StudySpec.free_steps))
    if not isinstance(steps, list) or any(
            not (s == "sweep" or (isinstance(s, (int, float)) and not isinstance(s, bool) and s > 0)) for s in steps):
        raise _bad("registration_study.free_steps", f'expected a list of positive numbers or "sweep" (empty: no '
                   f'free-search control), got {steps!r}')
    similarity = data.get("similarity", "grad_ncc")
    if similarity not in tuple(str(s) for s in Similarity):
        raise _bad("similarity", f"{similarity!r} not one of {[str(s) for s in Similarity]}")
    levels, kappa, z_min = data.get("levels", 4), data.get("kappa", 2.5), data.get("z_min_m", 3.0)
    if type(levels) is not int or levels < 1:
        raise _bad("levels", f"expected a positive integer, got {levels!r}")
    for key, v in (("kappa", kappa), ("z_min_m", z_min)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
            raise _bad(key, f"expected a positive number, got {v!r}")
    n_hyp = data.get("n_hypotheses", 16)
    if type(n_hyp) is not int or n_hyp < 2:
        raise _bad("n_hypotheses", f"expected an integer >= 2, got {n_hyp!r}")
    if type(data.get("fpn", True)) is not bool:
        raise _bad("fpn", "expected true or false")
    items = _choices(data, "items", ITEMS)
    if "zeroshot" in items:
        zeroshot = _choices(data, "zeroshot", tuple(ZEROSHOT_MODELS))
        if len(set(zeroshot)) != len(zeroshot):
            raise _bad("zeroshot", f"duplicate model names in {list(zeroshot)}")
        if Band.LWIR not in bands:
            raise _bad("bands", "the zeroshot item fuses VIS + LWIR; 'lwir' must be among the bands")
    elif "zeroshot" in data:
        raise _bad("zeroshot", "models are listed but 'zeroshot' is not in items")
    else:
        zeroshot = ()
    band_sets = _band_sets(data, bands) if "band_ablation" in items else ()
    if "band_sets" in data and "band_ablation" not in items:
        raise _bad("band_sets", "band sets are listed but 'band_ablation' is not in items")
    blocks = comp.get("view_group_blocks")
    if blocks is not None and (type(blocks) is not int or blocks < 2):
        raise _bad("complementarity.view_group_blocks", f"expected an integer >= 2, got {blocks!r}")
    return ProtocolConfig(
        name=data["name"], dataset=data["dataset"], bands=bands, vis_input=vis_input, ir_input=ir_input,
        items=items, registration=_choices(data, "registration", REGISTRATIONS),
        operators=_choices(data, "operators", OPERATORS), levels=levels, kappa=float(kappa),
        fpn=data.get("fpn", True), similarity=similarity, z_min_m=float(z_min), n_hypotheses=n_hyp,
        sensitivity=SensitivitySpec(
            bias=_numbers(sens.get("bias", list(SensitivitySpec.bias)), "sensitivity.bias"),
            field=_numbers(sens.get("field", list(SensitivitySpec.field)), "sensitivity.field"),
            corr_len_px=_corr_lengths(sens.get("corr_len_px", 8.0)), seed=int(sens.get("seed", 0)),
            inflation=bool(sens.get("inflation", True)),
            operators=_choices(sens, "operators", tuple(data.get("operators") or ()))
            if "operators" in sens else None,
            truth_modes=_choices(sens, "truth_modes", tuple(str(m) for m in TruthMode))
            if "truth_modes" in sens else None),
        registration_study=StudySpec(
            similarities=_choices(study, "similarities", tuple(str(s) for s in Similarity))
            if "similarities" in study else StudySpec.similarities,
            free_steps=tuple(s if s == "sweep" else float(s) for s in steps)),
        view_group_blocks=blocks, zeroshot=zeroshot, band_sets=band_sets,
    )


def _band_sets(data: dict, bands: tuple[Band, ...]) -> tuple[tuple[Band, ...], ...]:
    """@@band_sets@@: a non-empty list of distinct band lists, each of distinct configured bands
    starting with the reference "vis" (["vis"] alone is allowed)."""
    value = data.get("band_sets")
    allowed = [str(b) for b in bands]
    if not isinstance(value, list) or not value:
        raise _bad("band_sets", f"expected a non-empty list of band lists, got {value!r}")
    out = []
    for item in value:
        if (not isinstance(item, list) or not item or item[0] != str(Band.VIS) or len(set(item)) != len(item)
                or any(b not in allowed for b in item)):
            raise _bad("band_sets", f"each set must list distinct bands from {allowed} starting with 'vis', "
                       f"got {item!r}")
        out.append(tuple(Band(b) for b in item))
    if len({frozenset(s) for s in out}) != len(out):
        raise _bad("band_sets", f"duplicate band sets in {value!r}")
    return tuple(out)




class _Sample:
    """Lazily loaded scene, noise and warps of one sample."""

    def __init__(self, dataset: ThreeBandDataset, sample_id: str, cfg: ProtocolConfig):
        self.ds, self.sid, self.cfg = dataset, sample_id, cfg
        self.sample = dataset.sample(sample_id)
        self._scene: EvalScene | None = None
        self._warps: dict[str, dict[Band, Warp]] = {}
        self._noise: dict | None = None
        self._reg_inputs: dict[Band, np.ndarray] = {}
        self._zs_scene: EvalScene | None = None

    @property
    def has_references(self) -> bool:
        return bool(self.sample.references)

    @property
    def scene(self) -> EvalScene:
        if self._scene is None:
            spec = InputSpec(vis_normalizer(self.cfg.vis_input), ir_normalizer(self.cfg.ir_input))
            self._scene = load_scene(self.ds, self.sid, spec, self.cfg.bands)
        return self._scene

    @property
    def zeroshot_scene(self) -> EvalScene:
        """VIS + LWIR in the zero-shot input domain: VIS display luminance, LWIR through the
        protocol's fixed temperature window; clean co-view images in the same domain."""
        if self._zs_scene is None:
            spec = InputSpec(VisDisplay(), ir_normalizer(self.cfg.ir_input))
            self._zs_scene = load_scene(self.ds, self.sid, spec, (Band.VIS, Band.LWIR))
        return self._zs_scene

    def vis_display_rgb(self) -> np.ndarray:
        """The VIS display image as RGB in [0, 1] (what the zero-shot models are given)."""
        return normalized_input(self.ds, self.sample.observations[Band.VIS], VisDisplay())

    def reg_input(self, band: Band) -> np.ndarray:
        """Inputs the sweep's gates were tuned on (percentile scales)."""
        if band not in self._reg_inputs:
            norm = VisRawLinear(scale="percentile") if band == Band.VIS else IrLinear()
            img = normalized_input(self.ds, self.sample.observations[band], norm)
            self._reg_inputs[band] = luminance(img).astype(np.float32) if img.ndim == 3 else img
        return self._reg_inputs[band]

    def warps(self, registration: str) -> dict[Band, Warp]:
        if registration not in self._warps:
            ref = self.cfg.reference
            sources = [b for b in self.cfg.bands if b != ref]
            if registration == "oracle":
                self._warps[registration] = dict(self.scene.oracle_warps)
            else:
                reg = {"infinity_homography": InfinityHomographyRegistrar(), "resize": ResizeRegistrar(),
                       "epipolar_sweep": EpipolarSweepRegistrar(self.cfg.sweep_params(self.cfg.similarity))}[registration]
                obs = self.sample.observations
                self._warps[registration] = {
                    b: reg.estimate(obs[ref], obs[b], self.reg_input(ref), self.reg_input(b)) for b in sources}
        return self._warps[registration]

    def noise(self) -> dict:
        """Per-band native sigma (scalar or native-grid map), the VIS chain and provenance."""
        if self._noise is None:
            entry = {e.sample_id: e for e in self.ds.index.entries}[self.sid]
            doc = load_json(self.ds.root / entry.manifest)
            sigma, info, chains = {}, {}, {}
            for band in self.cfg.bands:
                obs = self.sample.observations[band]
                norm = vis_normalizer(self.cfg.vis_input) if band == Band.VIS else ir_normalizer(self.cfg.ir_input)
                normalized_input(self.ds, obs, norm)
                cfg = sample_noise_config(doc, str(band))
                kwargs = {}
                if cfg.detector == "photon":
                    raw = self.ds.load(obs, "rawdn", obs.product("rawdn").channels or None)
                    kwargs["total_e"] = electrons_from_raw(raw, cfg)
                    if isinstance(norm, IrTemperature):
                        kwargs["dsdt"] = estimate_dsdt(self.ds.load(obs, "corrected"), self.ds.load(obs, "tapp"))
                s, i = band_sigma(cfg, norm, fpn=self.cfg.fpn, **kwargs)
                sigma[band], info[str(band)] = s, {k: v for k, v in i.items() if not isinstance(v, np.ndarray)}
                if band == Band.VIS and isinstance(norm, VisRawLinear):
                    chains[band] = vis_luminance_chain(obs.readout)
            self._noise = {"sigma": sigma, "info": info, "chains": chains}
        return self._noise

    def sigma_on_reference(self, warps: dict[Band, Warp]) -> dict:
        """Native sigma per band given on the reference grid (variance resampled through the warp)."""
        n = self.noise()
        out = {}
        for band, s in n["sigma"].items():
            if np.ndim(s) == 0 or band == self.cfg.reference:
                out[band] = s
                continue
            var = apply_warp(np.asarray(s, np.float64) ** 2, warps[band], "linear")
            fill = float(np.nanmedian(var)) if np.isfinite(var).any() else float(np.median(np.asarray(s) ** 2))
            out[band] = np.sqrt(np.where(np.isfinite(var), var, fill))
        return out

    def operator(self, name: str, warps: dict[Band, Warp]):
        if name == "laplacian_pyramid_masked":
            return create(name, levels=self.cfg.levels)
        n = self.noise()
        return create(name, sigma=self.sigma_on_reference(warps), levels=self.cfg.levels, kappa=self.cfg.kappa,
                      chains=n["chains"], noise_info=n["info"])




def _operator_spec(cfg: ProtocolConfig, name: str) -> dict:
    spec = {"operator": name, "levels": cfg.levels}
    if name == "noise_aware_pyramid":
        spec |= {"kappa": cfg.kappa, "fpn": cfg.fpn, "gain_model": "chain"}
    return spec


def _base_spec(cfg: ProtocolConfig) -> dict:
    return {"bands": [str(b) for b in cfg.bands], "vis_input": cfg.vis_input, "ir_input": cfg.ir_input}


def _label(spec: dict) -> str:
    return "|".join(f"{k}={spec[k]}" for k in sorted(spec) if k not in ("bands", "vis_input", "ir_input"))


def _numeric(row: dict) -> dict[str, float]:
    out = {}
    for k, v in row.items():
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
    return out


def _truth_modes(sample: _Sample) -> list[TruthMode | None]:
    return [None, *TruthMode] if sample.has_references else [None]


def _layer_metric(group: str) -> str:
    return f"layer_{group}"


_GROUPS = REPORT_GROUPS


def _zeroshot_method(name: str):
    """Instantiate a zero-shot model; importing fusion.zeroshot needs torch (ImportError without)."""
    from ..fusion import zeroshot  # noqa: F401  (registers the models)

    return create(name)


def band_ablation_spec(cfg: ProtocolConfig, bands: tuple[Band, ...]) -> dict:
    """Spec of one band set: the set, the band set of the common truth reference (all configured
    bands), oracle warps and the masked pyramid -- or the VIS input passed through for ``("vis",)``."""
    single = len(bands) == 1
    return _base_spec(cfg) | {
        "item": "band_ablation", "band_set": [str(b) for b in bands],
        "reference_band_set": [str(b) for b in cfg.bands], "registration": "none" if single else "oracle",
        "truth_mode": TruthMode.FUSION.value, "truth_rule": BASELINE_RULE,
    } | ({"operator": "reference_only"} if single else _operator_spec(cfg, "laplacian_pyramid_masked"))


def zeroshot_input_domain(cfg: ProtocolConfig) -> str:
    """Name of the zero-shot input domain, e.g. ``vis_display+lwir_temperature_260-340K``."""
    lo, hi = ir_normalizer(cfg.ir_input).window_k
    return f"vis_display+lwir_{cfg.ir_input}_{lo:g}-{hi:g}K"


def zeroshot_base_spec(cfg: ProtocolConfig) -> dict:
    """Spec keys shared by every zero-shot row (models and anchor): the input domain and the truth mode."""
    return {"bands": [str(Band.VIS), str(Band.LWIR)], "vis_input": "display", "ir_input": cfg.ir_input,
            "ir_window_k": list(ir_normalizer(cfg.ir_input).window_k), "input_domain": zeroshot_input_domain(cfg),
            "item": "zeroshot", "truth_mode": TruthMode.FUSION.value, "truth_rule": BASELINE_RULE}


def zeroshot_anchor_spec(cfg: ProtocolConfig) -> dict:
    return zeroshot_base_spec(cfg) | {"model": "anchor", "registration": "oracle"} | \
        _operator_spec(cfg, "laplacian_pyramid_masked")


def zeroshot_model_spec(cfg: ProtocolConfig, identity: dict[str, str]) -> dict:
    """``identity`` from ``fusion.zeroshot_assets.model_identity``: model, code commit, weight hashes."""
    return zeroshot_base_spec(cfg) | {"registration": "model_own"} | identity


class _Run:
    def __init__(self, cfg: ProtocolConfig, dataset: ThreeBandDataset, store: ResultStore, out_dir: Path,
                 error_log: _ErrorLog | None = None):
        self.cfg, self.ds, self.store, self.out_dir = cfg, dataset, store, out_dir
        self.variants: dict[str, dict] = {}
        self.errors = 0
        self.error_log = error_log
        self._zs_identity: dict[str, dict[str, str]] = {}
        self._zs_methods: dict = {}

    def key(self, sid: str, spec: dict, metric: str) -> ResultKey:
        fp = fingerprint(spec)
        self.variants[_label(spec)] = {"fingerprint": fp, "spec": spec}
        return ResultKey(self.ds.dataset_id, sid, fp, metric, PROTOCOL_VERSION)

    def pending(self, sid: str, spec: dict, metrics) -> bool:
        return any(not self.store.has(self.key(sid, spec, m)) for m in metrics)

    def put(self, sid: str, spec: dict, metric: str, values: dict[str, float]) -> None:
        key = self.key(sid, spec, metric)
        if not self.store.has(key):
            self.store.append(key, values)

    def matrix_specs(self, sample: _Sample):
        for reg in self.cfg.registration:
            for op in self.cfg.operators:
                for mode in _truth_modes(sample):
                    yield reg, op, mode, _base_spec(self.cfg) | {
                        "item": "matrix", "registration": reg, "truth_mode": mode.value if mode else "none",
                        "similarity": self.cfg.similarity if reg == "epipolar_sweep" else "",
                    } | ({"z_min_m": self.cfg.z_min_m, "n_hypotheses": self.cfg.n_hypotheses}
                         if reg == "epipolar_sweep" else {}) | _operator_spec(self.cfg, op)

    def matrix(self, sample: _Sample) -> None:
        todo = [s for s in self.matrix_specs(sample) if self.pending(sample.sid, s[3], map(_layer_metric, _GROUPS))]
        for reg, op, mode, spec in todo:
            warps = sample.warps(reg)
            epe = epe_columns(sample.scene, warps)
            rows = evaluate_point(sample.scene, sample.operator(op, warps), warps, mode)
            for row in rows:
                values = _numeric(row) | epe | {"calibrated": float(is_calibrated(reg))}
                self.put(sample.sid, spec, _layer_metric(row["layer"]), values)

    def band_ablation(self, sample: _Sample) -> None:
        if not sample.has_references:
            return
        for bands in self.cfg.band_sets:
            spec = band_ablation_spec(self.cfg, bands)
            if not self.pending(sample.sid, spec, map(_layer_metric, _GROUPS)):
                continue
            op = create("reference_only") if len(bands) == 1 else create("laplacian_pyramid_masked",
                                                                            levels=self.cfg.levels)
            for row in evaluate_band_subset(sample.scene, bands, op):
                self.put(sample.sid, spec, _layer_metric(row["layer"]), _numeric(row))

    def sensitivity(self, sample: _Sample) -> None:
        if not sample.has_references:
            return
        s = self.cfg.sensitivity
        # The inverse-depth bias has no correlation length (recorded as None); the smooth field gets
        # one sweep per configured correlation length.
        sweeps = [(PerturbationKind.INVERSE_DEPTH_BIAS, s.bias, None),
                  *((PerturbationKind.SMOOTH_FIELD, s.field, c) for c in s.corr_len_px)]
        modes = [m for m in TruthMode if s.truth_modes is None or str(m) in s.truth_modes]
        for op in (s.operators or self.cfg.operators):
            for kind, amps, corr in sweeps:
                corr_arg = s.corr_len_px[0] if corr is None else corr  # unused by the bias sweep
                for mode in modes:
                    specs = {a: _base_spec(self.cfg) | {"item": "sensitivity", "perturbation": kind.value,
                                                         "amplitude": a, "truth_mode": mode.value,
                                                         "corr_len_px": corr, "seed": s.seed}
                             | _operator_spec(self.cfg, op) for a in amps}
                    if not any(self.pending(sample.sid, sp, map(_layer_metric, _GROUPS)) for sp in specs.values()):
                        continue
                    oracle = sample.warps("oracle")
                    rows = sensitivity_curve(sample.scene, sample.operator(op, oracle),
                                             Sweep(kind, amps, mode, corr_arg, s.seed))
                    for row in rows:
                        self.put(sample.sid, specs[row["amplitude"]], _layer_metric(row["layer"]), _numeric(row))
                if s.inflation:
                    spec = _base_spec(self.cfg) | {"item": "inflation", "perturbation": kind.value,
                                                   "amplitudes": list(amps), "corr_len_px": corr,
                                                   "seed": s.seed} | _operator_spec(self.cfg, op)
                    metrics = [f"amplitude_{a:g}" for a in amps]
                    if self.pending(sample.sid, spec, metrics):
                        rows = inflation_experiment(sample.scene, sample.operator(op, sample.warps("oracle")),
                                                    Sweep(kind, amps, None, corr_arg, s.seed))
                        for row in rows:
                            self.put(sample.sid, spec, f"amplitude_{row['amplitude']:g}", _numeric(row))

    def study_methods(self):
        yield "infinity_homography", "", None, lambda: InfinityHomographyRegistrar()
        for sim in self.cfg.registration_study.similarities:
            p = self.cfg.sweep_params(sim)
            yield "epipolar_sweep", sim, None, lambda p=p: EpipolarSweepRegistrar(p)
            for step in self.cfg.registration_study.free_steps:
                yield "free_search", sim, step, lambda p=p, step=step: FreeSearchRegistrar(p, step_px=step)

    def registration_study(self, sample: _Sample) -> None:
        ref = self.cfg.reference
        for band in (b for b in self.cfg.bands if b != ref):
            for method, sim, step, make in self.study_methods():
                spec = _base_spec(self.cfg) | {"item": "registration_study", "band": str(band), "method": method,
                                               "similarity": sim, "free_step": "" if step is None else step,
                                               "z_min_m": self.cfg.z_min_m, "n_hypotheses": self.cfg.n_hypotheses}
                if not self.pending(sample.sid, spec, ["epe"]):
                    continue
                obs = sample.sample.observations
                t0 = time.perf_counter()
                warp = make().estimate(obs[ref], obs[band], sample.reg_input(ref), sample.reg_input(band))
                seconds = time.perf_counter() - t0
                oracle = sample.scene.oracle_warps[band]
                corr = registration_error(warp, oracle)
                layers = layer_map(sample.scene.views[ref], sample.scene.views[band])
                edge = (layers == Layer.EDGE) & warp.valid & np.isfinite(warp.map_xy).all(axis=2) & \
                    np.isfinite(oracle.map_xy).all(axis=2)
                epe_edge = np.linalg.norm(warp.map_xy - oracle.map_xy, axis=2)[edge]
                fx = sample.scene.geometry[band].fx
                values = {**corr, "epe_median_mrad": corr["epe_median"] / fx * 1000.0,
                          "edge_epe_median": float(np.median(epe_edge)) if epe_edge.size else float("nan"),
                          "edge_n": float(epe_edge.size), "seconds": seconds,
                          "step_px": float(warp.provenance.get("step_px", float("nan")))}
                if "confident" in warp.provenance:
                    values["confident_fraction"] = float(warp.provenance["confident"][oracle.valid].mean())
                self.put(sample.sid, spec, "epe", values)

    def _error(self, sid: str, spec: dict | None, label: str, err: Exception) -> None:
        self.errors += 1
        if self.error_log is not None:
            self.error_log.write(self.ds.dataset_id, sid, fingerprint(spec) if spec else "", label, err)
        else:
            print(f"  error {sid} {label}: {err!r}")

    def zeroshot(self, sample: _Sample) -> None:
        if not sample.has_references:
            return
        spec = zeroshot_anchor_spec(self.cfg)
        try:
            if self.pending(sample.sid, spec, map(_layer_metric, _GROUPS)):
                scene = sample.zeroshot_scene
                op = create("laplacian_pyramid_masked", levels=self.cfg.levels)
                epe = epe_columns(scene, scene.oracle_warps)
                for row in evaluate_point(scene, op, scene.oracle_warps, TruthMode.FUSION):
                    self.put(sample.sid, spec, _layer_metric(row["layer"]),
                             _numeric(row) | epe | {"calibrated": float(is_calibrated("oracle"))})
        except Exception as e:
            self._error(sample.sid, spec, _label(spec), e)
        for name in self.cfg.zeroshot:
            spec = None
            try:
                if name not in self._zs_identity:
                    self._zs_identity[name] = model_identity(name)
                spec = zeroshot_model_spec(self.cfg, self._zs_identity[name])
                if not self.pending(sample.sid, spec, map(_layer_metric, _GROUPS)):
                    continue
                if name not in self._zs_methods:
                    self._zs_methods[name] = _zeroshot_method(name)
                rows = evaluate_zeroshot(sample.zeroshot_scene, sample.vis_display_rgb(), self._zs_methods[name])
                for row in rows:
                    self.put(sample.sid, spec, _layer_metric(row["layer"]), _numeric(row))
            except Exception as e:
                self._error(sample.sid, spec, _label(spec) if spec else f"item=zeroshot|model={name}", e)

    def complementarity(self) -> None:
        target = self.out_dir / "complementarity.md"
        if target.exists():
            print("complementarity: report exists, skipped")
            return
        report = analyse_dataset(self.ds, AnalysisSettings(view_group_blocks=self.cfg.view_group_blocks))
        tmp = target.with_suffix(".md.tmp")
        tmp.write_text(render_markdown(report), encoding="utf-8", newline="\n")
        tmp.replace(target)
        print(f"complementarity: wrote {target}")




def summarize_protocol(out_dir: Path, dataset: ThreeBandDataset) -> tuple[Path, Path]:
    """protocol_summary.csv (every value, mean over samples per spec x metric x condition)
    and protocol_summary.md (the matrix item's layered table)."""
    variants = load_json(out_dir / "variants.json")
    spec_of = {v["fingerprint"]: v["spec"] for v in variants.values()}
    conditions = {s.sample_id: s.condition.value for s in dataset}
    groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with ResultStore(out_dir / "results.jsonl") as store:
        for key, values in store.rows():
            spec = spec_of.get(key.method_fingerprint)
            if spec is None:
                continue
            for cond in (conditions.get(key.sample_id, "unknown"), "all"):
                g = groups[(key.method_fingerprint, key.metric, cond)]
                for name, v in values.items():
                    g[name].append(v)
    spec_keys = sorted({k for s in spec_of.values() for k in s if k not in ("bands", "vis_input", "ir_input")})
    value_keys = sorted({n for g in groups.values() for n in g})
    rows = []
    for (fp, metric, cond), g in sorted(groups.items(), key=lambda kv: (str(spec_of[kv[0][0]]), kv[0][1:])):
        means = {}
        for n in value_keys:
            finite = [v for v in g.get(n, []) if math.isfinite(v)]
            means[n] = sum(finite) / len(finite) if finite else float("nan")
        n_samples = max((len(v) for v in g.values()), default=0)
        rows.append({**{k: spec_of[fp].get(k, "") for k in spec_keys}, "metric": metric, "condition": cond,
                     "n_samples": n_samples, **means})
    csv_path = out_dir / "protocol_summary.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=[*spec_keys, "metric", "condition", "n_samples", *value_keys],
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    tmp.replace(csv_path)

    md = ["# Protocol summary: matrix (means over samples)", "",
          "| condition | layer | registration | operator | truth mode | n | truth PSNR | SCD | Qabf | "
          "hp dev x1e-3 | EPE px |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    def fmt(v, scale=1.0, digits=2):
        return "" if not isinstance(v, float) or not math.isfinite(v) else f"{v * scale:.{digits}f}"
    for r in rows:
        if r.get("item") != "matrix":
            continue
        md.append(f"| {r['condition']} | {r['metric'].removeprefix('layer_')} | {r['registration']} | "
                  f"{r['operator']} | {r['truth_mode']} | {r['n_samples']} | {fmt(r.get('truth_PSNR'))} | "
                  f"{fmt(r.get('SCD'), digits=3)} | {fmt(r.get('Qabf'), digits=3)} | "
                  f"{fmt(r.get('hp_dev_RMSE'), 1e3)} | {fmt(r.get('epe_mean_px'))} |")
    ba = [r for r in rows if r.get("item") == "band_ablation"]
    if ba:
        md += ["", "# Protocol summary: band_ablation (means over samples; FUSION truth mode, common reference "
               "from all configured bands)", "",
               "| condition | layer | band set | reference bands | n | truth PSNR | demeaned PSNR | mean offset | "
               "truth SSIM | cleansrc Qabf |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in ba:
            md.append(f"| {r['condition']} | {r['metric'].removeprefix('layer_')} | {'+'.join(r['band_set'])} | "
                      f"{'+'.join(r['reference_band_set'])} | {r['n_samples']} | {fmt(r.get('truth_PSNR'))} | "
                      f"{fmt(r.get('truth_PSNR_demeaned'))} | {fmt(r.get('truth_mean_offset'), digits=4)} | "
                      f"{fmt(r.get('truth_SSIM'), digits=3)} | {fmt(r.get('cleansrc_Qabf'), digits=3)} |")
    zs = [r for r in rows if r.get("item") == "zeroshot"]
    if zs:
        md += ["", "# Protocol summary: zeroshot (means over samples; FUSION truth mode, zero-shot input domain)", "",
               "| condition | layer | model | registration | n | truth PSNR | truth SSIM | hp dev x1e-3 | EPE px | "
               "runtime s |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in zs:
            md.append(f"| {r['condition']} | {r['metric'].removeprefix('layer_')} | {r['model']} | "
                      f"{r['registration']} | {r['n_samples']} | {fmt(r.get('truth_PSNR'))} | "
                      f"{fmt(r.get('truth_SSIM'), digits=3)} | {fmt(r.get('hp_dev_RMSE'), 1e3)} | "
                      f"{fmt(r.get('epe_mean_px'))} | {fmt(r.get('runtime_s'), digits=3)} |")
    md_path = out_dir / "protocol_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    return csv_path, md_path




def run_protocol(config_path: Path, *, limit: int | None = None, samples: list[str] | None = None,
                 out_root: Path | None = None) -> int:
    """Run (or resume) the protocol; 0 = clean, 1 = some (sample, item) failed (retried next run)."""
    from .driver import repo_root

    cfg = load_protocol_config(config_path)
    dataset = ThreeBandDataset(native_path(cfg.dataset))
    out_dir = (out_root or repo_root() / "build" / "experiments") / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    copy = out_dir / "config.toml"
    src = Path(config_path).read_bytes()
    if copy.exists() and copy.read_bytes() != src:
        raise ValueError(f"{copy} differs from {config_path}; a changed config needs a new experiment name")
    if not copy.exists():
        copy.write_bytes(src)
    ids = dataset.sample_ids()
    if samples is not None:
        unknown = set(samples) - set(ids)
        if unknown:
            raise ValueError(f"--samples not in dataset: {sorted(unknown)}")
        ids = tuple(s for s in ids if s in set(samples))
    if limit is not None:
        ids = ids[:limit]
    per_sample = [i for i in ("matrix", "band_ablation", "sensitivity", "registration_study", "zeroshot")
                  if i in cfg.items]
    t0 = time.monotonic()
    error_log = _ErrorLog(out_dir / "errors.jsonl")
    with ResultStore(out_dir / "results.jsonl") as store:
        run = _Run(cfg, dataset, store, out_dir, error_log)
        variants_path = out_dir / "variants.json"
        if variants_path.exists():
            run.variants.update(load_json(variants_path))
        for n, sid in enumerate(ids, 1):
            before = len(list(store.rows()))
            sample = _Sample(dataset, sid, cfg)
            for item in per_sample:
                try:
                    getattr(run, item)(sample)
                except Exception as e:
                    run.errors += 1
                    print(f"  error {sid} {item}: {e!r}")
            atomic_json(variants_path, run.variants)
            added = len(list(store.rows())) - before
            print(f"sample {n}/{len(ids)} {sid}: rows +{added}, errors {run.errors}, "
                  f"elapsed {time.monotonic() - t0:.1f}s", flush=True)
        if "complementarity" in cfg.items:
            try:
                run.complementarity()
            except Exception as e:
                run.errors += 1
                print(f"  error complementarity: {e!r}")
        atomic_json(variants_path, run.variants)
    error_log.close()
    if run.variants:
        csv_path, md_path = summarize_protocol(out_dir, dataset)
        print(f"wrote {csv_path} and {md_path}")
    return 1 if run.errors else 0
