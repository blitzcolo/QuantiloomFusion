"""Experiments: config strictness, variant expansion, driver end-to-end."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.dataset.reader import ThreeBandDataset
from quantiloom_fusion.evaluation import ResultKey, ResultStore
from quantiloom_fusion.experiments import expand, load_config
from quantiloom_fusion.experiments.driver import run
from quantiloom_fusion.experiments.variants import fingerprint

import dataset_fixture

FULL_TOML = """
name = "t-exp"
dataset = "{dataset}"
reference = "vis"
band_sets = [["vis"], ["vis", "lwir"], ["vis", "mwir"], ["vis", "mwir", "lwir"]]
vis_input = ["display", "raw_percentile"]
ir_input = ["linear"]
registration = ["oracle", "infinity_homography", "resize"]
outputs = ["grey", "colour"]
metrics = ["EN", "SD", "SF", "AG", "SCD", "Qabf", "MS_SSIM"]

[[fusion]]
name = "average"

[[fusion]]
name = "max_abs"

[[fusion]]
name = "laplacian_pyramid"
levels = 4
"""

SMALL_TOML = """
name = "t-small"
dataset = "{dataset}"
reference = "vis"
band_sets = [["vis"], ["vis", "lwir"]]
vis_input = ["display"]
ir_input = ["linear"]
registration = ["resize"]
outputs = ["grey", "colour"]
metrics = ["EN", "SD"]

[[fusion]]
name = "average"
"""


def _write(tmp_path, body, dataset):
    path = tmp_path / "exp.toml"
    path.write_text(body.format(dataset=str(dataset).replace("\\", "/")),
                    encoding="utf-8", newline="\n")
    return path


def test_config_strictness(tmp_path):
    cfg = load_config(_write(tmp_path, FULL_TOML, "/data/ds"))
    assert cfg.name == "t-exp" and cfg.reference == Band.VIS
    assert cfg.fusion[2].params == {"levels": 4}

    for bad, key in [
        ('name="x"\ndataset="d"\nreference="vis"\nbogus=1\n', "bogus"),
        (FULL_TOML.replace('reference = "vis"', 'reference = "nir"'), "reference"),
        (FULL_TOML.replace('"vis", "lwir"', '"vis", "nir"'), "band_sets"),
        (FULL_TOML.replace('"display", ', '"s16", '), "vis_input"),
        (FULL_TOML.replace('"linear"', '"magic"'), "ir_input"),
        (FULL_TOML.replace('"oracle", ', '"sift", '), "registration"),
        (FULL_TOML.replace('"grey", ', '"hdr", '), "outputs"),
        (FULL_TOML.replace('"EN", ', '"ZZ", '), "metrics"),
        (FULL_TOML.replace('name = "average"', 'name = "nope"', 1), "fusion"),
    ]:
        with pytest.raises(ValueError, match=key):
            load_config(_write(tmp_path, bad, "/data/ds"))


def test_config_band_set_must_contain_reference(tmp_path):
    body = FULL_TOML.replace('band_sets = [["vis"], ["vis", "lwir"], ["vis", "mwir"], ["vis", "mwir", "lwir"]]',
                             'band_sets = [["vis"], ["mwir", "lwir"]]')
    with pytest.raises(ValueError, match="band_sets"):
        load_config(_write(tmp_path, body, "/data/ds"))


def test_variant_expansion_counts_and_collapse(tmp_path):
    cfg = load_config(_write(tmp_path, FULL_TOML, "/data/ds"))
    variants = expand(cfg)
    # ["vis"] collapses: 2 vis_in x 1 ir x 1 reg x 1 fusion x 2 outputs = 4
    single = [v for v in variants if v.band_set == (Band.VIS,)]
    assert len(single) == 4
    assert {(v.registration, v.fusion_name, v.ir_input) for v in single} == {
        ("none", "reference_only", "none")}
    # 3 multi sets x 2 vis x 1 ir x 3 reg x 3 fusion x 2 outputs = 108
    assert len(variants) == 4 + 108
    assert len({v.method_fingerprint for v in variants}) == len(variants)
    labels = {v.label for v in variants}
    assert "vis+lwir|vis=raw_percentile|ir=linear|reg=oracle|average|colour" in labels


def test_fingerprint_stability_and_sensitivity(tmp_path):
    a = load_config(_write(tmp_path, FULL_TOML, "/data/ds"))
    reordered = FULL_TOML.replace(
        'metrics = ["EN", "SD", "SF", "AG", "SCD", "Qabf", "MS_SSIM"]',
        'metrics = ["MS_SSIM", "AG", "EN", "SF", "Qabf", "SCD", "SD"]')
    b = load_config(_write(tmp_path, reordered, "/data/ds"))
    fa = {v.label: v.method_fingerprint for v in expand(a)}
    fb = {v.label: v.method_fingerprint for v in expand(b)}
    assert fa == fb  # metric order must not change fingerprints

    changed = FULL_TOML.replace("levels = 4", "levels = 5")
    c = load_config(_write(tmp_path, changed, "/data/ds"))
    fc = {v.label: v.method_fingerprint for v in expand(c)}
    assert fa.keys() == fc.keys()
    assert all(fa[l] == fc[l] for l in fa if "laplacian" not in l)
    assert all(fa[l] != fc[l] for l in fa if "laplacian" in l)


# --- driver -----------------------------------------------------------------

@pytest.fixture
def fixture_dataset(tmp_path):
    return dataset_fixture.build_fixture(tmp_path / "ds", samples=2)


def _run_small(tmp_path, fixture_dataset, config=None, **kw):
    path = _write(tmp_path, config or SMALL_TOML, fixture_dataset)
    return run(path, out_root=tmp_path / "out", **kw), tmp_path / "out" / "t-small"


def test_driver_run_produces_all_rows(tmp_path, fixture_dataset):
    rc, out = _run_small(tmp_path, fixture_dataset)
    assert rc == 0
    cfg = load_config(_write(tmp_path, SMALL_TOML, fixture_dataset))
    variants = expand(cfg)
    with ResultStore(out / "results.jsonl") as store:
        rows = list(store.rows())
        # 2 band sets: ["vis"] -> 2 variants (grey+colour x 1 vis x ref_only),
        # ["vis","lwir"] -> 1 vis x 1 ir x 1 reg x 1 fusion x 2 outputs = 2;
        # total 4 variants x 2 metrics x 2 samples = 16 rows + EPE rows.
        metric_rows = [r for r in rows if r[0].metric != "registration_epe"]
        epe_rows = [r for r in rows if r[0].metric == "registration_epe"]
        assert len(metric_rows) == 4 * 2 * 2
        assert len(epe_rows) == 2  # resize x lwir x 2 samples
        assert all(all(np.isfinite(v) for v in vals.values())
                   for _, vals in metric_rows)
    assert (out / "config.toml").read_bytes() == _write(
        tmp_path, SMALL_TOML, fixture_dataset).read_bytes()
    assert json.loads((out / "variants.json").read_text())
    assert (out / "summary.csv").exists()


def test_single_band_source_ref_metrics_record_nan(tmp_path, fixture_dataset):
    toml = (SMALL_TOML
            .replace('band_sets = [["vis"], ["vis", "lwir"]]',
                     'band_sets = [["vis"]]')
            .replace('metrics = ["EN", "SD"]', 'metrics = ["SCD", "EN"]'))
    cfg = _write(tmp_path, toml, fixture_dataset)
    out = tmp_path / "out" / "t-small"
    assert run(cfg, out_root=tmp_path / "out") == 0
    with ResultStore(out / "results.jsonl") as store:
        rows = list(store.rows())
    scd = [v for k, v in rows if k.metric == "SCD"]
    en = [v for k, v in rows if k.metric == "EN"]
    assert len(scd) == len(en) == 2 * 2  # 2 outputs x 2 samples
    assert all(math.isnan(v["SCD"]) for v in scd)
    assert all(math.isfinite(v["EN"]) for v in en)


def test_second_run_appends_nothing_and_loads_nothing(tmp_path, fixture_dataset,
                                                    monkeypatch):
    _run_small(tmp_path, fixture_dataset)
    calls = []
    real_load = ThreeBandDataset.load

    def counting(self, *a, **k):
        calls.append(a)
        return real_load(self, *a, **k)

    monkeypatch.setattr(ThreeBandDataset, "load", counting)
    rc, out = _run_small(tmp_path, fixture_dataset)
    assert rc == 0
    assert calls == []
    with ResultStore(out / "results.jsonl") as store:
        assert len(list(store.rows())) == 18  # unchanged


def test_resume_after_crash(tmp_path, fixture_dataset, monkeypatch):
    # Kill the run mid-store: an exception type the driver's per-variant
    # handler does not swallow.
    real_append = ResultStore.append
    state = {"n": 0}

    def crash_after_5(self, key, values):
        state["n"] += 1
        if state["n"] == 6:
            raise KeyboardInterrupt("simulated ctrl+c")
        return real_append(self, key, values)

    monkeypatch.setattr(ResultStore, "append", crash_after_5)
    with pytest.raises(KeyboardInterrupt):
        _run_small(tmp_path, fixture_dataset)
    monkeypatch.undo()
    rc, out = _run_small(tmp_path, fixture_dataset)
    assert rc == 0
    with ResultStore(out / "results.jsonl") as store:
        rows = list(store.rows())
        assert len(rows) == 18  # complete, no duplicates


def test_failing_variant_lands_in_errors_jsonl(tmp_path, fixture_dataset,
                                             monkeypatch):
    import quantiloom_fusion.experiments.driver as drv

    real_create = drv.create

    def explode(name, **kw):
        if name == "max_abs":
            raise RuntimeError("boom")
        return real_create(name, **kw)

    cfg_toml = SMALL_TOML + '\n[[fusion]]\nname = "max_abs"\n'
    monkeypatch.setattr(drv, "create", explode)
    rc, out = _run_small(tmp_path, fixture_dataset, config=cfg_toml)
    assert rc == 1
    errors = [json.loads(l) for l in
              (out / "errors.jsonl").read_text().splitlines() if l.strip()]
    assert errors and all("max_abs" in e["label"] for e in errors)
    # Non-failed variants still completed: variants are 2 (vis only) + 4
    # (vis+lwir x 2 fusions x 2 outputs); the 4 max_abs rows per sample are
    # missing but reference_only and average rows exist.
    with ResultStore(out / "results.jsonl") as store:
        metric_rows = [k for k, _ in store.rows() if k.metric != "registration_epe"]
        assert len(metric_rows) == (2 + 2) * 2 * 2  # ref_only + average, 2 metrics, 2 samples


def test_changed_config_refuses(tmp_path, fixture_dataset):
    _run_small(tmp_path, fixture_dataset)
    cfg_path = _write(tmp_path, SMALL_TOML.replace('metrics = ["EN", "SD"]',
                                                   'metrics = ["EN"]'),
                      fixture_dataset)
    with pytest.raises(ValueError, match="differs"):
        run(cfg_path, out_root=tmp_path / "out")


def test_summary_csv_contents(tmp_path, fixture_dataset):
    rc, out = _run_small(tmp_path, fixture_dataset)
    assert rc == 0
    rows = list(csv.DictReader((out / "summary.csv").open(newline="")))
    assert rows and set(rows[0]) == {"label", "condition", "metric", "n", "mean", "std"}
    assert {r["condition"] for r in rows} >= {"all", "day"}
    en = [r for r in rows if r["metric"] == "EN" and r["condition"] == "all"]
    assert en and all(int(r["n"]) == 2 for r in en)
    epe = [r for r in rows if r["metric"] == "epe_mean"]
    assert epe and all(r["label"].startswith("reg=resize") for r in epe)


# --- real dataset -----------------------------------------------------------

DATASET = os.environ.get("QUANTILOOM_TEST_DATASET")


@pytest.mark.real_dataset
@pytest.mark.skipif(not DATASET, reason="set QUANTILOOM_TEST_DATASET for real dataset tests")
def test_real_dataset_one_sample_per_registrar(tmp_path):
    toml = """
name = "t-real"
dataset = "{dataset}"
reference = "vis"
band_sets = [["vis", "lwir"]]
vis_input = ["raw_percentile"]
ir_input = ["linear"]
registration = ["{reg}"]
outputs = ["grey"]
metrics = ["EN"]

[[fusion]]
name = "average"
"""
    first = ThreeBandDataset(DATASET).sample_ids()[0]
    for reg in ("oracle", "infinity_homography", "resize"):
        path = tmp_path / f"exp-{reg}.toml"
        path.write_text(toml.format(dataset=str(Path(DATASET).resolve()).replace("\\", "/"), reg=reg),
                        encoding="utf-8", newline="\n")
        rc = run(path, out_root=tmp_path / f"out-{reg}", samples=[first])
        assert rc == 0
        out = tmp_path / f"out-{reg}" / "t-real"
        with ResultStore(out / "results.jsonl") as store:
            rows = list(store.rows())
            assert any(k.metric == "EN" for k, _ in rows)
            if reg != "oracle":
                assert any(k.metric == "registration_epe" for k, _ in rows)
