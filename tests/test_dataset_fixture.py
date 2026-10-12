"""The dataset fixture produces sample.jsons the schema parser accepts."""

from pathlib import Path

from quantiloom_fusion.core import jsonio
from quantiloom_fusion.core.bands import BANDS
from quantiloom_fusion.dataset import schema

from dataset_fixture import build_fixture


def test_fixture_samples_parse(tmp_path):
    root = build_fixture(tmp_path / "ds", samples=2)
    index = schema.parse_index(jsonio.load_json(root / "dataset.index.json"))
    plan = schema.parse_plan_summary(jsonio.load_json(root / "dataset.plan.json"))
    assert index.dataset_id == plan.dataset_id and index.complete_triplets == 2
    for entry in index.entries:
        s = schema.parse_sample(jsonio.load_json(root / entry.manifest))
        assert set(s.observations) == set(BANDS)
        for band, obs in s.observations.items():
            for name in ("measurement", "rawdn", "corrected", "display", "rendered_reference",
                         "truth_camera_z", "truth_instance_id", "truth_material_id", "preview"):
                assert name in obs.products, (band, name)
            if band.value != "vis":
                assert "tapp" in obs.products
