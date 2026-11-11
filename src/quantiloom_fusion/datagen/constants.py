"""Dataset schema identity and per-band constants for the generator."""

SCHEMA = "quantiloomfusion.env_dataset"
VERSION = 1
SAMPLE_SCHEMA = "quantiloomfusion.sample"
# v2 adds "references": the co-view noise-free renders of reference_subset.
SAMPLE_VERSION = 2
# Capture key of a co-view reference: state/attempts/samples use "ref_<band>".
REFERENCE_PREFIX = "ref_"
BANDS = ("vis", "mwir", "lwir")
MODES = {"vis": "vis_hero", "mwir": "mwir_fused", "lwir": "lwir_fused"}
PATH_KEYS = {"gltf", "usd", "file", "basis_file", "materials_json", "solar_lut",
             "forcing_file", "model_pack", "environment_map", "temperature_texture",
             "known_psf_path", "emissive_curve", "spectral_weight_texture"}
