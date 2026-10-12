"""Dataset schema identity and per-band constants for the generator."""

SCHEMA = "quantiloomfusion.env_dataset"
VERSION = 1
BANDS = ("vis", "mwir", "lwir")
MODES = {"vis": "vis_hero", "mwir": "mwir_fused", "lwir": "lwir_fused"}
PATH_KEYS = {"gltf", "usd", "file", "basis_file", "materials_json", "solar_lut",
             "forcing_file", "model_pack", "environment_map", "temperature_texture",
             "known_psf_path", "emissive_curve", "spectral_weight_texture"}
