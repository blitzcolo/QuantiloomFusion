"""Experiment configs, variant expansion, the sample-major driver, summaries. Depends on everything except datagen."""
from .config import ExperimentConfig, FusionSpec, load_config
from .variants import Variant, expand
