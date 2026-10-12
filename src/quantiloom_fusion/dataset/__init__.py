"""Strict parsing and typed reading of the published dataset contract (dataset.index.json, sample.json, products), EXR IO, split_group folds, and an optional torch adapter. Depends on core; never imports datagen."""
from .reader import ThreeBandDataset
from .schema import (
    DatasetIndex,
    IndexEntry,
    PlanSummary,
    parse_index,
    parse_plan_summary,
    parse_sample,
    product_name,
)
from .splits import leave_one_group_out
from .types import (
    CameraGeometry,
    Condition,
    Observation,
    Product,
    Sample,
    SampleParameters,
    SignalSemantics,
    condition_of,
)
