"""One immutable request record and lazy model-compatible feature cache."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .biomarkers import derive
from .compatibility import transformation_key
from .runtime import DiseaseModel
from .schema import LabPanel

DIFFERENTIAL = ["neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils"]


def cross_field_qc(values, tolerance: float = 0.10):
    values = dict(values)
    notes = []
    rejected = []
    if "wbc" in values and all(k in values for k in DIFFERENTIAL):
        if sum(values[k] for k in DIFFERENTIAL) > (1 + tolerance) * values["wbc"]:
            rejected = DIFFERENTIAL.copy()
            notes.append(
                f"Absolute differential sum exceeds total WBC by >{tolerance:.0%}; differential counts withheld"
            )
            for k in rejected:
                values.pop(k)
    return values, notes, rejected


@dataclass
class ProcessedRecord:
    values: Mapping[str, float]
    biomarkers: dict[str, Any]
    rejected: Mapping[str, str]
    counters: dict[str, int]
    frames: dict[tuple[str, ...], pd.DataFrame] = field(default_factory=dict)
    transformed: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)


def process_panel(panel: LabPanel) -> ProcessedRecord:
    accepted = {}
    rejected = {}
    normalized = panel.canonical_snapshot()
    conversions = len(panel.measurements)
    for name, m in panel.measurements.items():
        if not m.available:
            rejected[name] = "unavailable"
            continue
        if m.quality != "acceptable":
            rejected[name] = "quality_" + m.quality
            continue
        accepted[name] = normalized[name]
    accepted, notes, invalid = cross_field_qc(accepted)
    rejected.update({k: "inconsistent_differential_count" for k in invalid})
    features = derive(panel, canonical=accepted)
    features["notes"].extend(notes)
    values = dict(accepted)
    values.update(features["values"])
    # Project canonical WBC into the source CKD concentration unit.
    if "wbc" in values:
        values["wbc_cells"] = values["wbc"] * 1000
    values["age"] = panel.age
    if panel.equation_sex is not None:
        values["sex"] = {"female": 0.0, "male": 1.0}[panel.equation_sex]
    return ProcessedRecord(
        MappingProxyType(values),
        features,
        MappingProxyType(rejected),
        {
            "panel_processing_calls": 1,
            "canonical_measurements": conversions,
            "derived_features": len(features["values"]),
            "projection_builds": 0,
            "transform_calls": 0,
            "transform_cache_hits": 0,
        },
    )


def project(record: ProcessedRecord, columns: Sequence[str]) -> pd.DataFrame:
    key = tuple(columns)
    if key not in record.frames:
        record.frames[key] = pd.DataFrame([{k: record.values.get(k, np.nan) for k in columns}])
        record.counters["projection_builds"] += 1
    return record.frames[key]


def transform(record: ProcessedRecord, bundle: DiseaseModel, signature: str) -> np.ndarray:
    key = transformation_key(bundle, signature)
    if key not in record.transformed:
        a = bundle.preprocessor.transform(project(record, bundle.preprocessor.columns))
        a.setflags(write=False)
        record.transformed[key] = a
        record.counters["transform_calls"] += 1
    else:
        record.counters["transform_cache_hits"] += 1
    return record.transformed[key]
