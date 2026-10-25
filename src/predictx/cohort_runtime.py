"""Source-record reconstruction for retrospective inference parity checks."""

import numpy as np
import pandas as pd

from .breast_data import breast_senegal
from .data import LOADERS, derived
from .processing import process_panel
from .schema import SPECS, LabPanel

LOADERS_ACTIVE = {**LOADERS, "breast_senegal_qc": breast_senegal}


def source_record(row, record_id, *, female=False):
    age = row.get("age", np.nan)
    if pd.isna(age) or not 0 <= age <= 120:
        return None, None, {"age": "missing_or_invalid_patient_age"}
    sex = "female" if female else {0.0: "female", 1.0: "male"}.get(row.get("sex"))
    measurements = {}
    rejected = {}
    for k, v in row.items():
        if pd.isna(v):
            continue
        if k == "wbc_cells":
            k = "wbc"
            v = float(v) / 1000
        if k not in SPECS:
            continue
        unit, lo, hi = SPECS[k]
        if not np.isfinite(v) or not lo <= v <= hi or (k.endswith("_ordinal") and int(v) != v):
            rejected[k] = "invalid_source_measurement"
            continue
        measurements[k] = {
            "value": float(v),
            "unit": unit,
            "specimen": "urine" if k.startswith("urine_") else "serum",
            "collected_at": "2026-10-25T00:00:00Z",
            "quality": "acceptable",
        }
    panel = LabPanel(
        patient_id=str(record_id),
        age=float(age),
        equation_sex=sex,
        measurements=measurements,
    )
    record = process_panel(panel)
    # Retain both schema-level source masking and shared cross-field QC provenance.
    rejected.update(record.rejected)
    return panel, record, rejected


def canonical_frame(cohort, *, raw=None):
    rows = []
    diagnostics = []
    for i, row in (raw if raw is not None else cohort.X).iterrows():
        panel, record, rejected = source_record(
            row, cohort.ids[i], female=cohort.name.startswith("breast_senegal")
        )
        rows.append(dict(record.values) if record is not None else {})
        diagnostics.append({"panel": panel, "record": record, "rejected": rejected})
    return pd.DataFrame(rows).reindex(columns=derived(cohort.X).columns), diagnostics
