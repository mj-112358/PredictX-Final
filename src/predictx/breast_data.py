"""Routine CBC/CRP cohort of pretreatment Senegalese women."""

import numpy as np
import pandas as pd

from .data import ROOT, Cohort
from .processing import cross_field_qc
from .schema import SPECS

CBC_MAP = {
    "Age": "age",
    "CRP": "crp",
    "RBC": "rbc",
    "Hb": "hemoglobin",
    "Ht": "hematocrit",
    "MCV": "mcv",
    "MCH": "mch",
    "MCHC": "mchc",
    "Platelets": "platelets",
    "WBC": "wbc",
    "Neutrophils": "neutrophils",
    "Lymphocytes": "lymphocytes",
    "Monocytes": "monocytes",
    "Eosinophils": "eosinophils",
    "Basophils": "basophils",
}


def _clean(x):
    """Mask broad physically implausible source cells, record exclusions in the audit."""
    x = x.apply(pd.to_numeric, errors="coerce")
    for col in x:
        if col in SPECS:
            _, lo, hi = SPECS[col]
            x[col] = x[col].where(x[col].between(lo, hi))
    # Apply the same cross-field consistency rule used during request processing.
    for i, row in x.iterrows():
        _, _, rejected = cross_field_qc(row.dropna().to_dict())
        if rejected:
            x.loc[i, rejected] = np.nan
    return x


def breast_senegal(clean=True):
    frames = []
    ys = []
    ids = []
    stages = []
    subtypes = []
    for sheet, label in [("TNBC", 1), ("HDBC", 1), ("Healthy controls", 0)]:
        d = pd.read_excel(ROOT / "data/raw/breast_senegal_v1.xlsx", sheet_name=sheet)
        d.columns = d.columns.str.strip()
        raw = d[list(CBC_MAP)].rename(columns=CBC_MAP)
        frames.append(_clean(raw) if clean else raw.apply(pd.to_numeric, errors="coerce"))
        ys.extend([label] * len(d))
        ids.extend(d["Numéro"].astype(str).str.strip())
        stage_col = next((c for c in d if c.startswith("Clinical stage")), None)
        stages.extend(d[stage_col].astype(str) if stage_col else ["control"] * len(d))
        subtypes.extend([sheet] * len(d))
    x = pd.concat(frames, ignore_index=True)
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated source patient ID; group before splitting")
    return Cohort(
        "breast_senegal_qc",
        x,
        pd.DataFrame({"breast_cancer": ys}, dtype=float),
        np.array(ids),
        pd.DataFrame(
            {
                "age_band": np.where(x.age >= 50, "50+", "<50"),
                "subtype": subtypes,
                "stage": stages,
            }
        ),
        {
            "breast_cancer": "Pretreatment breast cancer (40 TNBC + 40 hormone-dependent cases) versus 40 healthy women, Senegal; clinically diagnosed case-control endpoint."
        },
    )
