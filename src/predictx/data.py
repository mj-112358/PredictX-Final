"""Source cohort adapters and participant identifiers for fixed-partition evaluation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .biomarkers import derived_frame as derived

__all__ = ["Cohort", "LOADERS", "ROOT", "derived"]

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Cohort:
    name: str
    X: pd.DataFrame
    y: pd.DataFrame
    ids: np.ndarray
    groups: pd.DataFrame
    definition: dict
    excluded: int = 0


def _cohort(name, x, y, definition):
    # Remove duplicate feature/label rows before constructing partition identifiers.
    all_data = pd.concat([x, y], axis=1)
    keep = ~all_data.duplicated()
    ids = np.array([f"{name}:row:{i}" for i in x.index[keep]])
    group = pd.DataFrame(index=x.index)
    if "age" in x:
        group["age_band"] = np.where(x.age >= 60, "60+", "<60")
    if "sex" in x:
        group["sex"] = x.sex
    return Cohort(
        name,
        x.loc[keep].reset_index(drop=True),
        y.loc[keep].reset_index(drop=True),
        ids,
        group.loc[keep].reset_index(drop=True),
        definition,
        int((~keep).sum()),
    )


def ilpd():
    cols = [
        "age",
        "sex",
        "bilirubin",
        "direct_bilirubin",
        "alp",
        "alt",
        "ast",
        "total_protein",
        "albumin",
        "albumin_globulin",
        "label",
    ]
    d = pd.read_csv(
        ROOT / "data/raw/Indian Liver Patient Dataset (ILPD).csv",
        header=None,
        names=cols,
    )
    d["sex"] = d.sex.map({"Male": 1.0, "Female": 0.0})
    # Published A/G is redundant with albumin and total protein; derived ablation recomputes it.
    return _cohort(
        "ilpd",
        d.drop(columns=["label", "albumin_globulin"]),
        pd.DataFrame({"liver_class": (d.label == 1).astype(float)}),
        {
            "liver_class": "Selector 1: liver patient; 2: non-liver patient. Heterogeneous liver-patient classification."
        },
    )


def ckd():
    d = pd.read_csv(ROOT / "data/raw/ckd.csv", na_values=["?", "\t?"])
    d = d.apply(lambda col: col.map(lambda v: v.strip() if isinstance(v, str) else v))
    cols = {
        "age": "age",
        "bgr": "random_glucose",
        "bu": "urea",
        "sc": "creatinine",
        "sod": "sodium",
        "pot": "potassium",
        "hemo": "hemoglobin",
        "pcv": "hematocrit",
        "wbcc": "wbc_cells",
        "rbcc": "rbc",
        "sg": "urine_specific_gravity",
        "al": "urine_albumin_ordinal",
        "su": "urine_glucose_ordinal",
    }
    x = d[list(cols)].apply(pd.to_numeric, errors="coerce").rename(columns=cols)
    y = pd.DataFrame({"ckd_class": d["class"].map({"ckd": 1.0, "notckd": 0.0})})
    if y.isna().any().any():
        raise ValueError("Unknown CKD label")
    return _cohort(
        "ckd",
        x,
        y,
        {
            "ckd_class": "UCI hospital CKD class. No adjudicated chronicity or stage; includes biomarkers used in diagnosis. Excludes comorbidity labels, symptoms and disease-history features."
        },
    )


NH_MAP = {
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "LBXSGL": "random_glucose",
    "LBXGH": "hba1c",
    "LBXSCR": "creatinine",
    "LBXSBU": "bun",
    "LBXSUA": "uric_acid",
    "LBXSATSI": "alt",
    "LBXSASSI": "ast",
    "LBXSAPSI": "alp",
    "LBXSGTSI": "ggt",
    "LBXSTB": "bilirubin",
    "LBXSAL": "albumin",
    "LBXSTP": "total_protein",
    "LBXPLTSI": "platelets",
    "LBXWBCSI": "wbc",
    "LBDNENO": "neutrophils",
    "LBDLYMNO": "lymphocytes",
    "LBDMONO": "monocytes",
    "LBXRBCSI": "rbc",
    "LBXHGB": "hemoglobin",
    "LBXHCT": "hematocrit",
    "LBDHDD": "hdl",
    "LBXTC": "total_cholesterol",
    "URDACT": "urine_acr",
}


def nhanes():
    files = [
        "DEMO",
        "BIOPRO",
        "CBC",
        "GHB",
        "HDL",
        "TCHOL",
        "ALB_CR",
        "DIQ",
        "KIQ_U",
        "MCQ",
    ]
    parts = [pd.read_sas(ROOT / f"data/raw/{p}_J.xpt").set_index("SEQN") for p in files]
    d = parts[0]
    for p in parts[1:]:
        if not p.index.is_unique:
            raise ValueError("Duplicate participant")
        d = d.join(p, how="left", rsuffix="_duplicate", validate="one_to_one")
    d = d[d.RIDAGEYR >= 20].copy()
    x = d[list(NH_MAP)].rename(columns=NH_MAP)
    x["sex"] = d.RIAGENDR.map({1.0: 1.0, 2.0: 0.0})
    y = pd.DataFrame(index=d.index)
    y["diabetes_report"] = d.DIQ010.map({1.0: 1.0, 2.0: 0.0})
    y["kidney_report"] = d.KIQ022.map({1.0: 1.0, 2.0: 0.0})
    cv = d[["MCQ160C", "MCQ160E", "MCQ160F"]]
    y["cvd_report"] = np.where(
        (cv == 1).any(axis=1), 1.0, np.where((cv == 2).all(axis=1), 0.0, np.nan)
    )
    y["liver_report"] = d.MCQ160L.map({1.0: 1.0, 2.0: 0.0})
    keep = (x.drop(columns=["age", "sex"]).notna().sum(axis=1) >= 1) & y.notna().any(axis=1)
    groups = pd.DataFrame(
        {
            "sex": x.sex,
            "age_band": np.where(x.age >= 60, "60+", "<60"),
            "race_ethnicity": d.RIDRETH3,
        },
        index=d.index,
    )
    return Cohort(
        "nhanes",
        x.loc[keep].reset_index(drop=True),
        y.loc[keep].reset_index(drop=True),
        d.index[keep].astype(int).to_numpy(),
        groups.loc[keep].reset_index(drop=True),
        {
            "diabetes_report": "DIQ010 1 versus 2, borderline/refused/unknown excluded. Self-reported physician-diagnosed diabetes status.",
            "kidney_report": "KIQ022 1 versus 2: ever told weak/failing kidneys, excluding kidney stones/infection. Self-reported kidney-condition history.",
            "cvd_report": "Any MCQ160C/E/F=1 (ever CHD/MI/stroke), negative only if all three=2. Self-reported prevalent cardiovascular history.",
            "liver_report": "MCQ160L 1 versus 2: self-reported history of a liver condition.",
        },
        int((~keep).sum()),
    )


LOADERS = {"ilpd": ilpd, "ckd": ckd, "nhanes": nhanes}
