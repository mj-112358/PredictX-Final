from math import isfinite
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

# Canonical units and physical plausibility domains.
SPECS = {
    "fasting_glucose": ("mg/dL", 1, 2000),
    "random_glucose": ("mg/dL", 1, 2000),
    "hba1c": ("%", 1, 30),
    "insulin": ("uU/mL", 0, 10000),
    "triglycerides": ("mg/dL", 0, 20000),
    "hdl": ("mg/dL", 0, 500),
    "ldl": ("mg/dL", 0, 2000),
    "total_cholesterol": ("mg/dL", 0, 3000),
    "creatinine": ("mg/dL", 0.01, 100),
    "urea": ("mg/dL", 0, 1000),
    "bun": ("mg/dL", 0, 500),
    "uric_acid": ("mg/dL", 0, 100),
    "urine_acr": ("mg/g", 0, 100000),
    "urine_protein": ("mg/dL", 0, 10000),
    "urine_glucose": ("mg/dL", 0, 10000),
    "urine_rbc": ("cells/uL", 0, 100000),
    "alt": ("U/L", 0, 100000),
    "ast": ("U/L", 0, 100000),
    "alp": ("U/L", 0, 100000),
    "ggt": ("U/L", 0, 100000),
    "bilirubin": ("mg/dL", 0, 100),
    "direct_bilirubin": ("mg/dL", 0, 100),
    "albumin": ("g/dL", 0, 15),
    "total_protein": ("g/dL", 0, 30),
    "platelets": ("10^9/L", 0, 5000),
    "crp": ("mg/L", 0, 1000),
    "wbc": ("10^9/L", 0, 1000),
    "neutrophils": ("10^9/L", 0, 1000),
    "lymphocytes": ("10^9/L", 0, 1000),
    "monocytes": ("10^9/L", 0, 1000),
    "eosinophils": ("10^9/L", 0, 1000),
    "basophils": ("10^9/L", 0, 1000),
    "rbc": ("10^12/L", 0, 20),
    "hemoglobin": ("g/dL", 0, 30),
    "hematocrit": ("%", 0, 90),
    "mcv": ("fL", 0, 200),
    "mch": ("pg", 0, 100),
    "mchc": ("g/dL", 0, 60),
    "rdw": ("%", 0, 60),
    "mpv": ("fL", 0, 50),
    "sodium": ("mmol/L", 50, 250),
    "potassium": ("mmol/L", 0.1, 20),
    "urine_specific_gravity": ("1", 1, 1.1),
    "urine_albumin_ordinal": ("ordinal", 0, 5),
    "urine_glucose_ordinal": ("ordinal", 0, 5),
}
CONVERSIONS = {
    ("fasting_glucose", "mmol/L"): 18.0182,
    ("random_glucose", "mmol/L"): 18.0182,
    ("creatinine", "umol/L"): 1 / 88.4,
    ("triglycerides", "mmol/L"): 88.57,
    **{(k, "mmol/L"): 38.67 for k in ["hdl", "ldl", "total_cholesterol"]},
    **{(k, "g/L"): 0.1 for k in ["albumin", "total_protein", "hemoglobin"]},
    ("bilirubin", "umol/L"): 1 / 17.104,
    ("direct_bilirubin", "umol/L"): 1 / 17.104,
    ("urine_acr", "mg/mmol"): 8.84,
    **{
        (k, "cells/mm3"): 0.001
        for k in [
            "wbc",
            "neutrophils",
            "lymphocytes",
            "monocytes",
            "eosinophils",
            "basophils",
            "platelets",
        ]
    },
}


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    value: float
    unit: str
    reference_low: float | None = None
    reference_high: float | None = None
    collected_at: AwareDatetime
    specimen: Literal["blood", "serum", "plasma", "urine"]
    laboratory: str | None = None
    available: bool = True
    quality: Literal["acceptable", "hemolysed", "lipemic", "unknown"] = "unknown"

    @model_validator(mode="after")
    def reference_order(self):
        if (
            self.reference_low is not None
            and self.reference_high is not None
            and self.reference_low > self.reference_high
        ):
            raise ValueError("reference interval reversed")
        return self


class LabPanel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    patient_id: str
    age: float = Field(ge=0, le=120)
    equation_sex: Literal["female", "male"] | None = None
    fasting_hours: float | None = Field(default=None, ge=0, le=48)
    creatinine_idms: bool = False
    acute_illness: bool = False
    urine_microscopy: str | None = None
    measurements: dict[str, Measurement]
    _canonical: dict = PrivateAttr(default_factory=dict)
    _validated_fingerprint: tuple = PrivateAttr(default=())

    def _fingerprint(self):
        return (
            self.patient_id,
            self.age,
            self.equation_sex,
            self.fasting_hours,
            self.creatinine_idms,
            self.acute_illness,
            self.urine_microscopy,
            tuple(
                (
                    k,
                    m.value,
                    m.unit,
                    m.specimen,
                    m.available,
                    m.quality,
                    m.collected_at,
                    m.reference_low,
                    m.reference_high,
                    m.laboratory,
                )
                for k, m in sorted(self.measurements.items())
            ),
        )

    def canonical_snapshot(self) -> dict[str, float]:
        if self._fingerprint() != self._validated_fingerprint:
            raise ValueError(
                "Patient record modified after validation; construct a new validated record"
            )
        return self._canonical.copy()

    @model_validator(mode="after")
    def validate_measurements(self):
        canonical = {}
        for name, m in self.measurements.items():
            if name not in SPECS:
                raise ValueError(f"Unknown biomarker: {name}")
            canonical[name] = canonical_value(name, m)
        if self.measurements:
            times = [m.collected_at for m in self.measurements.values() if m.available]
            if times and (max(times) - min(times)).total_seconds() > 86400:
                raise ValueError("Panel measurements must be within 24 hours")
        self._canonical = canonical
        self._validated_fingerprint = self._fingerprint()
        return self


def canonical_value(name: str, m: Measurement) -> float:
    unit, low, high = SPECS[name]
    source_unit = m.unit.replace("µ", "u").replace("μ", "u").replace("³", "3").replace("^3", "3")
    if source_unit == unit:
        v = m.value
    elif (name, source_unit) in CONVERSIONS:
        v = m.value * CONVERSIONS[name, source_unit]
    else:
        raise ValueError(f"Unsupported unit {m.unit} for {name}; expected {unit}")
    if name.endswith("_ordinal") and v != int(v):
        raise ValueError("Ordinal urine observations must be integers 0 to 5")
    if not isfinite(v) or not low <= v <= high:
        raise ValueError(f"Implausible {name}")
    if name.startswith("urine_") and m.specimen != "urine":
        raise ValueError("Urine biomarker requires urine specimen")
    if not name.startswith("urine_") and m.specimen == "urine":
        raise ValueError("Blood biomarker requires blood/serum/plasma")
    return v
