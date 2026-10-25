"""Dependency-controlled biochemical recipes in canonical measurement units."""

import math

from .schema import LabPanel


def ratio(a, b):
    if not all(math.isfinite(v) and v >= 0 for v in [a, b]) or b == 0:
        raise ValueError("Ratio requires finite nonnegative values and positive denominator")
    value = a / b
    if not math.isfinite(value):
        raise ValueError("Ratio result must be finite")
    return value


def homa_ir(glucose_mg_dl, insulin_uU_ml):
    if glucose_mg_dl <= 0:
        raise ValueError("Glucose must be positive")
    return ratio(glucose_mg_dl * insulin_uU_ml, 405.0)


def egfr_2021(creatinine_mg_dl, age, sex, *, idms):
    if (
        not idms
        or not 18 <= age <= 120
        or sex not in ("male", "female")
        or not math.isfinite(creatinine_mg_dl)
        or creatinine_mg_dl <= 0
    ):
        raise ValueError("Adult age, equation sex, positive IDMS-traceable creatinine required")
    k, a = (0.7, -0.241) if sex == "female" else (0.9, -0.302)
    r = creatinine_mg_dl / k
    return (
        142 * min(r, 1) ** a * max(r, 1) ** (-1.2) * 0.9938**age * (1.012 if sex == "female" else 1)
    )


def fib4(age, ast, alt, platelets, *, acute_illness=False):
    if acute_illness or not 35 <= age <= 120 or alt <= 0:
        raise ValueError(
            "FIB-4 unsuitable below 35 years or in acute illness; ALT must be positive"
        )
    return ratio(age * ast, platelets * math.sqrt(alt))


def non_hdl(total: float, hdl: float) -> float:
    if total < hdl:
        raise ValueError("HDL exceeds total cholesterol")
    return total - hdl


NUMERICAL_RECIPES = {
    "ast_alt": (("ast", "alt"), ratio),
    "albumin_globulin": (("albumin", "total_protein"), lambda a, t: ratio(a, t - a)),
    "tg_hdl": (("triglycerides", "hdl"), ratio),
    "non_hdl": (("total_cholesterol", "hdl"), non_hdl),
    "nlr": (("neutrophils", "lymphocytes"), ratio),
    "plr": (("platelets", "lymphocytes"), ratio),
    "mlr": (("monocytes", "lymphocytes"), ratio),
}


def derived_frame(frame):
    """Apply the source-model feature schema using shared biochemical recipes."""
    import numpy as np

    x = frame.copy()
    for name in [
        "ast_alt",
        "albumin_globulin",
        "non_hdl",
        "nlr",
        "plr",
        "mlr",
        "homa_ir",
    ]:
        dependencies, fn = (
            (("fasting_glucose", "insulin"), homa_ir)
            if name == "homa_ir"
            else NUMERICAL_RECIPES[name]
        )
        if set(dependencies) <= set(x):

            def calculate(row):
                try:
                    return fn(*row)
                except ValueError:
                    return np.nan

            x[name] = x[list(dependencies)].apply(calculate, axis=1)
    return x.replace([np.inf, -np.inf], np.nan)


def derive(panel: LabPanel, canonical=None):
    x = (
        canonical
        if canonical is not None
        else {
            k: v
            for k, v in panel.canonical_snapshot().items()
            if panel.measurements[k].available and panel.measurements[k].quality == "acceptable"
        }
    )
    out = {}
    notes = []

    def fasting_homa(g, i):
        if panel.fasting_hours is None or panel.fasting_hours < 8:
            raise ValueError("Fasting >=8 hours required")
        return homa_ir(g, i)

    recipes = {
        **NUMERICAL_RECIPES,
        "homa_ir": (("fasting_glucose", "insulin"), fasting_homa),
        "egfr_2021": (
            ("creatinine",),
            lambda c: egfr_2021(c, panel.age, panel.equation_sex, idms=panel.creatinine_idms),
        ),
        "fib4": (
            ("ast", "alt", "platelets"),
            lambda a, b, p: fib4(panel.age, a, b, p, acute_illness=panel.acute_illness),
        ),
    }
    for name, (keys, fn) in recipes.items():
        if all(k in x for k in keys):
            try:
                out[name] = fn(*(x[k] for k in keys))
            except ValueError as e:
                notes.append(f"{name}: {e}")
    if panel.age >= 65:
        notes.append("FIB-4 interpretation requires age-adjusted thresholds in older adults")
    return {
        "values": out,
        "notes": notes,
        "availability": {k: k in x for k in panel.measurements},
    }
