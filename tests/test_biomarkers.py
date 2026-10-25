from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from predictx.biomarkers import derive, egfr_2021, fib4, homa_ir, ratio
from predictx.schema import LabPanel, Measurement


def measurement(value, unit, specimen="serum", quality="acceptable"):
    return Measurement(
        value=value,
        unit=unit,
        specimen=specimen,
        quality=quality,
        collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_homa_reference():
    # 90 mg/dL = 5 mmol/L by conventional factor 18; 5*9/22.5=2.
    assert homa_ir(90, 9) == pytest.approx(2.0)


def test_fib4_reference():
    # 50*40 / (200*sqrt(25)) = 2.
    assert fib4(50, 40, 25, 200) == pytest.approx(2.0)


def test_egfr_reference():
    # Independently expanded high-creatinine male branch: 142*(1.2/.9)^-1.2*.9938^50.
    assert egfr_2021(1.2, 50, "male", idms=True) == pytest.approx(73.6735302971, abs=0.001)
    assert egfr_2021(0.7, 40, "female", idms=True) == pytest.approx(112.0543067988, abs=0.02)


def test_egfr_requires_prerequisites():
    for age, sex, idms in [(17, "male", True), (40, None, True), (40, "male", False)]:
        with pytest.raises(ValueError):
            egfr_2021(1.0, age, sex, idms=idms)


@pytest.mark.parametrize(
    "a,b", [(1, 0), (-1, 2), (1, float("nan")), (float("inf"), 2), (1, 1e-320)]
)
def test_invalid_ratio(a, b):
    with pytest.raises(ValueError):
        ratio(a, b)


def test_fib4_applicability():
    for age, acute in [(20, False), (50, True)]:
        with pytest.raises(ValueError):
            fib4(age, 40, 25, 200, acute_illness=acute)


def test_unit_conversion_and_missing_quality():
    p = LabPanel(
        patient_id="unit-test",
        age=50,
        equation_sex="male",
        creatinine_idms=True,
        fasting_hours=8,
        measurements={
            "creatinine": measurement(106.08, "umol/L"),
            "fasting_glucose": measurement(90, "mg/dL"),
            "insulin": measurement(9, "uU/mL"),
            "ast": measurement(40, "U/L", quality="hemolysed"),
        },
    )
    d = derive(p)
    assert d["values"]["homa_ir"] == 2
    assert d["values"]["egfr_2021"] == pytest.approx(73.6735302971, abs=0.001)
    assert not d["availability"]["ast"]


@pytest.mark.parametrize(
    "name,value,unit,specimen",
    [
        ("creatinine", 1.0, "mmol/L", "serum"),
        ("alt", -1.0, "U/L", "serum"),
        ("urine_acr", 10.0, "mg/g", "serum"),
        ("hdl", 1.0, "mg/dL", "urine"),
        ("made_up", 1.0, "mg/L", "serum"),
    ],
)
def test_schema_rejects_invalid(name, value, unit, specimen):
    with pytest.raises((ValueError, ValidationError)):
        LabPanel(
            patient_id="test",
            age=50,
            measurements={name: measurement(value, unit, specimen)},
        )


def test_all_ratios_and_non_hdl():
    p = LabPanel(
        patient_id="test",
        age=50,
        measurements={
            k: measurement(v, u)
            for k, v, u in [
                ("ast", 40, "U/L"),
                ("alt", 20, "U/L"),
                ("triglycerides", 150, "mg/dL"),
                ("hdl", 50, "mg/dL"),
                ("total_cholesterol", 200, "mg/dL"),
                ("neutrophils", 4, "10^9/L"),
                ("lymphocytes", 2, "10^9/L"),
                ("platelets", 200, "10^9/L"),
                ("monocytes", 0.4, "10^9/L"),
                ("albumin", 4, "g/dL"),
                ("total_protein", 6, "g/dL"),
            ]
        },
    )
    d = derive(p)["values"]
    assert {
        k: d[k]
        for k in [
            "ast_alt",
            "tg_hdl",
            "non_hdl",
            "nlr",
            "plr",
            "mlr",
            "albumin_globulin",
        ]
    } == {
        "ast_alt": 2,
        "tg_hdl": 3,
        "non_hdl": 150,
        "nlr": 2,
        "plr": 100,
        "mlr": 0.2,
        "albumin_globulin": 2,
    }
