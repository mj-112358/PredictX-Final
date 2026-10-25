"""Regression tests target failure modes found by the abstention and parity audits."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictx.api import app
from predictx.breast_data import breast_senegal
from predictx.cohort_runtime import LOADERS_ACTIVE, source_record
from predictx.data import ROOT
from predictx.inference import evaluate, load_model, registry
from predictx.processing import process_panel, project, transform
from predictx.research_metrics import coverage_metrics, thresholds
from predictx.schema import LabPanel


@pytest.fixture
def payload():
    return json.loads((ROOT / "results/refinement/SYNTHETIC_benchmark_input.json").read_text())


def test_optional_differential_qc_preserved_without_blanket_abstention(payload):
    payload["measurements"]["wbc"]["value"] = 2
    out = evaluate(LabPanel.model_validate(payload))
    b = out["outputs"][-1]
    assert b["score"] is not None and "optional_inputs_imputed" in b["warning_codes"]
    assert all(
        out["quality_rejections"][k] == "inconsistent_differential_count"
        for k in ["neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils"]
    )
    assert {"nlr", "plr", "mlr"} <= set(b["missing_optional_inputs"])
    assert any("inconsistent_differential_count" in x for x in b["warning_codes"])


def test_renal_urine_optional_but_blood_anchor_required(payload):
    for k in [
        "urine_albumin_ordinal",
        "urine_specific_gravity",
        "urine_glucose_ordinal",
    ]:
        payload["measurements"].pop(k)
    out = evaluate(LabPanel.model_validate(payload))["outputs"][1]
    assert out["score"] is not None
    assert "urine_albumin_ordinal" in out["missing_optional_inputs"]
    payload["measurements"]["urea"]["quality"] = "unknown"
    out = evaluate(LabPanel.model_validate(payload))["outputs"][1]
    assert out["score"] is None and "missing_essential:urea" in out["reasons"]


@pytest.mark.parametrize("essential", ["wbc", "platelets"])
def test_breast_core_quality_failure_still_abstains(payload, essential):
    payload["measurements"][essential]["quality"] = "hemolysed"
    out = evaluate(LabPanel.model_validate(payload))["outputs"][-1]
    assert out["score"] is None and "missing_essential:" + essential in out["reasons"]


@pytest.mark.parametrize("kind", ["invalid_units", "invalid_value", "invalid_specimen"])
def test_invalid_lab_request_has_machine_readable_reason(payload, kind):
    m = payload["measurements"]["creatinine"]
    if kind == "invalid_units":
        m["unit"] = "bananas"
    elif kind == "invalid_value":
        m["value"] = -1
    else:
        m["specimen"] = "urine"
    r = TestClient(app).post("/research/predict", json=payload)
    assert r.status_code == 422 and "invalid_laboratory_measurement" in r.json()["reason_codes"]


def test_domain_flags_are_distinct_from_invalid_values(payload):
    payload["measurements"]["creatinine"]["value"] = 99.9
    out = evaluate(LabPanel.model_validate(payload))["outputs"][1]
    assert out["score"] is not None and "empirical_range_extrapolation" in out["warning_codes"]
    assert not out["reasons"] and "creatinine" in out["out_of_range_features"]


@pytest.mark.parametrize("scenario", ["age", "sex", "acute"])
def test_inapplicable_population_still_abstains(payload, scenario):
    if scenario == "age":
        payload["age"] = 12
    if scenario == "sex":
        payload["equation_sex"] = "male"
    if scenario == "acute":
        payload["acute_illness"] = True
    b = evaluate(LabPanel.model_validate(payload))["outputs"][-1]
    assert b["score"] is None and any(x.startswith("population_") for x in b["reasons"])


def test_optional_missing_indicator_is_zero_not_silently_normal(payload):
    payload["measurements"].pop("crp")
    record = process_panel(LabPanel.model_validate(payload))
    b = load_model("breast_cancer")
    j = b.preprocessor.columns.index("crp")
    x = transform(record, b, registry()["breast_cancer"]["preprocessing_signature"])
    assert np.isnan(project(record, b.preprocessor.columns)["crp"].iloc[0])
    assert np.isfinite(x).all() and x[0, len(b.preprocessor.columns) + j] == 0


def test_artifact_hash_tampering_fails_closed():
    cfg = copy.deepcopy(registry())
    cfg["kidney"]["sha256"] = "wrong"
    load_model.cache_clear()
    try:
        with patch("predictx.inference.registry", return_value=cfg):
            with pytest.raises(ValueError, match="hash mismatch"):
                load_model("kidney")
    finally:
        load_model.cache_clear()


def test_model_endpoint_provenance_tampering_fails_closed():
    cfg = copy.deepcopy(registry())
    cfg["liver"]["outcome_label"] = "future fibrosis"
    load_model.cache_clear()
    try:
        with patch("predictx.inference.registry", return_value=cfg):
            with pytest.raises(ValueError, match="provenance mismatch"):
                load_model("liver")
    finally:
        load_model.cache_clear()


def test_batch_reports_isolate_patients_and_expose_endpoints(payload):
    b = copy.deepcopy(payload)
    b["patient_id"] = "second"
    b["age"] = 70
    b["measurements"].pop("creatinine")
    r = TestClient(app).post("/research/predict/batch", json=[payload, b])
    assert r.status_code == 200
    a, z = r.json()["reports"]
    assert a["outputs"][1]["score"] is not None and z["outputs"][1]["score"] is None
    assert all(
        o["outcome_label"]
        and o["model_sha256"]
        and o["validation_status"] == "exploratory_internal_retrospective_classification"
        and not o["longitudinal_outcome"]
        for o in a["outputs"]
    )
    assert (
        a["patient_id"] != z["patient_id"]
        and a["processing"]["panel_processing_calls"]
        == z["processing"]["panel_processing_calls"]
        == 1
    )
    assert a["outputs"][-1]["assessment"] == "ranking_only"


@pytest.mark.parametrize(
    "disease", ["diabetes", "kidney", "liver", "cardiovascular", "breast_cancer"]
)
def test_genuine_saved_test_record_matches_actual_runtime(disease):
    cfg = registry()[disease]
    c = LOADERS_ACTIVE[cfg["dataset"]]()
    raw = breast_senegal(False).X if disease == "breast_cancer" else c.X
    p = pd.read_csv(
        ROOT / f"results/refinement/{disease}_test_predictions.csv",
        dtype={"record_id": str},
    )
    p = p[p.eligible].iloc[0]
    i = [str(v) for v in c.ids].index(p.record_id)
    panel, record, _ = source_record(raw.iloc[i], p.record_id, female=disease == "breast_cancer")
    from predictx.inference import assess

    out = assess(panel, record, disease)
    assert out["score"] == pytest.approx(p.p, rel=1e-6, abs=1e-7)


def test_complete_accounting_does_not_hide_abstained_cases():
    out = coverage_metrics(
        [1, 1, 1, 0, 0],
        [0.9, 0.1, 0.9, 0.1, 0.8],
        0.5,
        [True, True, False, True, False],
    )
    assert (out["tp"], out["fn"], out["abstained_positive"]) == (1, 1, 1)
    assert out["overall_positive_detection"] == pytest.approx(1 / 3)
    assert out["scored_sensitivity"] == 0.5


def test_threshold_methods_respect_development_constraints():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.2, 0.5, 0.7, 0.8])
    t = thresholds(y, p)
    assert (p[y == 1] >= t["sensitivity90"]).mean() >= 0.9
    assert (p[y == 0] < t["specificity80"]).mean() >= 0.8
