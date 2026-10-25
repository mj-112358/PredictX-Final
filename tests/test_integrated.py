import importlib.util
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from predictx.api import app
from predictx.data import ROOT
from predictx.inference import DISEASES, evaluate, load_model, registry
from predictx.processing import process_panel
from predictx.schema import LabPanel

spec = importlib.util.spec_from_file_location(
    "benchmark_fixture", ROOT / "scripts/benchmark_integration.py"
)
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


def test_all_five_models_independent_and_scalar():
    assert set(registry()) == set(DISEASES)
    for disease in DISEASES:
        model = load_model(disease)
        assert len(model.models) == len(model.tasks) == 1
        assert model.metadata["version"] == "0.4.0"


def test_exactly_five_outputs_and_one_normalization():
    p = LabPanel.model_validate(bm.fixture())
    r = evaluate(p)
    assert len(r["outputs"]) == 5
    assert [o["disease"] for o in r["outputs"]] == list(DISEASES)
    assert all(o["score"] is not None for o in r["outputs"]), r["outputs"]
    assert r["processing"]["panel_processing_calls"] == 1
    assert r["processing"]["canonical_measurements"] == len(p.measurements)
    assert r["processing"]["transform_calls"] <= 5


def test_essential_input_abstention_is_per_disease():
    x = bm.fixture()
    x["measurements"].pop("creatinine")
    r = evaluate(LabPanel.model_validate(x))
    kidney = next(o for o in r["outputs"] if o["disease"] == "kidney")
    assert kidney["assessment"] == "abstained" and kidney["score"] is None
    assert "creatinine" in kidney["missing_essential_inputs"]
    assert next(o for o in r["outputs"] if o["disease"] == "breast_cancer")["score"] is not None


def test_quality_and_population_abstain():
    x = bm.fixture()
    x["measurements"]["lymphocytes"]["quality"] = "hemolysed"
    r = evaluate(LabPanel.model_validate(x))
    assert r["outputs"][-1]["score"] is not None
    assert "optional_inputs_imputed" in r["outputs"][-1]["warning_codes"]
    x = bm.fixture()
    x["equation_sex"] = "male"
    assert evaluate(LabPanel.model_validate(x))["outputs"][-1]["score"] is None


def test_inconsistent_differential_is_withheld():
    x = bm.fixture()
    x["measurements"]["wbc"]["value"] = 2
    r = process_panel(LabPanel.model_validate(x))
    assert "lymphocytes" not in r.values
    assert "inconsistent_differential_count" in r.rejected.values()


def test_shared_signature_cache_reuses_only_identical_preprocessing():
    r = evaluate(LabPanel.model_validate(bm.fixture()))
    a = registry()["diabetes"]
    b = registry()["cardiovascular"]
    if a["preprocessing_signature"] == b["preprocessing_signature"]:
        assert r["processing"]["transform_cache_hits"] >= 1
    assert (
        registry()["kidney"]["preprocessing_signature"]
        != registry()["liver"]["preprocessing_signature"]
    )


def test_api_complete_and_empty_record():
    client = TestClient(app)
    r = client.post("/research/predict", json=bm.fixture())
    assert r.status_code == 200
    assert all(o["score"] is not None for o in r.json()["outputs"])
    empty = client.post(
        "/research/predict", json={"patient_id": "test", "age": 45, "measurements": {}}
    ).json()
    assert len(empty["outputs"]) == 5 and all(o["score"] is None for o in empty["outputs"])


def test_unit_conversion_occurs_once_per_measurement_across_five_models():
    from predictx.schema import canonical_value

    payload = bm.fixture()
    with patch("predictx.schema.canonical_value", wraps=canonical_value) as converted:
        panel = LabPanel.model_validate(payload)
        evaluate(panel)
        assert converted.call_count == len(payload["measurements"])


def test_post_validation_record_mutation_fails_closed():
    import pytest

    panel = LabPanel.model_validate(bm.fixture())
    panel.measurements["wbc"].value = 7.0
    with pytest.raises(ValueError, match="modified after validation"):
        evaluate(panel)


def test_fitted_signature_tampering_fails_closed():
    import copy

    import pytest

    cfg = copy.deepcopy(registry())
    cfg["diabetes"]["preprocessing_signature"] = "invalid"
    load_model.cache_clear()
    try:
        with patch("predictx.inference.registry", return_value=cfg):
            with pytest.raises(ValueError, match="signature mismatch"):
                load_model("diabetes")
    finally:
        load_model.cache_clear()


def test_request_transform_cache_does_not_cross_patient_records():
    from predictx.processing import transform

    a = bm.fixture()
    b = bm.fixture()
    b["patient_id"] = "SECOND_SYNTHETIC_ENGINEERING_FIXTURE"
    b["age"] = 46
    ra = process_panel(LabPanel.model_validate(a))
    rb = process_panel(LabPanel.model_validate(b))
    model = load_model("diabetes")
    sig = registry()["diabetes"]["preprocessing_signature"]
    xa = transform(ra, model, sig)
    xb = transform(rb, model, sig)
    assert xa is not xb and not np.array_equal(xa, xb)
    assert ra.counters["transform_cache_hits"] == rb.counters["transform_cache_hits"] == 0


@pytest.mark.parametrize("count_unit", ["cells/mm3", "cells/mm³", "cells/mm^3"])
def test_equivalent_cell_concentrations_preserve_five_scores(count_unit):
    payload = bm.fixture()
    reference = evaluate(LabPanel.model_validate(payload))
    for name in [
        "wbc",
        "neutrophils",
        "lymphocytes",
        "monocytes",
        "eosinophils",
        "basophils",
        "platelets",
    ]:
        payload["measurements"][name]["value"] *= 1000
        payload["measurements"][name]["unit"] = count_unit
    actual = evaluate(LabPanel.model_validate(payload))
    np.testing.assert_allclose(
        [o["score"] for o in actual["outputs"]],
        [o["score"] for o in reference["outputs"]],
        rtol=1e-7,
    )
    assert actual["processing"]["transform_calls"] == 4


def test_reference_metadata_mutation_invalidates_canonical_snapshot():
    panel = LabPanel.model_validate(bm.fixture())
    panel.measurements["creatinine"].laboratory = "changed-source"
    with pytest.raises(ValueError, match="modified after validation"):
        evaluate(panel)


def test_engine_version_partitions_request_transform_cache():
    import copy

    from predictx.processing import transform

    record = process_panel(LabPanel.model_validate(bm.fixture()))
    model = load_model("diabetes")
    changed = copy.deepcopy(model)
    changed.metadata["feature_engine_version"] = "different-engine"
    signature = registry()["diabetes"]["preprocessing_signature"]
    first = transform(record, model, signature)
    second = transform(record, changed, signature)
    assert first is not second
    assert record.counters["transform_calls"] == 2
