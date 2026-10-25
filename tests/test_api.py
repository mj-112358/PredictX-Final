from fastapi.testclient import TestClient

from predictx.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["research_only"] is True


def test_rejects_untyped_units():
    r = client.post(
        "/research/predict",
        json={
            "patient_id": "test",
            "age": 50,
            "measurements": {
                "creatinine": {
                    "value": 1,
                    "unit": "bananas",
                    "specimen": "serum",
                    "quality": "acceptable",
                    "collected_at": "2026-01-01T00:00:00Z",
                }
            },
        },
    )
    assert r.status_code == 422


def test_no_fabricated_prediction_on_empty_panel():
    r = client.post("/research/predict", json={"patient_id": "test", "age": 50, "measurements": {}})
    assert r.status_code == 200
    assert all(v.get("probability") is None for v in r.json()["outputs"])
    assert len(r.json()["outputs"]) == 5


def test_real_checkpoint_inference():
    import json

    from predictx.data import ROOT

    payload = json.loads((ROOT / "results/refinement/SYNTHETIC_benchmark_input.json").read_text())
    r = client.post("/research/predict", json=payload)
    assert r.status_code == 200
    bc = [o for o in r.json()["outputs"] if o.get("disease") == "breast_cancer"][0]
    assert 0 <= bc["score"] <= 1
    assert "not_population_screening_risk" in bc["estimate_type"]
