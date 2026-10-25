from streamlit.testing.v1 import AppTest

from predictx.data import ROOT


def test_demo_renders_and_withholds_unknown_quality():
    at = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=30).run()
    assert not at.exception
    assert "PredictX" in at.title[0].value
    at.button[0].click().run()
    assert not at.exception
    assert any("Source-cohort" in e.value for e in at.info)


def test_demo_computes_routine_cbc_assessment():
    import json

    at = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=30).run()
    next(item for item in at.selectbox if item.label.startswith("Sex parameter")).select("female")
    at.number_input(key="wbc").set_value(6.0)
    at.number_input(key="platelets").set_value(250.0)
    at.selectbox(key="wbcq").select("acceptable")
    at.selectbox(key="plateletsq").select("acceptable")
    at.button[0].click().run()
    assert not at.exception
    outputs = [json.loads(item.value) for item in at.json]
    breast = next(row for row in outputs if row.get("disease") == "breast_cancer")
    assert 0 <= breast["score"] <= 1
    assert breast["outcome_label"] and breast["model_sha256"]
