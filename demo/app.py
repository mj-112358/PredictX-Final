import json
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from predictx.inference import evaluate, registry
from predictx.schema import SPECS, LabPanel

st.set_page_config(page_title="PredictX research", layout="wide")
st.title("PredictX v2 | laboratory research")
st.caption(
    "Source-cohort research assessments with endpoint-specific calibration and model provenance."
)
st.caption(
    "One standardized record, five independently trained disease models. Breast model uses routine CBC and CRP; outputs can abstain independently."
)
with st.sidebar:
    age = st.number_input("Age (years)", 0, 120, 50)
    sex = st.selectbox(
        "Sex parameter used by source models/eGFR",
        ["unknown", "female", "male"],
        key="equation_sex",
    )
    fasting = st.number_input("Fasting hours", 0.0, 48.0, 8.0)
    idms = st.checkbox("Creatinine assay is IDMS traceable")
    acute = st.checkbox("Acute illness")
    collection_zone = ZoneInfo("Asia/Kolkata")
    collection_now = datetime.now(collection_zone)
    collection_date = st.date_input(
        "Collection date (all selected tests)", value=collection_now.date()
    )
    collection_time = st.time_input(
        "Collection time (Asia/Kolkata)",
        value=collection_now.time().replace(tzinfo=None),
    )
    collected_at = datetime.combine(
        collection_date, collection_time, tzinfo=collection_zone
    ).isoformat()
st.write(
    "Select available measurements, enter values in canonical units, and record collection quality."
)
selected = st.multiselect(
    "Available laboratory measurements",
    list(SPECS),
    default=[
        name for name in registry().get("breast_cancer", {}).get("features", []) if name in SPECS
    ],
)
measurements = {}
for k in selected:
    a, b, c = st.columns([2, 1, 1])
    unit, low, high = SPECS[k]
    value = a.number_input(
        f"{k} ({unit})", min_value=float(low), max_value=float(high), value=None, key=k
    )
    quality = b.selectbox(
        f"{k} quality", ["unknown", "acceptable", "hemolysed", "lipemic"], key=k + "q"
    )
    specimen = c.selectbox(
        f"{k} specimen",
        ["urine"] if k.startswith("urine_") else ["serum", "plasma", "blood"],
        key=k + "s",
    )
    if value is not None:
        measurements[k] = {
            "value": value,
            "unit": unit,
            "specimen": specimen,
            "quality": quality,
            "collected_at": collected_at,
        }
if st.button("Run research inference"):
    try:
        panel = LabPanel(
            patient_id="local-demo",
            age=age,
            equation_sex=None if sex == "unknown" else sex,
            fasting_hours=fasting,
            creatinine_idms=idms,
            acute_illness=acute,
            measurements=measurements,
        )
        result = evaluate(panel)
        st.subheader("Calculated biomarkers")
        st.json(result["biomarkers"])
        for row in result["outputs"]:
            st.subheader(row.get("outcome_label", row["disease"]))
            st.json(row)
        st.download_button(
            "Download integrated research report (JSON)",
            data=json.dumps(result, indent=2),
            file_name="predictx_research_report.json",
            mime="application/json",
        )
        st.info("Source-cohort calibration and input-support details accompany each assessment.")
        st.write(result["clinical_pathway"])
    except Exception as e:
        st.error(str(e))
