from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .inference import evaluate
from .schema import SPECS, LabPanel

app = FastAPI(
    title="PredictX v2 research API",
    version="0.4.0",
    description="Five independent disease models with shared laboratory processing and explicit abstention. Research use only.",
)


@app.get("/health")
def health():
    return {"status": "ok", "research_only": True}


@app.get("/schema/biomarkers")
def biomarkers():
    return {
        k: {"canonical_unit": v[0], "plausibility_min": v[1], "plausibility_max": v[2]}
        for k, v in SPECS.items()
    }


@app.post("/research/predict")
def predict(panel: LabPanel):
    try:
        return evaluate(panel)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    issues = [
        {
            "reason_code": "invalid_laboratory_measurement"
            if "measurements" in e["loc"]
            or any(
                s in e["msg"]
                for s in [
                    "biomarker",
                    "unit",
                    "specimen",
                    "Implausible",
                    "Panel measurements",
                    "Ordinal",
                ]
            )
            else "invalid_patient_record",
            "location": list(e["loc"]),
            "message": e["msg"],
        }
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "status": "invalid_request",
            "reason_codes": sorted(set(e["reason_code"] for e in issues)),
            "detail": issues,
        },
    )


@app.post("/research/predict/batch")
def predict_batch(panels: list[LabPanel]):
    if len(panels) > 128:
        raise HTTPException(422, "Maximum batch size is 128 isolated patient records")
    try:
        return {
            "research_only": True,
            "batch_semantics": "isolated scalar inference and request-local caches per patient",
            "reports": [evaluate(p) for p in panels],
        }
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
