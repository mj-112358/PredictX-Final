"""Unified request processing and five independent, endpoint-explicit research assessments."""

import hashlib
import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np

from .compatibility import FEATURE_ENGINE_VERSION, preprocessing_signature
from .data import ROOT
from .eligibility import inspect_inputs
from .processing import ProcessedRecord, process_panel, transform
from .runtime import DiseaseModel
from .schema import LabPanel

DISEASES = ("diabetes", "kidney", "liver", "cardiovascular", "breast_cancer")


@lru_cache(maxsize=1)
def registry() -> dict[str, Any]:
    path = ROOT / "configs/model_registry.json"
    return json.loads(path.read_text()) if path.exists() else {}


@lru_cache(maxsize=5)
def load_model(disease: str) -> DiseaseModel | None:
    cfg = registry().get(disease)
    if cfg is None:
        return None
    path = ROOT / cfg["artifact"]
    if not path.exists():
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {disease}")
    bundle = joblib.load(path)
    if preprocessing_signature(bundle.preprocessor) != cfg["preprocessing_signature"]:
        raise ValueError("Preprocessing signature mismatch")
    if list(bundle.preprocessor.columns) != cfg["features"] or bundle.tasks != [cfg["task"]]:
        raise ValueError("Model schema or task mismatch")
    if (
        cfg.get("feature_engine_version") != FEATURE_ENGINE_VERSION
        or bundle.metadata.get("feature_engine_version") != FEATURE_ENGINE_VERSION
    ):
        raise ValueError("Unsupported feature engine version")
    if (
        bundle.metadata.get("outcome_label") != cfg.get("outcome_label")
        or bundle.metadata.get("version") != cfg["version"]
    ):
        raise ValueError("Model provenance mismatch")
    return bundle


def prepare_assessment(panel, record, disease, bundle=None):
    cfg = registry().get(disease)
    result = {
        "disease": disease,
        "task": cfg["task"] if cfg else disease,
        "model": cfg["source_run"] if cfg else None,
        "probability": None,
        "score": None,
        "assessment": "abstained",
        "reasons": [],
    }
    if cfg is None:
        result.update(status="artifact_unavailable", reasons=["model_not_registered"])
        return result, cfg, bundle
    result.update(
        version=cfg["version"],
        model_sha256=cfg["sha256"],
        outcome_label=cfg["outcome_label"],
        validation_status=cfg["validation_status"],
        definition=cfg["definition"],
        clinical_screening_validated=False,
        longitudinal_outcome=False,
        evidence_limitations=cfg.get("evidence_limitations", []),
    )
    bundle = bundle if bundle is not None else load_model(disease)
    if bundle is None:
        result.update(status="artifact_unavailable", reasons=["model_file_unavailable"])
        return result, cfg, bundle
    support = inspect_inputs(panel, record, cfg, bundle)
    result.update(support)
    if not support["eligible"]:
        result.update(status="abstained")
        return result, cfg, bundle
    return result, cfg, bundle


def assess(
    panel: LabPanel, record: ProcessedRecord, disease: str, bundle: DiseaseModel | None = None
) -> dict[str, Any]:
    result, cfg, bundle = prepare_assessment(panel, record, disease, bundle)
    if not result.get("eligible", False):
        return result
    x = transform(record, bundle, cfg["preprocessing_signature"])
    score = float(bundle.predict_preprocessed(x)[0, 0])
    return finish_assessment(result, cfg, bundle, score, x)


def finish_assessment(result, cfg, bundle, score, x):
    threshold = cfg.get("operating_threshold", bundle.threshold)
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model emitted invalid score")
    binary = "above_development_threshold" if score >= threshold else "below_development_threshold"
    result.update(
        status="internal_research_benchmark_only",
        assessment="ranking_only" if cfg.get("reporting_mode") == "ranking_only" else binary,
        score=score,
        development_threshold=threshold,
        research_threshold_comparison=binary,
        operating_point_method=cfg.get("operating_point_method", "validation_sensitivity90"),
        reporting_mode=cfg.get("reporting_mode", "research_threshold"),
        estimate_type="within_source_cohort_classification_score_not_population_screening_risk",
    )
    if hasattr(bundle.models[0], "coef_"):
        coefs = bundle.models[0].coef_[0][: len(bundle.preprocessor.columns)]
        contribution = x[0, : len(coefs)] * coefs
        result["influential_linear_features"] = [
            {
                "feature": bundle.preprocessor.columns[i],
                "logit_contribution": float(contribution[i]),
            }
            for i in np.argsort(-np.abs(contribution))[:5]
        ]
    return result


def evaluate(panel: LabPanel) -> dict[str, Any]:
    record = process_panel(panel)
    outputs = [assess(panel, record, disease) for disease in DISEASES]
    return {
        "research_only": True,
        "architecture": "shared_processing_independent_models",
        "version": "0.4.0",
        "patient_id": panel.patient_id,
        "biomarkers": record.biomarkers,
        "quality_rejections": dict(record.rejected),
        "processing": record.counters,
        "outputs": outputs,
        "clinical_pathway": "Source-cohort research assessment; clinical interpretation requires independent validation and clinician review.",
    }
