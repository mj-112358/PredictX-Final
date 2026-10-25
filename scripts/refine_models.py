"""Nested training-partition comparisons and validation-selected model families."""

import os

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
import copy
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from predictx.cohort_runtime import LOADERS_ACTIVE, canonical_frame
from predictx.data import ROOT, derived
from predictx.evaluation import bootstrap, metrics
from predictx.models import Preprocessor
from predictx.runtime import DiseaseModel, FittedFeatures, scalar_scores

OUT = ROOT / "results/reproduced/development"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(exist_ok=True)
BASE = ROOT
REG = json.loads((BASE / "configs/model_registry.json").read_text())
SPECS = (
    [
        {"family": "logistic", "C": c, "weight": w, "features": f}
        for c, w, f in [
            (0.01, None, "all"),
            (0.1, None, "all"),
            (1.0, None, "all"),
            (0.1, "balanced", "all"),
            (0.1, None, "core"),
        ]
    ]
    + [
        {"family": "boosted", "leaves": leaves, "l2": r, "features": f}
        for leaves, r, f in [
            (3, 3.0, "all"),
            (7, 10.0, "all"),
            (15, 10.0, "all"),
            (7, 3.0, "core"),
        ]
    ]
    + [
        {"family": "forest", "features": "all"},
        {"family": "ensemble", "features": "all"},
    ]
)
CORE = {
    "liver": [
        "age",
        "sex",
        "bilirubin",
        "direct_bilirubin",
        "alt",
        "ast",
        "albumin",
        "alp",
        "albumin_globulin",
        "ast_alt",
    ],
    "cardiovascular": [
        "age",
        "sex",
        "hba1c",
        "random_glucose",
        "creatinine",
        "hdl",
        "total_cholesterol",
        "non_hdl",
        "urine_acr",
        "nlr",
        "plr",
    ],
}


def estimator(spec):
    family = spec["family"]
    if family == "logistic":
        return LogisticRegression(
            C=spec["C"], class_weight=spec["weight"], max_iter=2000, random_state=2026
        )
    if family == "boosted":
        return HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=spec["leaves"],
            l2_regularization=spec["l2"],
            learning_rate=0.06,
            early_stopping=False,
            random_state=2026,
        )
    if family == "forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            max_features=0.7,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=2026,
        )
    return VotingClassifier(
        [
            ("lr", LogisticRegression(C=0.1, max_iter=2000)),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    max_leaf_nodes=7,
                    l2_regularization=10,
                    early_stopping=False,
                    random_state=2026,
                ),
            ),
        ],
        voting="soft",
        n_jobs=1,
    )


def fit(spec, x, y):
    prep = FittedFeatures.from_fitted(Preprocessor().fit(x))
    model = estimator(spec).fit(prep.transform(x), y)
    return prep, model


def proba(prep, model, x):
    return model.predict_proba(prep.transform(x))[:, 1]


def cv_candidates(x, y, disease, indices, nfold=3):
    rows = []
    for h, spec in enumerate(SPECS):
        cols = [k for k in CORE[disease] if k in x] if spec["features"] == "core" else list(x)
        scores = []
        cv = StratifiedKFold(nfold, shuffle=True, random_state=2026)
        for fold, (a, b) in enumerate(cv.split(indices, y[indices])):
            tr, va = indices[a], indices[b]
            prep, model = fit(spec, x.iloc[tr][cols], y[tr])
            p = proba(prep, model, x.iloc[va][cols])
            scores.append(log_loss(y[va], p, labels=[0, 1]))
        rows.append({"candidate": h, "cv_log_loss": float(np.mean(scores)), **spec})
    return rows, min(rows, key=lambda z: (z["cv_log_loss"], z["candidate"]))["candidate"]


def main():
    allcv = []
    nested = []
    selections = []
    newreg = copy.deepcopy(REG)
    for disease in ["liver", "cardiovascular"]:
        cfg = REG[disease]
        c = LOADERS_ACTIVE[cfg["dataset"]]()
        x, diag = canonical_frame(c)
        x = derived(x)
        b = joblib.load(BASE / cfg["artifact"])
        parts = {
            k: np.array(
                [
                    i
                    for i, v in enumerate(c.ids)
                    if str(v) in set(ids) and pd.notna(c.y.iloc[i][cfg["task"]])
                ]
            )
            for k, ids in b.metadata["partitions"].items()
        }
        y = c.y[cfg["task"]].to_numpy()
        tr, va, ca = parts["train"], parts["validation"], parts["calibration"]
        # Fit preprocessing within each nested training fold.
        for fold, (a, z) in enumerate(
            StratifiedKFold(3, shuffle=True, random_state=2027).split(tr, y[tr])
        ):
            inner, winner = cv_candidates(x, y, disease, tr[a], 2)
            spec = SPECS[winner]
            cols = [k for k in CORE[disease] if k in x] if spec["features"] == "core" else list(x)
            prep, model = fit(spec, x.iloc[tr[a]][cols], y[tr[a]])
            p = proba(prep, model, x.iloc[tr[z]][cols])
            nested.append(
                {
                    "disease": disease,
                    "outer_fold": fold,
                    "selected_candidate": winner,
                    "n": len(z),
                    **metrics(y[tr[z]], p, 0.5),
                }
            )
            print(
                "Nested train-only fold",
                disease,
                fold,
                "AUROC",
                nested[-1]["auroc"],
                flush=True,
            )
        cv, _ = cv_candidates(x, y, disease, tr, 3)
        allcv.extend([{"disease": disease, **r} for r in cv])
        candidates = [
            (
                "original_baseline",
                b,
                cfg["architecture"],
                {
                    "calibrated_validation_log_loss": log_loss(
                        y[va], b.predict_proba(x.iloc[va])[:, 0]
                    )
                },
            )
        ]
        # Best train-CV hyperparameter in each family; calibrated validation chooses family.
        for family in ["logistic", "boosted", "forest", "ensemble"]:
            h = min(
                [r for r in cv if r["family"] == family],
                key=lambda z: (z["cv_log_loss"], z["candidate"]),
            )["candidate"]
            spec = SPECS[h]
            cols = [k for k in CORE[disease] if k in x] if spec["features"] == "core" else list(x)
            prep, model = fit(spec, x.iloc[tr][cols], y[tr])
            raw = scalar_scores(model, prep.transform(x.iloc[ca][cols]))
            cal = LogisticRegression(C=1.0, max_iter=1000).fit(raw, y[ca])
            md = {
                **b.metadata,
                "run": f"{disease}_refined_{family}_v0.4",
                "source_run": f"{disease}_refined_{family}_v0.4",
                "version": "0.3.0",
                "architecture": family,
                "derived": True,
                "development_protocol": "nested training-only CV; original calibration; validation-only family selection; exploratory test",
                "hyperparameters": spec,
            }
            obj = DiseaseModel(prep, model, cal, b.threshold, md)
            p = obj.predict_proba(x.iloc[va])[:, 0]
            record = {
                "candidate": h,
                "calibrated_validation_log_loss": log_loss(y[va], p),
                "validation_auroc": roc_auc_score(y[va], p),
                "validation_brier": float(np.mean((y[va] - p) ** 2)),
                "cv_log_loss": [r for r in cv if r["candidate"] == h][0]["cv_log_loss"],
            }
            name = md["run"]
            joblib.dump(obj, OUT / "models" / f"{name}.joblib")
            candidates.append((name, obj, family, record))
        # A 1% relative validation log-loss improvement is required to replace existing functioning fit.
        base_loss = candidates[0][3]["calibrated_validation_log_loss"]
        best = min(candidates, key=lambda z: (z[3]["calibrated_validation_log_loss"], z[0]))
        chosen = (
            best if best[3]["calibrated_validation_log_loss"] < 0.99 * base_loss else candidates[0]
        )
        for name, obj, family, record in candidates:
            selections.append(
                {
                    "disease": disease,
                    "run": name,
                    "architecture": family,
                    "selected": name == chosen[0],
                    **record,
                }
            )
        # Record the validation-selected model before test evaluation.
        (OUT / f"{disease}_locked_selection.json").write_text(
            json.dumps(
                {
                    "selected": chosen[0],
                    "rule": "minimum validation calibrated log loss; replace baseline only if >=1% relative reduction",
                    "candidates": selections[-len(candidates) :],
                    "test_used_for_selection": False,
                },
                indent=2,
            )
        )
        for name, obj, family, record in candidates:
            te = parts["test"]
            p = obj.predict_proba(x.iloc[te])[:, 0]
            vp = obj.predict_proba(x.iloc[va])[:, 0]
            pd.DataFrame(
                {"record_id": c.ids[te], "y": y[te], "p": p, "threshold": obj.threshold}
            ).to_csv(OUT / f"{disease}_{name}_test_predictions.csv", index=False)
            pd.DataFrame({"record_id": c.ids[va], "y": y[va], "p": vp}).to_csv(
                OUT / f"{disease}_{name}_validation_predictions.csv", index=False
            )
            row = {
                "disease": disease,
                "run": name,
                "architecture": family,
                "selected": name == chosen[0],
                **metrics(y[te], p, obj.threshold),
                **bootstrap(y[te], p, obj.threshold),
            }
            (OUT / f"{disease}_{name}_test_metrics.json").write_text(json.dumps(row, indent=2))
        if chosen[0] != "original_baseline":
            cfg = {
                **cfg,
                "architecture": chosen[2],
                "source_run": chosen[0],
                "features": chosen[1].preprocessor.columns,
                "selection_protocol": "nested train-only CV + separate calibration + validation-only family choice; 1% loss replacement margin",
                "source_checkpoint": str(
                    (OUT / "models" / f"{chosen[0]}.joblib").relative_to(ROOT)
                ),
            }
            cfg["artifact"] = cfg["source_checkpoint"]
            cfg["sha256"] = hashlib.sha256((ROOT / cfg["artifact"]).read_bytes()).hexdigest()
            cfg["preprocessing_signature"] = joblib.hash(chosen[1].preprocessor)
            newreg[disease] = cfg
        print("Locked", disease, chosen[0], flush=True)
    pd.DataFrame(allcv).to_csv(OUT / "nested_inner_cv.csv", index=False)
    pd.DataFrame(nested).to_csv(OUT / "nested_outer_cv.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "model_development_selection.csv", index=False)
    (OUT / "candidate_registry.json").write_text(json.dumps(newreg, indent=2))
    tests = [json.loads(p.read_text()) for p in OUT.glob("*_test_metrics.json")]
    pd.DataFrame(tests).to_csv(OUT / "model_comparisons.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
