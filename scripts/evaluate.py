"""Recompute retained model metrics, preprocessing provenance and runtime parity."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predictx.breast_data import breast_senegal
from predictx.cohort_runtime import LOADERS_ACTIVE, canonical_frame
from predictx.data import ROOT, derived
from predictx.evaluation import assert_disjoint, audit_sigmoid_calibration, bootstrap, metrics
from predictx.inference import assess, load_model, registry
from predictx.research_metrics import coverage_metrics, thresholds
from predictx.runtime import scalar_scores


def verify_calibration(bundle, cohort, source, indices, task):
    observed = indices[np.isfinite(cohort.y[task].iloc[indices].to_numpy())]
    raw = scalar_scores(bundle.model, bundle.preprocessor.transform(source.iloc[observed]))
    return audit_sigmoid_calibration(bundle.calibrator, raw, cohort.y[task].iloc[observed])


def verify_candidate_selection(output: Path) -> None:
    reference = pd.read_csv(ROOT / "results/independent/architecture_selection.csv")
    rows = []
    for disease, cfg in registry().items():
        cohort = LOADERS_ACTIVE[cfg["dataset"]]()
        frame = derived(cohort.X) if disease == "breast_cancer" else cohort.X
        scores = []
        for candidate in reference[reference.disease == disease].to_dict("records"):
            path = ROOT / f"results/models/{candidate['run']}.joblib"
            metadata = json.loads(path.with_suffix(".json").read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["checkpoint_sha256"]
            bundle = joblib.load(path)
            index = bundle.tasks.index(cfg["task"])
            indices = [
                i
                for i, v in enumerate(cohort.ids)
                if str(v) in set(bundle.metadata["partitions"]["validation"])
                and pd.notna(cohort.y.iloc[i][cfg["task"]])
            ]
            p = bundle.predict_proba(frame.iloc[indices])[:, index]
            result = metrics(cohort.y[cfg["task"]].iloc[indices], p, bundle.thresholds[index])
            np.testing.assert_allclose(
                result["log_loss"], candidate["validation_log_loss"], rtol=1e-9
            )
            np.testing.assert_allclose(result["brier"], candidate["validation_brier"], rtol=1e-9)
            scores.append((result["log_loss"], result["brier"], candidate["run"]))
            rows.append(
                {
                    "disease": disease,
                    "run": candidate["run"],
                    "validation_log_loss": result["log_loss"],
                    "validation_brier": result["brier"],
                }
            )
        assert min(scores)[2] == cfg["source_run"]
    pd.DataFrame(rows).to_csv(output / "architecture_selection.csv", index=False)


def evaluate_models(output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(ROOT / "results/refinement/selected_metrics.csv").set_index("disease")
    rows, comparisons, coverage = [], [], []
    for disease, cfg in registry().items():
        bundle = load_model(disease)
        cohort = LOADERS_ACTIVE[cfg["dataset"]]()
        parts = {
            name: np.array([i for i, v in enumerate(cohort.ids) if str(v) in set(ids)])
            for name, ids in bundle.metadata["partitions"].items()
        }
        assert_disjoint(cohort.ids, list(parts.values()))
        source = derived(cohort.X) if bundle.metadata["derived"] else cohort.X
        train = source.iloc[parts["train"]][cfg["features"]]
        fitted = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
        ).fit(train)
        for actual, expected in [
            (bundle.preprocessor.pipe[0].statistics_, fitted[0].statistics_),
            (bundle.preprocessor.pipe[1].mean_, fitted[1].mean_),
            (bundle.preprocessor.pipe[1].scale_, fitted[1].scale_),
            (bundle.preprocessor.lo, train.quantile(0.005).to_numpy()),
            (bundle.preprocessor.hi, train.quantile(0.995).to_numpy()),
        ]:
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        calibration_audit = verify_calibration(
            bundle, cohort, source, parts["calibration"], cfg["task"]
        )
        raw = breast_senegal(False).X if disease == "breast_cancer" else cohort.X
        canonical, diagnostics = canonical_frame(cohort, raw=raw)
        frame = derived(canonical)
        labels = cohort.y[cfg["task"]].to_numpy()
        test = parts["test"][np.isfinite(labels[parts["test"]])]
        validation = parts["validation"][np.isfinite(labels[parts["validation"]])]
        eligible = np.array(
            [
                d["panel"] is not None
                and assess(d["panel"], d["record"], disease, bundle)["score"] is not None
                for d in diagnostics
            ]
        )
        validation = validation[eligible[validation]]
        validation_probability = bundle.predict_proba(frame.iloc[validation])[:, 0]
        chosen = thresholds(labels[validation], validation_probability)
        np.testing.assert_allclose(
            chosen["youden"], cfg["operating_threshold"], rtol=1e-6, atol=1e-7
        )
        retained = json.loads(
            (ROOT / f"results/refinement/{disease}_locked_thresholds.json").read_text()
        )
        validation_metrics = metrics(labels[validation], validation_probability, chosen["youden"])
        for field in ["n", "positive", "tp", "tn", "fp", "fn"]:
            assert validation_metrics[field] == retained["validation_youden"][field], (
                disease,
                field,
            )
        probability = bundle.predict_proba(frame.iloc[test])[:, 0]
        saved = pd.read_csv(
            ROOT / f"results/refinement/{disease}_test_predictions.csv", dtype={"record_id": str}
        )
        observed = pd.DataFrame(
            {
                "record_id": [str(v) for v in cohort.ids[test]],
                "y": labels[test].astype(int),
                "p": probability,
                "eligible": eligible[test],
                "threshold": cfg["operating_threshold"],
            }
        )
        paired = observed.merge(
            saved, on="record_id", suffixes=("_new", "_reference"), validate="one_to_one"
        )
        assert len(paired) == len(saved) == len(observed)
        np.testing.assert_array_equal(paired.y_new, paired.y_reference)
        np.testing.assert_array_equal(paired.eligible_new, paired.eligible_reference)
        np.testing.assert_allclose(paired.p_new, paired.p_reference, rtol=1e-6, atol=1e-7)
        row = {
            "disease": disease,
            "architecture": cfg["architecture"],
            **metrics(observed.y, probability, cfg["operating_threshold"]),
            **bootstrap(observed.y, probability, cfg["operating_threshold"], seed=2026, n_boot=500),
        }
        for field in [
            "n",
            "positive",
            "auroc",
            "auroc_low",
            "auroc_high",
            "brier",
            "log_loss",
            "tp",
            "tn",
            "fp",
            "fn",
        ]:
            np.testing.assert_allclose(
                row[field], reference.loc[disease, field], rtol=1e-6, atol=1e-7
            )
        for j, i in enumerate(test):
            if eligible[i]:
                result = assess(diagnostics[i]["panel"], diagnostics[i]["record"], disease, bundle)
                np.testing.assert_allclose(result["score"], probability[j], rtol=1e-6, atol=1e-7)
        observed.to_csv(output / f"{disease}_test_predictions.csv", index=False)
        rows.append(row)
        coverage.append(
            {
                "disease": disease,
                **coverage_metrics(
                    observed.y, probability, cfg["operating_threshold"], eligible[test]
                ),
            }
        )
        comparisons.append(
            {
                "disease": disease,
                "sha256": cfg["sha256"],
                "test_n": len(test),
                "max_absolute_prediction_delta": float(
                    abs(paired.p_new - paired.p_reference).max()
                ),
                "training_preprocessing_verified": True,
                "calibration_refit_verified": True,
                "calibration_audit": calibration_audit,
                "threshold_reconstruction_absolute_delta": abs(
                    chosen["youden"] - cfg["operating_threshold"]
                ),
                "validation_confusion_counts_verified": True,
                "runtime_parity_verified": True,
            }
        )
        plot_predictions(observed, disease, output / "figures")
    result = pd.DataFrame(rows)
    result.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    (output / "verification.json").write_text(
        json.dumps(
            {
                "models": comparisons,
                "seed": 2026,
                "bootstrap_replicates": 500,
                "bootstrap_method": "paired IID record resampling; omit single-class draws; percentile 2.5/97.5; fitted predictions held fixed",
                "source_metadata_assumption": "Source quality, timing and acute-illness context are accepted for retrospective parity reconstruction.",
            },
            indent=2,
        )
        + "\n"
    )
    verify_candidate_selection(output)
    return result


def plot_predictions(predictions: pd.DataFrame, disease: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    RocCurveDisplay.from_predictions(
        predictions.y, predictions.p, ax=axes[0], name=disease.replace("_", " ")
    )
    observed, predicted = calibration_curve(
        predictions.y, predictions.p, n_bins=5, strategy="quantile"
    )
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].plot(predicted, observed, "o-")
    axes[1].set(
        xlabel="Mean calibrated score",
        ylabel="Observed source-class fraction",
        title="Calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ConfusionMatrixDisplay.from_predictions(
        predictions.y,
        predictions.p >= predictions.threshold,
        labels=[0, 1],
        ax=axes[2],
        colorbar=False,
    )
    axes[2].set_title(f"Component held-out records (n={len(predictions)})")
    fig.suptitle(disease.replace("_", " ").title())
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"{disease}.{suffix}", dpi=160)
    plt.close(fig)


def verify_sources() -> None:
    manifest = json.loads((ROOT / "data/raw/source_manifest.json").read_text())
    for item in manifest:
        path = ROOT / "data/raw" / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["file"]


if __name__ == "__main__":
    verify_sources()
    print(
        evaluate_models(ROOT / "results/reproduced/evaluation")[
            ["disease", "n", "auroc", "auroc_low", "auroc_high", "brier", "log_loss"]
        ].to_string(index=False)
    )
