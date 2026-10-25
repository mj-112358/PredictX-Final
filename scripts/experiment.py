"""Independent candidate training with fixed train/validation/calibration/test partitions."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from predictx.data import LOADERS, ROOT, derived
from predictx.evaluation import (
    assert_disjoint,
    bootstrap,
    metrics,
    sensitivity_threshold,
)
from predictx.models import Preprocessor, ResearchModel, raw_scores, train_neural

CFG = json.loads((ROOT / "configs/experiment.json").read_text())
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def split(c, seed):
    i = np.arange(len(c.X))
    strata = c.y.iloc[:, 0].fillna(-1)
    train, rest = train_test_split(i, test_size=0.6, random_state=seed, stratify=strata)
    val, rest = train_test_split(
        rest, test_size=2 / 3, random_state=seed, stratify=strata.iloc[rest]
    )
    cal, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=strata.iloc[rest])
    assert_disjoint(c.ids, [train, val, cal, test])
    return train, val, cal, test


def fit(c, arch, seed, use_derived=False, indicators=True):
    train, val, cal, test = split(c, CFG["seed"])
    x = derived(c.X) if use_derived else c.X
    prep = Preprocessor(indicators).fit(x.iloc[train])
    a, b, cc, tt = [prep.transform(x.iloc[ix]) for ix in [train, val, cal, test]]
    tasks = ["diabetes_report", "cvd_report"] if c.name == "nhanes" else list(c.y)
    y = c.y[tasks].to_numpy()
    tuning = []
    model = []
    for k, task in enumerate(tasks):
        tr = np.isfinite(y[train, k])
        va = np.isfinite(y[val, k])
        candidates = []
        for h in [0, 1]:
            if arch == "logistic":
                m = LogisticRegression(C=[0.1, 1.0][h], max_iter=2000, random_state=seed)
                m.fit(a[tr], y[train, k][tr])
            elif arch == "boosted":
                m = HistGradientBoostingClassifier(
                    max_iter=100,
                    max_leaf_nodes=[7, 15][h],
                    l2_regularization=1,
                    early_stopping=False,
                    random_state=seed,
                )
                m.fit(a[tr], y[train, k][tr])
            elif arch == "independent":
                m, record = train_neural(
                    a,
                    y[train, k : k + 1],
                    b,
                    y[val, k : k + 1],
                    "shared",
                    [0.003, 0.01][h],
                    seed + list(c.y).index(task),
                    CFG["neural_epochs"],
                )
            else:
                raise ValueError(arch)
            raw = raw_scores(m, b[va]).ravel()
            pv = 1 / (1 + np.exp(-np.clip(raw, -40, 40)))
            score = log_loss(y[val, k][va], pv, labels=[0, 1])
            candidates.append((score, m))
            tuning.append({"task": task, "candidate": h, "val_logloss": score})
        model.append(min(candidates, key=lambda t: t[0])[1])

    def scores(v):
        return (
            np.column_stack([raw_scores(m, v).ravel() for m in model])
            if isinstance(model, list)
            else raw_scores(model, v)
        )

    sc, sv = scores(cc), scores(b)
    calibrators = []
    thresholds = []
    for k in range(len(tasks)):
        obs = np.isfinite(y[cal, k])
        calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(
            sc[obs, k : k + 1], y[cal, k][obs]
        )
        calibrators.append(calibrator)
        obs = np.isfinite(y[val, k])
        vp = calibrator.predict_proba(sv[obs, k : k + 1])[:, 1]
        thresholds.append(
            sensitivity_threshold(y[val, k][obs], vp, CFG["sensitivity_development_target"])
        )
    run = f"{c.name}_{arch}_s{seed}_f1_d{int(use_derived)}_m{int(indicators)}"
    md = {
        "run": run,
        "dataset": c.name,
        "architecture": arch,
        "seed": seed,
        "fraction": 1.0,
        "derived": use_derived,
        "indicators": indicators,
        "version": "0.2.0",
        "validation": "internal retrospective benchmark only",
        "task_definitions": c.definition,
        "tuning": tuning,
        "train_n": len(train),
        "partitions": {
            n: [str(v) for v in c.ids[ix]]
            for n, ix in zip(
                ["train", "validation", "calibration", "test"], [train, val, cal, test]
            )
        },
    }
    md["input_features"] = prep.columns
    md["source_manifest"] = "data/raw/manifest.json"
    md["source_sha256"] = {
        record["file"]: record["sha256"]
        for record in json.loads((ROOT / "data/raw/manifest.json").read_text())
        if record.get("sha256")
    }
    bundle = ResearchModel(prep, model, tasks, calibrators, thresholds, md)
    path = OUT / "models" / f"{run}.joblib"
    joblib.dump(bundle, path)
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    md["checkpoint_sha256"] = checkpoint_hash
    (OUT / "models" / f"{run}.json").write_text(json.dumps(md, indent=2))
    probs = bundle.predict_proba(x.iloc[test])
    pred = []
    rows = []
    for k, task in enumerate(tasks):
        obs = np.isfinite(y[test, k])
        yy = y[test, k][obs]
        pp = probs[obs, k]
        threshold = thresholds[k]
        r = dict(
            run=run,
            dataset=c.name,
            architecture=arch,
            task=task,
            seed=seed,
            fraction=1.0,
            derived=use_derived,
            indicators=indicators,
            partition="test",
            checkpoint_sha256=checkpoint_hash,
            **metrics(yy, pp, threshold),
            **bootstrap(yy, pp, threshold, n_boot=CFG["bootstrap_replicates"]),
        )
        r["sensitivity_target_achieved_point"] = (
            r["sensitivity"] >= CFG["sensitivity_development_target"]
        )
        rows.append(r)
        for idx, predp in zip(test[obs], pp):
            pred.append(
                dict(
                    run=run,
                    dataset=c.name,
                    task=task,
                    record_id=str(c.ids[idx]),
                    y=int(y[idx, k]),
                    p=float(predp),
                    threshold=threshold,
                )
            )
    pd.DataFrame(pred).to_csv(OUT / f"{run}_predictions.csv", index=False)
    return rows


def main():
    import argparse

    from predictx.breast_data import breast_senegal

    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced/training")
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    records = []
    for loader in [LOADERS["ckd"], LOADERS["ilpd"], LOADERS["nhanes"], breast_senegal]:
        cohort = loader()
        for arch in ["logistic", "boosted", "independent"]:
            rows = fit(cohort, arch, 2026, use_derived=cohort.name == "breast_senegal_qc")
            records.extend(rows)
    pd.DataFrame(records).to_csv(OUT / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
