import numpy as np
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def sensitivity_threshold(y, p, target=0.9):
    y = np.asarray(y)
    p = np.asarray(p)
    p = p[np.asarray(y) == 1]
    if len(p) == 0:
        raise ValueError("No positive development labels")
    # Largest score cutoff that retains ceil(target*n_positive) positives, including ties.
    return float(np.sort(p)[len(p) - int(np.ceil(target * len(p)))])


def metrics(y, p, t):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p)
    z = p >= t
    tn, fp, fn, tp = confusion_matrix(y, z, labels=[0, 1]).ravel()

    def safe(a, b):
        return float(a / b) if b else None

    return dict(
        n=len(y),
        positive=int(y.sum()),
        threshold=float(t),
        auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        auprc=float(average_precision_score(y, p)) if y.sum() else None,
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        sensitivity=safe(tp, tp + fn),
        specificity=safe(tn, tn + fp),
        ppv=safe(tp, tp + fp),
        npv=safe(tn, tn + fn),
        referral_rate=float(z.mean()),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        net_benefit=float((tp - fp * t / (1 - t)) / len(y)) if 0 < t < 1 else None,
    )


def bootstrap(y, p, t, seed=2026, n_boot=500):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p)
    auc = []
    sens = []
    spec = []
    ap = []
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        a = y[i]
        b = p[i]
        if len(np.unique(a)) < 2:
            continue
        z = b >= t
        auc.append(roc_auc_score(a, b))
        ap.append(average_precision_score(a, b))
        bs.append(np.mean((a - b) ** 2))
        sens.append(z[a == 1].mean())
        spec.append((~z[a == 0]).mean())
    out = {}
    for name, values in [
        ("auroc", auc),
        ("auprc", ap),
        ("brier", bs),
        ("sensitivity", sens),
        ("specificity", spec),
    ]:
        out[name + "_low"], out[name + "_high"] = (
            [float(v) for v in np.quantile(values, [0.025, 0.975])] if values else [None, None]
        )
    out["bootstrap_valid"] = len(auc)
    return out


def assert_disjoint(ids, parts):
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Participant identifiers are not unique")
    sets = [set(ids[p]) for p in parts]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError("Partition leakage")
    if sum(map(len, sets)) != len(ids):
        raise ValueError("Partition coverage mismatch")


def audit_sigmoid_calibration(
    reference: LogisticRegression, raw: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Verify sigmoid reconstruction against its fitted objective and score contract."""
    if (
        reference.solver != "lbfgs"
        or reference.penalty != "l2"
        or reference.class_weight is not None
        or not reference.fit_intercept
    ):
        raise ValueError("Calibration audit requires unweighted L2 L-BFGS with an intercept")
    np.testing.assert_array_equal(reference.classes_, [0, 1])
    x = np.asarray(raw, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=float)
    fitted = clone(reference).fit(x, y)
    gradients = []
    for model in [reference, fitted]:
        residual = expit(x @ model.coef_[0] + model.intercept_[0]) - y
        gradient = np.r_[x.T @ residual + model.coef_[0] / model.C, residual.sum()] / len(y)
        norm = float(np.max(np.abs(gradient)))
        if norm > model.tol:
            raise AssertionError(
                f"Calibration gradient {norm} exceeds fitted solver tolerance {model.tol}"
            )
        gradients.append(norm)
    original = reference.predict_proba(x)
    reconstructed = fitted.predict_proba(x)
    np.testing.assert_allclose(reconstructed, original, rtol=1e-6, atol=1e-7)
    return {
        "coefficient_max_absolute_delta": float(np.max(np.abs(fitted.coef_ - reference.coef_))),
        "intercept_max_absolute_delta": float(
            np.max(np.abs(fitted.intercept_ - reference.intercept_))
        ),
        "probability_max_absolute_delta": float(np.max(np.abs(reconstructed - original))),
        "saved_gradient_infinity_norm": gradients[0],
        "refitted_gradient_infinity_norm": gradients[1],
        "fitted_solver_tolerance": float(reference.tol),
    }
