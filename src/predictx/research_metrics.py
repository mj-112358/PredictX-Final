"""Explicit study-cohort operating-point and coverage accounting with uncertainty."""

import numpy as np

from .evaluation import bootstrap, metrics


def wilson(a, n, z=1.95996398454):
    if not n:
        return [None, None]
    p = a / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, float(c - h)), min(1.0, float(c + h))]


def detailed_metrics(y, p, t):
    out = metrics(y, p, t)
    out.update(bootstrap(y, p, t, n_boot=500))
    out["precision"] = out["ppv"]
    out["recall"] = out["sensitivity"]
    for name, a, n in [
        ("precision", out["tp"], out["tp"] + out["fp"]),
        ("recall", out["tp"], out["tp"] + out["fn"]),
        ("specificity", out["tn"], out["tn"] + out["fp"]),
    ]:
        out[name + "_wilson_low"], out[name + "_wilson_high"] = wilson(a, n)
    for name in ["tp", "tn", "fp", "fn"]:
        # Marginal count uncertainty under n exchangeable source-cohort draws.
        a, b = wilson(out[name], out["n"])
        out[name + "_count_low"] = a * out["n"]
        out[name + "_count_high"] = b * out["n"]
    return out


def coverage_metrics(y, p, t, ok):
    y = np.asarray(y, int)
    p = np.asarray(p)
    ok = np.asarray(ok, bool)
    n = len(y)
    pos = y == 1
    neg = ~pos
    z = p >= t
    tp = int(np.sum(ok & pos & z))
    fn = int(np.sum(ok & pos & ~z))
    tn = int(np.sum(ok & neg & ~z))
    fp = int(np.sum(ok & neg & z))
    ap = int(np.sum(~ok & pos))
    an = int(np.sum(~ok & neg))

    def safe(a, b):
        return float(a / b) if b else None

    out = dict(
        total_n=n,
        positive_n=int(pos.sum()),
        negative_n=int(neg.sum()),
        scored_n=int(ok.sum()),
        abstained_n=int((~ok).sum()),
        coverage=safe(ok.sum(), n),
        positive_coverage=safe((ok & pos).sum(), pos.sum()),
        negative_coverage=safe((ok & neg).sum(), neg.sum()),
        abstention_rate_positive=safe(ap, pos.sum()),
        abstention_rate_negative=safe(an, neg.sum()),
        tp=tp,
        fn=fn,
        tn=tn,
        fp=fp,
        abstained_positive=ap,
        abstained_negative=an,
        scored_sensitivity=safe(tp, tp + fn),
        scored_specificity=safe(tn, tn + fp),
        scored_precision=safe(tp, tp + fp),
        overall_positive_detection=safe(tp, pos.sum()),
        overall_scored_miss_fraction=safe(fn, pos.sum()),
        overall_correct_scored_fraction=safe(tp + tn, n),
        positive_accounting="positive_n = tp + fn + abstained_positive",
        negative_accounting="negative_n = tn + fp + abstained_negative",
    )
    assert tp + fn + ap == pos.sum() and tn + fp + an == neg.sum()
    for name, a, den in [
        ("coverage", int(ok.sum()), n),
        ("positive_coverage", int((ok & pos).sum()), int(pos.sum())),
        ("negative_coverage", int((ok & neg).sum()), int(neg.sum())),
        ("scored_sensitivity", tp, tp + fn),
        ("scored_specificity", tn, tn + fp),
        ("scored_precision", tp, tp + fp),
        ("overall_positive_detection", tp, int(pos.sum())),
    ]:
        out[name + "_low"], out[name + "_high"] = wilson(a, den)
    return out


def thresholds(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p)
    t = np.r_[np.unique(p), np.nextafter(float(np.max(p)), np.inf)]
    sens = np.array([(p[y == 1] >= v).mean() for v in t])
    spec = np.array([(p[y == 0] < v).mean() for v in t])
    cost = np.array([4 * np.sum((p < v) & (y == 1)) + np.sum((p >= v) & (y == 0)) for v in t])

    # np.flatnonzero and highest threshold make ties deterministic.
    def take(mask):
        return float(t[np.flatnonzero(mask)[-1]])

    return {
        "sensitivity90": take(sens >= 0.9),
        "youden": take(np.isclose(sens + spec, np.max(sens + spec))),
        "cost_fn4_fp1": take(cost == cost.min()),
        "specificity80": float(t[np.flatnonzero(spec >= 0.8)[0]]),
    }
