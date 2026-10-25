"""Fitted preprocessing, partition integrity and compatibility regression tests."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from predictx.compatibility import preprocessing_signature
from predictx.evaluation import assert_disjoint, sensitivity_threshold
from predictx.models import Preprocessor, SharedNetwork, masked_bce
from predictx.runtime import FittedFeatures


def test_missing_outcome_gradient():
    logits = torch.tensor([[0.0], [0.0]], requires_grad=True)
    y = torch.tensor([[1.0], [float("nan")]])
    loss = masked_bce(logits, y)
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    assert logits.grad[1, 0].item() == 0
    assert logits.grad[0, 0].item() == pytest.approx(-0.5)


def test_independent_perceptron_gradient():
    model = SharedNetwork(4)
    loss = masked_bce(model(torch.ones((3, 4))), torch.ones((3, 1)))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert len(model.heads) == 1


def test_preprocess_training_only():
    train = pd.DataFrame({"a": [1.0, 3.0], "b": [np.nan, 4.0]})
    fitted = Preprocessor().fit(train)
    median = fitted.pipe[0].statistics_.copy()
    out = fitted.transform(pd.DataFrame({"a": [1000.0], "b": [np.nan]}))
    np.testing.assert_array_equal(fitted.pipe[0].statistics_, median)
    assert np.isfinite(out).all() and out.shape == (1, 5)


def test_signature_covers_fitted_numerical_state():
    fitted = FittedFeatures.from_fitted(
        Preprocessor().fit(pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 4.0]}))
    )
    original = preprocessing_signature(fitted)
    for field in ["imputation", "mean", "scale", "indicators", "low", "high", "order"]:
        changed = copy.deepcopy(fitted)
        if field == "imputation":
            changed.pipe[0].statistics_[0] += 1
        elif field == "mean":
            changed.pipe[1].mean_[0] += 1
        elif field == "scale":
            changed.pipe[1].scale_[0] += 1
        elif field == "indicators":
            changed.indicators = False
        elif field == "low":
            changed.lo[0] -= 1
        elif field == "high":
            changed.hi[0] += 1
        else:
            changed.columns.reverse()
        assert preprocessing_signature(changed) != original


def test_equivalent_preprocessors_have_identical_signatures():
    fitted = FittedFeatures.from_fitted(Preprocessor().fit(pd.DataFrame({"a": [1.0, 3.0]})))
    assert preprocessing_signature(fitted) == preprocessing_signature(copy.deepcopy(fitted))


def test_partition_leakage_and_identifier_guards():
    with pytest.raises(ValueError):
        assert_disjoint(np.array([1, 2, 3]), [[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        assert_disjoint(np.array([1, 1]), [[0], [1]])


def test_threshold_ties_sensitivity():
    y = np.array([1, 1, 1, 0])
    p = np.array([0.2, 0.2, 0.8, 0.4])
    cutoff = sensitivity_threshold(y, p, 0.9)
    assert cutoff == 0.2 and np.mean(p[y == 1] >= cutoff) >= 0.9


def test_calibration_audit_accepts_float32_roundoff():
    from sklearn.linear_model import LogisticRegression

    from predictx.evaluation import audit_sigmoid_calibration

    raw = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    labels = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    reference = LogisticRegression(C=1.0, max_iter=1000).fit(raw, labels)
    reference.intercept_ += np.finfo(np.float32).eps / 4
    result = audit_sigmoid_calibration(reference, raw, labels)
    assert result["probability_max_absolute_delta"] < 1e-7
    assert result["saved_gradient_infinity_norm"] <= reference.tol


def test_calibration_audit_rejects_inconsistent_fitted_parameters():
    from sklearn.linear_model import LogisticRegression

    from predictx.evaluation import audit_sigmoid_calibration

    raw = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    labels = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    reference = LogisticRegression(C=1.0, max_iter=1000).fit(raw, labels)
    reference.intercept_ += 0.05
    with pytest.raises(AssertionError, match="Calibration gradient"):
        audit_sigmoid_calibration(reference, raw, labels)
