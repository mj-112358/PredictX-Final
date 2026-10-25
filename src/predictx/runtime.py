"""Fitted feature transformations and independently calibrated scalar predictors."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FittedFeatures:
    columns: list[str]
    pipe: Any
    lo: np.ndarray
    hi: np.ndarray
    indicators: bool = True

    @classmethod
    def from_fitted(cls, p):
        return cls(list(p.columns), p.pipe, p.lo, p.hi, p.indicators)

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        x = x[self.columns]
        a = np.clip(self.pipe.transform(x), -10, 10)
        if self.indicators:
            mask = x.notna().to_numpy(dtype=float)
            a = np.c_[a, mask, mask.mean(1)]
        return a.astype(np.float32)

    def ood(self, x):
        a = x[self.columns].to_numpy(float)
        return np.mean((a < self.lo) | (a > self.hi), axis=1)


def scalar_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x)).reshape(-1, 1)
    if hasattr(model, "predict_proba"):
        p = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)
    import torch

    with torch.no_grad():
        return model(torch.from_numpy(x if x.flags.writeable else x.copy())).numpy()


@dataclass
class DiseaseModel:
    preprocessor: FittedFeatures
    model: Any
    calibrator: Any
    threshold: float
    metadata: dict[str, Any]

    @property
    def models(self):
        return [self.model]

    @property
    def thresholds(self):
        return [self.threshold]

    @property
    def tasks(self):
        return self.metadata["tasks"]

    def predict_preprocessed(self, x: np.ndarray) -> np.ndarray:
        return self.calibrator.predict_proba(scalar_scores(self.model, x))[:, 1:2]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.predict_preprocessed(self.preprocessor.transform(x))
