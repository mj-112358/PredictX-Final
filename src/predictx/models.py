"""Independent model fitting, training-only preprocessing and sigmoid calibration."""

import copy

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


def masked_bce(logits, targets, weights=None):
    mask = torch.isfinite(targets)
    safe = torch.nan_to_num(targets, nan=0.0)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, safe, reduction="none")
    per_task = (loss * mask).sum(0) / mask.sum(0).clamp_min(1)
    return (per_task * (weights if weights is not None else 1)).sum()


class SharedNetwork(nn.Module):
    """Single-output perceptron with the saved artifact's serialization identity."""

    def __init__(self, d, k=1, mode="shared", width=32):
        super().__init__()
        if k != 1 or mode != "shared":
            raise ValueError("Independent perceptron requires one output")
        self.mode = mode
        self.encoder = nn.Sequential(
            nn.Linear(d, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(width, 1)])
        self.private = None

    def forward(self, x):
        return self.heads[0](self.encoder(x))


class Preprocessor:
    def __init__(self, indicators=True):
        self.indicators = indicators

    def fit(self, x):
        self.columns = list(x.columns)
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
        )
        self.pipe.fit(x)
        self.lo = x.quantile(0.005).to_numpy()
        self.hi = x.quantile(0.995).to_numpy()
        return self

    def transform(self, x):
        x = x[self.columns]
        v = np.clip(self.pipe.transform(x), -10, 10)
        if self.indicators:
            mask = x.notna().to_numpy(dtype=float)
            # Append observation indicators and their per-record mean.
            v = np.c_[v, mask, mask.mean(1)]
        return v.astype(np.float32)

    def ood(self, x):
        v = x[self.columns].to_numpy(dtype=float)
        return np.nanmean((v < self.lo) | (v > self.hi), axis=1)


def train_neural(x, y, vx, vy, mode, lr, seed, epochs=120):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = SharedNetwork(x.shape[1], y.shape[1], mode)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.float32))
    vt = torch.from_numpy(vx)
    best = float("inf")
    state = None
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = masked_bce(model(xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            score = float(masked_bce(model(vt), torch.from_numpy(vy.astype(np.float32))))
        if score < best:
            best = score
            state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
    model.load_state_dict(state)
    model.eval()
    return model, {"lr": lr, "epoch": best_epoch, "validation_loss_sum": best}


def raw_scores(model, x):
    if isinstance(model, nn.Module):
        with torch.no_grad():
            return model(torch.from_numpy(x if x.flags.writeable else x.copy())).numpy()
    if hasattr(model, "decision_function"):
        return model.decision_function(x).reshape(-1, 1)
    p = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class ResearchModel:
    """Serialized model bundle; calibrators are fitted only on calibration partition."""

    def __init__(self, preprocessor, models, tasks, calibrators, thresholds, metadata):
        self.preprocessor = preprocessor
        self.models = models
        self.tasks = tasks
        self.calibrators = calibrators
        self.thresholds = thresholds
        self.metadata = metadata

    def scores(self, x):
        v = self.preprocessor.transform(x)
        return (
            np.c_[tuple(raw_scores(m, v).ravel() for m in self.models)]
            if isinstance(self.models, list)
            else raw_scores(self.models, v)
        )

    def predict_preprocessed(self, v):
        s = (
            np.column_stack([raw_scores(m, v).ravel() for m in self.models])
            if isinstance(self.models, list)
            else raw_scores(self.models, v)
        )
        return np.column_stack(
            [cal.predict_proba(s[:, k : k + 1])[:, 1] for k, cal in enumerate(self.calibrators)]
        )

    def predict_proba(self, x):
        s = self.scores(x)
        return np.c_[
            tuple(
                cal.predict_proba(s[:, k : k + 1])[:, 1] for k, cal in enumerate(self.calibrators)
            )
        ]
