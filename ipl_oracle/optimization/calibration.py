"""Platt scaling backed by sklearn LogisticRegression."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattScaler:
    """Fits P(y=1) = σ(A·logit(p) + B) on (predicted, actual) pairs."""

    def __init__(self):
        self._model: LogisticRegression | None = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def fit(self, predicted: list[float], actual: list[float]) -> None:
        if len(predicted) < 5:
            self._model = None
            return
        x = self._logit(np.asarray(predicted, dtype=float)).reshape(-1, 1)
        y = (np.asarray(actual, dtype=float) > 0.5).astype(int)
        if len(set(y.tolist())) < 2:
            self._model = None
            return
        self._model = LogisticRegression(max_iter=1000)
        self._model.fit(x, y)

    def transform(self, raw: float) -> float:
        if self._model is None:
            return raw
        x = self._logit(np.asarray([raw], dtype=float)).reshape(-1, 1)
        return float(self._model.predict_proba(x)[0, 1])

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
