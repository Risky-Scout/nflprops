"""OOF-only probability calibrators.

A calibrator is a post-processing model. It never sees the target sportsbook line as
a football feature; it learns whether raw forecast probabilities are systematically
too confident or too conservative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from nflprops.backtest.metrics import log_loss

EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


class ProbabilityCalibrator:
    def __init__(self, method: str = "logistic"):
        if method not in {"logistic", "beta", "isotonic"}:
            raise ValueError(method)
        self.method = method
        self.model = None

    def _features(self, p):
        p = _clip(p)
        if self.method == "logistic":
            return np.log(p / (1 - p)).reshape(-1, 1)
        if self.method == "beta":
            return np.column_stack([np.log(p), np.log(1 - p)])
        return p

    def fit(self, p_oof, y_true) -> ProbabilityCalibrator:
        p = _clip(p_oof)
        y = np.asarray(y_true, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("calibration requires both outcome classes")
        if self.method == "isotonic":
            self.model = IsotonicRegression(
                out_of_bounds="clip", y_min=EPS, y_max=1 - EPS
            ).fit(p, y)
        else:
            self.model = LogisticRegression(
                C=1e6, solver="lbfgs", max_iter=2000
            ).fit(self._features(p), y)
        return self

    def transform(self, p):
        if self.model is None:
            raise RuntimeError("calibrator is not fitted")
        pp = _clip(p)
        if self.method == "isotonic":
            return _clip(self.model.predict(pp))
        return _clip(self.model.predict_proba(self._features(pp))[:, 1])


@dataclass(frozen=True)
class CalibratorSelection:
    method: str
    oof_log_loss: float


def choose_calibrator(p_train_oof, y_train) -> tuple[ProbabilityCalibrator, CalibratorSelection]:
    """Choose method only by OOF log loss on the supplied calibration sample."""
    candidates = []
    for method in ("logistic", "beta", "isotonic"):
        cal = ProbabilityCalibrator(method).fit(p_train_oof, y_train)
        pred = cal.transform(p_train_oof)
        candidates.append((log_loss(y_train, pred), method, cal))
    score, method, cal = min(candidates, key=lambda x: x[0])
    return cal, CalibratorSelection(method=method, oof_log_loss=float(score))
