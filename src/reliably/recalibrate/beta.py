"""Beta calibration (3-parameter logistic on log-odds features)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from reliably._core.backend import clip_probs, to_numpy
from reliably.recalibrate.base import Calibrator

__all__ = ["BetaCalibrator"]


class BetaCalibrator(Calibrator):
    """Beta calibration: ``logit(p_cal) = c + a·log(s) − b·log(1 − s)``.

    Parameters
    ----------
    constrain_ab : bool
        If ``True`` (default), constrain ``a, b ≥ 0``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 300)
    >>> s = rng.uniform(0.05, 0.95, 300)
    >>> cal = BetaCalibrator().fit(s, y)
    >>> probs = cal.transform(s)
    >>> probs.shape == s.shape
    True
    """

    a_: float
    b_: float
    c_: float

    def __init__(self, constrain_ab: bool = True) -> None:
        self.constrain_ab = constrain_ab

    def fit(self, y_prob: Any, y_true: Any) -> "BetaCalibrator":
        """Fit beta calibration on calibration split.

        Parameters
        ----------
        y_prob : array-like
            Binary scores.
        y_true : array-like
            Binary labels.

        Returns
        -------
        BetaCalibrator
        """
        s = to_numpy(y_prob, dtype=np.float64)
        y = to_numpy(y_true, dtype=np.float64)
        if s.ndim == 2:
            s = s[:, 1]

        s_clipped = clip_probs(s)
        log_s = np.log(s_clipped)
        log_1ms = np.log(1.0 - s_clipped)

        def neg_log_lik(params: NDArray[np.float64]) -> float:
            a, b, c = params
            logit_p = c + a * log_s - b * log_1ms
            p = expit(logit_p)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        bounds = [(0.0, None), (0.0, None), (None, None)] if self.constrain_ab else None
        res = minimize(neg_log_lik, [1.0, 1.0, 0.0], method="L-BFGS-B", bounds=bounds)
        self.a_, self.b_, self.c_ = float(res.x[0]), float(res.x[1]), float(res.x[2])
        self._fitted = True
        return self

    def transform(self, y_prob: Any) -> NDArray[np.float64]:
        """Apply beta calibration.

        Parameters
        ----------
        y_prob : array-like
            Binary scores.

        Returns
        -------
        NDArray[np.float64]
            Calibrated probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        s = to_numpy(y_prob, dtype=np.float64)
        if s.ndim == 2:
            s = s[:, 1]
        s_clipped = clip_probs(s)
        logit_p = self.c_ + self.a_ * np.log(s_clipped) - self.b_ * np.log(1.0 - s_clipped)
        return np.array(expit(logit_p), dtype=np.float64)
