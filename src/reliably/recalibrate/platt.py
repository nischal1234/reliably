"""Platt scaling calibration (binary logistic fit)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from reliably._core.backend import clip_probs, to_numpy
from reliably.recalibrate.base import Calibrator

__all__ = ["PlattScaler"]


class PlattScaler(Calibrator):
    """Binary calibration via logistic regression: ``p_cal = σ(A·s + B)``.

    Parameters
    ----------
    None

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 300)
    >>> s = rng.uniform(0, 1, 300)
    >>> cal = PlattScaler().fit(s, y)
    >>> probs = cal.transform(s)
    >>> probs.shape == s.shape
    True
    """

    A_: float
    B_: float

    def fit(self, y_prob: Any, y_true: Any) -> "PlattScaler":
        """Fit logistic regression on calibration split.

        Parameters
        ----------
        y_prob : array-like
            Binary scores, shape ``(N,)``.
        y_true : array-like
            Binary labels.

        Returns
        -------
        PlattScaler
        """
        s = to_numpy(y_prob, dtype=np.float64)
        y = to_numpy(y_true, dtype=np.float64)
        if s.ndim == 2:
            s = s[:, 1]

        def neg_log_lik(params: NDArray[np.float64]) -> float:
            A, B = params
            p = expit(A * s + B)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        res = minimize(neg_log_lik, [1.0, 0.0], method="L-BFGS-B")
        self.A_ = float(res.x[0])
        self.B_ = float(res.x[1])
        self._fitted = True
        return self

    def transform(self, y_prob: Any) -> NDArray[np.float64]:
        """Apply Platt scaling.

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
        return np.array(expit(self.A_ * s + self.B_), dtype=np.float64)
