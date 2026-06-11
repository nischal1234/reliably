"""Temperature scaling calibration (single scalar T, multiclass)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from reliably._core.backend import clip_probs, softmax, to_numpy
from reliably.recalibrate.base import Calibrator

__all__ = ["TemperatureScaler"]


class TemperatureScaler(Calibrator):
    """Calibrate by dividing logits by a scalar temperature T > 0.

    Fits T by minimizing NLL on the calibration split using golden-section
    search.  Preserves the argmax (accuracy unchanged).

    Parameters
    ----------
    T_bounds : tuple[float, float]
        Search bounds for temperature.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y_true = rng.integers(0, 2, 200)
    >>> y_prob = rng.dirichlet([1, 1], 200)
    >>> cal = TemperatureScaler().fit(y_prob, y_true)
    >>> cal.T_ > 0
    True
    >>> cal_probs = cal.transform(y_prob)
    >>> np.allclose(cal_probs.sum(axis=1), 1.0, atol=1e-6)
    True
    """

    T_: float
    logits_: NDArray[np.float64]

    def __init__(self, T_bounds: tuple[float, float] = (0.01, 20.0)) -> None:
        self.T_bounds = T_bounds

    def fit(self, y_prob: Any, y_true: Any) -> "TemperatureScaler":
        """Fit temperature on calibration data.

        Parameters
        ----------
        y_prob : array-like
            Probabilities or logits, shape ``(N, K)`` or ``(N,)`` (binary).
        y_true : array-like
            Integer labels.

        Returns
        -------
        TemperatureScaler
        """
        y_prob_np = to_numpy(y_prob, dtype=np.float64)
        y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
        n = len(y_true_np)

        if y_prob_np.ndim == 1:
            # Binary: convert to 2-class
            y_prob_np = np.stack([1.0 - y_prob_np, y_prob_np], axis=1)

        # Recover pseudo-logits as log(p) — up to a constant, sufficient for
        # temperature scaling because softmax is shift-invariant
        p_clipped = clip_probs(y_prob_np)
        logits = np.log(p_clipped)
        self.logits_ = logits

        k = y_prob_np.shape[1]

        def nll_at_T(T: float) -> float:
            probs = softmax(logits / T)
            p_correct = clip_probs(probs[np.arange(n), y_true_np])
            return float(-np.log(p_correct).mean())

        result = minimize_scalar(nll_at_T, bounds=self.T_bounds, method="bounded")
        self.T_ = float(result.x)
        self._fitted = True
        return self

    def transform(self, y_prob: Any) -> NDArray[np.float64]:
        """Apply temperature scaling.

        Parameters
        ----------
        y_prob : array-like
            Probabilities to calibrate.

        Returns
        -------
        NDArray[np.float64]
            Calibrated probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        y_prob_np = to_numpy(y_prob, dtype=np.float64)
        binary = y_prob_np.ndim == 1
        if binary:
            y_prob_np = np.stack([1.0 - y_prob_np, y_prob_np], axis=1)

        p_clipped = clip_probs(y_prob_np)
        logits = np.log(p_clipped)
        cal = softmax(logits / self.T_)
        if binary:
            return cal[:, 1]
        return cal
