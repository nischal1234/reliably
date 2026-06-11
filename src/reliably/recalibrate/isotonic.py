"""Isotonic regression calibration (Pool-Adjacent-Violators)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import to_numpy
from reliably.recalibrate.base import Calibrator

__all__ = ["IsotonicCalibrator"]


class IsotonicCalibrator(Calibrator):
    """Nonparametric monotone calibration via isotonic regression.

    Wraps ``sklearn.isotonic.IsotonicRegression`` and requires the
    ``scikit-learn`` optional dependency.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 300)
    >>> s = rng.uniform(0, 1, 300)
    >>> cal = IsotonicCalibrator().fit(s, y)
    >>> probs = cal.transform(s)
    >>> probs.shape == s.shape
    True
    """

    def fit(self, y_prob: Any, y_true: Any) -> IsotonicCalibrator:
        """Fit isotonic regression on calibration split.

        Parameters
        ----------
        y_prob : array-like
            Scores, shape ``(N,)``.
        y_true : array-like
            Binary labels.

        Returns
        -------
        IsotonicCalibrator
        """
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for IsotonicCalibrator. "
                "Install with: pip install reliably[sklearn]"
            ) from exc

        s = to_numpy(y_prob, dtype=np.float64)
        y = to_numpy(y_true, dtype=np.float64)
        if s.ndim == 2:
            s = s[:, 1]

        self._ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._ir.fit(s, y)
        self._fitted = True
        return self

    def transform(self, y_prob: Any) -> NDArray[np.float64]:
        """Apply isotonic calibration.

        Parameters
        ----------
        y_prob : array-like
            Scores.

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
        return np.array(self._ir.transform(s), dtype=np.float64)
