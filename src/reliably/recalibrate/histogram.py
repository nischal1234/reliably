"""Histogram binning calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import adaptive_bins, bin_stats, equal_width_bins, to_numpy
from reliably.recalibrate.base import Calibrator

__all__ = ["HistogramCalibrator"]


class HistogramCalibrator(Calibrator):
    """Replace each bin's score with its empirical accuracy on the calibration split.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins.
    binning : str
        ``"equal_width"`` or ``"adaptive"``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 300)
    >>> s = rng.uniform(0, 1, 300)
    >>> cal = HistogramCalibrator().fit(s, y)
    >>> probs = cal.transform(s)
    >>> probs.shape == s.shape
    True
    """

    edges_: NDArray[np.float64]
    bin_acc_: NDArray[np.float64]

    def __init__(self, n_bins: int = 15, binning: str = "adaptive") -> None:
        self.n_bins = n_bins
        self.binning = binning

    def fit(self, y_prob: Any, y_true: Any) -> "HistogramCalibrator":
        """Fit histogram binning on calibration split.

        Parameters
        ----------
        y_prob : array-like
            Scores.
        y_true : array-like
            Binary labels.

        Returns
        -------
        HistogramCalibrator
        """
        s = to_numpy(y_prob, dtype=np.float64)
        y = to_numpy(y_true, dtype=np.float64)
        if s.ndim == 2:
            s = s[:, 1]

        edges = (
            equal_width_bins(self.n_bins)
            if self.binning == "equal_width"
            else adaptive_bins(s, self.n_bins)
        )
        _, bin_acc, bin_n = bin_stats(s, y, edges)
        # Where a bin is empty, use the base rate
        base_rate = float(y.mean())
        bin_acc[bin_n == 0] = base_rate

        self.edges_ = edges
        self.bin_acc_ = bin_acc
        self._fitted = True
        return self

    def transform(self, y_prob: Any) -> NDArray[np.float64]:
        """Apply histogram calibration.

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

        out = np.empty_like(s)
        n_bins = len(self.edges_) - 1
        for m in range(n_bins):
            lo, hi = self.edges_[m], self.edges_[m + 1]
            if m == n_bins - 1:
                mask = (s >= lo) & (s <= hi)
            else:
                mask = (s >= lo) & (s < hi)
            out[mask] = self.bin_acc_[m]
        return out
