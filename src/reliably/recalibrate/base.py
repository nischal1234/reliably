"""Abstract base class for all calibrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import to_numpy

__all__ = ["Calibrator"]


class Calibrator(ABC):
    """Abstract calibrator: fit on a calibration split, transform test scores.

    Subclasses implement :meth:`fit` and :meth:`transform`.

    Examples
    --------
    See concrete subclasses in the recalibrate sub-package.
    """

    _fitted: bool = False

    @abstractmethod
    def fit(self, y_prob: Any, y_true: Any) -> "Calibrator":
        """Fit the calibrator on a calibration split.

        Parameters
        ----------
        y_prob : array-like
            Predicted probabilities.
        y_true : array-like
            True labels.

        Returns
        -------
        Calibrator
            ``self``, for chaining.
        """
        ...

    @abstractmethod
    def transform(self, y_prob: Any) -> NDArray[np.float64]:
        """Apply calibration to new predictions.

        Parameters
        ----------
        y_prob : array-like
            Predicted probabilities.

        Returns
        -------
        NDArray[np.float64]
            Calibrated probabilities.
        """
        ...
