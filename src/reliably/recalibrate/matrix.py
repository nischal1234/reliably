"""Vector and matrix scaling for multiclass calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from reliably._core.backend import clip_probs, softmax, to_numpy
from reliably.recalibrate.base import Calibrator

__all__ = ["VectorScaler", "MatrixScaler"]


class VectorScaler(Calibrator):
    """Per-class temperature scaling: ``p_cal = softmax(w ⊙ logits + b)``.

    More expressive than scalar temperature but less prone to overfitting
    than full matrix scaling.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 3, 300)
    >>> p = rng.dirichlet([1, 1, 1], 300)
    >>> cal = VectorScaler().fit(p, y)
    >>> probs = cal.transform(p)
    >>> np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    True
    """

    W_: NDArray[np.float64]
    b_: NDArray[np.float64]

    def fit(self, y_prob: Any, y_true: Any) -> VectorScaler:
        """Fit per-class vector scaling.

        Parameters
        ----------
        y_prob : array-like
            Probabilities ``(N, K)``.
        y_true : array-like
            Integer labels.

        Returns
        -------
        VectorScaler
        """
        y_prob_np = to_numpy(y_prob, dtype=np.float64)
        y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
        n = len(y_true_np)
        k = y_prob_np.shape[1] if y_prob_np.ndim == 2 else 2

        if y_prob_np.ndim == 1:
            y_prob_np = np.stack([1.0 - y_prob_np, y_prob_np], axis=1)

        logits = np.log(clip_probs(y_prob_np))

        def neg_nll(params: NDArray[np.float64]) -> float:
            weights = params[:k]
            b = params[k:]
            z = logits * weights[None, :] + b[None, :]
            probs = softmax(z)
            p_correct = clip_probs(probs[np.arange(n), y_true_np])
            return float(-np.log(p_correct).mean())

        x0 = np.concatenate([np.ones(k), np.zeros(k)])
        res = minimize(neg_nll, x0, method="L-BFGS-B")
        self.W_ = res.x[:k]
        self.b_ = res.x[k:]
        self._fitted = True
        return self

    def transform(self, y_prob: Any) -> NDArray[np.float64]:
        """Apply vector scaling.

        Parameters
        ----------
        y_prob : array-like
            Probabilities.

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
        logits = np.log(clip_probs(y_prob_np))
        z = logits * self.W_[None, :] + self.b_[None, :]
        cal = softmax(z)
        if binary:
            return cal[:, 1]
        return cal


class MatrixScaler(Calibrator):
    """Full K×K affine map on logits: ``p_cal = softmax(W·logits + b)``.

    More expressive; gate behind ``method="matrix"``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 3, 300)
    >>> p = rng.dirichlet([1, 1, 1], 300)
    >>> cal = MatrixScaler().fit(p, y)
    >>> probs = cal.transform(p)
    >>> np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    True
    """

    W_: NDArray[np.float64]
    b_: NDArray[np.float64]

    def fit(self, y_prob: Any, y_true: Any) -> MatrixScaler:
        """Fit full matrix scaling.

        Parameters
        ----------
        y_prob : array-like
            Probabilities ``(N, K)``.
        y_true : array-like
            Integer labels.

        Returns
        -------
        MatrixScaler
        """
        y_prob_np = to_numpy(y_prob, dtype=np.float64)
        y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
        n = len(y_true_np)
        k = y_prob_np.shape[1] if y_prob_np.ndim == 2 else 2

        if y_prob_np.ndim == 1:
            y_prob_np = np.stack([1.0 - y_prob_np, y_prob_np], axis=1)

        logits = np.log(clip_probs(y_prob_np))

        def neg_nll(params: NDArray[np.float64]) -> float:
            weight_mat = params[: k * k].reshape(k, k)
            b = params[k * k :]
            z = logits @ weight_mat.T + b[None, :]
            probs = softmax(z)
            p_correct = clip_probs(probs[np.arange(n), y_true_np])
            return float(-np.log(p_correct).mean())

        x0 = np.concatenate([np.eye(k).ravel(), np.zeros(k)])
        res = minimize(neg_nll, x0, method="L-BFGS-B")
        self.W_ = res.x[: k * k].reshape(k, k)
        self.b_ = res.x[k * k :]
        self._fitted = True
        return self

    def transform(self, y_prob: Any) -> NDArray[np.float64]:
        """Apply matrix scaling.

        Parameters
        ----------
        y_prob : array-like
            Probabilities.

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
        logits = np.log(clip_probs(y_prob_np))
        z = logits @ self.W_.T + self.b_[None, :]
        cal = softmax(z)
        if binary:
            return cal[:, 1]
        return cal
