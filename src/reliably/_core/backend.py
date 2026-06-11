"""Backend utilities: array conversion, RNG, binning, and numerical helpers."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "to_numpy",
    "make_rng",
    "clip_probs",
    "equal_width_bins",
    "adaptive_bins",
]

EPS: float = 1e-12


def to_numpy(x: Any, *, dtype: type = np.float64) -> NDArray[np.float64]:
    """Convert any array-like (numpy, torch, jax, list) to a float64 numpy array.

    Parameters
    ----------
    x : array-like
        Input data.
    dtype : type
        Target numpy dtype, default ``float64``.

    Returns
    -------
    NDArray[np.float64]
        Contiguous C-order array.

    Examples
    --------
    >>> to_numpy([1, 2, 3]).dtype
    dtype('float64')
    """
    # Handle torch tensors
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        x = x.numpy()
    # Handle jax arrays
    if hasattr(x, "__jax_array__"):
        import jax.numpy as jnp  # type: ignore

        x = jnp.asarray(x)
    return np.asarray(x, dtype=dtype)


def make_rng(seed: int | np.random.Generator) -> np.random.Generator:
    """Create or pass through a numpy random generator.

    Parameters
    ----------
    seed : int | np.random.Generator
        Integer seed or an existing generator.

    Returns
    -------
    np.random.Generator

    Examples
    --------
    >>> rng = make_rng(42)
    >>> isinstance(rng, np.random.Generator)
    True
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def clip_probs(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clip probabilities to ``[EPS, 1-EPS]`` for numerical safety.

    Parameters
    ----------
    p : NDArray[np.float64]
        Probability array of any shape.

    Returns
    -------
    NDArray[np.float64]
        Clipped array.

    Examples
    --------
    >>> import numpy as np
    >>> clip_probs(np.array([0.0, 0.5, 1.0]))
    array([1.e-12, 5.e-01, 1.e+00])
    """
    return np.clip(p, EPS, 1.0 - EPS)


def equal_width_bins(n_bins: int = 15) -> NDArray[np.float64]:
    """Bin edges for equal-width binning on ``[0, 1]``.

    Parameters
    ----------
    n_bins : int
        Number of bins.

    Returns
    -------
    NDArray[np.float64]
        Array of ``n_bins + 1`` edges.

    Examples
    --------
    >>> equal_width_bins(5).tolist()
    [0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0]
    """
    return np.linspace(0.0, 1.0, n_bins + 1).astype(np.float64)


def adaptive_bins(
    confidences: NDArray[np.float64], n_bins: int = 15
) -> NDArray[np.float64]:
    """Bin edges for equal-mass (adaptive) binning.

    Parameters
    ----------
    confidences : NDArray[np.float64]
        1-D confidence scores.
    n_bins : int
        Number of bins.

    Returns
    -------
    NDArray[np.float64]
        Monotone array of ``n_bins + 1`` bin edges.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> c = rng.uniform(0, 1, 100)
    >>> edges = adaptive_bins(c, 5)
    >>> len(edges)
    6
    """
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    edges: NDArray[np.float64] = np.percentile(confidences, quantiles).astype(np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    return edges


def bin_stats(
    confidences: NDArray[np.float64],
    accuracies: NDArray[np.float64],
    edges: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Compute per-bin mean confidence, mean accuracy, and count.

    Parameters
    ----------
    confidences : NDArray[np.float64]
        Top-label confidence per sample, shape ``(N,)``.
    accuracies : NDArray[np.float64]
        Binary correctness per sample, shape ``(N,)``.
    edges : NDArray[np.float64]
        Bin edges, shape ``(M+1,)``.

    Returns
    -------
    bin_conf : NDArray[np.float64]
        Mean confidence per bin, shape ``(M,)``.
    bin_acc : NDArray[np.float64]
        Mean accuracy per bin, shape ``(M,)``.
    bin_n : NDArray[np.int64]
        Sample count per bin, shape ``(M,)``.

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array([0.1, 0.5, 0.9])
    >>> a = np.array([0.0, 1.0, 1.0])
    >>> edges = np.array([0.0, 0.5, 1.0])
    >>> bc, ba, bn = bin_stats(c, a, edges)
    >>> bn.tolist()
    [2, 1]
    """
    n_bins = len(edges) - 1
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_n = np.zeros(n_bins, dtype=np.int64)

    for m in range(n_bins):
        lo, hi = edges[m], edges[m + 1]
        if m == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        cnt = int(mask.sum())
        bin_n[m] = cnt
        if cnt > 0:
            bin_conf[m] = confidences[mask].mean()
            bin_acc[m] = accuracies[mask].mean()

    return bin_conf, bin_acc, bin_n


def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax along last axis.

    Parameters
    ----------
    z : NDArray[np.float64]
        Logits array of shape ``(..., K)``.

    Returns
    -------
    NDArray[np.float64]
        Probabilities, same shape as ``z``.

    Examples
    --------
    >>> import numpy as np
    >>> softmax(np.array([1.0, 2.0, 3.0])).sum()
    1.0
    """
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def validate_and_warn_probabilities(
    y_prob: NDArray[np.float64], tol: float = 1e-4
) -> NDArray[np.float64]:
    """Validate that rows sum to 1; auto-normalize with a warning if not.

    Parameters
    ----------
    y_prob : NDArray[np.float64]
        Probability array of shape ``(N, K)`` or ``(N,)``.
    tol : float
        Tolerance for sum-to-one check.

    Returns
    -------
    NDArray[np.float64]
        Normalized probability array.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([[0.4, 0.7]])
    >>> validate_and_warn_probabilities(p).sum(axis=1)
    array([1.])
    """
    if y_prob.ndim == 1:
        # Binary scalar scores — just validate range
        if not np.all((y_prob >= 0) & (y_prob <= 1)):
            raise ValueError("Binary scores must be in [0, 1].")
        return y_prob

    row_sums = y_prob.sum(axis=1)
    if not np.all(np.abs(row_sums - 1.0) <= tol):
        warnings.warn(
            "Probability rows do not sum to 1 (max deviation "
            f"{np.abs(row_sums - 1.0).max():.6f}); auto-normalizing.",
            stacklevel=3,
        )
        y_prob = y_prob / row_sums[:, None]
    if not np.all(y_prob >= 0):
        raise ValueError("Probabilities must be non-negative.")
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("Probabilities must be finite.")
    return y_prob
