"""Proper scoring rules: Brier score (+ Murphy decomposition) and NLL."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import (
    adaptive_bins,
    bin_stats,
    clip_probs,
    to_numpy,
)
from reliably._core.results import MetricResult
from reliably.stats.bootstrap import vectorized_bootstrap_ci

__all__ = ["brier", "nll"]


def _one_hot(y: NDArray[np.int64], k: int) -> NDArray[np.float64]:
    """Convert integer labels to one-hot matrix."""
    out = np.zeros((len(y), k), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def brier(
    y_true: Any,
    y_prob: Any,
    *,
    decompose: bool = False,
    n_bins: int = 15,
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Brier score with optional Murphy decomposition.

    Parameters
    ----------
    y_true : array-like
        Integer labels, shape ``(N,)``.
    y_prob : array-like
        Probability matrix ``(N, K)`` or binary scores ``(N,)``.
    decompose : bool
        If ``True``, include the Murphy decomposition (binary only) in
        ``MetricResult.extra``.
    n_bins : int
        Bins for the Murphy decomposition.
    ci : str | None
        CI method.
    n_bootstrap : int
        Bootstrap resamples.
    level : float
        Nominal coverage.
    seed : int
        RNG seed.

    Returns
    -------
    MetricResult
        Named ``"Brier"``. If ``decompose=True``, ``extra`` contains
        ``{"reliability", "resolution", "uncertainty"}``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 300)
    >>> p = rng.uniform(0, 1, 300)
    >>> r = brier(y, p, ci=None)
    >>> 0.0 <= r.value <= 1.0
    True
    """
    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_prob_np = to_numpy(y_prob, dtype=np.float64)
    n = len(y_true_np)
    binary = y_prob_np.ndim == 1

    if binary:
        # Per-sample squared error
        per_sample = (y_prob_np - y_true_np.astype(np.float64)) ** 2
    else:
        k = y_prob_np.shape[1]
        oh = _one_hot(y_true_np, k)
        per_sample = ((y_prob_np - oh) ** 2).sum(axis=1)

    point = float(per_sample.mean())

    extra: dict[str, float] | None = None
    if decompose and binary:
        extra = _murphy_decomposition(y_true_np, y_prob_np, n_bins=n_bins)

    if ci is None:
        return MetricResult(name="Brier", value=point, ci=None, n=n, extra=extra)

    ci_result = vectorized_bootstrap_ci(
        per_sample, point=point, n_boot=n_bootstrap, level=level, method=ci, seed=seed
    )
    return MetricResult(name="Brier", value=point, ci=ci_result, n=n, extra=extra)


def _murphy_decomposition(
    y_true: NDArray[np.int64],
    scores: NDArray[np.float64],
    n_bins: int = 15,
) -> dict[str, float]:
    """Murphy calibration–resolution–uncertainty decomposition (binary)."""
    n = len(y_true)
    base_rate = float(y_true.mean())
    edges = adaptive_bins(scores, n_bins)
    bc, ba, bn = bin_stats(scores, y_true.astype(np.float64), edges)

    weights = bn / n
    reliability = float(np.sum(weights * (bc - ba) ** 2))
    resolution = float(np.sum(weights * (ba - base_rate) ** 2))
    uncertainty = base_rate * (1.0 - base_rate)

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }


def nll(
    y_true: Any,
    y_prob: Any,
    *,
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Negative log-likelihood (log loss).

    Parameters
    ----------
    y_true : array-like
        Integer labels, shape ``(N,)``.
    y_prob : array-like
        Probability matrix ``(N, K)`` or binary scores ``(N,)``.
    ci : str | None
        CI method.
    n_bootstrap : int
        Bootstrap resamples.
    level : float
        Nominal coverage.
    seed : int
        RNG seed.

    Returns
    -------
    MetricResult
        Named ``"NLL"``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 300)
    >>> p = rng.uniform(0.1, 0.9, 300)
    >>> r = nll(y, p, ci=None)
    >>> r.value > 0.0
    True
    """
    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_prob_np = to_numpy(y_prob, dtype=np.float64)
    n = len(y_true_np)
    binary = y_prob_np.ndim == 1

    if binary:
        p_clipped = clip_probs(y_prob_np)
        p_correct = np.where(y_true_np == 1, p_clipped, 1.0 - p_clipped)
    else:
        p_clipped = clip_probs(y_prob_np)
        p_correct = p_clipped[np.arange(n), y_true_np]

    per_sample = -np.log(p_correct)
    point = float(per_sample.mean())

    if ci is None:
        return MetricResult(name="NLL", value=point, ci=None, n=n)

    ci_result = vectorized_bootstrap_ci(
        per_sample, point=point, n_boot=n_bootstrap, level=level, method=ci, seed=seed
    )
    return MetricResult(name="NLL", value=point, ci=ci_result, n=n)
