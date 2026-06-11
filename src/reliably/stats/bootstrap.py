"""Nonparametric bootstrap: vectorized percentile and BCa confidence intervals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from reliably._core.backend import make_rng
from reliably._core.results import CI

__all__ = ["bootstrap_ci", "bootstrap_replicate_indices"]


def bootstrap_replicate_indices(
    n: int,
    n_boot: int,
    *,
    seed: int | np.random.Generator = 0,
) -> NDArray[np.intp]:
    """Pre-draw the full ``(n_boot, n)`` resample index matrix.

    Parameters
    ----------
    n : int
        Dataset size.
    n_boot : int
        Number of bootstrap replicates.
    seed : int | np.random.Generator
        RNG seed for reproducibility.

    Returns
    -------
    NDArray[np.intp]
        Shape ``(n_boot, n)`` integer index matrix.

    Examples
    --------
    >>> idx = bootstrap_replicate_indices(10, 5, seed=0)
    >>> idx.shape
    (5, 10)
    """
    rng = make_rng(seed)
    dtype = np.int32 if n < 2**31 else np.int64
    return rng.integers(0, n, size=(n_boot, n), dtype=dtype)


def _bca_ci(
    boot: NDArray[np.float64],
    point: float,
    jack: NDArray[np.float64],
    level: float,
) -> CI:
    """Compute BCa CI from pre-computed bootstrap and jackknife arrays."""
    alpha = 1.0 - level
    n_boot = len(boot)

    # Bias correction
    n_below = float(np.sum(boot < point))
    z0 = norm.ppf((n_below + 0.5) / (n_boot + 1))

    # Acceleration via jackknife
    jbar = float(jack.mean())
    diff = jbar - jack
    num = float(np.sum(diff**3))
    den = 6.0 * float(np.sum(diff**2) ** 1.5) + 1e-12
    a = num / den

    zL = norm.ppf(alpha / 2)
    zU = norm.ppf(1.0 - alpha / 2)

    def adj(zq: float) -> float:
        denom = 1.0 - a * (z0 + zq)
        if abs(denom) < 1e-12:
            return float(norm.cdf(z0 + zq))
        return float(norm.cdf(z0 + (z0 + zq) / denom))

    alpha1 = adj(zL)
    alpha2 = adj(zU)
    # Clamp to valid quantile range
    alpha1 = float(np.clip(alpha1, 1e-6, 1.0 - 1e-6))
    alpha2 = float(np.clip(alpha2, 1e-6, 1.0 - 1e-6))

    lo, hi = np.quantile(boot, [alpha1, alpha2])
    return CI(float(lo), float(hi), level, "bca")


def bootstrap_ci(
    estimator: Callable[[NDArray[Any]], float],
    n: int,
    *,
    point: float,
    n_boot: int = 2000,
    level: float = 0.95,
    method: str = "bca",
    seed: int | np.random.Generator = 0,
) -> CI:
    """Compute a bootstrap confidence interval for any scalar estimator.

    The estimator is called once per bootstrap replicate with a resample
    index array; use vectorized implementations where possible.

    Parameters
    ----------
    estimator : Callable[[NDArray], float]
        Function ``f(idx) -> float`` applied to each resample.
    n : int
        Dataset size.
    point : float
        Point estimate (from the full dataset, not resampled).
    n_boot : int
        Number of resamples (default 2000).
    level : float
        Nominal coverage (default 0.95).
    method : str
        ``"bca"`` (default) or ``"percentile"``.
    seed : int | np.random.Generator
        RNG seed.

    Returns
    -------
    CI
        Confidence interval object.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.default_rng(0).normal(0, 1, 200)
    >>> ci = bootstrap_ci(lambda idx: data[idx].mean(), len(data),
    ...                   point=data.mean(), seed=0)
    >>> ci.low < ci.high
    True
    """
    rng = make_rng(seed)
    idx_matrix = bootstrap_replicate_indices(n, n_boot, seed=rng)

    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boot[b] = estimator(idx_matrix[b])

    alpha = 1.0 - level
    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2, 1.0 - alpha / 2])
        return CI(float(lo), float(hi), level, "percentile")

    # BCa: jackknife for acceleration
    full = np.arange(n)
    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        mask = np.concatenate([full[:i], full[i + 1 :]])
        jack[i] = estimator(mask)

    return _bca_ci(boot, point, jack, level)


def vectorized_bootstrap_ci(
    data: NDArray[np.float64],
    *,
    point: float,
    n_boot: int = 2000,
    level: float = 0.95,
    method: str = "bca",
    seed: int | np.random.Generator = 0,
) -> CI:
    """BCa/percentile CI for the **mean** of a 1-D data array.

    Uses a fully vectorized ``(n_boot × n)`` resample matrix — no Python
    loop over bootstrap replicates, which is critical for performance.

    Parameters
    ----------
    data : NDArray[np.float64]
        1-D array of values whose mean is the metric.
    point : float
        Point estimate (should equal ``data.mean()``).
    n_boot : int
        Number of resamples.
    level : float
        Nominal coverage.
    method : str
        ``"bca"`` or ``"percentile"``.
    seed : int | np.random.Generator
        RNG seed.

    Returns
    -------
    CI

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.default_rng(0).normal(0, 1, 500)
    >>> ci = vectorized_bootstrap_ci(data, point=data.mean(), seed=0)
    >>> ci.low < 0 < ci.high
    True
    """
    n = len(data)
    rng = make_rng(seed)
    idx_matrix = bootstrap_replicate_indices(n, n_boot, seed=rng)

    # Vectorized: all replicates as one matrix operation
    boot = data[idx_matrix].mean(axis=1)  # shape (n_boot,)

    alpha = 1.0 - level
    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2, 1.0 - alpha / 2])
        return CI(float(lo), float(hi), level, "percentile")

    # Jackknife: leave-one-out means via rank-1 update trick
    total = data.sum()
    jack = (total - data) / (n - 1)

    return _bca_ci(boot, point, jack, level)
