"""Separated Attribute Predictability (SAP) metric."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import to_numpy
from reliably._core.results import MetricResult
from reliably.stats.bootstrap import bootstrap_ci

__all__ = ["sap"]


def _r2_score(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """R² coefficient of determination."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _linear_r2(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """R² of linear regression of y on x."""
    # OLS closed form
    x_m = x - x.mean()
    y_m = y - y.mean()
    denom = float(np.dot(x_m, x_m))
    if denom < 1e-12:
        return 0.0
    beta = float(np.dot(x_m, y_m)) / denom
    y_pred = x * beta + (y.mean() - beta * x.mean())
    return _r2_score(y, y_pred)


def _sap_from_arrays(
    z: NDArray[np.float64],
    factors: NDArray[np.float64],
) -> float:
    """Compute SAP score."""
    d = z.shape[1]
    k = factors.shape[1]

    # Score matrix S[j, fk] = R² of latent j predicting factor fk
    S = np.zeros((d, k))
    for j in range(d):
        for fk in range(k):
            S[j, fk] = _linear_r2(z[:, j], factors[:, fk])

    # SAP = mean over factors of (top1 - top2) score
    total = 0.0
    for fk in range(k):
        col = np.sort(S[:, fk])
        if len(col) >= 2:
            total += col[-1] - col[-2]
        else:
            total += col[-1]
    return total / k


def sap(
    z: Any,
    factors: Any,
    *,
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Separated Attribute Predictability (SAP).

    Parameters
    ----------
    z : array-like
        Latent codes, shape ``(N, D)``.
    factors : array-like
        Ground-truth factors, shape ``(N, K)``.
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
        Named ``"SAP"``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(0, 1, (200, 4))
    >>> f = rng.normal(0, 1, (200, 3))
    >>> r = sap(z, f, ci=None)
    >>> r.value >= 0.0
    True
    """
    z_np = to_numpy(z, dtype=np.float64)
    f_np = to_numpy(factors, dtype=np.float64)
    if z_np.ndim == 1:
        z_np = z_np[:, None]
    if f_np.ndim == 1:
        f_np = f_np[:, None]
    n = z_np.shape[0]

    point = _sap_from_arrays(z_np, f_np)

    if ci is None:
        return MetricResult(name="SAP", value=point, ci=None, n=n)

    def _est(idx: NDArray[np.intp]) -> float:
        return _sap_from_arrays(z_np[idx], f_np[idx])

    ci_result = bootstrap_ci(_est, n, point=point, n_boot=n_bootstrap,
                             level=level, method=ci, seed=seed)
    return MetricResult(name="SAP", value=point, ci=ci_result, n=n)
