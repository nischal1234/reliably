"""DCI: Disentanglement, Completeness, Informativeness metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import to_numpy
from reliably._core.results import MetricResult
from reliably.stats.bootstrap import bootstrap_ci

__all__ = ["dci"]


def _entropy_normalized(p: NDArray[np.float64], base: int) -> float:
    """Normalized entropy in [0, 1] with given base."""
    p = p[p > 0]
    if len(p) == 0 or base <= 1:
        return 0.0
    h = float(-np.sum(p * np.log(p)))
    return h / np.log(base)


def _importance_matrix(
    z: NDArray[np.float64],
    factors: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute D×K relative-importance matrix using gradient boosted trees.

    Falls back to Lasso if sklearn is unavailable (unlikely since sklearn
    is an optional dep but let's be safe).
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-not-found]
    except ImportError:
        return _importance_lasso(z, factors)

    d = z.shape[1]
    k = factors.shape[1]
    R = np.zeros((d, k))
    for fk in range(k):
        gbr = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=0)
        gbr.fit(z, factors[:, fk])
        imp = np.abs(gbr.feature_importances_)
        R[:, fk] = imp / (imp.sum() + 1e-12)
    return R


def _importance_lasso(
    z: NDArray[np.float64],
    factors: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fallback: importance via |Lasso coefficients|."""
    from scipy.linalg import lstsq

    d = z.shape[1]
    k = factors.shape[1]
    R = np.zeros((d, k))
    for fk in range(k):
        coef, _, _, _ = lstsq(z, factors[:, fk])
        imp = np.abs(coef)
        R[:, fk] = imp / (imp.sum() + 1e-12)
    return R


def _dci_from_arrays(
    z: NDArray[np.float64],
    factors: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Return (disentanglement, completeness, informativeness)."""
    d = z.shape[1]
    k = factors.shape[1]

    R = _importance_matrix(z, factors)

    # Disentanglement: how exclusively does each latent encode one factor
    disentanglement_scores = np.zeros(d)
    rho = np.zeros(d)
    for j in range(d):
        row_sum = R[j, :].sum()
        rho[j] = row_sum
        if row_sum < 1e-12:
            disentanglement_scores[j] = 0.0
        else:
            p = R[j, :] / row_sum
            disentanglement_scores[j] = 1.0 - _entropy_normalized(p, k)
    rho_total = rho.sum()
    weights = rho / (rho_total + 1e-12)
    D_score = float(np.sum(weights * disentanglement_scores))

    # Completeness: how exclusively is each factor captured by one latent
    completeness_scores = np.zeros(k)
    for fk in range(k):
        col_sum = R[:, fk].sum()
        if col_sum < 1e-12:
            completeness_scores[fk] = 0.0
        else:
            p = R[:, fk] / col_sum
            completeness_scores[fk] = 1.0 - _entropy_normalized(p, d)
    C_score = float(completeness_scores.mean())

    # Informativeness: mean R² of factor predictions (using importance model)
    try:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-not-found]
        info_scores = []
        for fk in range(k):
            gbr = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=0)
            gbr.fit(z, factors[:, fk])
            info_scores.append(gbr.score(z, factors[:, fk]))
        I_score = float(np.mean(info_scores))
    except ImportError:
        I_score = float(R.max(axis=0).mean())

    return D_score, C_score, I_score


def dci(
    z: Any,
    factors: Any,
    *,
    ci: str | None = "bca",
    n_bootstrap: int = 200,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """DCI: Disentanglement, Completeness, Informativeness.

    Parameters
    ----------
    z : array-like
        Latent codes, shape ``(N, D)``.
    factors : array-like
        Ground-truth factors, shape ``(N, K)``.
    ci : str | None
        CI method (bootstrap; note: DCI is slow, so default n_bootstrap=200).
    n_bootstrap : int
        Bootstrap resamples.
    level : float
        Nominal coverage.
    seed : int
        RNG seed.

    Returns
    -------
    MetricResult
        Named ``"DCI"`` with ``extra`` containing
        ``{"disentanglement", "completeness", "informativeness"}``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(0, 1, (100, 4))
    >>> f = rng.normal(0, 1, (100, 3))
    >>> r = dci(z, f, ci=None)
    >>> 0.0 <= r.value <= 1.0
    True
    """
    z_np = to_numpy(z, dtype=np.float64)
    f_np = to_numpy(factors, dtype=np.float64)
    if z_np.ndim == 1:
        z_np = z_np[:, None]
    if f_np.ndim == 1:
        f_np = f_np[:, None]
    n = z_np.shape[0]

    D, C, I = _dci_from_arrays(z_np, f_np)
    point = (D + C) / 2.0
    extra = {"disentanglement": D, "completeness": C, "informativeness": I}

    if ci is None:
        return MetricResult(name="DCI", value=point, ci=None, n=n, extra=extra)

    def _est(idx: NDArray[np.intp]) -> float:
        d_, c_, _ = _dci_from_arrays(z_np[idx], f_np[idx])
        return (d_ + c_) / 2.0

    ci_result = bootstrap_ci(_est, n, point=point, n_boot=n_bootstrap,
                             level=level, method=ci, seed=seed)
    return MetricResult(name="DCI", value=point, ci=ci_result, n=n, extra=extra)
