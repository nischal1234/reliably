"""Fast DeLong variance and two-sample AUROC test (O(N log N) via midranks)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

__all__ = ["delong_var_components", "delong_test", "auroc_delong"]


def _midrank(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute midranks (average rank for ties) of a 1-D array.

    Parameters
    ----------
    x : NDArray[np.float64]
        1-D array of values.

    Returns
    -------
    NDArray[np.float64]
        Midranks, same length as ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> _midrank(np.array([3.0, 1.0, 2.0]))
    array([3., 1., 2.])
    """
    order = np.argsort(x, kind="stable")
    ranked = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i, N = 0, len(x)
    while i < N:
        j = i + 1
        while j < N and sx[j] == sx[i]:
            j += 1
        # Midrank = average of 1-based positions i+1 .. j
        mid = 0.5 * (i + j - 1) + 1  # 1-based midrank
        ranked[order[i:j]] = mid
        i = j
    return ranked


def delong_var_components(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Compute AUC and placement-value arrays via the DeLong method.

    Parameters
    ----------
    scores : NDArray[np.float64]
        Prediction scores, shape ``(N,)``.
    labels : NDArray[np.int64]
        Binary labels ``{0, 1}``, shape ``(N,)``.

    Returns
    -------
    auc : float
        AUROC estimate.
    var : float
        Variance of the AUROC estimate.
    v10 : NDArray[np.float64]
        Placement values for positives, shape ``(m,)``.
    v01 : NDArray[np.float64]
        Placement values for negatives, shape ``(n,)``.

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array([0.9, 0.4, 0.8, 0.3])
    >>> y = np.array([1, 0, 1, 0])
    >>> auc, var, v10, v01 = delong_var_components(s, y)
    >>> round(auc, 4)
    1.0
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos = scores[pos_mask]
    neg = scores[neg_mask]
    m, n = len(pos), len(neg)

    if m == 0 or n == 0:
        raise ValueError("Need at least one positive and one negative sample.")

    # Midranks in the combined array (positives first)
    combined = np.concatenate([pos, neg])
    tz = _midrank(combined)
    tx = _midrank(pos)
    ty = _midrank(neg)

    # AUC from midranks: equivalent to Wilcoxon-Mann-Whitney statistic
    auc = float((tz[:m].sum() / m - (m + 1) / 2.0) / n)

    # Placement values (structural components)
    v10 = (tz[:m] - tx) / n          # shape (m,)
    v01 = 1.0 - (tz[m:] - ty) / m   # shape (n,)

    s10 = float(np.var(v10, ddof=1) / m) if m > 1 else 0.0
    s01 = float(np.var(v01, ddof=1) / n) if n > 1 else 0.0
    var = s10 + s01

    return auc, var, v10, v01


def auroc_delong(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    level: float = 0.95,
) -> tuple[float, float, float, float]:
    """Compute AUROC with an analytic (DeLong) CI and normal-approximation p-value.

    Parameters
    ----------
    scores : NDArray[np.float64]
        Prediction scores.
    labels : NDArray[np.int64]
        Binary labels.
    level : float
        Nominal CI level.

    Returns
    -------
    auc : float
        AUROC point estimate.
    ci_low : float
        Lower CI bound.
    ci_high : float
        Upper CI bound.
    var : float
        Estimated variance.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 100)
    >>> s = rng.uniform(0, 1, 100)
    >>> auc, lo, hi, var = auroc_delong(s, y)
    >>> lo < auc < hi
    True
    """
    auc, var, _v10, _v01 = delong_var_components(scores, labels)
    se = float(np.sqrt(max(var, 0.0)))
    z = norm.ppf((1.0 + level) / 2.0)
    return auc, auc - z * se, auc + z * se, var


def delong_test(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> tuple[float, float, float]:
    """Compare two correlated AUROCs on the same test set (DeLong 1988).

    Parameters
    ----------
    scores_a : NDArray[np.float64]
        Scores from model A.
    scores_b : NDArray[np.float64]
        Scores from model B.
    labels : NDArray[np.int64]
        Shared binary labels.

    Returns
    -------
    delta : float
        ``AUC_a - AUC_b``.
    p_value : float
        Two-sided p-value.
    se : float
        Standard error of the difference.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> y = rng.integers(0, 2, 200)
    >>> sa = rng.uniform(0, 1, 200)
    >>> sb = rng.uniform(0, 1, 200)
    >>> delta, p, se = delong_test(sa, sb, y)
    >>> 0.0 <= p <= 1.0
    True
    """
    auc_a, _var_a, v10a, v01a = delong_var_components(scores_a, labels)
    auc_b, _var_b, v10b, v01b = delong_var_components(scores_b, labels)

    # Correlated covariance (same test set)
    cov10 = float(np.cov(v10a, v10b)[0, 1]) / len(v10a) if len(v10a) > 1 else 0.0
    cov01 = float(np.cov(v01a, v01b)[0, 1]) / len(v01a) if len(v01a) > 1 else 0.0

    var_a = float(np.var(v10a, ddof=1) / len(v10a) + np.var(v01a, ddof=1) / len(v01a))
    var_b = float(np.var(v10b, ddof=1) / len(v10b) + np.var(v01b, ddof=1) / len(v01b))

    var_diff = var_a + var_b - 2.0 * (cov10 + cov01)
    se = float(np.sqrt(max(var_diff, 0.0)))
    delta = auc_a - auc_b
    z = delta / se if se > 0 else 0.0
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return float(delta), p_value, se
