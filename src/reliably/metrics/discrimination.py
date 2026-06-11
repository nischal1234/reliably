"""Discrimination metrics: AUROC with DeLong analytic CI."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import to_numpy
from reliably._core.results import CI, MetricResult
from reliably.stats.delong import auroc_delong

__all__ = ["auroc"]


def auroc(
    y_true: Any,
    y_score: Any,
    *,
    ci: str | None = "bca",
    level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> MetricResult:
    """Area under the ROC curve (AUROC) with DeLong or bootstrap CI.

    For binary tasks the DeLong analytic CI is used by default
    (``ci="bca"`` routes through the analytic DeLong variance).
    For the bootstrap path, use ``ci="percentile"``.

    Parameters
    ----------
    y_true : array-like
        Binary labels ``{0, 1}``, shape ``(N,)``.
    y_score : array-like
        Predicted scores (higher = more likely positive), shape ``(N,)``.
    ci : str | None
        ``"bca"`` uses DeLong analytic; ``"percentile"`` uses bootstrap;
        ``None`` skips CI.
    level : float
        Nominal CI coverage.
    n_bootstrap : int
        Bootstrap resamples (only used when ``ci="percentile"``).
    seed : int
        RNG seed for bootstrap.

    Returns
    -------
    MetricResult
        Named ``"AUROC"``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 200)
    >>> s = rng.uniform(0, 1, 200)
    >>> r = auroc(y, s, ci=None)
    >>> 0.0 <= r.value <= 1.0
    True
    """
    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_score_np = to_numpy(y_score, dtype=np.float64)
    n = len(y_true_np)

    if y_score_np.ndim == 2:
        # Use score of the positive class (index 1)
        y_score_np = y_score_np[:, 1]

    if ci is None:
        auc_val, _, _, _ = auroc_delong(y_score_np, y_true_np, level=level)
        return MetricResult(name="AUROC", value=float(auc_val), ci=None, n=n)

    if ci in ("bca", "analytic"):
        auc_val, lo, hi, _ = auroc_delong(y_score_np, y_true_np, level=level)
        ci_obj = CI(float(lo), float(hi), level, "analytic")
        return MetricResult(name="AUROC", value=float(auc_val), ci=ci_obj, n=n)

    # Bootstrap path
    from reliably.stats.bootstrap import bootstrap_ci
    from reliably.stats.delong import delong_var_components

    auc_val, _, _, _ = auroc_delong(y_score_np, y_true_np, level=level)

    def _est(idx: NDArray[np.intp]) -> float:
        s = y_score_np[idx]
        y = y_true_np[idx]
        if y.sum() == 0 or y.sum() == len(y):
            return 0.5
        a, _, _, _ = delong_var_components(s, y)
        return float(a)

    ci_result = bootstrap_ci(
        _est, n, point=float(auc_val), n_boot=n_bootstrap, level=level, method=ci, seed=seed
    )
    return MetricResult(name="AUROC", value=float(auc_val), ci=ci_result, n=n)
