"""Paired bootstrap difference test and multiple-comparison corrections."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import make_rng
from reliably._core.results import CI
from reliably.stats.bootstrap import bootstrap_replicate_indices

__all__ = ["paired_bootstrap_test", "holm_bonferroni", "benjamini_hochberg"]


def paired_bootstrap_test(
    estimator_a: Callable[[NDArray[Any]], float],
    estimator_b: Callable[[NDArray[Any]], float],
    n: int,
    *,
    point_a: float,
    point_b: float,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int | np.random.Generator = 0,
) -> tuple[float, CI, float]:
    """Paired bootstrap test for any pair of scalar estimators.

    Both estimators are applied to the *same* resample indices so the
    comparison is paired.  This works for any metric, unlike DeLong.

    Parameters
    ----------
    estimator_a : Callable[[NDArray], float]
        ``f(idx) -> float`` for model A.
    estimator_b : Callable[[NDArray], float]
        ``f(idx) -> float`` for model B.
    n : int
        Dataset size.
    point_a : float
        Full-data point estimate for model A.
    point_b : float
        Full-data point estimate for model B.
    n_boot : int
        Number of resamples.
    level : float
        Nominal CI coverage.
    seed : int | np.random.Generator
        RNG seed.

    Returns
    -------
    delta : float
        ``point_a - point_b``.
    ci : CI
        Bootstrap CI on the difference (percentile by default).
    p_value : float
        Two-sided p-value via bootstrap hypothesis convention.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(0, 1, 200)
    >>> delta, ci, p = paired_bootstrap_test(
    ...     lambda idx: x[idx].mean(), lambda idx: (-x)[idx].mean(),
    ...     len(x), point_a=x.mean(), point_b=(-x).mean(), seed=0
    ... )
    >>> p < 0.01
    True
    """
    rng = make_rng(seed)
    idx_matrix = bootstrap_replicate_indices(n, n_boot, seed=rng)

    delta_boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = idx_matrix[b]
        delta_boot[b] = estimator_a(idx) - estimator_b(idx)

    delta = point_a - point_b
    alpha = 1.0 - level
    lo, hi = np.quantile(delta_boot, [alpha / 2, 1.0 - alpha / 2])
    ci = CI(float(lo), float(hi), level, "percentile")

    # Bootstrap hypothesis p-value
    n_le = int(np.sum(delta_boot <= 0))
    n_ge = int(np.sum(delta_boot >= 0))
    p_value = float(2.0 * min((n_le + 1) / (n_boot + 1), (n_ge + 1) / (n_boot + 1)))

    return delta, ci, p_value


def holm_bonferroni(p_values: list[float], level: float = 0.05) -> list[bool]:
    """Holm–Bonferroni step-down FWER correction.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values.
    level : float
        Family-wise error rate.

    Returns
    -------
    list[bool]
        Significance flags (``True`` means rejected) in the original order.

    Examples
    --------
    >>> holm_bonferroni([0.01, 0.04, 0.2], level=0.05)
    [True, True, False]
    """
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    rejected = [False] * n
    for rank, idx in enumerate(order):
        threshold = level / (n - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            # Once one test is not rejected, stop
            break
    return rejected


def benjamini_hochberg(p_values: list[float], level: float = 0.05) -> list[bool]:
    """Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values.
    level : float
        False discovery rate.

    Returns
    -------
    list[bool]
        Significance flags in the original order.

    Examples
    --------
    >>> benjamini_hochberg([0.01, 0.04, 0.2], level=0.05)
    [True, False, False]
    """
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    rejected = [False] * n
    # Step up: find largest k such that p_(k) <= k/n * level
    last_rejected = -1
    for rank, idx in enumerate(order):
        if p_values[idx] <= (rank + 1) / n * level:
            last_rejected = rank
    for rank in range(last_rejected + 1):
        rejected[order[rank]] = True
    return rejected


def apply_correction(
    p_values: list[float],
    correction: str | None,
    *,
    level: float = 0.05,
) -> list[bool]:
    """Apply a multiple-comparison correction by name.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values.
    correction : str | None
        ``"holm"``, ``"bh"`` (Benjamini–Hochberg), or ``None``.
    level : float
        Error rate.

    Returns
    -------
    list[bool]
        Significance flags.

    Examples
    --------
    >>> apply_correction([0.01, 0.2], "holm")
    [True, False]
    """
    if correction is None:
        return [p < level for p in p_values]
    if correction == "holm":
        return holm_bonferroni(p_values, level)
    if correction in ("bh", "fdr"):
        return benjamini_hochberg(p_values, level)
    raise ValueError(f"Unknown correction {correction!r}. Use 'holm', 'bh', or None.")
