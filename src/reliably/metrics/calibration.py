"""Calibration metrics: ECE family, MCE, smECE, classwise ECE."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from reliably._core.backend import (
    adaptive_bins,
    bin_stats,
    clip_probs,
    equal_width_bins,
    make_rng,
    to_numpy,
)
from reliably._core.results import CI, MetricResult
from reliably.stats.bootstrap import vectorized_bootstrap_ci

__all__ = [
    "ece",
    "adaptive_ece",
    "mce",
    "debiased_ece",
    "smece",
    "classwise_ece",
    "reliability_curve",
]


def _top_label_conf_acc(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract top-label confidence and binary correctness.

    Top-label confidence = probability assigned to the predicted class,
    i.e. max(p, 1-p) for binary scalar inputs.
    """
    if y_prob.ndim == 1:
        pred = (y_prob >= 0.5).astype(np.int64)
        # Top-label confidence: probability of the predicted class
        conf = np.where(y_prob >= 0.5, y_prob, 1.0 - y_prob)
    else:
        conf = y_prob.max(axis=1)
        pred = y_prob.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)
    return conf, acc


def _ece_from_bins(
    bin_conf: NDArray[np.float64],
    bin_acc: NDArray[np.float64],
    bin_n: NDArray[np.int64],
    n: int,
) -> float:
    """Compute ECE from pre-computed bin statistics."""
    weights = bin_n / n
    return float(np.sum(weights * np.abs(bin_acc - bin_conf)))


def ece(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = 15,
    binning: str = "equal_width",
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Expected Calibration Error (ECE) with equal-width or adaptive binning.

    Parameters
    ----------
    y_true : array-like
        Integer labels, shape ``(N,)``.
    y_prob : array-like
        Probability matrix ``(N, K)`` or binary scores ``(N,)``.
    n_bins : int
        Number of bins (default 15).
    binning : str
        ``"equal_width"`` or ``"adaptive"``.
    ci : str | None
        CI method: ``"bca"``, ``"percentile"``, or ``None``.
    n_bootstrap : int
        Bootstrap resamples.
    level : float
        Nominal coverage.
    seed : int
        RNG seed.

    Returns
    -------
    MetricResult
        Named ``"ECE"`` or ``"adaptive_ECE"``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 200)
    >>> p = rng.uniform(0, 1, 200)
    >>> result = ece(y, p, ci=None)
    >>> 0.0 <= result.value <= 1.0
    True
    """
    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_prob_np = to_numpy(y_prob, dtype=np.float64)
    n = len(y_true_np)

    conf, acc = _top_label_conf_acc(y_true_np, y_prob_np)
    name = "ECE" if binning == "equal_width" else "adaptive_ECE"

    def _est(idx: NDArray[np.intp]) -> float:
        c, a = conf[idx], acc[idx]
        edges = (
            equal_width_bins(n_bins)
            if binning == "equal_width"
            else adaptive_bins(c, n_bins)
        )
        bc, ba, bn = bin_stats(c, a, edges)
        return _ece_from_bins(bc, ba, bn, len(idx))

    edges = (
        equal_width_bins(n_bins) if binning == "equal_width" else adaptive_bins(conf, n_bins)
    )
    bc, ba, bn = bin_stats(conf, acc, edges)
    point = _ece_from_bins(bc, ba, bn, n)

    if ci is None:
        return MetricResult(name=name, value=point, ci=None, n=n)

    ci_obj = vectorized_bootstrap_ci if False else None  # use generic for ECE

    # Use per-sample absolute gap as the data for vectorized bootstrap
    # For binned metrics we fall back to the generic estimator
    from reliably.stats.bootstrap import bootstrap_ci

    ci_result = bootstrap_ci(
        _est, n, point=point, n_boot=n_bootstrap, level=level, method=ci, seed=seed
    )
    return MetricResult(name=name, value=point, ci=ci_result, n=n)


def adaptive_ece(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = 15,
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Adaptive ECE (equal-mass / quantile binning).

    Parameters
    ----------
    y_true : array-like
        Integer labels.
    y_prob : array-like
        Probability matrix or binary scores.
    n_bins : int
        Number of bins.
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

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> y = rng.integers(0, 2, 300)
    >>> p = rng.uniform(0, 1, 300)
    >>> r = adaptive_ece(y, p, ci=None)
    >>> 0.0 <= r.value <= 1.0
    True
    """
    return ece(
        y_true,
        y_prob,
        n_bins=n_bins,
        binning="adaptive",
        ci=ci,
        n_bootstrap=n_bootstrap,
        level=level,
        seed=seed,
    )


def mce(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = 15,
    binning: str = "adaptive",
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Maximum Calibration Error (MCE).

    Parameters
    ----------
    y_true : array-like
        Integer labels.
    y_prob : array-like
        Probability matrix or binary scores.
    n_bins : int
        Number of bins.
    binning : str
        ``"equal_width"`` or ``"adaptive"``.
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
        Named ``"MCE"``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> y = rng.integers(0, 2, 200)
    >>> p = rng.uniform(0, 1, 200)
    >>> r = mce(y, p, ci=None)
    >>> 0.0 <= r.value <= 1.0
    True
    """
    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_prob_np = to_numpy(y_prob, dtype=np.float64)
    n = len(y_true_np)

    conf, acc = _top_label_conf_acc(y_true_np, y_prob_np)

    def _est(idx: NDArray[np.intp]) -> float:
        c, a = conf[idx], acc[idx]
        edges = (
            equal_width_bins(n_bins) if binning == "equal_width" else adaptive_bins(c, n_bins)
        )
        bc, ba, bn = bin_stats(c, a, edges)
        gaps = np.abs(ba - bc)
        gaps[bn == 0] = 0.0
        return float(gaps.max())

    edges = (
        equal_width_bins(n_bins) if binning == "equal_width" else adaptive_bins(conf, n_bins)
    )
    bc, ba, bn = bin_stats(conf, acc, edges)
    gaps = np.abs(ba - bc)
    gaps[bn == 0] = 0.0
    point = float(gaps.max())

    if ci is None:
        return MetricResult(name="MCE", value=point, ci=None, n=n)

    from reliably.stats.bootstrap import bootstrap_ci

    ci_result = bootstrap_ci(
        _est, n, point=point, n_boot=n_bootstrap, level=level, method=ci, seed=seed
    )
    return MetricResult(name="MCE", value=point, ci=ci_result, n=n)


def debiased_ece(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = 15,
    binning: str = "adaptive",
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Debiased ECE² (bias-corrected squared calibration error).

    Removes the finite-sample positive bias of the plug-in estimator.

    Parameters
    ----------
    y_true : array-like
        Integer labels.
    y_prob : array-like
        Probability matrix or binary scores.
    n_bins : int
        Number of bins.
    binning : str
        ``"equal_width"`` or ``"adaptive"``.
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
        Named ``"debiased_ECE2"`` (square-root reported as value).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(3)
    >>> y = rng.integers(0, 2, 500)
    >>> p = rng.uniform(0, 1, 500)
    >>> r = debiased_ece(y, p, ci=None)
    >>> r.value >= 0.0
    True
    """
    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_prob_np = to_numpy(y_prob, dtype=np.float64)
    n = len(y_true_np)

    conf, acc = _top_label_conf_acc(y_true_np, y_prob_np)

    def _est_sq(idx: NDArray[np.intp]) -> float:
        c, a = conf[idx], acc[idx]
        nn = len(idx)
        edges = (
            equal_width_bins(n_bins) if binning == "equal_width" else adaptive_bins(c, n_bins)
        )
        bc, ba, bn = bin_stats(c, a, edges)
        sq_gap = (ba - bc) ** 2
        # Bias term: acc*(1-acc)/(|B|-1), zero for empty or singleton bins
        bias = np.where(bn > 1, ba * (1.0 - ba) / (bn - 1), 0.0)
        debiased = np.maximum(sq_gap - bias, 0.0)
        return float(np.sum(bn / nn * debiased))

    edges = (
        equal_width_bins(n_bins) if binning == "equal_width" else adaptive_bins(conf, n_bins)
    )
    bc, ba, bn = bin_stats(conf, acc, edges)
    sq_gap = (ba - bc) ** 2
    bias = np.where(bn > 1, ba * (1.0 - ba) / (bn - 1), 0.0)
    debiased = np.maximum(sq_gap - bias, 0.0)
    point_sq = float(np.sum(bn / n * debiased))
    point = float(np.sqrt(point_sq))

    if ci is None:
        return MetricResult(name="debiased_ECE2", value=point, ci=None, n=n)

    from reliably.stats.bootstrap import bootstrap_ci

    ci_result = bootstrap_ci(
        lambda idx: float(np.sqrt(_est_sq(idx))),
        n,
        point=point,
        n_boot=n_bootstrap,
        level=level,
        method=ci,
        seed=seed,
    )
    return MetricResult(name="debiased_ECE2", value=point, ci=ci_result, n=n)


def smece(
    y_true: Any,
    y_prob: Any,
    *,
    bandwidth: float | None = None,
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Smooth (kernel) Expected Calibration Error (smECE).

    Uses Gaussian kernel regression of correctness on confidence.
    Bandwidth is selected by a Silverman-style rule-of-thumb if not provided.

    Parameters
    ----------
    y_true : array-like
        Integer labels.
    y_prob : array-like
        Probability matrix or binary scores.
    bandwidth : float | None
        Kernel bandwidth ``h``; ``None`` uses ``h ∝ N^{-1/5}``.
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
        Named ``"smECE"``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(4)
    >>> y = rng.integers(0, 2, 400)
    >>> p = rng.uniform(0, 1, 400)
    >>> r = smece(y, p, ci=None)
    >>> 0.0 <= r.value <= 1.0
    True
    """
    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_prob_np = to_numpy(y_prob, dtype=np.float64)
    n = len(y_true_np)

    conf, acc = _top_label_conf_acc(y_true_np, y_prob_np)

    def _smece_from_arrays(c: NDArray[np.float64], a: NDArray[np.float64]) -> float:
        nn = len(c)
        h = bandwidth if bandwidth is not None else max(0.1 * nn ** (-0.2), 0.01)
        # Kernel matrix: nn x nn, K_h(c_i - c_j)
        diff = c[:, None] - c[None, :]   # (nn, nn)
        K = np.exp(-(diff**2) / (2.0 * h**2))
        K_sum = K.sum(axis=1)            # normalizing constant per query point
        r_hat = K @ a / K_sum            # kernel regression estimate
        return float(np.abs(r_hat - c).mean())

    point = _smece_from_arrays(conf, acc)

    if ci is None:
        return MetricResult(name="smECE", value=point, ci=None, n=n)

    from reliably.stats.bootstrap import bootstrap_ci

    def _est(idx: NDArray[np.intp]) -> float:
        return _smece_from_arrays(conf[idx], acc[idx])

    ci_result = bootstrap_ci(
        _est, n, point=point, n_boot=n_bootstrap, level=level, method=ci, seed=seed
    )
    return MetricResult(name="smECE", value=point, ci=ci_result, n=n)


def classwise_ece(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = 15,
    binning: str = "adaptive",
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Classwise / marginal ECE for multiclass models.

    Parameters
    ----------
    y_true : array-like
        Integer labels.
    y_prob : array-like
        Probability matrix ``(N, K)``.
    n_bins : int
        Number of bins per class.
    binning : str
        ``"equal_width"`` or ``"adaptive"``.
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
        Named ``"cwECE"``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(5)
    >>> y = rng.integers(0, 3, 300)
    >>> p = rng.dirichlet([1, 1, 1], 300)
    >>> r = classwise_ece(y, p, ci=None)
    >>> 0.0 <= r.value <= 1.0
    True
    """
    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_prob_np = to_numpy(y_prob, dtype=np.float64)
    n = len(y_true_np)

    if y_prob_np.ndim == 1:
        # Binary: treat as 2-class
        y_prob_np = np.stack([1.0 - y_prob_np, y_prob_np], axis=1)

    k = y_prob_np.shape[1]

    def _cw_ece(idx: NDArray[np.intp]) -> float:
        total = 0.0
        nn = len(idx)
        yt = y_true_np[idx]
        yp = y_prob_np[idx]
        for cls in range(k):
            conf_k = yp[:, cls]
            acc_k = (yt == cls).astype(np.float64)
            edges = (
                equal_width_bins(n_bins)
                if binning == "equal_width"
                else adaptive_bins(conf_k, n_bins)
            )
            bc, ba, bn = bin_stats(conf_k, acc_k, edges)
            total += _ece_from_bins(bc, ba, bn, nn)
        return total / k

    point = _cw_ece(np.arange(n))

    if ci is None:
        return MetricResult(name="cwECE", value=point, ci=None, n=n)

    from reliably.stats.bootstrap import bootstrap_ci

    ci_result = bootstrap_ci(
        _cw_ece, n, point=point, n_boot=n_bootstrap, level=level, method=ci, seed=seed
    )
    return MetricResult(name="cwECE", value=point, ci=ci_result, n=n)


def reliability_curve(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = 15,
    binning: str = "adaptive",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Compute the binned reliability curve (mean confidence vs. mean accuracy).

    Parameters
    ----------
    y_true : array-like
        Integer labels.
    y_prob : array-like
        Probability matrix or binary scores.
    n_bins : int
        Number of bins.
    binning : str
        ``"equal_width"`` or ``"adaptive"``.

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
    >>> rng = np.random.default_rng(6)
    >>> y = rng.integers(0, 2, 200)
    >>> p = rng.uniform(0, 1, 200)
    >>> bc, ba, bn = reliability_curve(y, p)
    >>> len(bc) == 15
    True
    """
    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_prob_np = to_numpy(y_prob, dtype=np.float64)
    conf, acc = _top_label_conf_acc(y_true_np, y_prob_np)
    edges = (
        equal_width_bins(n_bins) if binning == "equal_width" else adaptive_bins(conf, n_bins)
    )
    return bin_stats(conf, acc, edges)
