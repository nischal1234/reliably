"""Mutual Information Gap (MIG) disentanglement metric."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import to_numpy
from reliably._core.results import MetricResult
from reliably.stats.bootstrap import bootstrap_ci

__all__ = ["mig"]


def _discretize(x: NDArray[np.float64], n_bins: int = 20) -> NDArray[np.int64]:
    """Discretize a continuous array into ``n_bins`` equal-width bins."""
    edges = np.linspace(x.min(), x.max() + 1e-12, n_bins + 1)
    return np.digitize(x, edges[:-1]).astype(np.int64) - 1


def _entropy(x: NDArray[np.int64]) -> float:
    """Shannon entropy (nats) of a discrete variable."""
    vals, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def _mutual_info(
    z_disc: NDArray[np.int64], v_disc: NDArray[np.int64]
) -> float:
    """Mutual information I(z; v) via joint histogram."""
    n = len(z_disc)
    z_vals = int(z_disc.max()) + 1
    v_vals = int(v_disc.max()) + 1
    joint = np.zeros((z_vals, v_vals), dtype=np.float64)
    np.add.at(joint, (z_disc, v_disc), 1.0 / n)
    pz = joint.sum(axis=1, keepdims=True)
    pv = joint.sum(axis=0, keepdims=True)
    # Avoid log(0)
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log(joint[mask] / (pz * pv + 1e-12)[mask])))
    return max(mi, 0.0)


def _mig_from_arrays(
    z: NDArray[np.float64],
    factors: NDArray[np.float64],
    n_bins: int = 20,
) -> float:
    """Compute MIG score from latent codes and factors."""
    n, d = z.shape
    k = factors.shape[1]

    z_disc = np.stack([_discretize(z[:, j], n_bins) for j in range(d)], axis=1)
    f_disc = np.stack([_discretize(factors[:, fk], n_bins) for fk in range(k)], axis=1)

    total = 0.0
    for fk in range(k):
        h_fk = _entropy(f_disc[:, fk])
        if h_fk < 1e-6:
            continue
        mi_scores = np.array([_mutual_info(z_disc[:, j], f_disc[:, fk]) for j in range(d)])
        top2 = np.sort(mi_scores)[-2:]
        if len(top2) >= 2:
            gap = top2[-1] - top2[-2]
        else:
            gap = top2[-1]
        total += gap / h_fk
    return float(total / k)


def mig(
    z: Any,
    factors: Any,
    *,
    n_bins: int = 20,
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Mutual Information Gap (MIG) disentanglement metric.

    Parameters
    ----------
    z : array-like
        Latent codes, shape ``(N, D)``.
    factors : array-like
        Ground-truth generative factors, shape ``(N, K)``.
    n_bins : int
        Bins for MI estimation.
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
        Named ``"MIG"``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(0, 1, (200, 4))
    >>> f = rng.normal(0, 1, (200, 3))
    >>> r = mig(z, f, ci=None)
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

    point = _mig_from_arrays(z_np, f_np, n_bins)

    if ci is None:
        return MetricResult(name="MIG", value=point, ci=None, n=n)

    def _est(idx: NDArray[np.intp]) -> float:
        return _mig_from_arrays(z_np[idx], f_np[idx], n_bins)

    ci_result = bootstrap_ci(_est, n, point=point, n_boot=n_bootstrap, level=level,
                             method=ci, seed=seed)
    return MetricResult(name="MIG", value=point, ci=ci_result, n=n)
