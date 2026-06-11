"""Interventional Robustness Score (IRS) — Suter et al. 2019."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import make_rng, to_numpy
from reliably._core.results import MetricResult
from reliably.stats.bootstrap import bootstrap_ci

__all__ = ["irs"]


def _irs_from_arrays(
    z: NDArray[np.float64],
    factors: NDArray[np.float64],
    *,
    n_interventions: int = 100,
    rng: np.random.Generator,
) -> float:
    """Compute IRS score.

    For each pair (target factor, nuisance factor):
    - Sample two groups that agree on the target factor but differ on the nuisance.
    - Find the best-matching latent for the target factor.
    - Measure the maximum change in that latent across groups (lower = more robust).
    Aggregate and invert to report higher = better.
    """
    n, d = z.shape
    k = factors.shape[1]

    # Discretize factors for grouping
    n_bins = min(10, max(2, n // 20))
    f_disc = np.zeros_like(factors, dtype=np.int64)
    for fk in range(k):
        edges = np.linspace(factors[:, fk].min(), factors[:, fk].max() + 1e-12, n_bins + 1)
        f_disc[:, fk] = np.digitize(factors[:, fk], edges[:-1]) - 1

    # Normalize latents by global std
    std = z.std(axis=0) + 1e-12
    z_norm = z / std[None, :]

    max_deviations: list[float] = []

    for _ in range(n_interventions):
        target_factor = int(rng.integers(0, k))
        nuisance_factor = int(rng.integers(0, k))
        if nuisance_factor == target_factor and k > 1:
            nuisance_factor = (target_factor + 1) % k

        # Sample a fixed target level and two different nuisance levels
        target_val = int(rng.integers(0, n_bins))
        nuisance_vals = rng.choice(n_bins, size=2, replace=False)

        group_a_mask = (f_disc[:, target_factor] == target_val) & \
                       (f_disc[:, nuisance_factor] == int(nuisance_vals[0]))
        group_b_mask = (f_disc[:, target_factor] == target_val) & \
                       (f_disc[:, nuisance_factor] == int(nuisance_vals[1]))

        if group_a_mask.sum() < 2 or group_b_mask.sum() < 2:
            continue

        mean_a = z_norm[group_a_mask].mean(axis=0)
        mean_b = z_norm[group_b_mask].mean(axis=0)
        deviation = float(np.abs(mean_a - mean_b).max())
        max_deviations.append(deviation)

    if not max_deviations:
        return 0.0

    # IRS: mean max deviation → lower is better → report 1 - normalized deviation
    mean_dev = float(np.mean(max_deviations))
    # Normalize to [0, 1] using a soft clamp
    irs_score = float(np.exp(-mean_dev))
    return irs_score


def irs(
    z: Any,
    factors: Any,
    *,
    n_interventions: int = 100,
    ci: str | None = "bca",
    n_bootstrap: int = 200,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Interventional Robustness Score (IRS).

    Measures maximum change in the matched latent under interventions on
    nuisance factors while the target factor is held fixed (Suter et al., 2019).
    Higher score = more robust / better disentanglement.

    Parameters
    ----------
    z : array-like
        Latent codes, shape ``(N, D)``.
    factors : array-like
        Ground-truth factors, shape ``(N, K)``.
    n_interventions : int
        Number of random interventions to sample.
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
        Named ``"IRS"``, value in ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(0, 1, (300, 4))
    >>> f = rng.normal(0, 1, (300, 3))
    >>> r = irs(z, f, ci=None)
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

    rng = make_rng(seed)
    point = _irs_from_arrays(z_np, f_np, n_interventions=n_interventions, rng=rng)

    if ci is None:
        return MetricResult(name="IRS", value=point, ci=None, n=n)

    def _est(idx: NDArray[np.intp]) -> float:
        sub_rng = make_rng(int(idx[:3].sum()))
        return _irs_from_arrays(z_np[idx], f_np[idx],
                                n_interventions=n_interventions, rng=sub_rng)

    ci_result = bootstrap_ci(_est, n, point=point, n_boot=n_bootstrap,
                             level=level, method=ci, seed=seed)
    return MetricResult(name="IRS", value=point, ci=ci_result, n=n)
