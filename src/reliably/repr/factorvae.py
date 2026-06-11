"""FactorVAE metric for disentanglement evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import make_rng, to_numpy
from reliably._core.results import MetricResult
from reliably.stats.bootstrap import bootstrap_ci

__all__ = ["factorvae_metric"]


def _factorvae_from_arrays(
    z: NDArray[np.float64],
    factors: NDArray[np.float64],
    *,
    n_votes: int = 800,
    batch_size: int = 64,
    rng: np.random.Generator,
) -> float:
    """Compute FactorVAE metric score.

    For each vote: fix a factor, generate a batch varying all other factors,
    compute normalized variance of each latent, vote for the lowest-variance
    latent as encoding the fixed factor.  Train and evaluate a majority-vote
    classifier.
    """
    n, d = z.shape
    k = factors.shape[1]

    # Normalize latents by global std
    std = z.std(axis=0) + 1e-12
    z_norm = z / std[None, :]

    votes: list[tuple[int, int]] = []
    for _ in range(n_votes):
        # Pick a random factor
        fixed_factor = int(rng.integers(0, k))
        # Sample a batch
        idx = rng.integers(0, n, size=batch_size)
        z_batch = z_norm[idx]
        # Variance of each latent in this batch
        var = z_batch.var(axis=0)
        voted_latent = int(np.argmin(var))
        votes.append((voted_latent, fixed_factor))

    # Build majority-vote classifier: for each (voted_latent, factor) -> factor
    # Use a simple lookup: most common factor per latent vote
    from collections import defaultdict, Counter

    vote_dict: dict[int, list[int]] = defaultdict(list)
    for latent_vote, factor in votes:
        vote_dict[latent_vote].append(factor)

    # Classifier: latent -> predicted factor (majority vote)
    classifier: dict[int, int] = {}
    for latent, factor_list in vote_dict.items():
        classifier[latent] = Counter(factor_list).most_common(1)[0][0]

    # Evaluate on the same votes
    correct = sum(
        1 for (latent_vote, factor) in votes if classifier.get(latent_vote, -1) == factor
    )
    return correct / len(votes)


def factorvae_metric(
    z: Any,
    factors: Any,
    *,
    n_votes: int = 800,
    batch_size: int = 64,
    ci: str | None = "bca",
    n_bootstrap: int = 200,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """FactorVAE disentanglement metric.

    Parameters
    ----------
    z : array-like
        Latent codes, shape ``(N, D)``.
    factors : array-like
        Ground-truth factors, shape ``(N, K)``.
    n_votes : int
        Number of voting rounds.
    batch_size : int
        Batch size per round.
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
        Named ``"FactorVAE"``, value in ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(0, 1, (300, 4))
    >>> f = rng.normal(0, 1, (300, 3))
    >>> r = factorvae_metric(z, f, ci=None)
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
    point = _factorvae_from_arrays(z_np, f_np, n_votes=n_votes, batch_size=batch_size, rng=rng)

    if ci is None:
        return MetricResult(name="FactorVAE", value=point, ci=None, n=n)

    def _est(idx: NDArray[np.intp]) -> float:
        sub_rng = make_rng(int(idx[:3].sum()))
        return _factorvae_from_arrays(
            z_np[idx], f_np[idx], n_votes=n_votes, batch_size=min(batch_size, len(idx)),
            rng=sub_rng
        )

    ci_result = bootstrap_ci(_est, n, point=point, n_boot=n_bootstrap,
                             level=level, method=ci, seed=seed)
    return MetricResult(name="FactorVAE", value=point, ci=ci_result, n=n)
