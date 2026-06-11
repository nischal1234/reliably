"""Representation quality metrics: MIG, SAP, DCI, FactorVAE, IRS."""

from __future__ import annotations

from reliably.repr.dci import dci
from reliably.repr.factorvae import factorvae_metric
from reliably.repr.irs import irs
from reliably.repr.mig import mig
from reliably.repr.sap import sap

__all__ = ["mig", "sap", "dci", "factorvae_metric", "irs", "disentanglement"]


def disentanglement(
    z: object,
    factors: object,
    metrics: tuple[str, ...] = ("mig", "sap", "dci", "factorvae", "irs"),
    *,
    ci: str | None = "bca",
    n_bootstrap: int = 200,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, object]:
    """Compute a suite of disentanglement metrics.

    Parameters
    ----------
    z : array-like
        Latent codes, shape ``(N, D)``.
    factors : array-like
        Ground-truth factors, shape ``(N, K)``.
    metrics : tuple[str, ...]
        Metrics to compute.
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
    dict[str, MetricResult]
        Results keyed by metric name.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(0, 1, (100, 4))
    >>> f = rng.normal(0, 1, (100, 3))
    >>> results = disentanglement(z, f, metrics=("mig",), ci=None)
    >>> "mig" in results
    True
    """
    from reliably._core.results import MetricResult

    kw = dict(ci=ci, n_bootstrap=n_bootstrap, level=level, seed=seed)
    results: dict[str, MetricResult] = {}
    if "mig" in metrics:
        results["mig"] = mig(z, factors, **kw)  # type: ignore[arg-type]
    if "sap" in metrics:
        results["sap"] = sap(z, factors, **kw)  # type: ignore[arg-type]
    if "dci" in metrics:
        results["dci"] = dci(z, factors, **kw)  # type: ignore[arg-type]
    if "factorvae" in metrics:
        results["factorvae"] = factorvae_metric(z, factors, **kw)  # type: ignore[arg-type]
    if "irs" in metrics:
        results["irs"] = irs(z, factors, **kw)  # type: ignore[arg-type]
    return results
