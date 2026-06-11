"""Reliability diagram with confidence band and confidence histogram."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

__all__ = ["reliability_diagram", "confidence_histogram"]


def reliability_diagram(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = 15,
    binning: str = "adaptive",
    band: bool = True,
    n_bootstrap: int = 200,
    seed: int = 0,
    ax: "matplotlib.axes.Axes | None" = None,
    title: str = "Reliability Diagram",
) -> "matplotlib.axes.Axes":
    """Plot a reliability diagram with optional bootstrap confidence band.

    The smooth kernel curve (smECE) is plotted with a shaded bootstrap band.
    The binned points are overlaid as scatter.

    Parameters
    ----------
    y_true : array-like
        Integer labels.
    y_prob : array-like
        Probability matrix or binary scores.
    n_bins : int
        Number of bins for the scatter overlay.
    binning : str
        ``"equal_width"`` or ``"adaptive"``.
    band : bool
        Whether to show the bootstrap confidence band.
    n_bootstrap : int
        Resamples for the confidence band.
    seed : int
        RNG seed.
    ax : matplotlib.axes.Axes | None
        Axes to plot on; creates a new figure if ``None``.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 300)
    >>> p = rng.uniform(0, 1, 300)
    >>> ax = reliability_diagram(y, p, band=False)
    >>> ax is not None
    True
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for reliability_diagram. "
            "Install with: pip install reliably[viz]"
        ) from exc

    from reliably._core.backend import to_numpy
    from reliably.metrics.calibration import _top_label_conf_acc, reliability_curve
    from reliably.stats.bootstrap import bootstrap_replicate_indices

    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_prob_np = to_numpy(y_prob, dtype=np.float64)
    conf, acc = _top_label_conf_acc(y_true_np, y_prob_np)
    n = len(conf)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # --- Diagonal (perfect calibration) ---
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")

    # --- Smooth kernel curve ---
    h = max(0.1 * n ** (-0.2), 0.01)
    c_grid = np.linspace(0.0, 1.0, 200)
    diff = c_grid[:, None] - conf[None, :]
    K = np.exp(-(diff**2) / (2.0 * h**2))
    K_sum = K.sum(axis=1)
    r_hat = K @ acc / K_sum

    ax.plot(c_grid, r_hat, color="royalblue", linewidth=2.0, label="Kernel estimate")

    # --- Bootstrap confidence band ---
    if band:
        idx_matrix = bootstrap_replicate_indices(n, n_bootstrap, seed=seed)
        boot_curves = np.empty((n_bootstrap, len(c_grid)))
        for b in range(n_bootstrap):
            idx = idx_matrix[b]
            c_b, a_b = conf[idx], acc[idx]
            K_b = np.exp(-((c_grid[:, None] - c_b[None, :]) ** 2) / (2.0 * h**2))
            boot_curves[b] = K_b @ a_b / K_b.sum(axis=1)
        lo = np.percentile(boot_curves, 2.5, axis=0)
        hi = np.percentile(boot_curves, 97.5, axis=0)
        ax.fill_between(c_grid, lo, hi, alpha=0.25, color="royalblue", label="95% CI band")

    # --- Binned scatter overlay ---
    bin_conf, bin_acc, bin_n = reliability_curve(y_true_np, y_prob_np, n_bins=n_bins,
                                                  binning=binning)
    mask = bin_n > 0
    ax.scatter(bin_conf[mask], bin_acc[mask], s=bin_n[mask] / n * 2000,
               color="royalblue", alpha=0.6, zorder=5, label="Binned mean")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return ax


def confidence_histogram(
    y_prob: Any,
    *,
    n_bins: int = 20,
    ax: "matplotlib.axes.Axes | None" = None,
    title: str = "Confidence Histogram",
) -> "matplotlib.axes.Axes":
    """Plot a histogram of top-label confidence scores.

    Parameters
    ----------
    y_prob : array-like
        Probability matrix or binary scores.
    n_bins : int
        Number of histogram bins.
    ax : matplotlib.axes.Axes | None
        Axes to plot on.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0, 1, 300)
    >>> ax = confidence_histogram(p)
    >>> ax is not None
    True
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required. Install with: pip install reliably[viz]"
        ) from exc

    from reliably._core.backend import to_numpy

    y_prob_np = to_numpy(y_prob, dtype=np.float64)
    conf = y_prob_np if y_prob_np.ndim == 1 else y_prob_np.max(axis=1)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.hist(conf, bins=n_bins, range=(0.0, 1.0), color="royalblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return ax
