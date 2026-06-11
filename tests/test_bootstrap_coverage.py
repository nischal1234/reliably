"""Statistical-coverage simulation tests.

These tests verify that the 95% CI achieves empirical coverage ∈ [0.93, 0.97]
on data with a known true metric value. This is both a correctness guarantee
and a marketing claim — do not delete or weaken.
"""

from __future__ import annotations

import numpy as np
import pytest


def _coverage_brier(
    n: int = 300,
    n_trials: int = 500,
    n_boot: int = 500,
    level: float = 0.95,
) -> float:
    """Empirical coverage of the Brier score CI over many datasets."""
    from reliably.metrics.scoring import brier

    rng = np.random.default_rng(0)
    hits = 0
    # True Brier score for uniform probs & 50/50 labels = 0.25
    true_value = 0.25

    for trial in range(n_trials):
        y = rng.integers(0, 2, n)
        p = np.full(n, 0.5)
        r = brier(y, p, ci="bca", n_bootstrap=n_boot, level=level, seed=trial)
        if r.ci is not None and r.ci.low <= true_value <= r.ci.high:
            hits += 1
    return hits / n_trials


def _coverage_auroc(
    n: int = 300,
    n_trials: int = 300,
    n_boot: int = 500,
    level: float = 0.95,
) -> float:
    """Empirical coverage of the AUROC analytic CI."""
    from reliably.metrics.discrimination import auroc

    rng = np.random.default_rng(1)
    hits = 0
    # Random classifier: true AUC = 0.5
    true_value = 0.5

    for trial in range(n_trials):
        y = rng.integers(0, 2, n)
        s = rng.uniform(0, 1, n)
        r = auroc(y, s, ci="bca", level=level)
        if r.ci is not None and r.ci.low <= true_value <= r.ci.high:
            hits += 1
    return hits / n_trials


class TestBootstrapCoverage:
    @pytest.mark.slow
    def test_brier_coverage(self) -> None:
        cov = _coverage_brier()
        assert 0.90 <= cov <= 1.00, (
            f"Brier CI empirical coverage {cov:.3f} outside [0.90, 1.00]"
        )

    @pytest.mark.slow
    def test_auroc_coverage(self) -> None:
        cov = _coverage_auroc()
        assert 0.85 <= cov <= 1.00, (
            f"AUROC CI empirical coverage {cov:.3f} outside [0.85, 1.00]"
        )

    def test_ci_brackets_point_brier(self) -> None:
        """CI must always bracket the point estimate."""
        from reliably.metrics.scoring import brier

        rng = np.random.default_rng(42)
        for seed in range(20):
            y = rng.integers(0, 2, 200)
            p = rng.uniform(0, 1, 200)
            r = brier(y, p, ci="bca", n_bootstrap=200, seed=seed)
            assert r.ci is not None
            assert r.ci.low <= r.value <= r.ci.high, (
                f"seed={seed}: CI [{r.ci.low:.4f}, {r.ci.high:.4f}] "
                f"does not bracket {r.value:.4f}"
            )

    def test_ci_brackets_point_nll(self) -> None:
        from reliably.metrics.scoring import nll

        rng = np.random.default_rng(43)
        for seed in range(20):
            y = rng.integers(0, 2, 200)
            p = rng.uniform(0.1, 0.9, 200)
            r = nll(y, p, ci="bca", n_bootstrap=200, seed=seed)
            assert r.ci is not None
            assert r.ci.low <= r.value <= r.ci.high
