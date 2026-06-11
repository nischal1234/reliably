"""Known-value and property tests for calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest

import reliably as rb
from reliably.metrics.calibration import (
    adaptive_ece,
    classwise_ece,
    debiased_ece,
    ece,
    mce,
    reliability_curve,
    smece,
)


class TestECEKnownValues:
    """Hand-computed ECE fixtures."""

    def test_perfect_calibration_ece_near_zero(self) -> None:
        """A perfectly calibrated model has ECE ≈ 0."""
        rng = np.random.default_rng(42)
        n = 2000
        conf = rng.uniform(0, 1, n)
        # Calibrated by construction: y ~ Bernoulli(conf)
        y = (rng.uniform(0, 1, n) < conf).astype(int)
        r = ece(y, conf, n_bins=15, ci=None)
        assert r.value < 0.05, f"ECE={r.value:.4f} too large for perfect calibration"

    def test_overconfident_model_ece_positive(self) -> None:
        """An always-confident model has ECE > 0."""
        n = 500
        y = np.zeros(n, dtype=int)
        y[:250] = 1
        p = np.full(n, 0.9)
        r = ece(y, p, ci=None)
        assert r.value > 0.3

    def test_ece_unit_interval(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300)
        p = rng.uniform(0, 1, 300)
        r = ece(y, p, ci=None)
        assert 0.0 <= r.value <= 1.0

    def test_ece_binary_2d_input(self) -> None:
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 200)
        p1 = rng.uniform(0, 1, 200)
        p2d = np.stack([1 - p1, p1], axis=1)
        r1 = ece(y, p1, ci=None)
        r2 = ece(y, p2d, ci=None)
        assert abs(r1.value - r2.value) < 1e-10

    def test_ece_ci_brackets_point_estimate(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 300)
        p = rng.uniform(0, 1, 300)
        r = ece(y, p, ci="bca", n_bootstrap=200, seed=0)
        assert r.ci is not None
        assert r.ci.low <= r.value <= r.ci.high, (
            f"CI [{r.ci.low:.4f}, {r.ci.high:.4f}] does not bracket {r.value:.4f}"
        )

    def test_ece_metric_result_fields(self) -> None:
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, 100)
        p = rng.uniform(0, 1, 100)
        r = ece(y, p, ci=None)
        assert r.name == "ECE"
        assert r.n == 100
        assert r.ci is None

    def test_ece_small_dataset(self) -> None:
        y = np.array([0, 1, 1])
        p = np.array([0.2, 0.8, 0.9])
        r = ece(y, p, n_bins=3, ci=None)
        assert 0.0 <= r.value <= 1.0


class TestAdaptiveECE:
    def test_adaptive_ece_valid_range(self) -> None:
        rng = np.random.default_rng(10)
        y = rng.integers(0, 2, 300)
        p = rng.uniform(0, 1, 300)
        r = adaptive_ece(y, p, ci=None)
        assert 0.0 <= r.value <= 1.0

    def test_adaptive_ece_name(self) -> None:
        y = np.array([0, 1])
        p = np.array([0.3, 0.7])
        r = adaptive_ece(y, p, ci=None)
        assert r.name == "adaptive_ECE"


class TestMCE:
    def test_mce_ge_ece(self) -> None:
        rng = np.random.default_rng(20)
        y = rng.integers(0, 2, 300)
        p = rng.uniform(0, 1, 300)
        r_ece = ece(y, p, ci=None)
        r_mce = mce(y, p, ci=None)
        assert r_mce.value >= r_ece.value - 1e-9

    def test_mce_unit_interval(self) -> None:
        rng = np.random.default_rng(21)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        r = mce(y, p, ci=None)
        assert 0.0 <= r.value <= 1.0


class TestDebiasedECE:
    def test_debiased_ece_nonnegative(self) -> None:
        rng = np.random.default_rng(30)
        y = rng.integers(0, 2, 300)
        p = rng.uniform(0, 1, 300)
        r = debiased_ece(y, p, ci=None)
        assert r.value >= 0.0

    def test_debiased_le_unbiased_plus_noise(self) -> None:
        rng = np.random.default_rng(31)
        n = 1000
        conf = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < conf).astype(int)
        r_ece = ece(y, conf, ci=None)
        r_deb = debiased_ece(y, conf, ci=None)
        # Debiased should be smaller (less biased upward)
        assert r_deb.value <= r_ece.value + 0.05


class TestSmECE:
    def test_smece_valid_range(self) -> None:
        rng = np.random.default_rng(40)
        y = rng.integers(0, 2, 300)
        p = rng.uniform(0, 1, 300)
        r = smece(y, p, ci=None)
        assert 0.0 <= r.value <= 1.0

    def test_smece_perfect_low(self) -> None:
        rng = np.random.default_rng(41)
        n = 1000
        conf = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < conf).astype(int)
        r = smece(y, conf, ci=None)
        assert r.value < 0.1

    def test_smece_ci_brackets(self) -> None:
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        r = smece(y, p, ci="bca", n_bootstrap=100, seed=0)
        assert r.ci is not None
        assert r.ci.low <= r.value <= r.ci.high


class TestClasswiseECE:
    def test_cwece_multiclass(self) -> None:
        rng = np.random.default_rng(50)
        y = rng.integers(0, 3, 300)
        p = rng.dirichlet([1, 1, 1], 300)
        r = classwise_ece(y, p, ci=None)
        assert 0.0 <= r.value <= 1.0

    def test_cwece_name(self) -> None:
        rng = np.random.default_rng(51)
        y = rng.integers(0, 3, 100)
        p = rng.dirichlet([1, 1, 1], 100)
        r = classwise_ece(y, p, ci=None)
        assert r.name == "cwECE"


class TestReliabilityCurve:
    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(60)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        bc, ba, bn = reliability_curve(y, p, n_bins=15)
        assert len(bc) == len(ba) == len(bn) == 15

    def test_bin_counts_sum_to_n(self) -> None:
        rng = np.random.default_rng(61)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        _, _, bn = reliability_curve(y, p, n_bins=15)
        assert bn.sum() == 200
