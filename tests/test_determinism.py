"""Determinism tests: same seed + same inputs → identical bytes."""

from __future__ import annotations

import numpy as np
import pytest

import reliably as rb


def _make_inputs(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = rng.uniform(0, 1, n)
    return y, p


class TestDeterminism:
    def test_evaluate_deterministic(self) -> None:
        y, p = _make_inputs()
        r1 = rb.evaluate(y, p, n_bootstrap=100, seed=42)
        r2 = rb.evaluate(y, p, n_bootstrap=100, seed=42)
        for name in r1.metrics:
            mr1 = r1.metrics[name]
            mr2 = r2.metrics[name]
            assert mr1.value == mr2.value, f"{name} values differ"
            if mr1.ci is not None and mr2.ci is not None:
                assert mr1.ci.low == mr2.ci.low, f"{name} CI.low differs"
                assert mr1.ci.high == mr2.ci.high, f"{name} CI.high differs"

    def test_different_seeds_may_differ(self) -> None:
        y, p = _make_inputs()
        r1 = rb.evaluate(y, p, n_bootstrap=100, seed=1)
        r2 = rb.evaluate(y, p, n_bootstrap=100, seed=2)
        # Point estimates must be identical (deterministic)
        for name in r1.metrics:
            mr1 = r1.metrics[name]
            mr2 = r2.metrics[name]
            assert mr1.value == mr2.value, f"{name} point estimate differs between seeds"
        # But at least one CI should differ (due to different bootstrap draws)
        diffs = []
        for name in r1.metrics:
            mr1, mr2 = r1.metrics[name], r2.metrics[name]
            if mr1.ci is not None and mr2.ci is not None:
                diffs.append(mr1.ci.low != mr2.ci.low or mr1.ci.high != mr2.ci.high)
        assert any(diffs), "All CIs are identical for different seeds (unexpected)"

    def test_bootstrap_ci_deterministic(self) -> None:
        from reliably.stats.bootstrap import vectorized_bootstrap_ci

        data = np.random.default_rng(0).normal(0, 1, 300)
        ci1 = vectorized_bootstrap_ci(data, point=data.mean(), seed=99, n_boot=500)
        ci2 = vectorized_bootstrap_ci(data, point=data.mean(), seed=99, n_boot=500)
        assert ci1.low == ci2.low
        assert ci1.high == ci2.high

    def test_recalibrate_deterministic(self) -> None:
        y, p = _make_inputs()
        cal1 = rb.recalibrate(p, y, method="temperature")
        cal2 = rb.recalibrate(p, y, method="temperature")
        assert cal1.T_ == cal2.T_

    def test_smece_deterministic(self) -> None:
        from reliably.metrics.calibration import smece

        y, p = _make_inputs()
        r1 = smece(y, p, ci="bca", n_bootstrap=100, seed=7)
        r2 = smece(y, p, ci="bca", n_bootstrap=100, seed=7)
        assert r1.value == r2.value
        assert r1.ci is not None and r2.ci is not None
        assert r1.ci.low == r2.ci.low
        assert r1.ci.high == r2.ci.high
