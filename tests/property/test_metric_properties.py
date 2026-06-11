"""Property-based tests using Hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from reliably.metrics.calibration import ece
from reliably.metrics.discrimination import auroc
from reliably.metrics.scoring import brier, nll


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(50, 400), seed=st.integers(0, 9999))
def test_ece_in_unit_interval(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = rng.uniform(0, 1, n)
    r = ece(y, p, ci=None)
    assert 0.0 <= r.value <= 1.0


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(50, 400), seed=st.integers(0, 9999))
def test_ece_ci_brackets_point(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = rng.uniform(0, 1, n)
    r = ece(y, p, ci="bca", n_bootstrap=200, seed=seed)
    assert r.ci is not None
    assert r.ci.low <= r.value + 1e-9
    assert r.value <= r.ci.high + 1e-9


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(100, 500), seed=st.integers(0, 9999))
def test_auroc_in_unit_interval_and_ci_brackets(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = rng.uniform(0, 1, n)
    r = auroc(y, s, seed=seed)
    assert 0.0 <= r.value <= 1.0
    if r.ci is not None:
        assert r.ci.low <= r.value + 1e-9
        assert r.value <= r.ci.high + 1e-9


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(50, 400), seed=st.integers(0, 9999))
def test_brier_in_unit_interval(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = rng.uniform(0, 1, n)
    r = brier(y, p, ci=None)
    assert 0.0 <= r.value <= 1.0


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(50, 400), seed=st.integers(0, 9999))
def test_nll_positive(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = rng.uniform(0.05, 0.95, n)
    r = nll(y, p, ci=None)
    assert r.value > 0.0


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(100, 400), seed=st.integers(0, 9999))
def test_perfect_calibration_has_low_ece(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    r = ece(y, np.c_[1 - p, p], ci=None)
    assert 0.0 <= r.value <= 0.25


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(100, 400), seed=st.integers(0, 9999))
def test_permutation_invariance_ece(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = rng.uniform(0, 1, n)
    perm = rng.permutation(n)
    r1 = ece(y, p, ci=None)
    r2 = ece(y[perm], p[perm], ci=None)
    assert abs(r1.value - r2.value) < 1e-10


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(100, 400), seed=st.integers(0, 9999))
def test_permutation_invariance_brier(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = rng.uniform(0, 1, n)
    perm = rng.permutation(n)
    r1 = brier(y, p, ci=None)
    r2 = brier(y[perm], p[perm], ci=None)
    assert abs(r1.value - r2.value) < 1e-10
