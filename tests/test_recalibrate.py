"""Tests for recalibration methods."""

from __future__ import annotations

import numpy as np
import pytest

from reliably.recalibrate.beta import BetaCalibrator
from reliably.recalibrate.histogram import HistogramCalibrator
from reliably.recalibrate.matrix import MatrixScaler, VectorScaler
from reliably.recalibrate.platt import PlattScaler
from reliably.recalibrate.temperature import TemperatureScaler


def _make_binary(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(rng.normal(0.5, 0.2, n), 0.01, 0.99)
    return y, p


def _make_multiclass(
    n: int = 300, k: int = 3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    p = rng.dirichlet([1] * k, n)
    return y, p


class TestTemperatureScaler:
    def test_fit_returns_self(self) -> None:
        y, p = _make_binary()
        cal = TemperatureScaler()
        result = cal.fit(p, y)
        assert result is cal

    def test_temperature_positive(self) -> None:
        y, p = _make_binary()
        cal = TemperatureScaler().fit(p, y)
        assert cal.T_ > 0

    def test_output_sums_to_one_multiclass(self) -> None:
        y, p = _make_multiclass()
        cal = TemperatureScaler().fit(p, y)
        cal_p = cal.transform(p)
        assert np.allclose(cal_p.sum(axis=1), 1.0, atol=1e-6)

    def test_output_shape_binary(self) -> None:
        y, p = _make_binary()
        cal = TemperatureScaler().fit(p, y)
        cal_p = cal.transform(p)
        assert cal_p.shape == p.shape

    def test_transform_before_fit_raises(self) -> None:
        cal = TemperatureScaler()
        with pytest.raises(RuntimeError):
            cal.transform(np.array([0.5]))

    def test_argmax_preserved(self) -> None:
        y, p = _make_multiclass()
        cal = TemperatureScaler().fit(p, y)
        cal_p = cal.transform(p)
        assert np.all(p.argmax(axis=1) == cal_p.argmax(axis=1))


class TestPlattScaler:
    def test_output_shape(self) -> None:
        y, p = _make_binary()
        cal = PlattScaler().fit(p, y)
        cal_p = cal.transform(p)
        assert cal_p.shape == p.shape

    def test_output_range(self) -> None:
        y, p = _make_binary()
        cal = PlattScaler().fit(p, y)
        cal_p = cal.transform(p)
        assert np.all((cal_p >= 0) & (cal_p <= 1))

    def test_has_fitted_params(self) -> None:
        y, p = _make_binary()
        cal = PlattScaler().fit(p, y)
        assert hasattr(cal, "A_") and hasattr(cal, "B_")


class TestBetaCalibrator:
    def test_output_range(self) -> None:
        y, p = _make_binary()
        cal = BetaCalibrator().fit(p, y)
        cal_p = cal.transform(p)
        assert np.all((cal_p >= 0) & (cal_p <= 1))

    def test_output_shape(self) -> None:
        y, p = _make_binary()
        cal = BetaCalibrator().fit(p, y)
        cal_p = cal.transform(p)
        assert cal_p.shape == p.shape

    def test_constrained_ab_nonnegative(self) -> None:
        y, p = _make_binary()
        cal = BetaCalibrator(constrain_ab=True).fit(p, y)
        assert cal.a_ >= 0 and cal.b_ >= 0


class TestHistogramCalibrator:
    def test_output_range(self) -> None:
        y, p = _make_binary()
        cal = HistogramCalibrator().fit(p, y)
        cal_p = cal.transform(p)
        assert np.all((cal_p >= 0) & (cal_p <= 1))

    def test_output_shape(self) -> None:
        y, p = _make_binary()
        cal = HistogramCalibrator().fit(p, y)
        cal_p = cal.transform(p)
        assert cal_p.shape == p.shape


class TestVectorScaler:
    def test_output_sums_to_one(self) -> None:
        y, p = _make_multiclass()
        cal = VectorScaler().fit(p, y)
        cal_p = cal.transform(p)
        assert np.allclose(cal_p.sum(axis=1), 1.0, atol=1e-6)

    def test_output_shape(self) -> None:
        y, p = _make_multiclass()
        cal = VectorScaler().fit(p, y)
        cal_p = cal.transform(p)
        assert cal_p.shape == p.shape


class TestMatrixScaler:
    def test_output_sums_to_one(self) -> None:
        y, p = _make_multiclass()
        cal = MatrixScaler().fit(p, y)
        cal_p = cal.transform(p)
        assert np.allclose(cal_p.sum(axis=1), 1.0, atol=1e-6)

    def test_output_shape(self) -> None:
        y, p = _make_multiclass()
        cal = MatrixScaler().fit(p, y)
        cal_p = cal.transform(p)
        assert cal_p.shape == p.shape


class TestIsotonicCalibrator:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sklearn"),
        reason="scikit-learn not installed",
    )
    def test_output_range(self) -> None:
        from reliably.recalibrate.isotonic import IsotonicCalibrator

        y, p = _make_binary()
        cal = IsotonicCalibrator().fit(p, y)
        cal_p = cal.transform(p)
        assert np.all((cal_p >= 0) & (cal_p <= 1))


class TestRecalibrateAPI:
    def test_temperature_via_api(self) -> None:
        import reliably as rb

        y, p = _make_binary()
        cal = rb.recalibrate(p, y, method="temperature")
        cal_p = cal.transform(p)
        assert cal_p.shape == p.shape

    def test_unknown_method_raises(self) -> None:
        import reliably as rb

        y, p = _make_binary()
        with pytest.raises(ValueError):
            rb.recalibrate(p, y, method="nonexistent")
