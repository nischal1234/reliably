"""Recalibration methods: temperature, platt, isotonic, beta, histogram, matrix."""

from __future__ import annotations

from reliably.recalibrate.base import Calibrator
from reliably.recalibrate.beta import BetaCalibrator
from reliably.recalibrate.histogram import HistogramCalibrator
from reliably.recalibrate.isotonic import IsotonicCalibrator
from reliably.recalibrate.matrix import MatrixScaler, VectorScaler
from reliably.recalibrate.platt import PlattScaler
from reliably.recalibrate.temperature import TemperatureScaler

__all__ = [
    "Calibrator",
    "TemperatureScaler",
    "PlattScaler",
    "IsotonicCalibrator",
    "BetaCalibrator",
    "HistogramCalibrator",
    "VectorScaler",
    "MatrixScaler",
]
