# Changelog

All notable changes to `reliably` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Complete v0.1 implementation per the Technical Design Document.
- `_core/results.py`: `CI`, `MetricResult`, `Report`, `ComparisonResult` dataclasses.
- `_core/backend.py`: array-API conversion, RNG, binning utilities, numerical helpers.
- `_core/validation.py`: public API input validation and task inference.
- `stats/bootstrap.py`: vectorized nonparametric bootstrap (percentile + BCa).
- `stats/delong.py`: fast O(N log N) DeLong AUROC variance and two-sample test.
- `stats/tests.py`: paired bootstrap test, Holm–Bonferroni, Benjamini–Hochberg.
- `metrics/calibration.py`: ECE (equal-width), adaptive ECE, MCE, debiased ECE², smECE, classwise ECE.
- `metrics/scoring.py`: Brier score with Murphy decomposition, NLL.
- `metrics/discrimination.py`: AUROC with DeLong analytic CI.
- `recalibrate/`: TemperatureScaler, PlattScaler, IsotonicCalibrator, BetaCalibrator, HistogramCalibrator, VectorScaler, MatrixScaler.
- `api.py`: `evaluate`, `compare`, `recalibrate` facade functions.
- `repr/`: MIG, SAP, DCI, FactorVAE, IRS disentanglement metrics with bootstrap CIs.
- `llm/`: verbalized confidence parser, LM-Polygraph/UQLM adapters, `evaluate` function.
- `viz/diagrams.py`: reliability diagram with bootstrap confidence band.
- `report/render.py`: HTML (Jinja2) and Markdown report export.
- Full test suite: known-value, property-based (Hypothesis), coverage simulation, determinism.
- `pyproject.toml` with hatchling build backend, optional extras, ruff/mypy config.
- `.github/workflows/ci.yml` for matrix CI.

---

## [0.1.0] — TBD

First tagged release. See `CLAUDE.md §10` for the definition of done.
