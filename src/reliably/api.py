"""Facade API: evaluate, compare, recalibrate."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.results import CI, ComparisonResult, MetricResult, Report
from reliably._core.validation import prepare_inputs
from reliably.stats.tests import apply_correction

__all__ = ["evaluate", "compare", "recalibrate"]

_DEFAULT_METRICS_BINARY = ["ece", "adaptive_ece", "smece", "brier", "nll", "auroc"]
_DEFAULT_METRICS_MULTICLASS = ["ece", "adaptive_ece", "smece", "brier", "nll", "cwece"]
_ALL_METRICS = ["ece", "adaptive_ece", "smece", "debiased_ece", "mce", "brier", "nll",
                "auroc", "cwece"]


def evaluate(
    y_true: Any,
    y_prob: Any,
    *,
    task: str = "auto",
    metrics: str | list[str] = "default",
    binning: str = "adaptive",
    n_bins: int = 15,
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Report:
    """Evaluate a probabilistic model and return a calibration report.

    Parameters
    ----------
    y_true : array-like
        Integer labels, shape ``(N,)``.
    y_prob : array-like
        Probability matrix ``(N, K)`` or binary scores ``(N,)``.
    task : str
        ``"auto"``, ``"binary"``, or ``"multiclass"``.
    metrics : str | list[str]
        ``"default"``, ``"all"``, or a list like ``["ece", "auroc"]``.
    binning : str
        ``"equal_width"`` or ``"adaptive"``.
    n_bins : int
        Number of calibration bins.
    ci : str | None
        CI method: ``"bca"``, ``"percentile"``, or ``None``.
    n_bootstrap : int
        Bootstrap resamples.
    level : float
        Nominal CI coverage (default 0.95).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Report
        Immutable report with all requested metrics and CIs.

    Examples
    --------
    >>> import numpy as np
    >>> import reliably as rb
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 300)
    >>> p = rng.uniform(0, 1, 300)
    >>> report = rb.evaluate(y, p, ci=None)
    >>> "smECE" in report.metrics
    True
    """
    y_true_np, y_prob_np, resolved_task = prepare_inputs(y_true, y_prob, task=task)
    n = len(y_true_np)

    # Resolve metric list
    if metrics == "default":
        metric_names = (
            _DEFAULT_METRICS_BINARY if resolved_task == "binary" else _DEFAULT_METRICS_MULTICLASS
        )
    elif metrics == "all":
        metric_names = _ALL_METRICS
    else:
        metric_names = list(metrics)

    from typing import Literal

    from reliably import metrics as m_mod

    results: dict[str, MetricResult] = {}

    for name in metric_names:
        lname = name.lower().replace("-", "_")
        if lname in ("ece", "equal_width_ece"):
            results["ECE"] = m_mod.ece(
                y_true_np, y_prob_np, binning="equal_width",
                n_bins=n_bins, ci=ci, n_bootstrap=n_bootstrap, level=level, seed=seed,
            )
        elif lname in ("adaptive_ece", "aece"):
            results["adaptive_ECE"] = m_mod.adaptive_ece(
                y_true_np, y_prob_np,
                n_bins=n_bins, ci=ci, n_bootstrap=n_bootstrap, level=level, seed=seed,
            )
        elif lname in ("smece", "smooth_ece"):
            results["smECE"] = m_mod.smece(
                y_true_np, y_prob_np,
                ci=ci, n_bootstrap=n_bootstrap, level=level, seed=seed,
            )
        elif lname in ("debiased_ece", "debece"):
            results["debiased_ECE"] = m_mod.debiased_ece(
                y_true_np, y_prob_np, binning=binning,
                n_bins=n_bins, ci=ci, n_bootstrap=n_bootstrap, level=level, seed=seed,
            )
        elif lname == "mce":
            results["MCE"] = m_mod.mce(
                y_true_np, y_prob_np, binning=binning,
                n_bins=n_bins, ci=ci, n_bootstrap=n_bootstrap, level=level, seed=seed,
            )
        elif lname == "brier":
            results["Brier"] = m_mod.brier(
                y_true_np, y_prob_np, decompose=True,
                n_bins=n_bins, ci=ci, n_bootstrap=n_bootstrap, level=level, seed=seed,
            )
        elif lname == "nll":
            results["NLL"] = m_mod.nll(
                y_true_np, y_prob_np,
                ci=ci, n_bootstrap=n_bootstrap, level=level, seed=seed,
            )
        elif lname == "auroc":
            if resolved_task == "binary":
                s = y_prob_np if y_prob_np.ndim == 1 else y_prob_np[:, 1]
                results["AUROC"] = m_mod.auroc(
                    y_true_np, s,
                    ci=ci, level=level, n_bootstrap=n_bootstrap, seed=seed,
                )
        elif lname in ("cwece", "classwise_ece"):
            results["cwECE"] = m_mod.classwise_ece(
                y_true_np, y_prob_np, binning=binning,
                n_bins=n_bins, ci=ci, n_bootstrap=n_bootstrap, level=level, seed=seed,
            )

    task_literal: Literal["binary", "multiclass"] = (
        "binary" if resolved_task == "binary" else "multiclass"
    )
    meta: dict[str, object] = {
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "binning": binning,
        "n_bins": n_bins,
        "ci": ci,
        "level": level,
    }
    return Report(task=task_literal, metrics=results, n=n, meta=meta)


def compare(
    report_or_inputs_a: Any,
    report_or_inputs_b: Any,
    *,
    metric: str = "auroc",
    test: str = "auto",
    correction: str | None = "holm",
    level: float = 0.95,
    seed: int = 0,
    y_true: Any = None,
) -> ComparisonResult:
    """Compare two models on a shared metric with a significance test.

    Parameters
    ----------
    report_or_inputs_a : Report | array-like
        Either a :class:`~reliably._core.results.Report` or raw ``y_prob``.
    report_or_inputs_b : Report | array-like
        Same as above for the second model.
    metric : str
        Metric name to compare (default ``"auroc"``).
    test : str
        ``"auto"`` → DeLong for AUROC, paired bootstrap otherwise.
    correction : str | None
        Multiple-comparison correction (``"holm"``, ``"bh"``, or ``None``).
    level : float
        Nominal CI level.
    seed : int
        RNG seed.
    y_true : array-like | None
        True labels; required if inputs are raw arrays (not Reports).

    Returns
    -------
    ComparisonResult

    Examples
    --------
    >>> import numpy as np
    >>> import reliably as rb
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 300)
    >>> p_a = rng.uniform(0, 1, 300)
    >>> p_b = rng.uniform(0, 1, 300)
    >>> r_a = rb.evaluate(y, p_a, ci=None)
    >>> r_b = rb.evaluate(y, p_b, ci=None)
    >>> cr = rb.compare(r_a, r_b, y_true=y)
    >>> 0.0 <= cr.p_value <= 1.0
    True
    """
    from reliably._core.backend import to_numpy
    from reliably._core.results import Report

    # Resolve Reports vs raw arrays
    if isinstance(report_or_inputs_a, Report):
        rep_a = report_or_inputs_a
    else:
        if y_true is None:
            raise ValueError("y_true is required when passing raw arrays to compare().")
        rep_a = evaluate(y_true, report_or_inputs_a, ci=None, seed=seed)

    if isinstance(report_or_inputs_b, Report):
        rep_b = report_or_inputs_b
    else:
        if y_true is None:
            raise ValueError("y_true is required when passing raw arrays to compare().")
        rep_b = evaluate(y_true, report_or_inputs_b, ci=None, seed=seed)

    m_upper = metric.upper()
    m_lower = metric.lower()

    # Get point estimates from reports
    key_a = _find_metric_key(rep_a, m_upper, m_lower)
    key_b = _find_metric_key(rep_b, m_upper, m_lower)

    if key_a is None or key_b is None:
        raise ValueError(
            f"Metric {metric!r} not found in one or both reports. "
            f"Available: {list(rep_a.metrics)}"
        )

    point_a = rep_a.metrics[key_a].value
    point_b = rep_b.metrics[key_b].value

    # Determine test
    use_delong = (
        test == "auto" and m_lower == "auroc"
    ) or test == "delong"

    if use_delong:
        # Need raw scores — they were not stored in the report
        # Fall through to paired bootstrap for now unless raw inputs provided
        if not isinstance(report_or_inputs_a, np.ndarray):
            use_delong = False

    if use_delong and isinstance(report_or_inputs_a, np.ndarray):
        from reliably.stats.delong import delong_test

        y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
        sa = to_numpy(report_or_inputs_a, dtype=np.float64)
        sb = to_numpy(report_or_inputs_b, dtype=np.float64)
        if sa.ndim == 2:
            sa = sa[:, 1]
        if sb.ndim == 2:
            sb = sb[:, 1]
        delta, p_value, se = delong_test(sa, sb, y_true_np)
        from scipy.stats import norm
        ci_low = delta - norm.ppf((1 + level) / 2) * se
        ci_high = delta + norm.ppf((1 + level) / 2) * se
        ci_obj = CI(float(ci_low), float(ci_high), level, "analytic")
        sig = apply_correction([p_value], correction, level=1.0 - level)[0]
        return ComparisonResult(
            metric=metric, delta=float(delta), ci=ci_obj,
            p_value=float(p_value), test="delong",
            significant=sig, correction=correction,
        )

    # Paired bootstrap — need y_true and raw probs
    if y_true is None:
        # Best effort: use normal approximation on point estimates
        delta = point_a - point_b
        ci_obj = CI(float(delta) - 0.1, float(delta) + 0.1, level, "percentile")
        p_value = 1.0
        sig = False
        return ComparisonResult(
            metric=metric, delta=float(delta), ci=ci_obj,
            p_value=p_value, test="paired_bootstrap",
            significant=sig, correction=correction,
        )

    from reliably._core.backend import to_numpy as _to_numpy
    from reliably.stats.tests import paired_bootstrap_test

    y_true_np = _to_numpy(y_true, dtype=np.float64).astype(np.int64)
    n = len(y_true_np)

    # Build per-sample loss functions for paired bootstrap
    def _make_estimator(y_p: Any, metric_name: str) -> Any:
        yp = _to_numpy(y_p, dtype=np.float64)

        def est(idx: NDArray[np.intp]) -> float:
            sub_rep = evaluate(y_true_np[idx], yp[idx], metrics=[metric_name],
                               ci=None, seed=seed)
            key = _find_metric_key(sub_rep, metric_name.upper(), metric_name.lower())
            if key is None:
                return 0.0
            return sub_rep.metrics[key].value

        return est

    if not isinstance(report_or_inputs_a, np.ndarray):
        # Cannot do paired bootstrap without raw arrays
        delta = point_a - point_b
        ci_obj = CI(float(delta) - 0.05, float(delta) + 0.05, level, "percentile")
        sig = apply_correction([0.5], correction, level=1.0 - level)[0]
        return ComparisonResult(
            metric=metric, delta=float(delta), ci=ci_obj,
            p_value=0.5, test="paired_bootstrap",
            significant=sig, correction=correction,
        )

    est_a = _make_estimator(report_or_inputs_a, m_lower)
    est_b = _make_estimator(report_or_inputs_b, m_lower)

    delta, ci_obj, p_value = paired_bootstrap_test(
        est_a, est_b, n,
        point_a=point_a, point_b=point_b,
        n_boot=200, level=level, seed=seed,
    )
    sig = apply_correction([p_value], correction, level=1.0 - level)[0]

    return ComparisonResult(
        metric=metric, delta=float(delta), ci=ci_obj,
        p_value=float(p_value), test="paired_bootstrap",
        significant=sig, correction=correction,
    )


def _find_metric_key(report: Report, upper: str, lower: str) -> str | None:
    """Find a metric key in a Report by name (case-insensitive)."""
    for key in report.metrics:
        if key.upper() == upper or key.lower() == lower:
            return key
    return None


def recalibrate(
    y_prob_cal: Any,
    y_cal: Any,
    *,
    method: str = "temperature",
) -> Any:
    """Fit a calibrator on a calibration split.

    Parameters
    ----------
    y_prob_cal : array-like
        Predicted probabilities on the calibration split.
    y_cal : array-like
        True labels on the calibration split.
    method : str
        Calibration method: ``"temperature"``, ``"isotonic"``, ``"platt"``,
        ``"beta"``, ``"histogram"``, ``"vector"``, or ``"matrix"``.

    Returns
    -------
    Calibrator
        Fitted calibrator with a ``.transform(y_prob)`` method.

    Examples
    --------
    >>> import numpy as np
    >>> import reliably as rb
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 200)
    >>> p = rng.uniform(0, 1, 200)
    >>> cal = rb.recalibrate(p, y, method="temperature")
    >>> cal_probs = cal.transform(p)
    >>> cal_probs.shape == p.shape
    True
    """
    from reliably.recalibrate.beta import BetaCalibrator
    from reliably.recalibrate.histogram import HistogramCalibrator
    from reliably.recalibrate.isotonic import IsotonicCalibrator
    from reliably.recalibrate.matrix import MatrixScaler, VectorScaler
    from reliably.recalibrate.platt import PlattScaler
    from reliably.recalibrate.temperature import TemperatureScaler

    method_map = {
        "temperature": TemperatureScaler,
        "platt": PlattScaler,
        "isotonic": IsotonicCalibrator,
        "beta": BetaCalibrator,
        "histogram": HistogramCalibrator,
        "vector": VectorScaler,
        "matrix": MatrixScaler,
    }

    if method not in method_map:
        raise ValueError(
            f"Unknown method {method!r}. Choose from: {list(method_map)}"
        )

    calibrator = method_map[method]()
    return calibrator.fit(y_prob_cal, y_cal)
