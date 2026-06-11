"""Core result types — every public function returns these, never bare floats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


@dataclass(frozen=True, slots=True)
class CI:
    """Confidence interval.

    Parameters
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound.
    level : float
        Nominal coverage, default 0.95.
    method : str
        One of ``"bca"``, ``"percentile"``, ``"analytic"``.

    Examples
    --------
    >>> ci = CI(low=0.1, high=0.3)
    >>> ci.low, ci.high
    (0.1, 0.3)
    """

    low: float
    high: float
    level: float = 0.95
    method: Literal["percentile", "bca", "analytic"] = "bca"


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Metric point estimate with optional confidence interval.

    Parameters
    ----------
    name : str
        Human-readable metric name, e.g. ``"smECE"``.
    value : float
        Point estimate.
    ci : CI | None
        Confidence interval; ``None`` only when CI computation is disabled.
    n : int
        Sample size on which the metric was computed.
    extra : Mapping[str, float] | None
        Optional extra scalars, e.g. Brier decomposition components.

    Examples
    --------
    >>> mr = MetricResult(name="ECE", value=0.05, ci=CI(0.03, 0.07), n=100)
    >>> "ECE" in str(mr)
    True
    """

    name: str
    value: float
    ci: CI | None
    n: int
    extra: Mapping[str, float] | None = None

    def __str__(self) -> str:
        c = f" [{self.ci.low:.4f}, {self.ci.high:.4f}]" if self.ci else ""
        return f"{self.name}={self.value:.4f}{c}"


@dataclass(frozen=True, slots=True)
class Report:
    """Immutable result of :func:`reliably.evaluate`.

    Parameters
    ----------
    task : str
        One of ``"binary"``, ``"multiclass"``.
    metrics : Mapping[str, MetricResult]
        All computed metrics, keyed by name.
    n : int
        Dataset size.
    meta : Mapping[str, object]
        Provenance: seed, n_bootstrap, binning, etc.

    Examples
    --------
    >>> r = Report(task="binary", metrics={}, n=100, meta={})
    >>> r.task
    'binary'
    """

    task: Literal["binary", "multiclass"]
    metrics: Mapping[str, MetricResult]
    n: int
    meta: Mapping[str, object]

    def __getitem__(self, name: str) -> MetricResult:
        return self.metrics[name]

    def summary(self) -> str:
        lines = [f"Report(task={self.task}, n={self.n})"]
        for mr in self.metrics.values():
            lines.append(f"  {mr}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Result of :func:`reliably.compare`.

    Parameters
    ----------
    metric : str
        Name of the compared metric.
    delta : float
        Point estimate of the difference (value_a - value_b).
    ci : CI
        Confidence interval on the difference.
    p_value : float
        Two-sided p-value.
    test : str
        Test used: ``"delong"`` or ``"paired_bootstrap"``.
    significant : bool
        ``True`` if ``p_value < 1 - ci.level`` after correction.
    correction : str | None
        Multiple-comparison correction applied (e.g. ``"holm"``).

    Examples
    --------
    >>> cr = ComparisonResult(
    ...     metric="auroc", delta=0.02, ci=CI(-0.01, 0.05),
    ...     p_value=0.19, test="delong", significant=False, correction="holm"
    ... )
    >>> cr.significant
    False
    """

    metric: str
    delta: float
    ci: CI
    p_value: float
    test: Literal["delong", "paired_bootstrap"]
    significant: bool
    correction: str | None
