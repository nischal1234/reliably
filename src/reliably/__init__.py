"""reliably — statistically rigorous model reliability evaluation.

Every metric carries a bootstrap confidence interval.
Every comparison carries a significance test.

Quick start::

    import numpy as np
    import reliably as rb

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p = rng.uniform(0, 1, 1000)

    report = rb.evaluate(y, p)
    print(report.summary())
"""

from __future__ import annotations

# Register the recalibrate subpackage in sys.modules before binding the api function,
# so rb.recalibrate stays callable after its first invocation.
import reliably.recalibrate as _recalibrate_pkg  # noqa: F401
from reliably import llm, metrics
from reliably import repr as repr  # noqa: A001
from reliably._core.results import CI, ComparisonResult, MetricResult, Report
from reliably.api import compare, evaluate, recalibrate

__version__ = "0.1.0"

__all__ = [
    "evaluate",
    "compare",
    "recalibrate",
    "CI",
    "MetricResult",
    "Report",
    "ComparisonResult",
    "metrics",
    "repr",
    "llm",
    "__version__",
]
