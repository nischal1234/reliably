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

# Pre-import the recalibrate subpackage so it is registered in sys.modules BEFORE
# we bind the `recalibrate` name to the api function.  Once a package is in
# sys.modules, subsequent `from reliably.recalibrate.xxx import ...` calls inside
# api.recalibrate() will NOT reset the parent-package attribute — so
# `rb.recalibrate` stays callable even after the first invocation.
import reliably.recalibrate as _recalibrate_pkg  # noqa: F401,E402

from reliably.api import compare, evaluate  # noqa: E402
from reliably.api import recalibrate  # noqa: E402  — overrides subpkg attr above
from reliably._core.results import CI, ComparisonResult, MetricResult, Report  # noqa: E402
from reliably import metrics  # noqa: E402
from reliably import repr as repr  # noqa: E402,A001
from reliably import llm  # noqa: E402

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
