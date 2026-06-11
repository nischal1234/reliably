"""LLM confidence evaluation: route parsed confidence into calibration machinery."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.results import Report
from reliably.llm.verbalized import parse_verbalized_batch

__all__ = ["evaluate"]


def evaluate(
    answers: list[str] | Any,
    correct: list[bool] | Any,
    confidence: list[str | float] | Any,
    *,
    kind: str = "verbalized",
    n_bins: int = 15,
    ci: str | None = "bca",
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Report:
    """Evaluate LLM calibration from answers, correctness, and confidence.

    Parameters
    ----------
    answers : list[str] | array-like
        Model answers (used for identification; not required for metric computation).
    correct : list[bool] | array-like
        Whether each answer is correct.
    confidence : list[str | float] | array-like
        Confidence values: either float probabilities or raw text strings
        (parsed when ``kind="verbalized"``).
    kind : str
        ``"verbalized"`` (parse text), ``"probability"`` (use floats directly),
        or ``"logprob"`` (exponentiate log-probs).
    n_bins : int
        Bins for calibration metrics.
    ci : str | None
        CI method.
    n_bootstrap : int
        Bootstrap resamples.
    level : float
        Nominal coverage.
    seed : int
        RNG seed.

    Returns
    -------
    Report
        Calibration report with ECE, smECE, Brier, NLL.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> correct = rng.integers(0, 2, 100).tolist()
    >>> conf = rng.uniform(0, 1, 100).tolist()
    >>> report = evaluate(["ans"] * 100, correct, conf, kind="probability", ci=None)
    >>> "smECE" in report.metrics
    True
    """
    import numpy as np

    y_true = np.asarray(correct, dtype=np.int64)

    if kind == "verbalized":
        conf_strs = [str(c) for c in confidence]
        parsed = parse_verbalized_batch(conf_strs)
        # Fill None with 0.5
        y_prob = np.array([p if p is not None else 0.5 for p in parsed], dtype=np.float64)
    elif kind == "logprob":
        from reliably.llm.adapters import logprobs_to_confidence
        y_prob = logprobs_to_confidence(np.asarray(confidence, dtype=np.float64))
    else:
        y_prob = np.asarray(confidence, dtype=np.float64)

    y_prob = np.clip(y_prob, 1e-12, 1.0 - 1e-12)

    # Evaluate using the main API
    from reliably import api

    return api.evaluate(
        y_true,
        y_prob,
        task="binary",
        metrics="default",
        binning="adaptive",
        n_bins=n_bins,
        ci=ci,
        n_bootstrap=n_bootstrap,
        level=level,
        seed=seed,
    )
