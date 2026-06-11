"""Input validation helpers for the public API boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reliably._core.backend import to_numpy, validate_and_warn_probabilities

__all__ = ["prepare_inputs"]


def prepare_inputs(
    y_true: Any,
    y_prob: Any,
    *,
    task: str = "auto",
) -> tuple[NDArray[np.int64], NDArray[np.float64], str]:
    """Validate, convert, and infer task type from raw inputs.

    Parameters
    ----------
    y_true : array-like
        Integer class labels, shape ``(N,)``.
    y_prob : array-like
        Probability matrix ``(N, K)`` or binary score vector ``(N,)``.
    task : str
        ``"auto"``, ``"binary"``, or ``"multiclass"``.

    Returns
    -------
    y_true : NDArray[np.int64]
        Validated labels.
    y_prob : NDArray[np.float64]
        Validated (and possibly normalized) probabilities.
    task : str
        Resolved task string (``"binary"`` or ``"multiclass"``).

    Raises
    ------
    ValueError
        On shape mismatch, invalid labels, or non-probability inputs.

    Examples
    --------
    >>> import numpy as np
    >>> yt = np.array([0, 1, 1])
    >>> yp = np.array([0.2, 0.8, 0.6])
    >>> labels, probs, t = prepare_inputs(yt, yp)
    >>> t
    'binary'
    """
    y_true_np = to_numpy(y_true, dtype=np.float64).astype(np.int64)
    y_prob_np = to_numpy(y_prob, dtype=np.float64)

    if y_true_np.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y_true_np.shape}.")

    n = len(y_true_np)

    if y_prob_np.ndim == 1:
        if len(y_prob_np) != n:
            raise ValueError(
                f"y_true length {n} != y_prob length {len(y_prob_np)}."
            )
        resolved_task = "binary"
    elif y_prob_np.ndim == 2:
        if y_prob_np.shape[0] != n:
            raise ValueError(
                f"y_true length {n} != y_prob.shape[0] {y_prob_np.shape[0]}."
            )
        k = y_prob_np.shape[1]
        resolved_task = "binary" if k == 2 else "multiclass"
    else:
        raise ValueError(f"y_prob must be 1-D or 2-D, got shape {y_prob_np.shape}.")

    # Override inferred task if explicitly given
    if task != "auto":
        if task not in ("binary", "multiclass"):
            raise ValueError(f"task must be 'auto', 'binary', or 'multiclass'; got {task!r}.")
        resolved_task = task

    y_prob_np = validate_and_warn_probabilities(y_prob_np)

    # Validate labels
    classes = np.unique(y_true_np)
    if not np.all(classes >= 0):
        raise ValueError("Labels must be non-negative integers.")

    return y_true_np, y_prob_np, resolved_task
