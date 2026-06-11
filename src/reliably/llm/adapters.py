"""Thin adapters that consume LM-Polygraph/UQLM outputs and route to reliably."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["from_lm_polygraph", "from_uqlm", "logprobs_to_confidence"]


def logprobs_to_confidence(
    logprobs: Any,
    *,
    length_normalize: bool = True,
) -> NDArray[np.float64]:
    """Convert sequence log-probabilities to a confidence score in ``[0, 1]``.

    Parameters
    ----------
    logprobs : array-like
        Per-token log-probabilities. Can be:
        - 1-D array for a single sequence.
        - List of 1-D arrays for a batch (variable length).
    length_normalize : bool
        If ``True``, divide the sum by the sequence length.

    Returns
    -------
    NDArray[np.float64]
        Confidence scores in ``[0, 1]``, shape ``(N,)`` for a batch.

    Examples
    --------
    >>> import numpy as np
    >>> lp = np.array([-0.1, -0.2, -0.3])
    >>> logprobs_to_confidence(lp)
    array([0.81873075])
    """
    import numpy as np

    if isinstance(logprobs, np.ndarray) and logprobs.ndim == 1:
        logprobs = [logprobs]

    scores = []
    for seq in logprobs:
        seq_arr = np.asarray(seq, dtype=np.float64)
        if length_normalize and len(seq_arr) > 0:
            mean_lp = seq_arr.mean()
        else:
            mean_lp = seq_arr.sum()
        scores.append(float(np.exp(mean_lp)))
    return np.array(scores, dtype=np.float64)


def from_lm_polygraph(result: Any) -> dict[str, NDArray[np.float64]]:
    """Extract confidence scores from an LM-Polygraph result object.

    This is a thin adapter — it does not import LM-Polygraph directly, but
    rather reads the output dict/object that the user already has.

    Parameters
    ----------
    result : dict or object
        LM-Polygraph ``UQResult`` or plain dict with keys like
        ``"confidence"``, ``"answers"``, etc.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        Dict with ``"confidence"`` key (and optionally ``"answers"``).

    Examples
    --------
    >>> from_lm_polygraph({"confidence": [0.8, 0.6]})
    {'confidence': array([0.8, 0.6])}
    """
    if isinstance(result, dict):
        out: dict[str, NDArray[np.float64]] = {}
        if "confidence" in result:
            out["confidence"] = np.asarray(result["confidence"], dtype=np.float64)
        if "scores" in result:
            out["confidence"] = np.asarray(result["scores"], dtype=np.float64)
        return out

    # Try attribute access for UQResult
    out = {}
    for attr in ("confidence", "scores", "uncertainty"):
        if hasattr(result, attr):
            out["confidence"] = np.asarray(getattr(result, attr), dtype=np.float64)
            break
    return out


def from_uqlm(result: Any) -> dict[str, NDArray[np.float64]]:
    """Extract confidence scores from a UQLM result object.

    Parameters
    ----------
    result : dict or object
        UQLM result dict or object.

    Returns
    -------
    dict[str, NDArray[np.float64]]

    Examples
    --------
    >>> from_uqlm({"probabilities": [0.9, 0.7]})
    {'confidence': array([0.9, 0.7])}
    """
    if isinstance(result, dict):
        for key in ("probabilities", "confidence", "scores"):
            if key in result:
                return {"confidence": np.asarray(result[key], dtype=np.float64)}

    # Fallback: try attribute access
    out: dict[str, NDArray[np.float64]] = {}
    for attr in ("probabilities", "confidence", "scores"):
        if hasattr(result, attr):
            out["confidence"] = np.asarray(getattr(result, attr), dtype=np.float64)
            break
    return out
