"""Parse verbalized confidence strings to float probabilities."""

from __future__ import annotations

import re

__all__ = ["parse_verbalized_confidence"]

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DECIMAL_RE = re.compile(r"\b(0\.\d+|1\.0+)\b")
_WORD_MAP: dict[str, float] = {
    "certain": 0.99,
    "sure": 0.90,
    "confident": 0.85,
    "probably": 0.75,
    "likely": 0.70,
    "maybe": 0.50,
    "possibly": 0.45,
    "uncertain": 0.30,
    "unsure": 0.25,
    "unlikely": 0.20,
    "doubtful": 0.15,
}


def parse_verbalized_confidence(text: str) -> float | None:
    """Parse a verbalized confidence string to a probability in ``[0, 1]``.

    Handles:
    - Percentage strings: ``"I'm 80% confident"`` → ``0.80``
    - Decimal fractions: ``"confidence 0.75"`` → ``0.75``
    - Word qualifiers: ``"probably correct"`` → ``0.75``

    Parameters
    ----------
    text : str
        Raw text from a language model response.

    Returns
    -------
    float | None
        Parsed probability or ``None`` if no confidence expression found.

    Examples
    --------
    >>> parse_verbalized_confidence("I am 85% sure.")
    0.85
    >>> parse_verbalized_confidence("probably correct")
    0.75
    >>> parse_verbalized_confidence("no confidence here") is None
    True
    """
    # Try percentage first
    m = _PERCENT_RE.search(text)
    if m:
        val = float(m.group(1)) / 100.0
        return float(min(max(val, 0.0), 1.0))

    # Try decimal
    m = _DECIMAL_RE.search(text)
    if m:
        return float(m.group(1))

    # Try word map
    lower = text.lower()
    for word, prob in sorted(_WORD_MAP.items(), key=lambda kv: -len(kv[0])):
        if word in lower:
            return prob

    return None


def parse_verbalized_batch(texts: list[str]) -> list[float | None]:
    """Parse a list of verbalized confidence strings.

    Parameters
    ----------
    texts : list[str]
        List of raw text outputs.

    Returns
    -------
    list[float | None]
        Parsed probabilities; ``None`` where parsing failed.

    Examples
    --------
    >>> parse_verbalized_batch(["90%", "maybe", "unknown"])
    [0.9, 0.5, None]
    """
    return [parse_verbalized_confidence(t) for t in texts]
