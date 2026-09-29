"""
ClarifySign - Uncertainty Estimation (wrapper around core/infogain.py)
=======================================================================

This module is a THIN WRAPPER that re-exposes uncertainty metrics in the
interface expected by the rest of the codebase (candidates as list-of-tuples or
list-of-floats) while delegating all computation to core.infogain primitives.

Functions exported:
    entropy(candidates, base=2.0)     -> float  (Shannon entropy)
    normalized_entropy(candidates)    -> float  (in [0,1])
    margin(candidates)                -> float  (top-1 minus top-2)
    top1_confidence(candidates)       -> float
    is_ambiguous(candidates, ...)     -> (bool, str, list[str])
    report(candidates, ...)           -> dict
"""

import numpy as np
from core.infogain import (
    entropy_bits,
    normalized_entropy as _normalized_entropy_arr,
    top2_margin,
    normalize,
)
from config import (
    CONFIDENCE_THRESHOLD,
    MARGIN_THRESHOLD,
    ENTROPY_THRESHOLD,
)


def _to_probs(candidates) -> np.ndarray:
    """Accept list-of-(key, prob) tuples OR list-of-floats; return normalised ndarray."""
    if not candidates:
        return np.array([])
    if isinstance(candidates[0], (tuple, list)):
        raw = np.array([float(c[1]) for c in candidates], dtype=np.float64)
    else:
        raw = np.array([float(c) for c in candidates], dtype=np.float64)
    total = raw.sum()
    return raw / total if total > 0 else raw


def entropy(candidates, base: float = 2.0) -> float:
    """
    Shannon entropy H(Y|X) in the specified base (default: bits).
    candidates: list of (class_name, probability) tuples or list of floats.
    """
    p = _to_probs(candidates)
    if len(p) == 0:
        return 0.0
    # entropy_bits always uses base 2; convert for other bases
    import math
    h = entropy_bits(p)
    if base != 2.0:
        h = h / math.log2(base)  # convert bits -> nats or other
    return float(max(0.0, h))


def normalized_entropy(candidates) -> float:
    """
    Normalised entropy H_norm in [0,1] relative to the uniform distribution over K classes.
    H_norm = H(Y|X) / log2(K)
    """
    p = _to_probs(candidates)
    if len(p) <= 1:
        return 0.0
    return float(min(1.0, max(0.0, _normalized_entropy_arr(p))))


def margin(candidates) -> float:
    """
    Top-2 margin Delta = p_(1) - p_(2).
    A small margin indicates high competition between top interpretations.
    """
    p = _to_probs(candidates)
    if len(p) == 0:
        return 0.0
    if len(p) == 1:
        return 1.0
    return float(top2_margin(p))


def top1_confidence(candidates) -> float:
    """
    Returns the top-1 raw probability score (not normalized).
    This preserves the original value from the candidates list.
    """
    if not candidates:
        return 0.0
    if isinstance(candidates[0], (tuple, list)):
        return float(candidates[0][1])
    return float(max(candidates))


def is_ambiguous(
    candidates,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
) -> tuple:
    """
    Multi-criteria ambiguity detector combining confidence, margin, and entropy.
    Returns:
        (is_ambig: bool, primary_reason: str, triggered_criteria: list[str])
    """
    if not candidates:
        return True, "no_candidates", ["no_candidates"]

    top_p = top1_confidence(candidates)
    m = margin(candidates)
    h = entropy(candidates, base=2.0)

    triggers = []
    if top_p < conf_threshold:
        triggers.append(f"low_top1_confidence ({top_p:.2f} < {conf_threshold:.2f})")
    if m < margin_threshold:
        triggers.append(f"narrow_margin ({m:.2f} < {margin_threshold:.2f})")
    if h > entropy_threshold:
        triggers.append(f"high_entropy ({h:.2f} > {entropy_threshold:.2f} bits)")

    is_ambig = len(triggers) > 0
    if not is_ambig:
        primary_reason = "confident_clear"
    elif m < margin_threshold:
        primary_reason = "narrow_margin"
    elif h > entropy_threshold:
        primary_reason = "high_entropy"
    else:
        primary_reason = "low_top1_confidence"

    return is_ambig, primary_reason, triggers


def report(
    candidates,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
) -> dict:
    """
    Generates a full diagnostic dictionary of uncertainty metrics.
    """
    top_p = top1_confidence(candidates)
    m = margin(candidates)
    ent_bits = entropy(candidates, base=2.0)
    norm_ent = normalized_entropy(candidates)
    ambig, primary_reason, triggers = is_ambiguous(
        candidates,
        conf_threshold=conf_threshold,
        margin_threshold=margin_threshold,
        entropy_threshold=entropy_threshold,
    )

    return {
        "top1_confidence": top_p,
        "margin": m,
        "entropy_bits": ent_bits,
        "normalized_entropy": norm_ent,
        "is_ambiguous": ambig,
        "primary_reason": primary_reason,
        "triggered_criteria": triggers,
        "num_candidates": len(candidates),
        "decision": "clarify" if ambig else "commit",
    }
