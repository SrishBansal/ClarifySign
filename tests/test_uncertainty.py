"""
Tests for core/uncertainty.py
Verifies Shannon entropy, margin, normalized entropy, ambiguity detection, and reporting.
"""

import math
import pytest
from core.uncertainty import (
    entropy,
    margin,
    normalized_entropy,
    top1_confidence,
    is_ambiguous,
    report
)


def test_entropy_deterministic():
    """Deterministic prediction should have exactly 0 entropy."""
    candidates = [("shirt", 1.0)]
    assert entropy(candidates) == pytest.approx(0.0, abs=1e-5)


def test_entropy_binary_uniform():
    """Binary 50-50 uniform distribution should have exactly 1 bit of entropy."""
    candidates = [("blue", 0.5), ("black", 0.5)]
    assert entropy(candidates, base=2.0) == pytest.approx(1.0, abs=1e-5)


def test_entropy_4_way_uniform():
    """4-way uniform distribution should have log2(4) = 2.0 bits of entropy."""
    candidates = [("a", 0.25), ("b", 0.25), ("c", 0.25), ("d", 0.25)]
    assert entropy(candidates, base=2.0) == pytest.approx(2.0, abs=1e-5)


def test_margin_computation():
    """Margin should correctly measure difference between top-1 and top-2."""
    candidates = [("blue", 0.65), ("black", 0.25), ("green", 0.10)]
    assert margin(candidates) == pytest.approx(0.40, abs=1e-5)


def test_margin_single_candidate():
    """Single candidate has a default margin of 1.0."""
    assert margin([("shirt", 1.0)]) == 1.0


def test_normalized_entropy_range():
    """Normalized entropy must strictly lie in [0, 1]."""
    c1 = [("a", 0.25), ("b", 0.25), ("c", 0.25), ("d", 0.25)]
    assert normalized_entropy(c1) == pytest.approx(1.0, abs=1e-4)

    c2 = [("a", 0.99), ("b", 0.01)]
    assert 0.0 <= normalized_entropy(c2) <= 1.0


def test_top1_confidence():
    candidates = [("bank", 0.72), ("storeorshop", 0.18)]
    assert top1_confidence(candidates) == pytest.approx(0.72, abs=1e-5)


def test_is_ambiguous_confident():
    """High confidence and wide margin should be classified as clear/unambiguous."""
    candidates = [("shoes", 0.94), ("hat", 0.04), ("tshirt", 0.02)]
    ambig, reason, triggers = is_ambiguous(candidates, conf_threshold=0.78, margin_threshold=0.20, entropy_threshold=1.25)
    assert not ambig
    assert reason == "confident_clear"
    assert len(triggers) == 0


def test_is_ambiguous_competing_margin():
    """Narrow margin should trigger ambiguity detection."""
    candidates = [("blue", 0.49), ("black", 0.42), ("red", 0.09)]
    ambig, reason, triggers = is_ambiguous(candidates, conf_threshold=0.78, margin_threshold=0.20, entropy_threshold=1.25)
    assert ambig
    assert "narrow_margin" in reason or "low_top1_confidence" in reason
    assert len(triggers) > 0


def test_report_structure():
    candidates = [("hello", 0.95), ("thankyou", 0.05)]
    rep = report(candidates)
    assert "entropy_bits" in rep
    assert "margin" in rep
    assert "top1_confidence" in rep
    assert "is_ambiguous" in rep
    assert "decision" in rep
    assert rep["decision"] == "commit"
