"""
Tests for core/clarification.py
Verifies Expected Information Gain (EIG), context scoring, answerability, interaction cost,
and optimal question selection.
"""

import pytest
from core.clarification import Planner, Question


def test_information_gain_positive_for_ambiguous():
    """
    Information gain must be strictly positive when reducing prior distribution entropy.

    Under the NOISY ANSWER MODEL (rho_n = max(0.5, rho0 - 0.03*(n-2))):
      - A binary choose-question (n=2 options) on a 50/50 distribution gives ~0.81 bits.
      - This is LESS than the old perfect-oracle value of 1.0 bit.
      - The expected_residual_entropy is no longer 0; there remains some uncertainty
        because the user may answer incorrectly with probability (1-rho).
    The old test threshold of 0.95 assumed the perfect oracle (H(Y|X,q,a=k)=0), which
    is the flaw this module corrects.  The new threshold is 0.70 (conservative lower bound).
    """
    planner = Planner()
    candidates = [("blue", 0.50), ("black", 0.50)]
    ig, res_h = planner.compute_information_gain(candidates, ["blue", "black"])
    assert ig > 0.70, f"IG should be substantial (got {ig:.3f})"
    assert ig < 1.0, "IG must be < H(prior)=1.0 bit since user answers noisily"
    assert res_h >= 0.0, "Residual entropy must be non-negative"



def test_answerability_monotonic():
    """Binary questions must have higher answerability score than 5-option questions."""
    planner = Planner()
    ans_2 = planner.compute_answerability(2)
    ans_4 = planner.compute_answerability(4)
    ans_6 = planner.compute_answerability(6)
    assert ans_2 > ans_4 > ans_6


def test_cost_penalty_increases_with_options():
    """Questions with more options impose higher cognitive interaction cost."""
    planner = Planner()
    cost_2 = planner.compute_cost(2)
    cost_5 = planner.compute_cost(5)
    assert cost_5 > cost_2


def test_context_score_affinity():
    """Context score should increase when options match active dialogue history."""
    planner = Planner()
    context = {"tshirt": 0.85, "blue": 0.60}
    # Color options matching context
    score_with_ctx = planner.compute_context_score(["blue", "black"], context)
    score_no_ctx = planner.compute_context_score(["blue", "black"], {})
    assert score_with_ctx > score_no_ctx


def test_planner_decide_clear_commits():
    """High-confidence gesture should commit immediately without asking questions."""
    planner = Planner()
    candidates = [("shirt", 0.93), ("shoe", 0.05), ("trouser", 0.02)]
    decision = planner.decide(candidates)
    assert decision["action"] == "commit"
    assert decision["candidate"] == "shirt"
    assert decision["reason"] == "confident_clear"


def test_planner_decide_ambiguous_clarifies():
    """Ambiguous gesture should trigger clarification with selected question."""
    planner = Planner()
    candidates = [("blue", 0.49), ("black", 0.42), ("green", 0.09)]
    decision = planner.decide(candidates)
    assert decision["action"] == "clarify"
    assert decision["question"] is not None
    assert decision["question"].information_gain > 0.0
    assert len(decision["question"].options) >= 2


def test_utility_threshold_override():
    """Very high threshold forces commit even when ambiguous."""
    strict_planner = Planner(threshold=5.0)  # Impossibly high utility threshold
    candidates = [("blue", 0.49), ("black", 0.42), ("green", 0.09)]
    decision = strict_planner.decide(candidates)
    assert decision["action"] == "commit"
    assert decision["reason"] == "utility_below_threshold"
