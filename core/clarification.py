"""
ClarifySign - Clarification Engine (backward-compatible adapter over core/infogain.py)
========================================================================================

All IG mathematics lives in core/infogain.py.  This module provides:
1. A backward-compatible Planner that:
   - accepts the old candidates-as-list-of-tuples interface
   - returns the old dict-based decision format
   - exposes the legacy helper methods (compute_information_gain, compute_context_score,
     compute_answerability, compute_cost) expected by existing tests
2. A backward-compatible Question dataclass with the fields the rest of the code reads
   (information_gain, options as label list, etc.)
3. Re-exports of infogain primitives so downstream imports keep working.

KEY DIFFERENCE FROM THE PREVIOUS IMPLEMENTATION:
  OLD: IG assumed P(answer_k | y=k) = 1.0 (perfect oracle) → H(Y|X,q,a=k) = 0
       IG = sum of probabilities of queried candidates (not a real mutual information)
  NEW: IG = I(Y;A) under an explicit noisy answer model where the user answers
       correctly with probability rho_n and makes noise with probability 1-rho_n.
       This means IG is always strictly lower than the prior entropy, and a
       confirm-question on a very unlikely candidate gives little IG even if that
       candidate has moderate probability.

See core/infogain.py for the full mathematical specification and all assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

import core.infogain as _ig
from core.infogain import (
    entropy_bits,
    top2_margin,
    normalized_entropy,
    fuse_context,
    apply_temperature,
    fit_temperature,
    ece,
    normalize,
    expected_information_gain,
    update_posterior,
    sample_answer_k,
    make_question,
    reliability,
)
from config import (
    CONFIDENCE_THRESHOLD,
    MARGIN_THRESHOLD,
    ENTROPY_THRESHOLD,
    UTILITY_THRESHOLD,
    ALPHA_CONTEXT,
    BETA_ANSWERABILITY,
    LAMBDA_COST,
    RHO0,
)


# ---------------------------------------------------------------------------
# Backward-compatible Question dataclass
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """
    Clarification question with full diagnostic fields.
    Wraps core.infogain.Question but exposes the legacy attribute names
    expected by tests and app.py (information_gain, options as label-strings,
    candidate_keys, question_type).
    """
    text: str
    options: list           # display strings shown to user
    candidate_keys: list    # label strings for the candidate classes
    information_gain: float
    expected_residual_entropy: float
    context_score: float
    answerability: float
    cost: float
    utility: float
    question_type: str = "candidate_choice"
    # New field: the backing infogain Question (for Bayesian updates)
    _ig_question: Optional[_ig.Question] = None


# ---------------------------------------------------------------------------
# Helpers to convert old-style inputs to arrays
# ---------------------------------------------------------------------------

def _extract_labels_probs(candidates) -> tuple[list, np.ndarray]:
    """candidates is list of (label, prob) tuples.  Returns (labels, probs_array)."""
    labels = [c[0] for c in candidates]
    probs = np.array([float(c[1]) for c in candidates], dtype=float)
    s = probs.sum()
    if s > 0:
        probs /= s
    return labels, probs


def _context_dict_to_prior(context: dict, labels: list) -> Optional[np.ndarray]:
    """Convert a {label: score} context dict to a per-class prior array."""
    if not context:
        return None
    prior = np.array([context.get(l, 0.0) for l in labels], dtype=float)
    return prior if prior.sum() > 0 else None


# ---------------------------------------------------------------------------
# Main Planner
# ---------------------------------------------------------------------------

class Planner:
    """
    Backward-compatible clarification decision planner.

    Accepts the old candidates interface (list of (label, prob) tuples) and
    returns the old dict-based decision format, while internally using the
    corrected infogain.py mathematics.
    """

    def __init__(
        self,
        threshold: float = UTILITY_THRESHOLD,   # alias for u_min (backward compat)
        alpha: float = ALPHA_CONTEXT,
        beta: float = BETA_ANSWERABILITY,
        lambda_cost: float = LAMBDA_COST,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        margin_threshold: float = MARGIN_THRESHOLD,
        entropy_threshold: float = ENTROPY_THRESHOLD,
        rho0: float = RHO0,
    ):
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.lambda_cost = lambda_cost
        self.conf_threshold = conf_threshold
        self.margin_threshold = margin_threshold
        self.entropy_threshold = entropy_threshold
        self.rho0 = rho0
        # Internal infogain planner
        self._planner = _ig.Planner(
            alpha=alpha,
            beta=beta,
            lam=lambda_cost,
            u_min=threshold,
            conf_thr=conf_threshold,
            margin_thr=margin_threshold,
            ent_thr=entropy_threshold,
            rho0=rho0,
        )

    # -----------------------------------------------------------------------
    # Legacy helper methods (used by tests and the old evaluation code)
    # -----------------------------------------------------------------------

    def compute_information_gain(self, candidates, question_candidate_keys: list) -> tuple:
        """
        Compute IG using the noisy-answer model for a CHOOSE question over
        question_candidate_keys, given the full candidates distribution.

        Returns (ig, expected_residual_entropy).

        NOTE: the previous implementation assumed P(a=k|y=k) = 1 (perfect oracle),
        which made H(Y|X,q,a=k) = 0.  The new implementation uses the noisy model
        (rho_n = max(0.5, rho0 - 0.03*(n-2))) so IG is strictly less than the old value.
        This is the mathematically correct mutual information.
        """
        if not candidates:
            return 0.0, 0.0
        labels, probs = _extract_labels_probs(candidates)
        # Map string keys to integer indices
        idx = [labels.index(k) for k in question_candidate_keys if k in labels]
        if not idx:
            return 0.0, 0.0
        kind = "confirm" if len(idx) == 1 else "choose"
        q = _ig.make_question(kind, idx, self.rho0)
        result = _ig.expected_information_gain(probs, q)
        return float(result["ig"]), float(result["h_post"])

    def compute_answerability(self, num_options: int) -> float:
        """Cognitive ease of answering: fewer choices → higher score in [0,1]."""
        return float(max(0.10, 1.0 - 0.12 * max(0, num_options - 2)))

    def compute_cost(self, num_options: int) -> float:
        """Interaction cost: more options → higher cost.  Uses lambda_cost weighting."""
        return float(self.lambda_cost * (1.0 + 0.10 * max(0, num_options - 2)))

    def compute_context_score(self, candidate_keys: list, context: dict) -> float:
        """
        Context relevance score in [0,1] based on active dialogue state.
        """
        if not context or not candidate_keys:
            return 0.0
        context_scores = []
        for k in candidate_keys:
            score = float(context.get(k, 0.0))
            # Domain coherence bonuses
            if any(c in context for c in ["shirt", "tshirt", "shoes", "clothes"]):
                if k in ["blue", "black", "red", "white", "small", "smalllittle", "biglarge", "large"]:
                    score = max(score, 0.45)
            if any(c in context for c in ["bank", "storeorshop", "cellphone"]):
                if k in ["thankyou", "pen", "bank"]:
                    score = max(score, 0.40)
            context_scores.append(score)
        return float(np.mean(context_scores)) if context_scores else 0.0

    # -----------------------------------------------------------------------
    # Question generation (returns backward-compatible Question objects)
    # -----------------------------------------------------------------------

    def generate_candidate_questions(self, candidates, context=None) -> list:
        """
        Generate Question objects for all candidate question types.
        Returns list of backward-compatible Question dataclass instances.
        """
        if not candidates:
            return []
        labels, probs = _extract_labels_probs(candidates)
        k = len(labels)
        questions = []
        c_keys = labels  # ordered by the input (already sorted by probability)

        # Q1: Binary Disambiguation (top-2)
        if k >= 2:
            top2_keys = c_keys[:2]
            ig, res_h = self.compute_information_gain(candidates, top2_keys)
            ctx = self.compute_context_score(top2_keys, context or {})
            options = [f"1. {top2_keys[0].capitalize()}", f"2. {top2_keys[1].capitalize()}", "3. Neither / Other"]
            ans = self.compute_answerability(len(options))
            cost = self.compute_cost(len(options))
            u = ig + self.alpha * ctx + self.beta * ans - cost
            idx = [c_keys.index(k_) for k_ in top2_keys if k_ in c_keys]
            q_ig = _ig.make_question("choose", idx, self.rho0)
            questions.append(Question(
                text=f"Did you intend '{top2_keys[0]}' or '{top2_keys[1]}'?",
                options=options,
                candidate_keys=top2_keys,
                information_gain=ig,
                expected_residual_entropy=res_h,
                context_score=ctx,
                answerability=ans,
                cost=cost,
                utility=u,
                question_type="binary_disambiguation",
                _ig_question=q_ig,
            ))

        # Q2: Confirmation on top-1 (Yes/No)
        top1_key = [c_keys[0]]
        ig_conf, res_h_conf = self.compute_information_gain(candidates, top1_key)
        ctx_conf = self.compute_context_score(top1_key, context or {})
        options_conf = [f"Yes, '{c_keys[0]}'", "No, something else"]
        ans_conf = self.compute_answerability(2)
        cost_conf = self.compute_cost(2)
        u_conf = ig_conf + self.alpha * ctx_conf + self.beta * ans_conf - cost_conf
        idx_conf = [c_keys.index(c_keys[0])]
        q_ig_conf = _ig.make_question("confirm", idx_conf, self.rho0)
        questions.append(Question(
            text=f"Did you mean '{c_keys[0]}'?",
            options=options_conf,
            candidate_keys=top1_key,
            information_gain=ig_conf,
            expected_residual_entropy=res_h_conf,
            context_score=ctx_conf,
            answerability=ans_conf,
            cost=cost_conf,
            utility=u_conf,
            question_type="single_confirmation",
            _ig_question=q_ig_conf,
        ))

        # Q3: Triplet choice (top-3)
        if k >= 3:
            top3_keys = c_keys[:3]
            ig_3, res_h_3 = self.compute_information_gain(candidates, top3_keys)
            ctx_3 = self.compute_context_score(top3_keys, context or {})
            options_3 = [f"{i+1}. {k_.capitalize()}" for i, k_ in enumerate(top3_keys)] + [f"{len(top3_keys)+1}. None of these"]
            ans_3 = self.compute_answerability(len(options_3))
            cost_3 = self.compute_cost(len(options_3))
            u_3 = ig_3 + self.alpha * ctx_3 + self.beta * ans_3 - cost_3
            idx_3 = [c_keys.index(k_) for k_ in top3_keys if k_ in c_keys]
            q_ig_3 = _ig.make_question("choose", idx_3, self.rho0)
            questions.append(Question(
                text="Which sign did you mean among these?",
                options=options_3,
                candidate_keys=top3_keys,
                information_gain=ig_3,
                expected_residual_entropy=res_h_3,
                context_score=ctx_3,
                answerability=ans_3,
                cost=cost_3,
                utility=u_3,
                question_type="triplet_choice",
                _ig_question=q_ig_3,
            ))

        return questions

    def plan(self, candidates, context=None) -> Optional[tuple]:
        """Select q* = argmax_q U(q).  Returns (best_q, all_questions) or None."""
        questions = self.generate_candidate_questions(candidates, context)
        if not questions:
            return None
        best_q = max(questions, key=lambda q: q.utility)
        return best_q, questions

    # -----------------------------------------------------------------------
    # Main decision method — backward-compatible dict output
    # -----------------------------------------------------------------------

    def decide(self, candidates, context=None) -> dict:
        """
        Main clarification decision:
          1. Evaluate ambiguity triggers (confidence, margin, entropy).
          2. If unambiguous → COMMIT top-1 candidate.
          3. If ambiguous:
             - Optimize question utility q* = argmax U(q).
             - If U(q*) >= threshold → CLARIFY using q*.
             - Else → COMMIT with uncertainty caveat.

        Returns a dict with keys:
            action       "commit" | "clarify" | "none"
            candidate    top-1 label string
            reason       human-readable reason string
            question     Question dataclass (or None)
            all_questions list of all evaluated Questions
            confidence   top-1 probability
            triggers     list of triggered ambiguity criteria strings
        """
        if not candidates:
            return {
                "action": "none",
                "candidate": None,
                "reason": "empty_candidates",
                "question": None,
                "all_questions": [],
            }

        from core.uncertainty import is_ambiguous as _is_ambiguous
        is_ambig, primary_reason, triggers = _is_ambiguous(
            candidates,
            conf_threshold=self.conf_threshold,
            margin_threshold=self.margin_threshold,
            entropy_threshold=self.entropy_threshold,
        )

        top_cand = candidates[0][0]
        top_prob = float(candidates[0][1])

        if not is_ambig:
            return {
                "action": "commit",
                "candidate": top_cand,
                "confidence": top_prob,
                "reason": "confident_clear",
                "triggers": triggers,
                "question": None,
                "all_questions": [],
            }

        plan_result = self.plan(candidates, context)
        if plan_result is None:
            return {
                "action": "commit",
                "candidate": top_cand,
                "confidence": top_prob,
                "reason": "no_questions_generated",
                "triggers": triggers,
                "question": None,
                "all_questions": [],
            }

        best_q, all_questions = plan_result

        if best_q.utility >= self.threshold:
            return {
                "action": "clarify",
                "candidate": top_cand,
                "confidence": top_prob,
                "reason": f"ambiguous_{primary_reason}",
                "triggers": triggers,
                "question": best_q,
                "all_questions": all_questions,
            }
        else:
            return {
                "action": "commit",
                "candidate": top_cand,
                "confidence": top_prob,
                "reason": "utility_below_threshold",
                "triggers": triggers,
                "question": best_q,
                "all_questions": all_questions,
            }


__all__ = [
    "Planner",
    "Question",
    "entropy_bits",
    "top2_margin",
    "normalized_entropy",
    "fuse_context",
    "apply_temperature",
    "fit_temperature",
    "ece",
    "normalize",
    "expected_information_gain",
    "update_posterior",
    "sample_answer_k",
    "make_question",
    "reliability",
]
