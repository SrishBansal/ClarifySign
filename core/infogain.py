"""
infogain.py — corrected expected-information-gain clarification planner
=========================================================================

Why this exists
---------------
The original core/clarification.py computed IG by assuming that when the user
confirms a queried candidate c, the interpretation is resolved to c and
H(Y|X, q, a=c) = 0.  That treats the user as a perfect oracle, so IG collapses
to the probability mass on the queried candidates and cannot discriminate between
question quality or candidate spread.

This module implements the formulation stated in the project:

    H(Y|X)  = - sum_y p(y|x) log2 p(y|x)
    IG(q)   = H(Y|X) - sum_a P(a|q,X) H(Y|X,q,a)         (= mutual information I(Y;A))
    U(q)    = IG(q) + alpha*Context(q) + beta*Answerability(q) - lambda*Cost(q)
    q*      = argmax_q U(q)

with an explicit, documented NOISY ANSWER MODEL so P(a|q,y) is a real likelihood:

  CONFIRM(c)  "Did you mean c?"      answers {yes, no}
      P(yes|y=c) = rho          P(yes|y!=c) = 1-rho
  CHOOSE(S)   "Which of S?"         answers S + {none}
      y in S : picks y w.p. rho, otherwise uniformly among the other options
      y notin S: says none w.p. rho, otherwise uniformly among S
  rho depends on option count:  rho_n = max(0.5, rho_0 - 0.03*(n-2))

Bounded / documented assumptions (do NOT describe as more than they are)
------------------------------------------------------------------------
* One-step (myopic) lookahead: IG is computed for the next question only.
* The answer model is a modelling assumption. rho_0 is NOT measured from real
  users.  Evaluation code sweeps rho_true independently of rho_0 to test
  robustness under model misspecification.
* Context enters in two places: (1) Bayesian fusion into the posterior
  p(y|x,ctx) ~ p(y|x) * prior_ctx(y)^gamma, and (2) a Context(q) utility term
  rewarding questions whose options are plausible under the context prior.
* IG (bits) and the [0,1] utility terms are added with hand-set weights.
  These are hyper-parameters, not derived quantities; tuned on validation split only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# Basic probability utilities
# ---------------------------------------------------------------------------

def normalize(p: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))


def entropy_bits(p: Sequence[float]) -> float:
    """Shannon entropy in bits.  Zeros are ignored."""
    p = np.asarray(p, dtype=float)
    p = p[p > EPS]
    return float(-(p * np.log2(p)).sum())


def normalized_entropy(p: Sequence[float]) -> float:
    k = len(p)
    return entropy_bits(p) / np.log2(k) if k > 1 else 0.0


def top2_margin(p: Sequence[float]) -> float:
    s = np.sort(np.asarray(p, dtype=float))[::-1]
    return float(s[0] - s[1]) if len(s) > 1 else float(s[0])


def fuse_context(
    p: Sequence[float],
    prior: Optional[Sequence[float]],
    gamma: float = 1.0
) -> np.ndarray:
    """Bayesian fusion  p(y|x,ctx) ~ p(y|x) * prior(y)^gamma.  gamma=0 disables context."""
    p = normalize(p)
    if prior is None or gamma == 0.0:
        return p
    prior = normalize(np.asarray(prior, dtype=float) + EPS)
    return normalize(p * np.power(prior, gamma))


# ---------------------------------------------------------------------------
# Calibration utilities
# ---------------------------------------------------------------------------

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error on top-1 confidence."""
    probs = np.asarray(probs)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total, n = 0.0, len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """softmax(log p / T).  Equivalent to scaling the original logits by 1/T."""
    z = np.log(np.clip(probs, EPS, 1.0)) / T
    z -= z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    lo: float = 0.05,
    hi: float = 10.0
) -> float:
    """
    Fit temperature T by minimising NLL on a VALIDATION set (never the test set).
    Returns the optimal temperature scalar.
    """
    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        q = apply_temperature(probs, T)
        return -np.log(np.clip(q[np.arange(len(labels)), labels], EPS, 1.0)).mean()

    result = minimize_scalar(nll, bounds=(lo, hi), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Question types and noisy answer model
# ---------------------------------------------------------------------------

@dataclass
class Question:
    kind: str                  # "confirm" | "choose"
    options: tuple             # class indices shown to the user
    rho: float                 # assumed answer reliability for this question
    cost: float                # normalised interaction cost in [0,1]
    answerability: float       # perceived ease in [0,1]

    @property
    def n_options(self) -> int:
        return len(self.options)

    def answers(self) -> list:
        """Answer labels.  confirm -> ['yes','no']; choose -> options + ['none']."""
        return ["yes", "no"] if self.kind == "confirm" else list(self.options) + ["none"]

    def likelihood(self, k: int, rho: Optional[float] = None) -> np.ndarray:
        """
        Matrix L[a, y] = P(answer=a | true_class=y).  Columns sum to 1.

        confirm(c):
            a=yes: P(yes|y=c) = rho;   P(yes|y!=c) = 1-rho
            a=no:  complement
        choose(opts):
            y in opts: P(y|y) = rho; rest distributed uniformly over the other options
            y notin opts: P(none|y) = rho; rest distributed uniformly over listed opts
        """
        r = self.rho if rho is None else rho

        if self.kind == "confirm":
            c = self.options[0]
            L = np.zeros((2, k))
            L[0, :] = 1.0 - r        # P(yes | y!=c) = noise
            L[0, c] = r              # P(yes | y=c)
            L[1, :] = 1.0 - L[0, :]
            return L

        # choose
        opts = list(self.options)
        n = len(opts)
        L = np.zeros((n + 1, k))
        for y in range(k):
            if y in opts:
                L[:, y] = (1.0 - r) / n
                L[opts.index(y), y] = r
            else:
                L[:, y] = (1.0 - r) / n
                L[n, y] = r    # "none" answer index
        return L


def reliability(n_options: int, rho0: float) -> float:
    """Degrade reliability for more options: rho_n = max(0.5, rho0 - 0.03*(n-2))."""
    return float(max(0.5, rho0 - 0.03 * max(0, n_options - 2)))


def make_question(kind: str, options: Sequence[int], rho0: float) -> Question:
    n = len(options) if kind == "choose" else 2
    cost = (0.5 + 0.15 * n) / (0.5 + 0.15 * 4)
    ans = float(np.clip(1.0 - 0.12 * (n - 2), 0.0, 1.0))
    return Question(
        kind=kind,
        options=tuple(int(o) for o in options),
        rho=reliability(n, rho0),
        cost=float(cost),
        answerability=ans
    )


# ---------------------------------------------------------------------------
# Core information-gain calculation (exact under noisy answer model)
# ---------------------------------------------------------------------------

def expected_information_gain(
    p: np.ndarray,
    q: Question,
    rho: Optional[float] = None
) -> dict:
    """
    Exact IG = I(Y;A) under the noisy answer model.

    Returns dict with keys:
        ig                       -- information gain in bits (>= 0)
        h_prior                  -- H(Y|X) before the question
        h_post                   -- expected H(Y|X,a) after the question
        p_answer                 -- P(a) for each answer
        expected_accuracy_after  -- E[max_y P(y|X,a)] across answers
    """
    p = normalize(p)
    L = q.likelihood(len(p), rho)   # (A, K)
    joint = L * p[None, :]          # P(a, y)
    p_a = joint.sum(1)              # P(a)
    h_prior = entropy_bits(p)
    h_post = 0.0
    for a_idx in range(L.shape[0]):
        if p_a[a_idx] > EPS:
            posterior = joint[a_idx] / p_a[a_idx]
            h_post += p_a[a_idx] * entropy_bits(posterior)
    exp_acc = float(joint.max(1).sum())
    return {
        "ig": max(0.0, h_prior - h_post),
        "h_prior": h_prior,
        "h_post": h_post,
        "p_answer": p_a,
        "expected_accuracy_after": exp_acc
    }


def update_posterior(
    p: np.ndarray,
    q: Question,
    answer_index: int,
    rho: Optional[float] = None
) -> np.ndarray:
    """
    Bayesian posterior update after receiving an answer.
    A noisy answer does NOT hard-set the label; it updates via the likelihood.
    """
    L = q.likelihood(len(p), rho)
    return normalize(np.asarray(p) * L[answer_index])


def sample_answer_k(
    q: Question,
    y_true: int,
    k: int,
    rng: np.random.Generator,
    rho_true: Optional[float] = None
) -> int:
    """
    Sample a simulated user answer for a given true class y_true.
    rho_true is the ACTUAL user reliability (may differ from q.rho used by the planner).
    """
    col = q.likelihood(k, rho_true)[:, y_true]
    return int(rng.choice(len(col), p=col / col.sum()))


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    action: str                                      # "commit" | "clarify"
    label: int                                       # argmax of the (context-fused) posterior
    question: Optional[Question] = None
    ambiguous: bool = False
    posterior: np.ndarray = field(default_factory=lambda: np.zeros(0))
    table: list = field(default_factory=list)        # per-question diagnostics (for the inspector)
    reason: str = ""


class Planner:
    """
    One-step information-theoretic clarification planner.

    Parameters (all are modelling assumptions / hyper-parameters):
        alpha    -- weight on Context(q) in the utility function
        beta     -- weight on Answerability(q)
        lam      -- weight on Cost(q) (penalty)
        u_min    -- minimum utility for a clarification to be worth asking
        conf_thr -- top-1 probability threshold below which the sign is flagged ambiguous
        margin_thr -- top-2 margin threshold
        ent_thr  -- entropy threshold
        rho0     -- assumed answer reliability (MODELLING ASSUMPTION, not measured)
        max_options -- maximum number of options in a choose question
        gamma    -- context fusion strength (0 = no context)
    """

    def __init__(
        self,
        alpha: float = 0.30,
        beta: float = 0.15,
        lam: float = 0.18,
        u_min: float = 0.05,
        conf_thr: float = 0.78,
        margin_thr: float = 0.20,
        ent_thr: float = 1.25,
        rho0: float = 0.95,
        max_options: int = 3,
        gamma: float = 1.0
    ):
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.u_min = u_min
        self.conf_thr = conf_thr
        self.margin_thr = margin_thr
        self.ent_thr = ent_thr
        self.rho0 = rho0
        self.max_options = max_options
        self.gamma = gamma

    def is_ambiguous(self, p: np.ndarray) -> bool:
        return bool(
            p.max() < self.conf_thr
            or top2_margin(p) < self.margin_thr
            or entropy_bits(p) > self.ent_thr
        )

    def candidate_questions(self, p: np.ndarray) -> list:
        order = np.argsort(p)[::-1]
        qs = [make_question("confirm", [order[0]], self.rho0)]
        for n in range(2, self.max_options + 1):
            if len(p) >= n:
                qs.append(make_question("choose", order[:n], self.rho0))
        return qs

    def score(
        self,
        p: np.ndarray,
        q: Question,
        prior: Optional[np.ndarray]
    ) -> dict:
        info = expected_information_gain(p, q)
        ctx_prior = normalize(
            np.ones(len(p)) if prior is None
            else np.asarray(prior, float) + EPS
        )
        ctx = float(sum(ctx_prior[o] for o in q.options))
        u = (info["ig"]
             + self.alpha * ctx
             + self.beta * q.answerability
             - self.lam * q.cost)
        return {
            "question": q,
            "ig": info["ig"],
            "context": ctx,
            "answerability": q.answerability,
            "cost": q.cost,
            "utility": float(u),
            "expected_accuracy_after": info["expected_accuracy_after"]
        }

    def decide(
        self,
        p_raw: Sequence[float],
        prior: Optional[Sequence[float]] = None
    ) -> Decision:
        """
        Main decision entry point.

        p_raw  -- raw class probabilities (need not sum to 1; will be normalised)
        prior  -- optional context prior; if provided, posterior is Bayesian-fused

        Returns a Decision with action "commit" or "clarify".
        """
        p = fuse_context(p_raw, prior, self.gamma)
        label = int(p.argmax())

        if not self.is_ambiguous(p):
            return Decision(
                action="commit",
                label=label,
                posterior=p,
                reason="confident: passes all uncertainty gates"
            )

        table = [self.score(p, q, prior) for q in self.candidate_questions(p)]
        best = max(table, key=lambda r: r["utility"])

        if best["utility"] >= self.u_min:
            return Decision(
                action="clarify",
                label=label,
                question=best["question"],
                ambiguous=True,
                posterior=p,
                table=table,
                reason="ambiguous: clarification has positive utility"
            )

        return Decision(
            action="commit",
            label=label,
            ambiguous=True,
            posterior=p,
            table=table,
            reason="ambiguous but no question exceeds the utility floor"
        )
