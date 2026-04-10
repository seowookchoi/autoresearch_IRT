"""
irt_parameters.py — 1PL (Rasch) Item Response Theory parameter assignment.

One-Parameter Logistic (Rasch) model
--------------------------------------
The probability that a solver with ability θ answers item i correctly is:

    P(θ) = 1 / (1 + exp(−(θ − b_i)))

Parameters
    a  : discrimination — FIXED at 1.0 for all items (Rasch model assumption)
    b  : difficulty     — the θ level at which P(correct) = 0.50; varies per item

Why 1PL instead of 2PL?
------------------------
We observe only two solver profiles per item:

    Vanilla   → θ_vanilla  ≈ 0.0   (baseline zero-shot LLM)
    Augmented → θ_augmented ≈ 2.5  (chain-of-thought + rule-checking agent)

With only 2 binary observations (pass/fail at two fixed θ levels), the
discrimination parameter `a` is NOT identifiable — infinitely many (a, b) pairs
are consistent with any observed (FAIL, PASS) pattern. Assigning a random `a`
per item would be statistically meaningless. The Rasch model fixes a=1.0 and
only estimates b, which IS constrained by the outcome pattern:

    Vanilla FAIL, Augmented PASS  → b ∈ (0.0, 2.5)   sampled uniformly
    Both PASS (easy)              → b ∈ (-2.5, 0.0)   sampled uniformly
    Both FAIL (too hard)          → b ~ Uniform(2.5, 5.0) — keeps item in θ MLE
    Vanilla PASS, Augmented FAIL  → discarded — anomalous ordering

A deterministic seed from task_id ensures reproducibility.
"""

import hashlib
import math
import random
from typing import Optional


# ---------------------------------------------------------------------------
# Solver θ constants (mock ability levels for the two profiles)
# ---------------------------------------------------------------------------

THETA_VANILLA   = 0.0   # zero-shot baseline
THETA_AUGMENTED = 2.5   # chain-of-thought + rule-checking agent

RASCH_A = 1.0           # fixed discrimination for all items (Rasch / 1PL model)


# ---------------------------------------------------------------------------
# Retention outcomes
# ---------------------------------------------------------------------------

class RetentionOutcome:
    RETAINED       = "retained"            # vanilla fail / augmented pass  (hard zone)
    RETAINED_EASY  = "retained_easy"       # both pass                      (easy zone)
    TOO_HARD       = "discarded_too_hard"  # both fail
    ANOMALOUS      = "discarded_anomalous" # vanilla pass / augmented fail


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def classify_item(vanilla_pass: bool, augmented_pass: bool) -> str:
    """
    Apply the filtration rule and return a RetentionOutcome string.
    """
    if vanilla_pass and augmented_pass:
        return RetentionOutcome.RETAINED_EASY
    if not vanilla_pass and not augmented_pass:
        return RetentionOutcome.TOO_HARD
    if vanilla_pass and not augmented_pass:
        return RetentionOutcome.ANOMALOUS
    # not vanilla_pass and augmented_pass
    return RetentionOutcome.RETAINED


def assign_rasch_parameters_easy(task_id: str) -> dict:
    """
    Assign Rasch (1PL) parameters for an easy item (both solvers pass).

    a = RASCH_A (fixed at 1.0)
    b ~ Uniform(-2.5, 0.0)  — difficulty below θ_vanilla, so both solvers pass
    """
    seed = int(hashlib.md5(("easy:" + task_id).encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)
    b    = round(rng.uniform(-2.5, 0.0), 4)
    return {"irt_a": RASCH_A, "irt_b": b}


def assign_rasch_parameters(task_id: str) -> dict:
    """
    Assign Rasch (1PL) parameters for a hard retained item (vanilla fail, augmented pass).

    a = RASCH_A (fixed at 1.0)
    b ~ Uniform(0.0, 2.5)   — difficulty between the two solver θ levels
    """
    seed = int(hashlib.md5(task_id.encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)
    b    = round(rng.uniform(0.0, 2.5), 4)
    return {"irt_a": RASCH_A, "irt_b": b}


def assign_rasch_parameters_too_hard(task_id: str) -> dict:
    """
    Assign Rasch (1PL) parameters for a too-hard item (both solvers fail).

    Option 2 — include too-hard items in the θ MLE response vector.
    Both solvers fail, so b must lie above θ_augmented = 2.5.

    a = RASCH_A (fixed at 1.0)
    b ~ Uniform(2.5, 5.0)   — difficulty above both solver θ levels
    """
    seed = int(hashlib.md5(("too_hard:" + task_id).encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)
    b    = round(rng.uniform(2.5, 5.0), 4)
    return {"irt_a": RASCH_A, "irt_b": b}


def estimate_b_from_soft_scores(
    p_vanilla: float,
    p_augmented: float,
    theta_vanilla: float = THETA_VANILLA,
    theta_augmented: float = THETA_AUGMENTED,
) -> Optional[float]:
    """
    Option 4 — empirical b estimation from judge logprob soft scores.

    Under the Rasch model:
        logit(P(θ)) = θ - b   →   b = θ - logit(P)

    We compute one b estimate from each solver's soft score, then take a
    precision-weighted average where the weight is the Fisher information
    at that ability level: w_i = P_i · (1 - P_i).

    Parameters
    ----------
    p_vanilla   : soft P(correct) for the vanilla solver (from logprobs)
    p_augmented : soft P(correct) for the augmented solver (from logprobs)

    Returns
    -------
    Estimated b, clipped to [-4.0, 8.0]; or None if both weights are ≈ 0.
    """
    # Clip to avoid log(0)
    eps = 1e-3
    p_v = max(eps, min(1 - eps, p_vanilla))
    p_a = max(eps, min(1 - eps, p_augmented))

    b_v = theta_vanilla   - math.log(p_v / (1 - p_v))
    b_a = theta_augmented - math.log(p_a / (1 - p_a))

    w_v = p_v * (1 - p_v)   # Fisher information weight
    w_a = p_a * (1 - p_a)

    total_w = w_v + w_a
    if total_w < 1e-9:
        return None

    b_est = (w_v * b_v + w_a * b_a) / total_w
    return round(max(-4.0, min(8.0, b_est)), 4)


# Keep old names as aliases so any external callers don't break
assign_2pl_parameters_easy = assign_rasch_parameters_easy
assign_2pl_parameters      = assign_rasch_parameters


def p_correct(theta: float, a: float, b: float) -> float:
    """
    2PL probability of a correct response.

    P(θ) = 1 / (1 + exp(−a * (θ − b)))
    """
    return 1.0 / (1.0 + math.exp(-a * (theta - b)))


def item_information(theta: float, a: float, b: float) -> float:
    """
    Fisher information for the 2PL at ability level θ.

    I(θ) = a² · P(θ) · (1 − P(θ))
    """
    p = p_correct(theta, a, b)
    return a**2 * p * (1.0 - p)


def estimate_theta_mle(
    responses: list[tuple[float, bool]],
    theta_init: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> float:
    """
    Estimate solver ability θ via MLE using Newton-Raphson on the Rasch score equation.

    Rasch model (a=1):  P(θ) = 1 / (1 + exp(-(θ - b)))
    Score equation:     f(θ)  = Σ [r_i - P(θ, b_i)] = 0
    Newton update:      θ ← θ - f(θ) / f'(θ)
    where              f'(θ) = -Σ P(θ, b_i) · (1 - P(θ, b_i))

    Parameters
    ----------
    responses   : list of (b_i, r_i) — item difficulty and pass/fail (True=1)
    theta_init  : starting point for Newton-Raphson (prior theta estimate)
    max_iter    : maximum Newton-Raphson iterations
    tol         : convergence tolerance on |f(θ)|

    Returns
    -------
    Estimated θ, clipped to [-4.0, 6.0] to avoid divergence on extreme response patterns.
    """
    if not responses:
        return theta_init

    # Perfect score or zero score → MLE is ±∞; return boundary value
    n_pass = sum(1 for _, r in responses if r)
    if n_pass == 0:
        return -4.0
    if n_pass == len(responses):
        return 6.0

    theta = float(theta_init)
    for _ in range(max_iter):
        f  = 0.0
        df = 0.0
        for b, r in responses:
            p   = p_correct(theta, RASCH_A, b)
            f  += (1.0 if r else 0.0) - p
            df -= p * (1.0 - p)
        if abs(df) < 1e-12:
            break
        step  = f / df
        theta -= step
        if abs(step) < tol:
            break

    return max(-4.0, min(6.0, round(theta, 4)))


def describe_item(task_id: str, a: float, b: float) -> str:
    """Human-readable summary of item parameters."""
    p_van = p_correct(THETA_VANILLA,   a, b)
    p_aug = p_correct(THETA_AUGMENTED, a, b)
    info  = item_information(b, a, b)   # max info at θ = b
    return (
        f"[{task_id}]  a={a:.4f}  b={b:.4f}  "
        f"P(correct|θ_vanilla={THETA_VANILLA:.1f})={p_van:.3f}  "
        f"P(correct|θ_aug={THETA_AUGMENTED:.1f})={p_aug:.3f}  "
        f"MaxInfo={info:.4f}"
    )
