"""
irt_parameters.py — 2PL Item Response Theory parameter assignment & filtration.

Two-Parameter Logistic (2PL) model
-----------------------------------
The probability that a person with ability θ answers item i correctly is:

    P(θ) = 1 / (1 + exp(−a_i * (θ − b_i)))

Parameters
    a  : discrimination — how sharply the item separates lower from higher θ
    b  : difficulty     — the θ level at which P(correct) = 0.50

Filtration logic
----------------
We observe two "solver profiles":

    Vanilla   → θ_vanilla  ≈ 0.0   (baseline zero-shot LLM)
    Augmented → θ_augmented ≈ 2.5  (chain-of-thought + rule-checking agent)

Four possible outcomes per item:

    Both PASS  → item is too easy  (b << 0);   discard  (low information)
    Both FAIL  → item is too hard  or ill-defined; discard (non-discriminating at any θ)
    Vanilla PASS, Augmented FAIL  → anomalous (augmented should dominate); discard
    Vanilla FAIL, Augmented PASS  → RETAIN: item lives in the discrimination band
                                    between the two θ levels → high b, high a

For retained items we assign mock 2PL parameters calibrated to the observation:

    b ~ N(2.0, σ=0.25), clipped to [1.5, 3.0]
      — difficulty sits between the two θ levels (0 < b < 2.5)
        but is closer to the augmented threshold

    a ~ N(2.2, σ=0.25), clipped to [1.5, 3.0]
      — high discrimination: the item cleanly separates the two profiles

A random seed is derived from the task_id string so output is reproducible
across runs for the same task.
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


# ---------------------------------------------------------------------------
# Retention outcomes
# ---------------------------------------------------------------------------

class RetentionOutcome:
    RETAINED   = "retained"          # vanilla fail / augmented pass
    TOO_EASY   = "discarded_too_easy"   # both pass
    TOO_HARD   = "discarded_too_hard"   # both fail
    ANOMALOUS  = "discarded_anomalous"  # vanilla pass / augmented fail


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def classify_item(vanilla_pass: bool, augmented_pass: bool) -> str:
    """
    Apply the filtration rule and return a RetentionOutcome string.
    """
    if vanilla_pass and augmented_pass:
        return RetentionOutcome.TOO_EASY
    if not vanilla_pass and not augmented_pass:
        return RetentionOutcome.TOO_HARD
    if vanilla_pass and not augmented_pass:
        return RetentionOutcome.ANOMALOUS
    # not vanilla_pass and augmented_pass
    return RetentionOutcome.RETAINED


def assign_2pl_parameters(task_id: str) -> dict:
    """
    Assign initial mock 2PL parameters for a retained item.

    A deterministic seed derived from task_id ensures reproducibility.
    """
    seed = int(hashlib.md5(task_id.encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)

    b = rng.gauss(2.0, 0.25)
    b = max(1.5, min(3.0, round(b, 4)))

    a = rng.gauss(2.2, 0.25)
    a = max(1.5, min(3.0, round(a, 4)))

    return {"irt_a": a, "irt_b": b}


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
