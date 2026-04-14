"""
calibrator.py — Stage 2: IRT Calibration Engine.

15-Profile Synthetic Population
---------------------------------
Items are calibrated against a population of 15 AI solver profiles spanning
a spectrum of regulatory expertise and temperature settings:

  Tier 0  — General Layperson          (temps 0.1, 0.4, 0.7)
  Tier 1  — Life Sciences Professional  (temps 0.2, 0.5, 0.8)
  Tier 2  — Regulatory Affairs Coordinator (temps 0.1, 0.4, 0.7)
  Tier 3  — Regulatory Affairs Specialist  (temps 0.2, 0.5, 0.8)
  Tier 4  — Senior FDA Auditor          (temps 0.1, 0.4, 0.7)

All profiles use the same simple user template; the system prompt encodes
the expertise level.  Difficulty b is computed from the empirical pass rate P:

    b = ln((1 − P) / P)   with P clipped to [0.01, 0.99]

The layperson profile (index 0) is stored as "vanilla" and the expert profile
(index 14) as "augmented" for backward compatibility with the DB schema and
fda_importer.py.
"""

import os
import re as _re

import openai

from evaluator import Evaluator
from irt_parameters import (
    RetentionOutcome,
    classify_item_from_pass_rate,
    compute_b_from_pass_rate,
    describe_item,
)

RASCH_A = 1.0


# ---------------------------------------------------------------------------
# 15-Profile solver population
# Each entry: (profile_name, temperature, system_prompt)
# ---------------------------------------------------------------------------

SOLVER_PROFILES: list[tuple[str, float, str]] = [
    # ── Tier 0: General Layperson ──────────────────────────────────────────
    (
        "layperson_t0.1", 0.1,
        "You are a general member of the public with no background in "
        "pharmaceutical or medical regulations. Answer questions using common "
        "sense and general knowledge only. Do not cite specific regulation numbers.",
    ),
    (
        "layperson_t0.4", 0.4,
        "You are a general member of the public with no background in "
        "pharmaceutical or medical regulations. Answer questions using common "
        "sense and general knowledge only. Do not cite specific regulation numbers.",
    ),
    (
        "layperson_t0.7", 0.7,
        "You are a general member of the public with no background in "
        "pharmaceutical or medical regulations. Answer questions using common "
        "sense and general knowledge only. Do not cite specific regulation numbers.",
    ),
    # ── Tier 1: Life Sciences Professional ────────────────────────────────
    (
        "lifesci_t0.2", 0.2,
        "You are a life sciences professional (e.g., lab scientist or clinical "
        "research coordinator) with general knowledge of good scientific "
        "practices but limited formal regulatory training. Answer based on "
        "your scientific background and general understanding of quality systems.",
    ),
    (
        "lifesci_t0.5", 0.5,
        "You are a life sciences professional (e.g., lab scientist or clinical "
        "research coordinator) with general knowledge of good scientific "
        "practices but limited formal regulatory training. Answer based on "
        "your scientific background and general understanding of quality systems.",
    ),
    (
        "lifesci_t0.8", 0.8,
        "You are a life sciences professional (e.g., lab scientist or clinical "
        "research coordinator) with general knowledge of good scientific "
        "practices but limited formal regulatory training. Answer based on "
        "your scientific background and general understanding of quality systems.",
    ),
    # ── Tier 2: Regulatory Affairs Coordinator ────────────────────────────
    (
        "ra_coord_t0.1", 0.1,
        "You are a regulatory affairs coordinator with 2–3 years of experience "
        "in biopharma compliance. You are familiar with FDA regulations at a "
        "high level but sometimes need to look up specific section numbers. "
        "Answer from your working knowledge of 21 CFR and GCP basics.",
    ),
    (
        "ra_coord_t0.4", 0.4,
        "You are a regulatory affairs coordinator with 2–3 years of experience "
        "in biopharma compliance. You are familiar with FDA regulations at a "
        "high level but sometimes need to look up specific section numbers. "
        "Answer from your working knowledge of 21 CFR and GCP basics.",
    ),
    (
        "ra_coord_t0.7", 0.7,
        "You are a regulatory affairs coordinator with 2–3 years of experience "
        "in biopharma compliance. You are familiar with FDA regulations at a "
        "high level but sometimes need to look up specific section numbers. "
        "Answer from your working knowledge of 21 CFR and GCP basics.",
    ),
    # ── Tier 3: Regulatory Affairs Specialist ─────────────────────────────
    (
        "ra_spec_t0.2", 0.2,
        "You are a senior regulatory affairs specialist with 8+ years of "
        "experience in FDA compliance. You are well-versed in 21 CFR Parts 11, "
        "50, 56, 211, GCP ICH E6(R2), and GMP regulations. Provide thorough, "
        "citation-backed answers citing specific section numbers where relevant.",
    ),
    (
        "ra_spec_t0.5", 0.5,
        "You are a senior regulatory affairs specialist with 8+ years of "
        "experience in FDA compliance. You are well-versed in 21 CFR Parts 11, "
        "50, 56, 211, GCP ICH E6(R2), and GMP regulations. Provide thorough, "
        "citation-backed answers citing specific section numbers where relevant.",
    ),
    (
        "ra_spec_t0.8", 0.8,
        "You are a senior regulatory affairs specialist with 8+ years of "
        "experience in FDA compliance. You are well-versed in 21 CFR Parts 11, "
        "50, 56, 211, GCP ICH E6(R2), and GMP regulations. Provide thorough, "
        "citation-backed answers citing specific section numbers where relevant.",
    ),
    # ── Tier 4: Senior FDA Auditor ─────────────────────────────────────────
    (
        "fda_auditor_t0.1", 0.1,
        "You are a senior FDA auditor with 15+ years of experience conducting "
        "GMP, GCP, and 21 CFR Part 11 compliance inspections. You have deep "
        "expertise in FDA enforcement actions, warning letters, and regulatory "
        "nuance. Provide precise, section-level regulatory analysis citing "
        "exact CFR provisions and ICH guidelines.",
    ),
    (
        "fda_auditor_t0.4", 0.4,
        "You are a senior FDA auditor with 15+ years of experience conducting "
        "GMP, GCP, and 21 CFR Part 11 compliance inspections. You have deep "
        "expertise in FDA enforcement actions, warning letters, and regulatory "
        "nuance. Provide precise, section-level regulatory analysis citing "
        "exact CFR provisions and ICH guidelines.",
    ),
    (
        "fda_auditor_t0.7", 0.7,
        "You are a senior FDA auditor with 15+ years of experience conducting "
        "GMP, GCP, and 21 CFR Part 11 compliance inspections. You have deep "
        "expertise in FDA enforcement actions, warning letters, and regulatory "
        "nuance. Provide precise, section-level regulatory analysis citing "
        "exact CFR provisions and ICH guidelines.",
    ),
]

# Indexes into SOLVER_PROFILES used for backward-compat DB columns
_IDX_VANILLA   = 0   # layperson_t0.1  → vanilla_*
_IDX_AUGMENTED = 14  # fda_auditor_t0.7 → augmented_*

assert len(SOLVER_PROFILES) == 15, "SOLVER_PROFILES must have exactly 15 entries"

# Single user template shared by all 15 profiles
_PROFILE_USER = """{context}

Question: {question}

Answer directly and concisely."""


# ---------------------------------------------------------------------------
# Calibrator class
# ---------------------------------------------------------------------------

class Calibrator:
    """
    Runs all 15 synthetic AI solver profiles against a compliance task, grades
    each response with the Evaluator, derives item difficulty b from the
    empirical pass rate, and returns a result dict ready for DB persistence.

    Parameters
    ----------
    client           : openai.OpenAI (Groq-compatible endpoint)
    solver_model     : model used for all 15 solver profiles
    judge_model      : model used by the Evaluator
    verbose          : print per-step progress to stdout
    augmented_config : accepted but ignored (backward-compat shim)
    """

    def __init__(
        self,
        client: openai.OpenAI,
        solver_model: str = "llama-3.3-70b-versatile",
        judge_model: str  = "llama-3.3-70b-versatile",
        verbose: bool = True,
        augmented_config: dict | None = None,
    ) -> None:
        self.client       = client
        self.solver_model = solver_model
        self.verbose      = verbose
        self._evaluator_strict  = Evaluator(client, model=judge_model, strict=True)
        self._evaluator_lenient = Evaluator(client, model=judge_model, strict=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, task: dict) -> dict:
        """
        Calibrate a single task against all 15 solver profiles.

        Returns a result dict with:
            task_id, vanilla_response, augmented_response,
            vanilla_pass, augmented_pass, p_vanilla, p_augmented,
            pass_rate, is_retained, retention_reason, irt_a, irt_b,
            _vanilla_reason, _augmented_reason,
            _profile_passes  (list[bool], 15 entries)
        """
        task_id       = task["task_id"]
        context       = task["context"]
        question      = task["question"]
        gold_standard = task["gold_standard"]
        is_easy       = task.get("is_easy", False)
        evaluator     = self._evaluator_lenient if is_easy else self._evaluator_strict

        self._log(f"\n  {'─'*60}")
        self._log(
            f"  Calibrating : {task_id}  [{'easy' if is_easy else 'hard'}]"
        )
        self._log(f"  Domain      : {task.get('domain', 'unknown')}")
        self._log(f"  Profiles    : {len(SOLVER_PROFILES)}")

        profile_responses : list[str]   = []
        profile_passes    : list[bool]  = []
        profile_soft_p    : list[float] = []

        profile_reasons: list[str] = []

        for idx, (name, temperature, system_prompt) in enumerate(SOLVER_PROFILES):
            response = self._run_profile(context, question, system_prompt, temperature)
            passed, reason, p_soft = evaluator.evaluate(
                question, gold_standard, response
            )
            profile_responses.append(response)
            profile_passes.append(passed)
            profile_soft_p.append(p_soft)
            profile_reasons.append(reason)
            self._log(
                f"  [{idx:02d}] {name:<22}  "
                f"{'PASS' if passed else 'FAIL'}  (p={p_soft:.3f})  {reason}"
            )

        # -- Pass rate and b estimation -----------------------------------
        n_pass    = sum(profile_passes)
        pass_rate = n_pass / len(SOLVER_PROFILES)
        irt_b     = compute_b_from_pass_rate(pass_rate)
        irt_a     = RASCH_A

        # -- Retention classification -------------------------------------
        outcome     = classify_item_from_pass_rate(pass_rate)
        is_retained = outcome in (RetentionOutcome.RETAINED, RetentionOutcome.RETAINED_EASY)

        self._log(
            f"\n  Pass rate   : {n_pass}/{len(SOLVER_PROFILES)} = {pass_rate:.3f}  "
            f"→  b = {irt_b:.4f}  ({outcome})"
        )
        if is_retained:
            self._log(f"  [IRT]       {describe_item(task_id, irt_a, irt_b)}")

        # -- Backward-compat DB fields ------------------------------------
        vanilla_response   = profile_responses[_IDX_VANILLA]
        augmented_response = profile_responses[_IDX_AUGMENTED]
        vanilla_pass       = profile_passes[_IDX_VANILLA]
        augmented_pass     = profile_passes[_IDX_AUGMENTED]
        p_vanilla          = profile_soft_p[_IDX_VANILLA]
        p_augmented        = profile_soft_p[_IDX_AUGMENTED]

        return {
            "task_id":             task_id,
            # Legacy two-profile fields (required by DB schema)
            "vanilla_response":    vanilla_response,
            "augmented_response":  augmented_response,
            "vanilla_pass":        vanilla_pass,
            "augmented_pass":      augmented_pass,
            "p_vanilla":           p_vanilla,
            "p_augmented":         p_augmented,
            # Population-level calibration
            "pass_rate":           pass_rate,
            "_profile_passes":     profile_passes,
            # Retention
            "is_retained":         is_retained,
            "retention_reason":    outcome,
            "irt_a":               irt_a,
            "irt_b":               irt_b,
            # Rationales for layperson (idx 0) and expert (idx 14)
            "_vanilla_reason":     profile_reasons[_IDX_VANILLA],
            "_augmented_reason":   profile_reasons[_IDX_AUGMENTED],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_profile(
        self,
        context: str,
        question: str,
        system_prompt: str,
        temperature: float,
    ) -> str:
        """Call the LLM for one solver profile with its specific system prompt and temperature."""
        prompt = _PROFILE_USER.format(context=context, question=question)
        try:
            msg = self.client.chat.completions.create(
                model=self.solver_model,
                max_tokens=512,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = msg.choices[0].message.content.strip()
            # Strip <think>…</think> reasoning blocks (DeepSeek-R1 style)
            stripped = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
            return stripped if len(stripped) >= 80 else raw
        except Exception as exc:
            return f"[Solver error: {exc}]"

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
