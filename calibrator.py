"""
calibrator.py — Stage 2: IRT Calibration Engine.

Two solver profiles
-------------------
Vanilla   : Zero-shot LLM.  Receives only the scenario context and question.
            No scaffolding, no system prompt beyond minimal framing.

Augmented : Simulated chain-of-thought + rule-checking agent.
            A structured multi-step prompt forces the model to:
              Step 1 — Identify and quote the potentially applicable regulation(s).
              Step 2 — Apply each regulation to the specific facts in the scenario.
              Step 3 — Critically reflect: what could go wrong in the analysis?
              Step 4 — State a final, precise compliance verdict.

The Calibrator runs both profiles against each task, grades them with the
Evaluator, applies the IRT filtration logic, and returns a result dict
ready for database persistence.
"""

import json
import os

import openai

from evaluator import Evaluator
from irt_parameters import (
    RetentionOutcome,
    assign_2pl_parameters,
    assign_2pl_parameters_easy,
    assign_rasch_parameters_too_hard,
    classify_item,
    describe_item,
    estimate_b_from_soft_scores,
)


# ---------------------------------------------------------------------------
# Solver prompts
# ---------------------------------------------------------------------------

_VANILLA_SYSTEM = (
    "You are a general life sciences professional answering a compliance question. "
    "Give a brief, direct answer based on your general understanding. "
    "Do not look up specific regulation numbers; answer from intuition and general knowledge."
)

_VANILLA_USER = """
{context}

Question: {question}

Answer directly and concisely.
"""

# ------------------------------------
# Augmented solver config — loaded from solver_config.json if present,
# otherwise falls back to the hardcoded baseline below.

_SOLVER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "solver_config.json")

_AUGMENTED_SYSTEM_DEFAULT = (
    "You are a senior biopharma regulatory affairs expert. "
    "You answer compliance questions through rigorous, structured analysis. "
    "You never skip steps, even when the answer seems obvious."
)

_AUGMENTED_USER_DEFAULT = """
A compliance question requires your expert analysis. Work through it systematically.

--- REGULATORY REFERENCE TOOLKIT ---
Draw on these key provisions when relevant (not exhaustive):
\u2022 21 CFR Part 11 (electronic records): \u00a711.10(a) validation, \u00a711.10(b) copies, \u00a711.10(c) retrieval, \u00a711.10(e) audit trails, \u00a711.10(f) authority checks, \u00a711.10(g) sequence checks, \u00a711.10(h) device checks, \u00a711.30 open systems, \u00a711.50 signatures, \u00a711.70 signature linking
\u2022 GCP/ICH E6(R2): \u00a74.8 informed consent, \u00a74.8.2 re-consent, \u00a75.18 monitoring, \u00a75.18.2 deviation reporting, \u00a75.21 non-compliance, 21 CFR 56.108(a)(3) IRB reportable changes, 21 CFR 312.62 investigator records
\u2022 Promotional materials: 21 CFR 202.1(e)(1) brief summary, \u00a7202.1(e)(2) reminder ads, \u00a7202.1(e)(3)(ii) brief summary exceptions, \u00a7202.1(e)(5) black box requirements, OPDP draft guidance on social media
\u2022 GMP manufacturing: 21 CFR 211.25(a) personnel qualification, \u00a7211.68 computerized systems, \u00a7211.100 production controls, \u00a7211.165(a) specifications, \u00a7211.192 batch record review, \u00a7211.194 OOS investigation
\u2022 Informed consent: 21 CFR 50.25(a)-(b) elements, \u00a750.25(c) additional elements, 45 CFR 46.116(a)-(c) required elements, \u00a746.116(f) waiver criteria, \u00a746.116(f)(3) alteration conditions

--- SCENARIO ---
{context}

--- QUESTION ---
{question}

Follow this EXACT four-step process before giving your final answer:

STEP 1 \u2014 REGULATION IDENTIFICATION
List every regulation, guideline, or guidance document that might apply to this scenario.
For each, quote the specific section number and its core obligation.

STEP 2 \u2014 FACTUAL APPLICATION
For each regulation identified, explicitly map it to the facts in the scenario.
State whether each regulatory requirement is met, violated, or uncertain given the facts.

STEP 3 \u2014 CRITICAL REFLECTION
Identify at least one way your analysis in Step 2 could be wrong or incomplete.
Consider edge cases, exceptions, safe harbors, or regulatory carve-outs that might change your conclusion.

STEP 4 \u2014 FINAL COMPLIANCE VERDICT
Based on the above analysis, state your definitive answer to the question.
Cite the single most controlling regulation SECTION (not just the Part) and state the precise obligation or violation.

Begin your response with "STEP 1 \u2014"
"""


def _load_solver_config() -> dict:
    """Load augmented solver config from solver_config.json, or return defaults."""
    try:
        with open(_SOLVER_CONFIG_PATH) as f:
            cfg = json.load(f)
        return {
            "model":         cfg.get("model", "llama-3.3-70b-versatile"),
            "max_tokens":    cfg.get("max_tokens", 1536),
            "system_prompt": cfg.get("system_prompt", _AUGMENTED_SYSTEM_DEFAULT),
            "user_template": cfg.get("user_template", _AUGMENTED_USER_DEFAULT),
        }
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "model":         "llama-3.3-70b-versatile",
            "max_tokens":    1536,
            "system_prompt": _AUGMENTED_SYSTEM_DEFAULT,
            "user_template": _AUGMENTED_USER_DEFAULT,
        }


# ---------------------------------------------------------------------------
# Calibrator class
# ---------------------------------------------------------------------------

class Calibrator:
    """
    Runs the Vanilla and Augmented solver profiles against a compliance task,
    grades both responses, applies 2PL filtration, and returns a result dict.

    Parameters
    ----------
    client         : anthropic.Anthropic
    solver_model   : Claude model used for both solver profiles
    judge_model    : Claude model used by the Evaluator (can be the same)
    verbose        : Print per-step progress to stdout
    """

    def __init__(
        self,
        client: openai.OpenAI,
        solver_model: str = "llama-3.3-70b-versatile",
        judge_model: str  = "llama-3.3-70b-versatile",
        verbose: bool = True,
        augmented_config: dict | None = None,
    ) -> None:
        self.client            = client
        self.solver_model      = solver_model
        self.evaluator_strict  = Evaluator(client, model=judge_model, strict=True)
        self.evaluator_lenient = Evaluator(client, model=judge_model, strict=False)
        self.verbose           = verbose
        # Augmented solver config: explicit override > solver_config.json > defaults
        cfg = augmented_config or _load_solver_config()
        self._aug_model     = cfg["model"]
        self._aug_max_tok   = cfg["max_tokens"]
        self._aug_system    = cfg["system_prompt"]
        self._aug_user_tmpl = cfg["user_template"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, task: dict) -> dict:
        """
        Calibrate a single task.

        Returns a result dict containing:
            task_id, vanilla_response, augmented_response,
            vanilla_pass, augmented_pass, is_retained,
            retention_reason, irt_a (or None), irt_b (or None)
        """
        task_id       = task["task_id"]
        context       = task["context"]
        question      = task["question"]
        gold_standard = task["gold_standard"]
        is_easy       = task.get("is_easy", False)
        evaluator     = self.evaluator_lenient if is_easy else self.evaluator_strict

        self._log(f"\n  {'─'*60}")
        self._log(f"  Calibrating: {task_id}  [{'easy' if is_easy else 'hard'}]")
        self._log(f"  Domain     : {task.get('domain', 'unknown')}")

        # -- Vanilla solver --------------------------------------------------
        self._log("  [Vanilla]   Running zero-shot solver…")
        vanilla_response = self._run_vanilla(context, question)
        vanilla_pass, vanilla_reason, p_vanilla = evaluator.evaluate(
            question, gold_standard, vanilla_response
        )
        self._log(
            f"  [Vanilla]   Verdict: {'PASS' if vanilla_pass else 'FAIL'} "
            f"(p={p_vanilla:.3f}) — {vanilla_reason}"
        )

        # -- Augmented solver ------------------------------------------------
        self._log("  [Augmented] Running chain-of-thought agent…")
        augmented_response = self._run_augmented(context, question)
        augmented_pass, augmented_reason, p_augmented = evaluator.evaluate(
            question, gold_standard, augmented_response
        )
        self._log(
            f"  [Augmented] Verdict: {'PASS' if augmented_pass else 'FAIL'} "
            f"(p={p_augmented:.3f}) — {augmented_reason}"
        )

        # -- Filtration & IRT parameters -------------------------------------
        outcome     = classify_item(vanilla_pass, augmented_pass)
        is_retained = outcome in (RetentionOutcome.RETAINED, RetentionOutcome.RETAINED_EASY)

        irt_a = irt_b = None

        if outcome == RetentionOutcome.RETAINED:
            # Prefer soft b estimate (Option 4); fall back to uniform band
            b_soft = estimate_b_from_soft_scores(p_vanilla, p_augmented)
            if b_soft is not None and 0.0 <= b_soft <= 2.5:
                irt_a, irt_b = 1.0, b_soft
            else:
                params = assign_2pl_parameters(task_id)
                irt_a, irt_b = params["irt_a"], params["irt_b"]
            self._log(
                f"  [IRT]       RETAINED (hard) — {describe_item(task_id, irt_a, irt_b)}"
            )

        elif outcome == RetentionOutcome.RETAINED_EASY:
            # Prefer soft b estimate; fall back to uniform band
            b_soft = estimate_b_from_soft_scores(p_vanilla, p_augmented)
            if b_soft is not None and -2.5 <= b_soft <= 0.0:
                irt_a, irt_b = 1.0, b_soft
            else:
                params = assign_2pl_parameters_easy(task_id)
                irt_a, irt_b = params["irt_a"], params["irt_b"]
            self._log(
                f"  [IRT]       RETAINED (easy) — {describe_item(task_id, irt_a, irt_b)}"
            )

        elif outcome == RetentionOutcome.TOO_HARD:
            # Option 2 — assign b above θ_augmented so both-fail items enter θ MLE.
            # Prefer soft b estimate; fall back to uniform(2.5, 5.0) band.
            b_soft = estimate_b_from_soft_scores(p_vanilla, p_augmented)
            if b_soft is not None and b_soft >= 2.5:
                irt_a, irt_b = 1.0, b_soft
            else:
                params = assign_rasch_parameters_too_hard(task_id)
                irt_a, irt_b = params["irt_a"], params["irt_b"]
            self._log(
                f"  [IRT]       TOO_HARD (b assigned) — {describe_item(task_id, irt_a, irt_b)}"
            )

        else:
            self._log(f"  [IRT]       Discarded ({outcome})")

        return {
            "task_id":            task_id,
            "vanilla_response":   vanilla_response,
            "augmented_response": augmented_response,
            "vanilla_pass":       vanilla_pass,
            "augmented_pass":     augmented_pass,
            "p_vanilla":          p_vanilla,
            "p_augmented":        p_augmented,
            "is_retained":        is_retained,
            "retention_reason":   outcome,
            "irt_a":              irt_a,
            "irt_b":              irt_b,
            # Evaluation rationales (not persisted to DB but useful for debugging)
            "_vanilla_reason":   vanilla_reason,
            "_augmented_reason": augmented_reason,
        }

    # ------------------------------------------------------------------
    # Solver implementations
    # ------------------------------------------------------------------

    def _run_vanilla(self, context: str, question: str) -> str:
        """Zero-shot solver: minimal framing, no scaffolding."""
        prompt = _VANILLA_USER.format(context=context, question=question)
        try:
            msg = self.client.chat.completions.create(
                model=self.solver_model,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": _VANILLA_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            return msg.choices[0].message.content.strip()
        except Exception as exc:
            return f"[Solver error: {exc}]"

    def _run_augmented(self, context: str, question: str) -> str:
        """
        Multi-step augmented agent. Config loaded from solver_config.json so
        the autoevolve loop can hot-swap prompts and models without code changes.
        """
        prompt = self._aug_user_tmpl.format(context=context, question=question)
        try:
            msg = self.client.chat.completions.create(
                model=self._aug_model,
                max_tokens=self._aug_max_tok,
                messages=[
                    {"role": "system", "content": self._aug_system},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = msg.choices[0].message.content.strip()
            # DeepSeek-R1 and similar reasoning models wrap their chain-of-thought
            # in <think>…</think> tags. Strip those so the judge sees only the answer.
            import re as _re
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
            return raw
        except Exception as exc:
            return f"[Solver error: {exc}]"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
