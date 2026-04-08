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

import openai

from evaluator import Evaluator
from irt_parameters import (
    RetentionOutcome,
    assign_2pl_parameters,
    classify_item,
    describe_item,
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

_AUGMENTED_SYSTEM = (
    "You are a senior biopharma regulatory affairs expert. "
    "You answer compliance questions through rigorous, structured analysis. "
    "You never skip steps, even when the answer seems obvious."
)

_AUGMENTED_USER = """
A compliance question requires your expert analysis. Work through it systematically.

--- SCENARIO ---
{context}

--- QUESTION ---
{question}

Follow this EXACT four-step process before giving your final answer:

STEP 1 — REGULATION IDENTIFICATION
List every regulation, guideline, or guidance document that might apply to this scenario.
For each, quote the specific section number and its core obligation.

STEP 2 — FACTUAL APPLICATION
For each regulation identified, explicitly map it to the facts in the scenario.
State whether each regulatory requirement is met, violated, or uncertain given the facts.

STEP 3 — CRITICAL REFLECTION
Identify at least one way your analysis in Step 2 could be wrong or incomplete.
Consider edge cases, conflicting requirements, or factual ambiguities in the scenario.

STEP 4 — FINAL COMPLIANCE VERDICT
Based on the above analysis, state your definitive answer to the question.
Cite the single most controlling regulation and state the precise obligation or violation.

Begin your response with "STEP 1 —"
"""


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
    ) -> None:
        self.client       = client
        self.solver_model = solver_model
        self.evaluator    = Evaluator(client, model=judge_model)
        self.verbose      = verbose

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
        task_id      = task["task_id"]
        context      = task["context"]
        question     = task["question"]
        gold_standard = task["gold_standard"]

        self._log(f"\n  {'─'*60}")
        self._log(f"  Calibrating: {task_id}")
        self._log(f"  Domain     : {task.get('domain', 'unknown')}")

        # -- Vanilla solver --------------------------------------------------
        self._log("  [Vanilla]   Running zero-shot solver…")
        vanilla_response = self._run_vanilla(context, question)
        vanilla_pass, vanilla_reason = self.evaluator.evaluate(
            question, gold_standard, vanilla_response
        )
        self._log(
            f"  [Vanilla]   Verdict: {'PASS' if vanilla_pass else 'FAIL'} "
            f"— {vanilla_reason}"
        )

        # -- Augmented solver ------------------------------------------------
        self._log("  [Augmented] Running chain-of-thought agent…")
        augmented_response = self._run_augmented(context, question)
        augmented_pass, augmented_reason = self.evaluator.evaluate(
            question, gold_standard, augmented_response
        )
        self._log(
            f"  [Augmented] Verdict: {'PASS' if augmented_pass else 'FAIL'} "
            f"— {augmented_reason}"
        )

        # -- Filtration & IRT parameters -------------------------------------
        outcome    = classify_item(vanilla_pass, augmented_pass)
        is_retained = outcome == RetentionOutcome.RETAINED

        irt_a = irt_b = None
        if is_retained:
            params = assign_2pl_parameters(task_id)
            irt_a  = params["irt_a"]
            irt_b  = params["irt_b"]
            self._log(
                f"  [IRT]       RETAINED — {describe_item(task_id, irt_a, irt_b)}"
            )
        else:
            self._log(f"  [IRT]       Discarded ({outcome})")

        return {
            "task_id":            task_id,
            "vanilla_response":   vanilla_response,
            "augmented_response": augmented_response,
            "vanilla_pass":       vanilla_pass,
            "augmented_pass":     augmented_pass,
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
        Multi-step augmented agent: forced chain-of-thought, regulation
        identification, factual application, reflection, and verdict.
        """
        prompt = _AUGMENTED_USER.format(context=context, question=question)
        try:
            msg = self.client.chat.completions.create(
                model=self.solver_model,
                max_tokens=1536,
                messages=[
                    {"role": "system", "content": _AUGMENTED_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            return msg.choices[0].message.content.strip()
        except Exception as exc:
            return f"[Solver error: {exc}]"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
