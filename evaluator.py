"""
evaluator.py — Binary (Pass / Fail) grader for solver responses.

Uses an LLM-as-judge approach: the judge receives the task question,
the gold-standard constraint, and the solver's response, then returns
a structured PASS or FAIL verdict with a brief rationale.

This keeps the grading objective and consistent across both the Vanilla
and Augmented solver profiles.
"""

import re
from typing import Optional

import anthropic


_JUDGE_SYSTEM = (
    "You are an impartial regulatory compliance examiner. "
    "Your sole job is to decide whether a response correctly addresses "
    "a biopharma regulatory question. You are strict: partial answers, "
    "vague generalisations, or responses that miss the specific regulatory "
    "citation required by the gold standard are marked FAIL. "
    "You do NOT give partial credit."
)

_JUDGE_TEMPLATE = """
You must evaluate whether the SOLVER RESPONSE correctly answers the compliance question
and satisfies the GOLD STANDARD requirement.

--- QUESTION ---
{question}

--- GOLD STANDARD (the correct, complete answer) ---
{gold_standard}

--- SOLVER RESPONSE ---
{response}

Evaluation criteria:
1. Does the response identify the correct regulatory obligation or prohibited action?
2. Does it cite the correct regulation section (or an equivalent that leads to the same answer)?
3. Is the core compliance guidance consistent with the gold standard?

Minor wording differences are acceptable. Critically wrong or missing key requirements → FAIL.

Respond with EXACTLY this format (two lines, nothing else):
VERDICT: <PASS or FAIL>
REASON: <one sentence explaining the verdict>
"""


class Evaluator:
    """
    LLM-as-judge binary grader.

    Parameters
    ----------
    client : anthropic.Anthropic
    model  : Model to use for the judge (can differ from the solver model)
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.client = client
        self.model = model

    def evaluate(
        self,
        question: str,
        gold_standard: str,
        response: str,
    ) -> tuple[bool, str]:
        """
        Grade `response` against `gold_standard`.

        Returns
        -------
        passed : bool  — True = PASS, False = FAIL
        reason : str   — Judge's one-sentence rationale
        """
        if not response or len(response.strip()) < 10:
            return False, "Response was empty or too short to evaluate."

        prompt = _JUDGE_TEMPLATE.format(
            question=question,
            gold_standard=gold_standard,
            response=response,
        )

        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            return self._parse_verdict(raw)
        except Exception as exc:
            # Fail-safe: treat judge errors as FAIL to avoid false positives
            return False, f"Judge error: {exc}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_verdict(raw: str) -> tuple[bool, str]:
        """
        Parse the judge's two-line response.

        Expected format:
            VERDICT: PASS
            REASON: <sentence>
        """
        verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", raw, re.IGNORECASE)
        reason_match  = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)

        if not verdict_match:
            # Fallback: scan for bare PASS/FAIL keywords
            upper = raw.upper()
            passed = "PASS" in upper and "FAIL" not in upper
            return passed, raw.strip()

        passed = verdict_match.group(1).upper() == "PASS"
        reason = reason_match.group(1).strip() if reason_match else raw
        return passed, reason
