"""
evaluator.py — Binary (Pass / Fail) grader for solver responses.

Uses an LLM-as-judge approach: the judge receives the task question,
the gold-standard constraint, and the solver's response, then returns
a structured PASS or FAIL verdict with a brief rationale.

Option 4 — soft scores via logprobs
------------------------------------
When the Groq API returns token-level log-probabilities, we extract
a soft P(correct) ∈ (0,1) from the logprob of the PASS vs. FAIL token
at the verdict position.  This continuous signal allows empirical b
estimation via logit inversion:

    b_i = θ - logit(P_soft(correct | θ))

which is more informative than band-based uniform sampling.
If logprobs are unavailable, p_correct defaults to 0.5 (uninformative).
"""

import math
import re
from typing import Optional

import openai


_JUDGE_SYSTEM_STRICT = (
    "You are an impartial regulatory compliance examiner. "
    "Your sole job is to decide whether a response correctly addresses "
    "a biopharma regulatory question. Focus on whether the core compliance "
    "determination is correct (compliant vs. non-compliant) and whether the "
    "response identifies the right regulatory obligation. Exact section numbers "
    "are a plus but not required if the substantive answer is correct. "
    "Mark FAIL only if the compliance conclusion is wrong, a key obligation is "
    "completely missed, or the answer is vague to the point of being useless."
)

_JUDGE_TEMPLATE_STRICT = """
You must evaluate whether the SOLVER RESPONSE correctly answers the compliance question
and satisfies the GOLD STANDARD requirement.

--- QUESTION ---
{question}

--- GOLD STANDARD (the correct, complete answer) ---
{gold_standard}

--- SOLVER RESPONSE ---
{response}

Evaluation criteria:
1. Does the response reach the correct compliance conclusion (e.g., compliant vs. non-compliant, major vs. minor deviation)?
2. Does it cite at least one specific regulation at the SECTION level (e.g., "21 CFR 11.10", "21 CFR 211.25", "ICH E6(R2) §4.8.2", "45 CFR 46.116") that is relevant to the answer? Citing only a Part or Title (e.g., "21 CFR Part 11", "21 CFR Part 211", "GMP regulations") WITHOUT a specific section number does NOT satisfy this criterion.
3. Is the substantive compliance reasoning consistent with the gold standard's key obligation?

Mark PASS only when ALL THREE criteria are met.
Mark FAIL when: the conclusion is wrong, no specific regulatory citation is provided, or the key obligation is missed.

Respond with EXACTLY this format (two lines, nothing else):
VERDICT: <PASS or FAIL>
REASON: <one sentence explaining the verdict>
"""

_JUDGE_SYSTEM_LENIENT = (
    "You are an impartial regulatory compliance examiner grading straightforward "
    "biopharma compliance questions. These are baseline questions with clear, "
    "well-established answers. Focus solely on whether the response reaches the "
    "correct compliance conclusion with coherent reasoning. "
    "Do NOT require specific regulatory section citations — general regulatory "
    "knowledge and correct reasoning are sufficient. "
    "Mark FAIL only if the compliance conclusion is outright wrong or the response "
    "is too vague to convey any meaningful answer."
)

_JUDGE_TEMPLATE_LENIENT = """
You must evaluate whether the SOLVER RESPONSE correctly answers the compliance question
and satisfies the GOLD STANDARD requirement.

--- QUESTION ---
{question}

--- GOLD STANDARD (the correct, complete answer) ---
{gold_standard}

--- SOLVER RESPONSE ---
{response}

Evaluation criteria:
1. Does the response reach the correct compliance conclusion?
2. Is the substantive compliance reasoning consistent with the gold standard's key obligation?

Mark PASS when BOTH criteria are met. Specific regulatory citations are NOT required.
Mark FAIL only when the conclusion is wrong or the answer is too vague to be useful.

Respond with EXACTLY this format (two lines, nothing else):
VERDICT: <PASS or FAIL>
REASON: <one sentence explaining the verdict>
"""


class Evaluator:
    """
    LLM-as-judge binary grader.

    Parameters
    ----------
    client : openai.OpenAI
    model  : Model to use for the judge (can differ from the solver model)
    strict : If True (default), require section-level regulatory citations.
             If False, only require a correct conclusion and coherent reasoning.
             Use strict=False for easy-domain items where vanilla is expected to pass.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "llama-3.3-70b-versatile",
        strict: bool = True,
    ) -> None:
        self.client = client
        self.model  = model
        self._system   = _JUDGE_SYSTEM_STRICT   if strict else _JUDGE_SYSTEM_LENIENT
        self._template = _JUDGE_TEMPLATE_STRICT if strict else _JUDGE_TEMPLATE_LENIENT

    def evaluate(
        self,
        question: str,
        gold_standard: str,
        response: str,
    ) -> tuple[bool, str, float]:
        """
        Grade `response` against `gold_standard`.

        Returns
        -------
        passed    : bool  — True = PASS, False = FAIL
        reason    : str   — Judge's one-sentence rationale
        p_correct : float — Soft probability of correctness from logprobs (0.5 if unavailable)
        """
        if not response or len(response.strip()) < 10:
            return False, "Response was empty or too short to evaluate.", 0.0

        prompt = self._template.format(
            question=question,
            gold_standard=gold_standard,
            response=response,
        )

        try:
            msg = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = msg.choices[0].message.content.strip()
            passed, reason = self._parse_verdict(raw)

            # Attempt to extract soft P(correct) from token logprobs if available.
            # Many inference backends (including Groq llama-3.3-70b-versatile) do not
            # support logprobs; we fall back to 0.5 (uninformative prior) gracefully.
            # In the 15-profile architecture, b is derived from the population pass
            # rate rather than from individual logprobs, so 0.5 is acceptable here.
            lp_content = None
            try:
                lp_content = getattr(msg.choices[0].logprobs, "content", None)
            except Exception:
                pass
            p_correct = self._extract_soft_score(lp_content) if lp_content else 0.5

            return passed, reason, p_correct

        except Exception as exc:
            # Fail-safe: treat judge errors as FAIL to avoid false positives
            return False, f"Judge error: {exc}", 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_soft_score(logprobs_content) -> float:
        """
        Scan token-level logprobs for the PASS/FAIL verdict token and return
        a soft P(PASS) by normalising the logprobs of both options.

        Falls back to 0.5 when neither token is found in the logprob stream.
        """
        for token_data in logprobs_content:
            token = token_data.token.strip().upper()
            if token not in ("PASS", "FAIL"):
                continue

            lp_pass = lp_fail = None

            # The generated token contributes its logprob directly
            if token == "PASS":
                lp_pass = token_data.logprob
            else:
                lp_fail = token_data.logprob

            # Alternatives may carry the missing verdict token
            for top in (token_data.top_logprobs or []):
                t = top.token.strip().upper()
                if t == "PASS" and lp_pass is None:
                    lp_pass = top.logprob
                elif t == "FAIL" and lp_fail is None:
                    lp_fail = top.logprob

            if lp_pass is not None and lp_fail is not None:
                # Numerically stable softmax over the two-token vocabulary
                m = max(lp_pass, lp_fail)
                ep = math.exp(lp_pass - m)
                ef = math.exp(lp_fail - m)
                return ep / (ep + ef)
            elif lp_pass is not None:
                return max(0.01, min(0.99, math.exp(lp_pass)))
            elif lp_fail is not None:
                return max(0.01, min(0.99, 1.0 - math.exp(lp_fail)))
            return 0.5  # verdict token found but no usable logprob pair

        return 0.5  # no verdict token in logprob stream

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
