"""
task_generator.py — Stage 1: Bio-Compliance Task Generator.

Uses an LLM to produce structured, domain-specific compliance scenarios
across FDA 21 CFR Part 11, GxP, clinical trial protocol deviations,
and promotional material review.
"""

import json
import re
import uuid
from typing import Optional

import openai


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAINS = [
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Electronic Records & Electronic Signatures",
        "hint": (
            "Create a scenario where the OBVIOUS answer is wrong. "
            "For example: a system appears compliant on the surface (has audit trails, "
            "validation docs, access controls) but has a single specific gap that only "
            "someone with deep Part 11 knowledge would catch — e.g., hybrid records "
            "where paper and electronic coexist, the 'predicate rule' scoping issue, "
            "or a closed system that performs an operation requiring open-system controls. "
            "The question should have a COUNTERINTUITIVE correct answer that a general "
            "compliance professional would likely get wrong."
        ),
    },
    {
        "key": "gcp_deviation",
        "label": "GCP — Clinical Trial Protocol Deviations & IRB/IEC Reporting",
        "hint": (
            "Create a scenario where the reporting obligation is COUNTERINTUITIVE. "
            "For example: a deviation looks minor but crosses a threshold that makes "
            "it legally major; OR a seemingly serious deviation actually falls under "
            "a pre-approved exception in the protocol; OR the correct answer hinges on "
            "whether the deviation was prospective vs. retrospective, or sponsor- vs. "
            "investigator-initiated. The common-sense classification should be WRONG. "
            "Cite precise ICH E6(R2) section numbers and FDA 21 CFR obligations."
        ),
    },
    {
        "key": "promo_review",
        "label": "FDA Promotional Material Review — Off-Label Promotion & Fair Balance",
        "hint": (
            "Focus on the fair balance rule under 21 CFR 202.1(e)(1) and the specific "
            "conditions that trigger or exempt the brief summary requirement. Create a "
            "scenario with a clearly non-compliant promotional piece where the violation "
            "is non-obvious — e.g., a piece that includes risk information but fails the "
            "fair balance test because benefits are presented more prominently; or an ad "
            "that omits a black box warning in a context where 21 CFR 202.1(e)(3)(ii) "
            "requires it. The correct answer must require knowing a SPECIFIC subsection "
            "of 21 CFR 202 or OPDP guidance. General 'fair balance' reasoning should "
            "point to the wrong conclusion or miss the critical element."
        ),
    },
    {
        "key": "gmp_deviation",
        "label": "GMP — Manufacturing Deviation Investigation & CAPA",
        "hint": (
            "Create a scenario where the batch disposition decision is COUNTERINTUITIVE. "
            "For example: a batch with an OOS result that seems obvious-reject is actually "
            "releasable under specific Phase II investigation findings; OR a batch that "
            "passed all specs must be rejected because of a procedural non-compliance "
            "discovered during investigation (e.g., unqualified analyst, equipment "
            "calibration lapse). The correct answer under 21 CFR 211.192 or 211.165 "
            "should surprise a non-expert."
        ),
    },
    {
        "key": "informed_consent",
        "label": "GCP — Informed Consent Documentation & Re-Consent Requirements",
        "hint": (
            "Create a scenario where re-consent is either NOT required when a "
            "non-expert would assume it is, or IS required in a situation that seems "
            "exempt. For example: a protocol amendment that adds a non-invasive "
            "questionnaire might not require re-consent under 45 CFR 46.116 waiver "
            "criteria; OR subjects who have completed their last study visit still "
            "require re-consent for a sample repository addition. The question must "
            "hinge on a specific regulatory provision (21 CFR 50.25, ICH E6(R2) §4.8.2, "
            "45 CFR 46) that overrides the intuitive answer."
        ),
    },
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a senior biopharma regulatory affairs specialist with 20+ years of "
    "experience across FDA submissions, GxP compliance, and clinical operations. "
    "You draft exam-quality compliance scenarios used to evaluate regulatory "
    "professionals. Your scenarios are factually accurate, cite real regulation "
    "sections, and are genuinely challenging — requiring precise knowledge, not "
    "general reasoning."
)

_USER_TEMPLATE = """
Generate ONE biopharma compliance scenario as a single JSON object.

Domain: {domain_label}
Guidance: {domain_hint}

Required JSON structure (use EXACTLY these keys, no extras):
{{
  "task_id": "{task_id}",
  "domain": "{domain_key}",
  "context": "<3-5 sentences. A specific, realistic situation at a named fictional company (e.g., NovaBio Inc., Helix Pharma). Include concrete details: system names, dates, staff roles, quantities. Make the compliance challenge non-obvious.>",
  "question": "<One precise question that has a single defensible correct answer grounded in regulation. Avoid opinion questions.>",
  "gold_standard": "<The exact compliance requirement that constitutes the correct answer. MUST cite the specific regulation section(s) (e.g., 21 CFR 11.10(e), ICH E6(R2) §4.8.2) and state the precise obligation or prohibited action.>"
}}

Rules:
- Respond with ONLY valid JSON — no markdown fences, no commentary.
- The gold_standard must be verifiable against published regulations.
- CRITICAL: The scenario must be COUNTERINTUITIVE. The common-sense or obvious answer must be WRONG. A knowledgeable professional without specific regulatory training should confidently give the wrong answer.
- The correct answer must hinge on a specific regulatory provision, exception, or threshold that only deep domain expertise reveals.
- Do NOT include the answer anywhere in the context or question fields.
"""


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class TaskGenerator:
    """
    Generates structured biopharma compliance tasks using an LLM.

    Parameters
    ----------
    client      : anthropic.Anthropic
    model       : Claude model ID to use for generation
    max_retries : Number of reprompt attempts if JSON parsing fails
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "llama-3.3-70b-versatile",
        max_retries: int = 2,
    ) -> None:
        self.client = client
        self.model = model
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n: int = 3) -> list[dict]:
        """
        Generate `n` compliance tasks, cycling through the domain registry.

        Returns a list of task dicts; malformed responses are skipped with
        a warning rather than crashing the pipeline.
        """
        tasks: list[dict] = []
        for i in range(n):
            domain = DOMAINS[i % len(DOMAINS)]
            task = self._generate_one(domain, attempt=0)
            if task is not None:
                tasks.append(task)
                print(
                    f"  [TaskGenerator] Generated task {task['task_id']} "
                    f"(domain: {domain['key']})"
                )
            else:
                print(
                    f"  [TaskGenerator] WARNING: failed to generate task "
                    f"for domain '{domain['key']}' after {self.max_retries + 1} attempts"
                )
        return tasks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_one(self, domain: dict, attempt: int) -> Optional[dict]:
        """Generate a single task, retrying on parse errors."""
        task_id = f"TASK-{domain['key'].upper()}-{uuid.uuid4().hex[:8].upper()}"
        prompt = _USER_TEMPLATE.format(
            domain_label=domain["label"],
            domain_hint=domain["hint"],
            domain_key=domain["key"],
            task_id=task_id,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            task = self._parse_json(raw)
            task = self._validate(task, required_task_id=task_id, domain=domain)
            return task

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            if attempt < self.max_retries:
                print(
                    f"  [TaskGenerator] Parse error on attempt {attempt + 1}: {exc} "
                    f"— retrying…"
                )
                return self._generate_one(domain, attempt + 1)
            return None

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from the model response, tolerating markdown fences."""
        # Strip ```json ... ``` fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        return json.loads(cleaned.strip())

    @staticmethod
    def _validate(task: dict, required_task_id: str, domain: dict) -> dict:
        """Ensure all required keys exist and patch task_id / domain if needed."""
        required = {"task_id", "domain", "context", "question", "gold_standard"}
        missing = required - task.keys()
        if missing:
            raise KeyError(f"Task JSON missing keys: {missing}")

        # Normalise: always use the ID we generated, not what the model produced
        task["task_id"] = required_task_id
        task["domain"] = domain["key"]

        for field in ("context", "question", "gold_standard"):
            if not isinstance(task[field], str) or len(task[field].strip()) < 20:
                raise ValueError(f"Field '{field}' is empty or too short")

        return task
