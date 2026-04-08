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

import anthropic


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAINS = [
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Electronic Records & Electronic Signatures",
        "hint": (
            "Focus on audit trail requirements, system validation, "
            "access controls, and the distinction between open and closed systems. "
            "Include a scenario where an SOP or system configuration creates a subtle "
            "Part 11 gap that a non-expert would miss."
        ),
    },
    {
        "key": "gcp_deviation",
        "label": "GCP — Clinical Trial Protocol Deviations & IRB/IEC Reporting",
        "hint": (
            "Focus on classifying deviations as major vs. minor, the timing of "
            "mandatory reporting to the IRB/IEC, and ICH E6(R2) sponsor obligations. "
            "Include a realistic scenario involving a missed visit window or "
            "incorrect IP dispensation with a borderline classification."
        ),
    },
    {
        "key": "promo_review",
        "label": "FDA Promotional Material Review — Off-Label Promotion & Fair Balance",
        "hint": (
            "Focus on OPDP requirements, the fair-balance rule for brief summary "
            "omission, and the distinction between reminder advertisements and "
            "product-claim ads. Include a nuanced scenario where a social media "
            "post or sales aid is borderline non-compliant."
        ),
    },
    {
        "key": "gmp_deviation",
        "label": "GMP — Manufacturing Deviation Investigation & CAPA",
        "hint": (
            "Focus on 21 CFR 211 out-of-specification (OOS) result investigation "
            "timelines, the Phase I / Phase II investigation structure, and when "
            "a batch must be rejected vs. conditionally released pending investigation. "
            "Include a scenario with an ambiguous root cause."
        ),
    },
    {
        "key": "informed_consent",
        "label": "GCP — Informed Consent Documentation & Re-Consent Requirements",
        "hint": (
            "Focus on 21 CFR 50, ICH E6(R2) Section 4.8, and 45 CFR 46 "
            "requirements for re-consent when protocol amendments occur mid-trial. "
            "Include a scenario where subjects are in a vulnerable population "
            "or where the amendment changes the risk/benefit assessment."
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
- The scenario must require domain expertise; a well-read generalist should fail it.
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
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
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
