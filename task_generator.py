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
            "Create a scenario focused specifically on audit trail requirements under "
            "21 CFR 11.10(e). The system has time-stamped audit trails but a specific "
            "gap in either: (1) review of audit trails — e.g., SOP does not require "
            "periodic review of audit trails, only their generation; OR (2) the audit "
            "trail does not capture all required metadata (e.g., misses system-initiated "
            "changes, or only captures user ID but not reason for change). "
            "The compliance gap must be non-obvious: a general professional assumes "
            "'has audit trails = compliant with 11.10(e)' but the specific obligation "
            "under 11.10(e) requires both generation AND review AND completeness. "
            "The gold standard must cite 21 CFR 11.10(e) specifically."
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


EASY_DOMAINS = [
    {
        "key": "easy_training",
        "label": "GMP — Basic Personnel Training & Qualification Requirements",
        "hint": (
            "Create a scenario where the compliance answer is OBVIOUS to any life "
            "sciences professional: a new employee or analyst begins performing GMP "
            "tasks without documented training. The question asks whether this is "
            "compliant. The correct answer (non-compliant) should be immediately "
            "apparent to any professional — cite 21 CFR 211.25(a). "
            "Do NOT add any counterintuitive twist. The scenario should be "
            "straightforward and the gold standard citation well-known."
        ),
    },
    {
        "key": "easy_consent",
        "label": "GCP — Basic Written Informed Consent Requirement",
        "hint": (
            "Create a scenario where a clinical site enrolls a participant without "
            "first obtaining signed written informed consent. The question asks whether "
            "this is permissible. The answer (no, it is not) should be obvious to any "
            "clinical professional — cite 21 CFR 50.25(a). "
            "Do NOT add any counterintuitive twist or exception. Keep it simple and clear."
        ),
    },
    {
        "key": "easy_batch_record",
        "label": "GMP — Basic Batch Production Record Requirement",
        "hint": (
            "Create a scenario where a manufacturer fails to complete a batch production "
            "record for a released product. The question asks whether this meets GMP "
            "requirements. The answer (no) is immediately clear to any GMP professional "
            "— cite 21 CFR 211.188. "
            "No counterintuitive twist. The scenario should feel routine and unambiguous."
        ),
    },
    {
        "key": "easy_irb",
        "label": "GCP — Basic IRB Approval Requirement Before Study Initiation",
        "hint": (
            "Create a scenario where a study site begins enrolling participants before "
            "receiving written IRB approval. The question asks if this is permitted. "
            "The answer (no) should be obvious — cite 21 CFR 56.108 or 45 CFR 46.109. "
            "No exceptions or ambiguity. The scenario should be completely unambiguous."
        ),
    },
    {
        "key": "easy_backup",
        "label": "FDA 21 CFR Part 11 — Basic Electronic Record Backup Requirement",
        "hint": (
            "Create a scenario where a company has no procedure for backing up its "
            "electronic records or copies are not readily retrievable. The question asks "
            "whether this meets 21 CFR Part 11 requirements. The answer (non-compliant) "
            "is well-known — cite 21 CFR 11.10(c). "
            "Keep it straightforward. No counterintuitive twist."
        ),
    },
]


# ---------------------------------------------------------------------------
# Targeted 21 CFR Part 11 sub-domains (Hybrid Systems + Legacy Validation)
# 10 entries; generate_targeted_21cfr11() cycles through them.
# ---------------------------------------------------------------------------

DOMAINS_21CFR11_TARGETED: list[dict] = [
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Hybrid Systems: Predicate Record Designation",
        "hint": (
            "Create a scenario where a company uses BOTH paper and electronic records "
            "for the same data (a 'hybrid system') but has never formally designated which "
            "record is the official 'predicate record' under 21 CFR 11.2(a). "
            "Red Herring: the company has both records and believes having two copies "
            "provides extra compliance coverage — an auditor reviewing the records sees "
            "a complete paper binder AND a validated LIMS with matching data. "
            "Actual gap: 21 CFR 11.2(a) requires a formal SOP designation of which "
            "record is the predicate; without it, there is no regulatory basis for relying "
            "on the electronic record as the official record, even if the data matches. "
            "Gold standard must cite 21 CFR 11.2(a) and the predicate rule designation "
            "requirement. Explain that data match does not substitute for formal designation."
        ),
    },
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Legacy System: Post-1997 Modifications and Re-validation",
        "hint": (
            "Create a scenario where a system was deployed before August 1997 (the 21 CFR "
            "Part 11 effective date) and the company believes it is exempt from Part 11 "
            "under the legacy exemption — BUT the system has since undergone three major "
            "software patches and an operating system migration, all post-1997. "
            "Red Herring: the original system installation predates Part 11, and the "
            "company has a thick binder of IQ/OQ/PQ documentation from 1996. "
            "Actual gap: the FDA's 2003 Part 11 guidance states that substantial "
            "modifications to a legacy system after the effective date bring it under "
            "Part 11 scope; the original pre-date status does not carry forward through "
            "material changes. Each post-1997 upgrade required a fresh impact assessment. "
            "Gold standard must cite FDA's 2003 Part 11 Guidance (Section III, 'Scope') "
            "and 21 CFR 11.10(a) continuous validation obligation."
        ),
    },
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Audit Trail: System-Initiated Changes Not Captured",
        "hint": (
            "Create a scenario where a validated LIMS captures audit trails for all "
            "operator-initiated entries but routes all automated batch import jobs and "
            "scheduled calculation updates through a service account that bypasses the "
            "audit trigger — so system-initiated changes are invisible in the log. "
            "Red Herring: the audit trail is voluminous (thousands of entries), a recent "
            "internal audit reviewed a 200-record sample and found zero gaps, and the "
            "system was validated for audit trail completeness at deployment. "
            "Actual gap: 21 CFR 11.10(e) requires audit trails to capture ALL changes to "
            "electronic records, not only operator-initiated ones; system-initiated changes "
            "from batch jobs are within scope. "
            "Gold standard must cite 21 CFR 11.10(e) and state that the obligation covers "
            "automated/system-initiated changes, not only manual operator entries."
        ),
    },
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Electronic Signature: Missing Meaning Component",
        "hint": (
            "Create a scenario where a validated system captures a login credential event "
            "as the signature for document approval, but the signature record does not "
            "display the human-readable meaning (e.g., 'reviewed and approved', 'authored') "
            "alongside the signer's name and timestamp, as required by 21 CFR 11.50(a)(2). "
            "Red Herring: every approved record has a timestamp and the approver's user ID "
            "clearly linked. An auditor traces each record to its approver with certainty. "
            "Actual gap: 21 CFR 11.50(a) requires each electronic signature to include "
            "(1) printed name, (2) date/time, and (3) the MEANING of the signature — all "
            "displayed on the record. The absence of the meaning component is a direct "
            "Part 11 violation even when the timestamp and name are present. "
            "Gold standard must cite 21 CFR 11.50(a)(1)-(3) and name all three required "
            "components."
        ),
    },
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Open System: Encryption and Integrity Controls",
        "hint": (
            "Create a scenario where a company stores validated electronic records in a "
            "cloud SaaS platform (an 'open system' under 21 CFR 11.3(b)(9)) that uses "
            "TLS in transit and SOC 2 Type II access controls, but does NOT encrypt "
            "records at rest and has no cryptographic mechanism to detect unauthorized "
            "record alteration. "
            "Red Herring: the vendor holds SOC 2 Type II, ISO 27001, and HIPAA certifications "
            "and all data transmissions are TLS-encrypted — an IT security review marks the "
            "system as 'fully secured'. "
            "Actual gap: 21 CFR 11.30 requires open systems to employ additional controls "
            "including document encryption and digital signatures to ensure record "
            "authenticity and integrity. Transport encryption (TLS) satisfies neither; "
            "third-party security certifications do not substitute for CFR compliance. "
            "Gold standard must cite 21 CFR 11.30 and distinguish closed-system (11.10) "
            "from open-system (11.30) requirements."
        ),
    },
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Hybrid System: Wet-Ink Signature on Printed Electronic Record",
        "hint": (
            "Create a scenario where a company captures data electronically, prints a "
            "formatted report, obtains a handwritten signature on the printout, then "
            "scans the signed page back into the EDMS. The company treats the scanned "
            "image as an equivalent to a handwritten signature on a paper record. "
            "Red Herring: every finalized record has a visible ink signature and a scan "
            "date stamp. An auditor confirms every batch record was 'signed'. "
            "Actual gap: a scanned image of a handwritten signature attached to an "
            "electronic record constitutes an 'electronic signature' under 21 CFR "
            "11.3(b)(7) — it must therefore satisfy all Part 11 e-signature requirements "
            "(11.50 for displayed attributes, 11.70 for cryptographic linkage to the record). "
            "A JPEG scan cannot meet 11.70 linkage requirements. If the intent is paper "
            "as the predicate record, the paper original must be retained as the official "
            "record and the EDMS copy treated as a copy only. "
            "Gold standard must cite 21 CFR 11.3(b)(7) and 11.70."
        ),
    },
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — CSV Change Control: OS Patch Without Impact Assessment",
        "hint": (
            "Create a scenario where IT applies a 'security-only' OS patch to the server "
            "hosting a validated EDMS. IT classifies the patch as 'low risk, no functional "
            "change'. Post-patch, all user acceptance tests pass and no functionality is "
            "altered. QA closes the event as 'validated state unaffected — no re-validation "
            "required.' No formal impact assessment or change control record is generated. "
            "Red Herring: the patch changes nothing the user sees; all smoke tests pass "
            "and the vendor's release notes confirm 'no application changes'. "
            "Actual gap: 21 CFR 11.10(a) and FDA's 2003 Part 11 guidance require that ANY "
            "change to a validated system — including infrastructure changes — undergo a "
            "documented impact assessment. The DECISION not to re-validate must be "
            "documented and justified; absent that record, the validation status is "
            "unsubstantiated regardless of the outcome. "
            "Gold standard must cite 21 CFR 11.10(a) and FDA's 2003 guidance Section IV."
        ),
    },
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Data Migration: Bulk Audit Entry vs. Field-Level Traceability",
        "hint": (
            "Create a scenario where historical data is migrated from a legacy system to "
            "a new LIMS. The migration reformats date fields, normalizes units, and maps "
            "old column names to new schema. The audit trail records a single entry: "
            "'Batch migration from LegacyLIMS v2.3 on [date]' per task — not a field-level "
            "record of original vs. new value for each transformed field. "
            "Red Herring: post-migration reconciliation scripts verified record counts, "
            "checksums, and a 5% random sample spot-check — all matching. The QA lead "
            "certifies the migration as 'validated and fully traceable.' "
            "Actual gap: 21 CFR 11.10(e) requires audit trail entries to capture the "
            "original value, the new value, who made the change, when, and why — at the "
            "individual record/field level. A bulk migration log entry does not allow "
            "reconstruction of what any individual field contained before migration. "
            "Gold standard must cite 21 CFR 11.10(e) and specify the field-level granularity "
            "requirement for audit trail entries."
        ),
    },
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Predicate Rule Primacy: GLP Archive Accessibility (21 CFR 58)",
        "hint": (
            "Create a scenario where a GLP laboratory has a fully Part 11-compliant "
            "electronic study management system, but all raw data is archived only as "
            "proprietary binary files (.esm format) within the active LIMS. The vendor "
            "contract includes a 5-year support window ending next year. No plan exists "
            "to export data to a human-readable or open format before decommissioning. "
            "Red Herring: the system has robust Part 11 controls — validated backup, "
            "access controls, audit trails, and 21 CFR 11.10(c) backup procedures all "
            "in place. A compliance reviewer concludes the system is fully compliant. "
            "Actual gap: 21 CFR 58.130(e) (the GLP predicate rule) requires that raw "
            "data remain readily retrievable for the retention period — including AFTER "
            "the originating system is decommissioned. Part 11 compliance does not "
            "satisfy the predicate rule's archive accessibility obligation. "
            "Gold standard must cite 21 CFR 58.130(e) as the controlling predicate rule "
            "and explain that Part 11 does not substitute for predicate rule requirements."
        ),
    },
    {
        "key": "21cfr11",
        "label": "FDA 21 CFR Part 11 — Hybrid System: Access Control Gaps for Post-Validation Roles",
        "hint": (
            "Create a scenario where a system was validated in 2018 with access roles "
            "for QA Analyst, QA Manager, and Administrator. Since then, two new roles "
            "have been added operationally (Regulatory Liaison and External Auditor Read-Only) "
            "by modifying the system configuration — but no formal validation addendum, "
            "role-based access re-validation, or updated User Requirements Specification "
            "was generated for these additions. "
            "Red Herring: all five roles are actively enforced in the system and the "
            "External Auditor role is genuinely read-only; an IT security review confirms "
            "no privilege escalation. The system behaves as intended. "
            "Actual gap: adding new user roles to a validated system is a system change "
            "under 21 CFR 11.10(a) and (d) that requires formal change control and "
            "validation addendum — the correct behavior of the roles does not validate "
            "the change; the absence of documented validation for the new roles means "
            "access controls for those roles are unvalidated per Part 11. "
            "Gold standard must cite 21 CFR 11.10(a) and 11.10(d) (access privilege controls)."
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
  "context": "<4-6 sentences. A specific, realistic situation at a named fictional company (e.g., NovaBio Inc., Helix Pharma). Include concrete details: system names, software versions, dates, staff roles, quantities. MUST include at least one RED HERRING — a process or element that appears suspicious or non-compliant on the surface but is actually fully compliant (e.g., a legacy system running correctly in read-only mode, a delayed but permissible timestamp, a deviation log entry that was properly handled). The actual compliance gap must be a SUBTLE PROCEDURAL LAPSE, SIGNATURE TIMING ISSUE, or SYSTEM-TO-SYSTEM DATA INTEGRITY GAP that is easy to overlook when the red herring draws attention elsewhere.>",
  "question": "<One precise question that has a single defensible correct answer grounded in regulation. Avoid opinion questions.>",
  "gold_standard": "<The exact compliance requirement that constitutes the correct answer. MUST cite the specific regulation section(s) (e.g., 21 CFR 11.10(e), ICH E6(R2) §4.8.2) and state the precise obligation or prohibited action. Explain briefly why the Red Herring is NOT the violation.>"
}}

Rules:
- Respond with ONLY valid JSON — no markdown fences, no commentary.
- The gold_standard must be verifiable against published regulations.
- HIGH COMPLEXITY REQUIREMENT — Red Herring: embed at least one compliant process that looks non-compliant. The solver should be pulled toward incorrectly flagging the Red Herring as the violation.
- HIGH COMPLEXITY REQUIREMENT — Subtle Violation: the actual violation must be a procedural lapse, signature timing issue, or system-to-system data integrity gap. NOT a blatant, obvious non-compliance.
- CRITICAL: The common-sense or obvious answer must be WRONG. A knowledgeable professional without specific regulatory training should confidently identify the Red Herring as the problem and miss the actual subtle violation.
- The correct answer must hinge on a specific regulatory provision, exception, or threshold that only deep domain expertise reveals.
- Do NOT include the answer anywhere in the context or question fields.
"""

_USER_TEMPLATE_EASY = """
Generate ONE biopharma compliance scenario as a single JSON object.

Domain: {domain_label}
Guidance: {domain_hint}

Required JSON structure (use EXACTLY these keys, no extras):
{{
  "task_id": "{task_id}",
  "domain": "{domain_key}",
  "context": "<3-5 sentences. A specific, realistic situation at a named fictional company. Include concrete details: roles, dates, systems. The compliance issue should be clear and unambiguous.>",
  "question": "<One precise question with an obvious, well-established correct answer.>",
  "gold_standard": "<The correct answer. Cite the specific regulation section (e.g., 21 CFR 211.25(a), 21 CFR 50.25(a)) and state the basic obligation clearly.>"
}}

Rules:
- Respond with ONLY valid JSON — no markdown fences, no commentary.
- The gold_standard must be verifiable against published regulations.
- The scenario must be STRAIGHTFORWARD. Any life sciences professional should immediately know the correct answer.
- Do NOT add counterintuitive twists, exceptions, or ambiguity.
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
        Generate `n` hard compliance tasks, cycling through DOMAINS.
        Returns a list of task dicts; malformed responses are skipped.
        """
        tasks: list[dict] = []
        for i in range(n):
            domain = DOMAINS[i % len(DOMAINS)]
            task = self._generate_one(domain, attempt=0, easy=False)
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

    def generate_targeted_21cfr11(self, n: int = 50) -> list[dict]:
        """
        Generate `n` targeted 21 CFR Part 11 items cycling through
        DOMAINS_21CFR11_TARGETED (Hybrid Systems + Legacy System Validation).
        All items use the High-Complexity hard template with Red Herring mandate.
        """
        tasks: list[dict] = []
        for i in range(n):
            domain = DOMAINS_21CFR11_TARGETED[i % len(DOMAINS_21CFR11_TARGETED)]
            task = self._generate_one(domain, attempt=0, easy=False)
            if task is not None:
                tasks.append(task)
                print(
                    f"  [TaskGenerator] [{i + 1}/{n}] Generated 21CFR11 targeted task "
                    f"{task['task_id']}  ({domain['label'][-50:]})"
                )
            else:
                print(
                    f"  [TaskGenerator] WARNING: failed to generate 21CFR11 targeted task "
                    f"for sub-domain '{domain['label'][-50:]}' after "
                    f"{self.max_retries + 1} attempts"
                )
        return tasks

    def generate_easy(self, n: int = 2) -> list[dict]:
        """
        Generate `n` easy compliance tasks, cycling through EASY_DOMAINS.
        These use a straightforward prompt (no counterintuitive twist) so that
        both the Vanilla and Augmented solvers are expected to pass.
        """
        tasks: list[dict] = []
        for i in range(n):
            domain = EASY_DOMAINS[i % len(EASY_DOMAINS)]
            task = self._generate_one(domain, attempt=0, easy=True)
            if task is not None:
                tasks.append(task)
                print(
                    f"  [TaskGenerator] Generated easy task {task['task_id']} "
                    f"(domain: {domain['key']})"
                )
            else:
                print(
                    f"  [TaskGenerator] WARNING: failed to generate easy task "
                    f"for domain '{domain['key']}' after {self.max_retries + 1} attempts"
                )
        return tasks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_one(self, domain: dict, attempt: int, easy: bool = False) -> Optional[dict]:
        """Generate a single task, retrying on parse errors."""
        task_id = f"TASK-{domain['key'].upper()}-{uuid.uuid4().hex[:8].upper()}"
        template = _USER_TEMPLATE_EASY if easy else _USER_TEMPLATE
        prompt = template.format(
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
                return self._generate_one(domain, attempt + 1, easy=easy)
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
