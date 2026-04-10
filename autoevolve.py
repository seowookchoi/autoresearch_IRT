"""
autoevolve.py — Iterative augmented-solver improvement loop.

Each iteration:
  1. Load current solver state and item bank statistics
  2. Build a failure report (domain rates, b-band rates, sample failure reasons)
  3. Ask an LLM to self-critique and propose one targeted change to the solver config
  4. Test the proposal on a stratified sample of items from the bank
  5. If delta_theta >= threshold: adopt the config, git commit + push
  6. Generate new calibration items with the current best solver
  7. Log everything to evolve.log and evolve_history.jsonl

What the LLM is allowed to change each iteration:
  - system_prompt
  - user_template (prompt structure, reference toolkit content)
  - model (any Groq-available model)
  - max_tokens

What never changes automatically:
  - Rasch math, judge prompts, database schema, item generation logic

Usage:
  source venv/bin/activate
  export $(cat .env | xargs)
  python3 autoevolve.py --iterations 10 --test-items 15 --min-delta 0.05
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import openai

from calibrator import Calibrator, _load_solver_config
from database import Database
from irt_parameters import estimate_theta_mle, THETA_AUGMENTED


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOLVER_CONFIG_PATH  = Path(__file__).parent / "solver_config.json"
EVOLVE_LOG_PATH     = Path(__file__).parent / "evolve.log"
EVOLVE_HISTORY_PATH = Path(__file__).parent / "evolve_history.jsonl"

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

PROPOSER_MODEL = "llama-3.3-70b-versatile"   # model used to generate proposals
JUDGE_MODEL    = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(EVOLVE_LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def build_failure_report(db: Database) -> dict:
    """
    Analyse the calibration_results table and return a structured failure
    report for the augmented solver.
    """
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT
                t.domain,
                c.retention_reason,
                c.augmented_pass,
                c.irt_b,
                c.p_augmented
            FROM calibration_results c
            JOIN tasks t ON t.task_id = c.task_id
            ORDER BY c.calibrated_at DESC
        """).fetchall()

    # Per-domain augmented failure rate
    domain_counts: dict = {}
    for r in rows:
        d = r["domain"] or "unknown"
        if d not in domain_counts:
            domain_counts[d] = {"total": 0, "aug_fail": 0}
        domain_counts[d]["total"] += 1
        if not r["augmented_pass"]:
            domain_counts[d]["aug_fail"] += 1

    domain_failure_rates = {
        d: round(v["aug_fail"] / v["total"], 3) if v["total"] else 0
        for d, v in domain_counts.items()
        if not d.startswith("easy_")   # exclude easy domains from signal
    }

    # b-band failure rates (only rows with a b value)
    bands = {"(-inf,0)": [0, 0], "[0,1)": [0, 0], "[1,2)": [0, 0],
             "[2,3)": [0, 0], "[3,5)": [0, 0], "[5,inf)": [0, 0]}

    def band(b):
        if b < 0:   return "(-inf,0)"
        if b < 1:   return "[0,1)"
        if b < 2:   return "[1,2)"
        if b < 3:   return "[2,3)"
        if b < 5:   return "[3,5)"
        return "[5,inf)"

    for r in rows:
        if r["irt_b"] is None:
            continue
        b = band(r["irt_b"])
        bands[b][0] += 1
        if not r["augmented_pass"]:
            bands[b][1] += 1

    b_band_failure_rates = {
        k: round(v[1] / v[0], 3) if v[0] else None
        for k, v in bands.items()
    }

    # Sample soft p_augmented scores for failed items (proxy for failure signal)
    failure_reasons = []
    for r in rows:
        if not r["augmented_pass"] and r["p_augmented"] is not None:
            failure_reasons.append(
                f"domain={r['domain']} b={r['irt_b']} p_aug={round(r['p_augmented'], 3)}"
            )
            if len(failure_reasons) >= 20:
                break

    # Augmented theta from current bank
    resp_vectors = db.fetch_response_vectors()
    aug_responses = resp_vectors.get("augmented", [])
    current_theta = estimate_theta_mle(aug_responses, theta_init=THETA_AUGMENTED)

    # Too-hard item count (structural signal)
    with db._connect() as conn:
        n_too_hard = conn.execute(
            "SELECT COUNT(*) FROM calibration_results WHERE retention_reason = 'discarded_too_hard'"
        ).fetchone()[0]
        n_retained = conn.execute(
            "SELECT COUNT(*) FROM calibration_results WHERE is_retained = 1"
        ).fetchone()[0]
        n_total = conn.execute(
            "SELECT COUNT(*) FROM calibration_results"
        ).fetchone()[0]

    return {
        "n_total_items":         n_total,
        "n_retained":            n_retained,
        "n_too_hard":            n_too_hard,
        "augmented_theta":       current_theta,
        "domain_failure_rates":  domain_failure_rates,
        "b_band_failure_rates":  b_band_failure_rates,
        "sample_failure_reasons": failure_reasons[:10],
    }


# ---------------------------------------------------------------------------
# Proposal generation (LLM self-questions)
# ---------------------------------------------------------------------------

_PROPOSER_SYSTEM = """
You are an AI research engineer tasked with improving a biopharma regulatory compliance solver.
The solver answers questions about FDA regulations (21 CFR Part 11, GCP, GMP, informed consent, promotional review).

You will receive:
1. A failure report showing where the current solver underperforms
2. The current solver configuration (system prompt and user template)
3. A history of past proposals and whether they improved performance

Your job: identify the most likely root cause of failures and propose ONE concrete, targeted change.

Rules:
- Propose only ONE change per iteration. Do not rewrite everything.
- The change must be specific and actionable (add a sentence, expand a section, swap a model, add a step).
- Do not change the judge prompts or Rasch math — only the solver config.
- If a past proposal failed, do not repeat it.
- Be self-critical: state what assumption might be wrong about your proposal.

Available Groq models (if suggesting a model swap):
  llama-3.3-70b-versatile, deepseek-r1-distill-llama-70b, qwen-qwq-32b,
  llama-3.1-8b-instant, gemma2-9b-it

Respond in EXACTLY this JSON format (no markdown, no extra text):
{
  "hypothesis": "one sentence describing the root cause of current failures",
  "change_type": "one of: prompt_system | prompt_toolkit | prompt_steps | model_swap | max_tokens",
  "description": "one sentence describing the specific change",
  "self_critique": "one sentence on what could go wrong with this proposal",
  "new_system_prompt": "<full updated system prompt, or null if unchanged>",
  "new_user_template": "<full updated user template, or null if unchanged>",
  "new_model": "<model id, or null if unchanged>",
  "new_max_tokens": <integer, or null if unchanged>
}
"""

_PROPOSER_USER = """
--- FAILURE REPORT ---
{failure_report}

--- CURRENT SOLVER CONFIG ---
Model: {model}
Max tokens: {max_tokens}

System prompt:
{system_prompt}

User template:
{user_template}

--- PROPOSAL HISTORY (most recent first) ---
{history}

Based on the failure report, what is the single most impactful change to make?
"""


def generate_proposal(
    client: openai.OpenAI,
    failure_report: dict,
    current_config: dict,
    history: list[dict],
) -> dict | None:
    """Ask the LLM to self-question and propose one improvement."""
    history_str = "\n".join(
        f"Iter {h['iteration']}: [{h['change_type']}] {h['description']} "
        f"→ delta_theta={h.get('delta_theta', 'N/A')} ({'ADOPTED' if h.get('adopted') else 'REJECTED'})"
        for h in history[-8:]   # last 8 attempts
    ) or "No history yet."

    prompt = _PROPOSER_USER.format(
        failure_report=json.dumps(failure_report, indent=2),
        model=current_config["model"],
        max_tokens=current_config["max_tokens"],
        system_prompt=current_config["system_prompt"],
        user_template=current_config["user_template"],
        history=history_str,
    )

    try:
        msg = client.chat.completions.create(
            model=PROPOSER_MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": _PROPOSER_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = msg.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[:-3]
        return json.loads(raw)
    except Exception as exc:
        log(f"  [Proposer] ERROR generating proposal: {exc}")
        return None


# ---------------------------------------------------------------------------
# Proposal testing
# ---------------------------------------------------------------------------

def build_test_config(proposal: dict, current_config: dict) -> dict:
    """Apply the proposed changes on top of the current config."""
    cfg = dict(current_config)
    if proposal.get("new_system_prompt"):
        cfg["system_prompt"] = proposal["new_system_prompt"]
    if proposal.get("new_user_template"):
        cfg["user_template"] = proposal["new_user_template"]
    if proposal.get("new_model"):
        cfg["model"] = proposal["new_model"]
    if proposal.get("new_max_tokens"):
        cfg["max_tokens"] = int(proposal["new_max_tokens"])
    return cfg


def sample_test_items(db: Database, n: int, seed: int = 42) -> list[dict]:
    """
    Sample N items from the bank, stratified toward the hard zone (b > 1.5)
    where the augmented solver is most likely to fail and improvements show up.
    Also always include any too-hard items so we can track if solver escapes
    the too-hard band.
    """
    all_tasks = db.fetch_all_tasks()
    task_map  = {t["task_id"]: t for t in all_tasks}

    with db._connect() as conn:
        rows = conn.execute("""
            SELECT task_id, irt_b, retention_reason
            FROM calibration_results
            WHERE irt_b IS NOT NULL
            ORDER BY irt_b DESC
        """).fetchall()

    hard_items    = [r for r in rows if r["irt_b"] >= 1.5]
    other_items   = [r for r in rows if r["irt_b"] < 1.5]

    rng = random.Random(seed)
    n_hard  = min(len(hard_items), max(1, int(n * 0.75)))
    n_other = min(len(other_items), n - n_hard)

    selected_rows = rng.sample(hard_items, n_hard) + rng.sample(other_items, n_other)

    items = []
    for r in selected_rows:
        task = task_map.get(r["task_id"])
        if task:
            task = dict(task)
            task["is_easy"] = r["irt_b"] < 0
            items.append(task)
    return items


def test_proposal(
    client:       openai.OpenAI,
    proposed_cfg: dict,
    test_items:   list[dict],
    current_theta: float,
    db:           Database,
    verbose:      bool = True,
) -> tuple[float, float]:
    """
    Run the proposed solver config on test_items, estimate delta_theta.

    Returns (new_theta, delta_theta).
    """
    calibrator = Calibrator(
        client=client,
        solver_model=proposed_cfg["model"],
        judge_model=JUDGE_MODEL,
        verbose=verbose,
        augmented_config=proposed_cfg,
    )

    new_responses: list[tuple[float, bool]] = []

    for task in test_items:
        try:
            result = calibrator.calibrate(task)
            # Use the b value from the DB (we know it), pair with new augmented pass
            with db._connect() as conn:
                row = conn.execute(
                    "SELECT irt_b FROM calibration_results WHERE task_id = ? AND irt_b IS NOT NULL",
                    (task["task_id"],),
                ).fetchone()
            if row:
                new_responses.append((row["irt_b"], result["augmented_pass"]))
        except Exception as exc:
            log(f"    [Test] Error on {task['task_id']}: {exc}")

    if not new_responses:
        return current_theta, 0.0

    new_theta   = estimate_theta_mle(new_responses, theta_init=current_theta)
    delta_theta = round(new_theta - current_theta, 4)
    return new_theta, delta_theta


# ---------------------------------------------------------------------------
# Config persistence and git
# ---------------------------------------------------------------------------

def save_config(cfg: dict, iteration: int, theta: float, description: str) -> None:
    cfg_out = dict(cfg)
    cfg_out["version"]           = iteration
    cfg_out["iteration"]         = iteration
    cfg_out["theta_at_adoption"] = theta
    cfg_out["adopted_at"]        = datetime.now(timezone.utc).isoformat()
    cfg_out["description"]       = description
    with open(SOLVER_CONFIG_PATH, "w") as f:
        json.dump(cfg_out, f, indent=2, ensure_ascii=False)


def append_history(entry: dict) -> None:
    with open(EVOLVE_HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_history() -> list[dict]:
    if not EVOLVE_HISTORY_PATH.exists():
        return []
    history = []
    with open(EVOLVE_HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return history


def git_commit_push(iteration: int, delta_theta: float, description: str) -> bool:
    """Stage changed files, commit, and push to origin/dev. Returns True on success."""
    repo_root = str(Path(__file__).parent)
    files_to_stage = ["solver_config.json", "evolve_history.jsonl", "evolve.log"]

    try:
        # Ensure we are on dev
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root, text=True
        ).strip()
        if branch != "dev":
            log(f"  [Git] WARNING: on branch '{branch}', expected 'dev'. Skipping push.")
            return False

        # Stage files
        for f in files_to_stage:
            path = os.path.join(repo_root, f)
            if os.path.exists(path):
                subprocess.run(["git", "add", path], cwd=repo_root, check=True)

        # Commit
        msg = (
            f"autoevolve iter {iteration:03d}: Δθ=+{delta_theta:.3f} — {description}\n\n"
            f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        )
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=repo_root, capture_output=True, text=True
        )
        if result.returncode != 0:
            if "nothing to commit" in result.stdout + result.stderr:
                log("  [Git] Nothing to commit, skipping push.")
                return True
            log(f"  [Git] Commit failed: {result.stderr.strip()}")
            return False

        # Push
        push = subprocess.run(
            ["git", "push", "origin", "dev"],
            cwd=repo_root, capture_output=True, text=True
        )
        if push.returncode != 0:
            log(f"  [Git] Push failed: {push.stderr.strip()}")
            return False

        log(f"  [Git] Committed and pushed iter {iteration:03d}.")
        return True

    except Exception as exc:
        log(f"  [Git] Exception during commit/push: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def evolve(
    n_iterations:  int   = 10,
    n_test_items:  int   = 15,
    min_delta:     float = 0.05,
    generate_new:  bool  = True,
    n_generate:    int   = 5,
    db_path:       str   = "compliance_bank.db",
    verbose:       bool  = False,
) -> None:

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY not set. Run: export $(cat .env | xargs)")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    db = Database(db_path)

    log("=" * 70)
    log(f"autoevolve started: {n_iterations} iterations, test_items={n_test_items}, min_delta={min_delta}")
    log("=" * 70)

    history = load_history()

    for iteration in range(1, n_iterations + 1):
        log(f"\n{'─'*60}")
        log(f"ITERATION {iteration}/{n_iterations}")
        log(f"{'─'*60}")

        try:
            # ── 1. Current state ─────────────────────────────────────────
            current_config = _load_solver_config()
            failure_report = build_failure_report(db)
            current_theta  = failure_report["augmented_theta"]

            log(f"  Current θ_augmented = {current_theta:.4f}")
            log(f"  Domain failure rates: {failure_report['domain_failure_rates']}")
            log(f"  Too-hard items: {failure_report['n_too_hard']} / {failure_report['n_total_items']}")

            # ── 2. Generate proposal ──────────────────────────────────────
            log("  [Proposer] Generating improvement proposal…")
            proposal = generate_proposal(client, failure_report, current_config, history)

            if proposal is None:
                log("  [Proposer] No valid proposal generated. Skipping iteration.")
                continue

            log(f"  [Proposer] Hypothesis : {proposal.get('hypothesis', '')}")
            log(f"  [Proposer] Change type: {proposal.get('change_type', '')}")
            log(f"  [Proposer] Description: {proposal.get('description', '')}")
            log(f"  [Proposer] Self-critique: {proposal.get('self_critique', '')}")

            # ── 3. Build and test proposed config ─────────────────────────
            proposed_config = build_test_config(proposal, current_config)
            test_items      = sample_test_items(db, n=n_test_items, seed=iteration)

            if not test_items:
                log("  [Test] No test items available. Run main.py first to build the bank.")
                continue

            log(f"  [Test] Testing on {len(test_items)} items…")
            new_theta, delta_theta = test_proposal(
                client, proposed_config, test_items, current_theta, db, verbose=verbose
            )
            log(f"  [Test] θ: {current_theta:.4f} → {new_theta:.4f}  (Δθ = {delta_theta:+.4f})")

            adopted = delta_theta >= min_delta

            # ── 4. Adopt or reject ────────────────────────────────────────
            if adopted:
                log(f"  [Decision] ADOPTED (Δθ={delta_theta:+.4f} ≥ {min_delta})")
                save_config(
                    proposed_config, iteration, new_theta,
                    proposal.get("description", "")
                )
                git_commit_push(iteration, delta_theta, proposal.get("description", ""))
            else:
                log(f"  [Decision] REJECTED (Δθ={delta_theta:+.4f} < {min_delta})")

            # ── 5. Log to history ─────────────────────────────────────────
            entry = {
                "iteration":    iteration,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                "hypothesis":   proposal.get("hypothesis", ""),
                "change_type":  proposal.get("change_type", ""),
                "description":  proposal.get("description", ""),
                "self_critique": proposal.get("self_critique", ""),
                "theta_before": current_theta,
                "theta_after":  new_theta,
                "delta_theta":  delta_theta,
                "adopted":      adopted,
                "n_test_items": len(test_items),
            }
            history.append(entry)
            append_history(entry)

            # ── 6. Generate new bank items with current best solver ────────
            if generate_new:
                log(f"  [Generate] Running main.py to add {n_generate} new items…")
                try:
                    subprocess.run(
                        [sys.executable, "main.py", f"--n={n_generate}", "--easy-n=1", "--quiet"],
                        cwd=str(Path(__file__).parent),
                        timeout=300,
                    )
                except subprocess.TimeoutExpired:
                    log("  [Generate] main.py timed out after 5 min.")
                except Exception as exc:
                    log(f"  [Generate] Error: {exc}")

        except KeyboardInterrupt:
            log("\nInterrupted by user.")
            break
        except Exception as exc:
            log(f"  [ERROR] Iteration {iteration} failed: {exc}")
            log(traceback.format_exc())
            continue

    log("\n" + "=" * 70)
    log("autoevolve complete.")
    adopted_count = sum(1 for h in history if h.get("adopted"))
    if history:
        final_theta = history[-1].get("theta_after", "?")
        first_theta = history[0].get("theta_before", "?")
        log(f"  Adopted {adopted_count}/{len(history)} proposals")
        log(f"  θ trajectory: {first_theta} → {final_theta}")
    log("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative augmented-solver evolution loop")
    parser.add_argument("--iterations",   type=int,   default=10,   help="Number of evolution iterations")
    parser.add_argument("--test-items",   type=int,   default=15,   help="Items sampled per test")
    parser.add_argument("--min-delta",    type=float, default=0.05, help="Min delta_theta to adopt a proposal")
    parser.add_argument("--no-generate",  action="store_true",      help="Skip generating new bank items each iteration")
    parser.add_argument("--n-generate",   type=int,   default=5,    help="New hard items to generate per iteration")
    parser.add_argument("--db",           default="compliance_bank.db")
    parser.add_argument("--verbose",      action="store_true",      help="Print calibration detail per item")
    args = parser.parse_args()

    evolve(
        n_iterations = args.iterations,
        n_test_items = args.test_items,
        min_delta    = args.min_delta,
        generate_new = not args.no_generate,
        n_generate   = args.n_generate,
        db_path      = args.db,
        verbose      = args.verbose,
    )
