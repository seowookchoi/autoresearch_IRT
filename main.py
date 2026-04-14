"""
main.py — End-to-end pipeline runner.

Pipeline
--------
Stage 1  TaskGenerator   →  generate N bio-compliance tasks (structured JSON)
Stage 2  Calibrator      →  run Vanilla + Augmented solvers, grade with Evaluator
Filtration               →  keep only Vanilla-FAIL / Augmented-PASS items
IRT                      →  assign mock 2PL a & b parameters to retained items
Persistence              →  write everything to compliance_bank.db (SQLite)

Usage
-----
    python main.py                # generate 3 tasks (default)
    python main.py --n 5          # generate 5 tasks
    python main.py --db custom.db # use a different database file
    python main.py --quiet        # suppress per-step logs
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

import openai

from calibrator import Calibrator
from database import Database
from fda_importer import print_comparison
from irt_parameters import describe_item, estimate_theta_mle, THETA_VANILLA, THETA_AUGMENTED
from task_generator import TaskGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str, width: int = 64) -> str:
    bar = "═" * width
    return f"\n╔{bar}╗\n║  {text:<{width - 2}}║\n╚{bar}╝"


def _section(text: str) -> str:
    return f"\n{'─' * 64}\n  {text}\n{'─' * 64}"


def _wrap(text: str, indent: int = 4, width: int = 80) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix,
                         subsequent_indent=prefix)


def _print_task_card(task: dict) -> None:
    print(f"\n  Task ID  : {task['task_id']}")
    print(f"  Domain   : {task['domain']}")
    print("  Context  :")
    print(_wrap(task["context"]))
    print("  Question :")
    print(_wrap(task["question"]))
    print("  Gold Std :")
    print(_wrap(task["gold_standard"]))


def _print_result_card(result: dict) -> None:
    reason = result["retention_reason"]
    if reason == "retained":
        status = "RETAINED (hard)"
    elif reason == "retained_easy":
        status = "RETAINED (easy)"
    else:
        status = f"DISCARDED ({reason})"
    print(f"\n  {result['task_id']}  →  {status}")
    print(f"    Vanilla   : {'PASS' if result['vanilla_pass'] else 'FAIL'}"
          f"  |  {result['_vanilla_reason']}")
    print(f"    Augmented : {'PASS' if result['augmented_pass'] else 'FAIL'}"
          f"  |  {result['_augmented_reason']}")
    if result["is_retained"]:
        print(
            f"    IRT params: a={result['irt_a']:.4f}  b={result['irt_b']:.4f}"
        )


def _print_db_summary(db: Database) -> None:
    s = db.fetch_summary()
    print(f"\n  Tasks generated    : {s['total_tasks']}")
    print(f"  Calibrations run   : {s['total_calibrations']}")
    print(f"  Retained (hard)    : {s['retained_hard']}")
    print(f"  Retained (easy)    : {s['retained_easy']}")
    print(f"  Discarded too-hard : {s['discarded_too_hard']}")
    print(f"  Discarded anomalous: {s['discarded_anomalous']}")


def _print_retained_items(db: Database) -> None:
    items = db.fetch_retained()
    if not items:
        print("\n  (no retained items to display)")
        return
    for item in items:
        band = "hard" if item["retention_reason"] == "retained" else "easy"
        vanilla_label   = "PASS" if item["vanilla_pass"]   else "FAIL"
        augmented_label = "PASS" if item["augmented_pass"] else "FAIL"
        print(f"\n  ┌─ {item['task_id']}  [{item['domain']}]  ({band})")
        print(f"  │  b (difficulty)    = {item['irt_b']:.4f}")
        print(f"  │  a (discrimination)= {item['irt_a']:.4f}")
        print(f"  │  Vanilla           : {vanilla_label}")
        print(f"  │  Augmented         : {augmented_label}")
        print(f"  └─ Gold Standard (excerpt):")
        excerpt = item['gold_standard'][:200].replace("\n", " ")
        print(f"     {excerpt}…" if len(item['gold_standard']) > 200 else f"     {excerpt}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(n: int, db_path: str, verbose: bool, theta_min_items: int = 10, n_easy: int = 1, compare: bool = False) -> None:
    print(_banner("Bio-Compliance IRT Pipeline"))

    # -- API client ----------------------------------------------------------
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n  ERROR: GROQ_API_KEY environment variable is not set.")
        print("  Add it to your Replit Secrets and restart.")
        sys.exit(1)

    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )

    # -- Database ------------------------------------------------------------
    db = Database(db_path)
    print(f"\n  Database  : {Path(db_path).resolve()}")

    # ========================================================================
    # STAGE 1 — Task Generation
    # ========================================================================
    print(_section(f"Stage 1 — Generating {n} hard + {n_easy} easy Bio-Compliance Tasks"))
    generator = TaskGenerator(client)

    hard_tasks = generator.generate(n=n)
    easy_tasks = generator.generate_easy(n=n_easy)
    for task in easy_tasks:
        task["is_easy"] = True

    tasks = hard_tasks + easy_tasks

    if not tasks and (n > 0 or n_easy > 0):
        print("\n  No tasks were generated. Check your API key and try again.")
        sys.exit(1)

    print(f"\n  Generated {len(hard_tasks)} hard + {len(easy_tasks)} easy task(s).")
    for task in tasks:
        _print_task_card(task)

    # Persist tasks immediately (before calibration, so they exist even if
    # the calibration crashes halfway through)
    for task in tasks:
        db.insert_task(task)

    # ========================================================================
    # STAGE 2 — Calibration
    # ========================================================================
    print(_section("Stage 2 — Calibration (15-Profile Synthetic Population)"))
    calibrator = Calibrator(client, verbose=verbose)

    results        : list[dict] = []
    retained_count : int        = 0

    for task in tasks:
        result = calibrator.calibrate(task)
        results.append(result)
        db.insert_calibration_result(result)
        if result["is_retained"]:
            retained_count += 1

    # ========================================================================
    # STAGE 3 — Filtration & IRT Report
    # ========================================================================
    print(_section("Stage 3 — Filtration & IRT Parameter Summary"))

    print("\n  Per-item outcome:")
    for result in results:
        _print_result_card(result)

    # ========================================================================
    # DATABASE SUMMARY
    # ========================================================================
    print(_section("Database Summary"))
    _print_db_summary(db)

    if retained_count > 0:
        print(_section("Retained Items (sorted by difficulty b ↓)"))
        _print_retained_items(db)
    else:
        print(
            "\n  No items were retained on this run.\n"
            "  This can happen when the LLM fails all tasks (too hard / underspecified).\n"
            "  Try running with a larger n or adjusting the generation prompts."
        )

    # ========================================================================
    # THETA ESTIMATION (once enough items with known b have accumulated)
    # ========================================================================
    vectors = db.fetch_response_vectors()
    n_items = len(vectors["vanilla"])
    if n_items >= theta_min_items:
        print(_section(f"Solver Ability Estimation (N={n_items} calibrated items)"))
        theta_van = estimate_theta_mle(vectors["vanilla"],   theta_init=THETA_VANILLA)
        theta_aug = estimate_theta_mle(vectors["augmented"], theta_init=THETA_AUGMENTED)
        print(f"\n  Vanilla   θ: prior={THETA_VANILLA:.2f}  →  MLE estimate={theta_van:.4f}")
        print(f"  Augmented θ: prior={THETA_AUGMENTED:.2f}  →  MLE estimate={theta_aug:.4f}")
        print(
            f"\n  (Based on {n_items} items with calibrated b values. "
            f"Estimates replace hardcoded priors once N ≥ {theta_min_items}.)"
        )
    else:
        print(
            f"\n  Theta estimation deferred — need ≥ {theta_min_items} calibrated items "
            f"(currently {n_items}). Using priors: "
            f"θ_vanilla={THETA_VANILLA:.1f}, θ_augmented={THETA_AUGMENTED:.1f}."
        )

    if compare:
        print_comparison(db)

    print(_banner("Pipeline Complete"))


# ---------------------------------------------------------------------------
# Targeted 21 CFR Part 11 generation
# ---------------------------------------------------------------------------

def run_targeted_21cfr11(n: int, db_path: str, verbose: bool) -> None:
    """
    Generate `n` targeted 21 CFR Part 11 items (Hybrid Systems + Legacy
    System Validation sub-domains), calibrate each against the 15-profile
    population, and persist to the DB.
    """
    print(_banner(f"Targeted 21 CFR Part 11 Generation — {n} items"))

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n  ERROR: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    db        = Database(db_path)
    generator = TaskGenerator(client)
    calibrator_inst = Calibrator(client, verbose=verbose)

    print(_section(f"Stage 1 — Generating {n} Targeted 21 CFR Part 11 Items"))
    tasks = generator.generate_targeted_21cfr11(n=n)
    print(f"\n  Generated {len(tasks)} task(s).")
    for task in tasks:
        _print_task_card(task)
        db.insert_task(task)

    print(_section("Stage 2 — Calibration (15-Profile Synthetic Population)"))
    retained_count = 0
    for task in tasks:
        result = calibrator_inst.calibrate(task)
        db.insert_calibration_result(result)
        if result["is_retained"]:
            retained_count += 1

    print(_section("Stage 3 — Results"))
    print(f"\n  Retained {retained_count}/{len(tasks)} targeted 21 CFR Part 11 items.")
    _print_db_summary(db)
    print(_banner("Targeted Generation Complete"))


# ---------------------------------------------------------------------------
# Full-bank recalibration (Rescue Mission)
# ---------------------------------------------------------------------------

def run_recalibration(db_path: str, verbose: bool) -> None:
    """
    Re-evaluate EVERY task in the DB against the current 15-profile population
    using the logit pass-rate b formula and widened retention thresholds
    (P ∈ [0.05, 0.95]).

    For each task:
      1. Delete all existing calibration_results rows for that task.
      2. Run the 15-profile calibrator (respecting is_easy flag from domain key).
      3. Insert a fresh calibration_results row.

    WARNING: This makes ~30 API calls per task. For 1,000+ items expect
    several hours of runtime. Run via `nohup` or `build_bank.sh`-style script.
    """
    print(_banner("Full-Bank Recalibration — All Tasks"))

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n  ERROR: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    db             = Database(db_path)
    calibrator_inst = Calibrator(client, verbose=verbose)

    tasks = db.fetch_all_tasks()
    total = len(tasks)
    print(f"\n  Tasks to recalibrate : {total}")
    print(f"  Retention thresholds  : P ∈ [0.05, 0.95]")
    print(f"  API calls (est.)      : {total * 30:,}")
    print(f"\n  Starting recalibration…\n")

    retained_new = 0
    for i, task in enumerate(tasks, 1):
        # Infer is_easy from domain key (not stored in tasks table)
        task["is_easy"] = task.get("domain", "").startswith("easy_")

        if not verbose:
            print(f"  [{i:>4}/{total}] {task['task_id']}  ", end="", flush=True)

        result = calibrator_inst.calibrate(task)

        # Replace stale calibration data
        deleted = db.delete_calibration_results_for_task(task["task_id"])
        db.insert_calibration_result(result)

        retained = "RETAINED" if result["is_retained"] else "DISCARDED"
        if not verbose:
            print(
                f"P={result['pass_rate']:.3f}  b={result['irt_b']:+.3f}  "
                f"{retained}  (replaced {deleted} old row(s))"
            )
        if result["is_retained"]:
            retained_new += 1

        if i % 50 == 0:
            print(f"\n  ── Checkpoint {i}/{total}: {retained_new} retained so far ──\n")

    print(_banner("Recalibration Complete"))
    print(f"\n  Newly retained items : {retained_new}/{total}")
    _print_db_summary(db)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bio-Compliance IRT Pipeline: generate, calibrate, and store items."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of compliance tasks to generate (default: 3)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="compliance_bank.db",
        help="SQLite database file path (default: compliance_bank.db)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step solver and judge logs",
    )
    parser.add_argument(
        "--theta-min-items",
        type=int,
        default=10,
        help="Minimum calibrated items before estimating solver theta via MLE (default: 10)",
    )
    parser.add_argument(
        "--easy-n",
        type=int,
        default=1,
        help="Number of easy tasks to generate per run (default: 1)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print real vs. synthetic difficulty distribution after pipeline completes",
    )
    parser.add_argument(
        "--gen-21cfr11",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Generate N targeted 21 CFR Part 11 items focusing on Hybrid Systems "
            "and Legacy System Validation. Skips the normal hard/easy generation. "
            "Example: --gen-21cfr11 50"
        ),
    )
    parser.add_argument(
        "--recalibrate-all",
        action="store_true",
        help=(
            "Re-evaluate ALL tasks in the DB with the current 15-profile population "
            "and new retention thresholds (P ∈ [0.05, 0.95]). Replaces old calibration "
            "rows. WARNING: ~30 API calls per task — very long for large banks."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.recalibrate_all:
        run_recalibration(db_path=args.db, verbose=not args.quiet)
    elif args.gen_21cfr11 > 0:
        run_targeted_21cfr11(n=args.gen_21cfr11, db_path=args.db, verbose=not args.quiet)
    else:
        run_pipeline(n=args.n, db_path=args.db, verbose=not args.quiet,
                     theta_min_items=args.theta_min_items, n_easy=args.easy_n,
                     compare=args.compare)
