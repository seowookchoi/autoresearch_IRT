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

import anthropic

from calibrator import Calibrator
from database import Database
from irt_parameters import describe_item
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
    status = "RETAINED" if result["is_retained"] else f"DISCARDED ({result['retention_reason']})"
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
    print(f"  Retained (hard)    : {s['retained']}")
    print(f"  Discarded too-easy : {s['discarded_too_easy']}")
    print(f"  Discarded too-hard : {s['discarded_too_hard']}")


def _print_retained_items(db: Database) -> None:
    items = db.fetch_retained()
    if not items:
        print("\n  (no retained items to display)")
        return
    for item in items:
        print(f"\n  ┌─ {item['task_id']}  [{item['domain']}]")
        print(f"  │  b (difficulty)    = {item['irt_b']:.4f}")
        print(f"  │  a (discrimination)= {item['irt_a']:.4f}")
        print(f"  │  Vanilla           : FAIL")
        print(f"  │  Augmented         : PASS")
        print(f"  └─ Gold Standard (excerpt):")
        excerpt = item['gold_standard'][:200].replace("\n", " ")
        print(f"     {excerpt}…" if len(item['gold_standard']) > 200 else f"     {excerpt}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(n: int, db_path: str, verbose: bool) -> None:
    print(_banner("Bio-Compliance IRT Pipeline"))

    # -- API client ----------------------------------------------------------
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("  Export it before running:  export ANTHROPIC_API_KEY=sk-ant-…")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # -- Database ------------------------------------------------------------
    db = Database(db_path)
    print(f"\n  Database  : {Path(db_path).resolve()}")

    # ========================================================================
    # STAGE 1 — Task Generation
    # ========================================================================
    print(_section(f"Stage 1 — Generating {n} Bio-Compliance Tasks"))
    generator = TaskGenerator(client, verbose=verbose) if hasattr(TaskGenerator, "verbose") \
                else TaskGenerator(client)

    tasks = generator.generate(n=n)

    if not tasks:
        print("\n  No tasks were generated. Check your API key and try again.")
        sys.exit(1)

    print(f"\n  Generated {len(tasks)} task(s).")
    for task in tasks:
        _print_task_card(task)

    # Persist tasks immediately (before calibration, so they exist even if
    # the calibration crashes halfway through)
    for task in tasks:
        db.insert_task(task)

    # ========================================================================
    # STAGE 2 — Calibration
    # ========================================================================
    print(_section("Stage 2 — Calibration (Vanilla vs. Augmented)"))
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
    print(_section("Stage 3 — Filtration & 2PL Parameter Summary"))

    print("\n  Per-item outcome:")
    for result in results:
        _print_result_card(result)

    # ========================================================================
    # DATABASE SUMMARY
    # ========================================================================
    print(_section("Database Summary"))
    _print_db_summary(db)

    if retained_count > 0:
        print(_section("Retained Hard Items (sorted by difficulty b ↓)"))
        _print_retained_items(db)
    else:
        print(
            "\n  No items were retained on this run.\n"
            "  This can happen when the LLM passes all tasks (too easy) or fails all\n"
            "  tasks (too hard / underspecified).  Try running with a larger n or\n"
            "  adjusting the generation prompts."
        )

    print(_banner("Pipeline Complete"))


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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(n=args.n, db_path=args.db, verbose=not args.quiet)
