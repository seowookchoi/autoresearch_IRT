# CLAUDE.md — autoresearch_IRT

Persistent context for Claude Code sessions. Update this file when significant decisions change.

---

## 1. Project Overview

This project builds an **automated bio-compliance Item Response Theory (IRT) question bank** for evaluating AI systems (and potentially human experts) on biopharma regulatory knowledge.

The core loop:
1. An LLM **generates** compliance scenarios across 5 regulatory domains (FDA 21 CFR Part 11, GCP deviations, promotional review, GMP, informed consent)
2. Two AI **solver profiles** attempt each scenario — zero-shot "Vanilla" and chain-of-thought "Augmented"
3. An LLM **judge** grades each response PASS or FAIL
4. Items are **filtered** by outcome pattern and assigned Rasch difficulty parameters (`b`)
5. Results persist to **SQLite** for cumulative calibration across runs
6. Once enough items accumulate (configurable threshold), solver **ability (θ) is estimated via MLE**

A parallel **real-world track** ingests actual FDA warning letters and converts them into calibration items, tagged `source_type="real"`. This grounds the synthetic bank in actual regulatory enforcement findings.

**Current bank size (as of last build):**
- Synthetic retained: 78 items
- Real-world retained: 54 items (from 19 FDA warning letters)
- Total calibrated: 139+ items with known b values

---

## 2. Architecture

### Repo layout (flat — single folder)

```
autoresearch_irt/               ← repo root, everything lives here
  main.py                       ← CLI entry point, orchestrates all stages
  task_generator.py             ← Stage 1: LLM generates compliance tasks
  calibrator.py                 ← Stage 2: runs Vanilla + Augmented solvers, grades them
  evaluator.py                  ← LLM-as-judge (strict and lenient modes)
  irt_parameters.py             ← Rasch model math, outcome classification, theta MLE
  database.py                   ← SQLite persistence (tasks + calibration_results tables)
  fda_importer.py               ← Real-world track: fetch FDA warning letters → task dicts
  compliance_bank.db            ← live SQLite database (gitignored... actually tracked)
  build_bank.sh                 ← bash script: runs 15 synthetic pipeline rounds in background
  build_real.sh                 ← bash script: Claude autonomously finds + imports FDA letters
  venv/                         ← Python virtualenv (gitignored)
  .env                          ← GROQ_API_KEY (gitignored)
  CLAUDE.md                     ← this file
```

### Note on repo history
The repo previously had a nested `autoresearch_IRT/` subdirectory inside a TypeScript monorepo scaffold (pnpm, Express, Drizzle). Both were consolidated into this flat structure. The TypeScript scaffold (lib/, artifacts/, scripts/) was removed entirely — it had no domain logic.

### Python module dependencies
```
main.py
  ├── task_generator.py   (generates hard + easy tasks)
  ├── calibrator.py
  │     ├── evaluator.py  (strict or lenient judge)
  │     └── irt_parameters.py
  ├── database.py
  └── fda_importer.py     (print_comparison)

fda_importer.py           (standalone CLI + helper used by main.py)
  ├── calibrator.py
  └── database.py
```

---

## 3. Tech Stack

- **Python 3.12** (Homebrew on macOS — externally managed, requires venv)
- **openai SDK** — pointed at Groq's OpenAI-compatible endpoint, not OpenAI directly
- **Groq API** (`https://api.groq.com/openai/v1`) — model: `llama-3.3-70b-versatile`
- **SQLite** (stdlib `sqlite3`) — no ORM, raw SQL, single persistent connection per `Database` instance
- **venv** at `./venv/` — activate before running anything

### Running the pipeline
```bash
cd /Users/awesomedasom/Desktop/autoresearch_irt
source venv/bin/activate
export $(cat .env | xargs)

python3 main.py --n 5 --easy-n 2 --quiet
python3 fda_importer.py --url <url> --max-items 4
```

### Background build scripts
```bash
# Synthetic (15 rounds of 5 hard + 2 easy tasks)
nohup bash build_bank.sh &

# Real FDA (Claude finds + imports warning letters autonomously)
nohup bash build_real.sh &

# Monitor
tail -f bank_build.log
tail -f bank_build_real.log
```

---

## 4. Key Conventions

### Task dict schema
```python
{
  "task_id":      "TASK-{DOMAIN_KEY}-{8HEX}",         # synthetic hard
                  "TASK-REAL-{DOMAIN[:8]}-{8HEX}",    # real-world
  "domain":       str,         # e.g. "21cfr11", "gcp_deviation", "easy_training"
  "context":      str,
  "question":     str,
  "gold_standard": str,
  "source_type":  "synthetic" | "real",
  "source_ref":   str | None,  # URL for real items
  "is_easy":      bool,        # runtime only, not persisted to DB
}
```

### Domain keys
- **Hard synthetic**: `21cfr11`, `gcp_deviation`, `promo_review`, `gmp_deviation`, `informed_consent`
- **Easy synthetic**: `easy_training`, `easy_consent`, `easy_batch_record`, `easy_irb`, `easy_backup`
- **Real-world**: domain assigned by LLM during extraction, same keys as hard synthetic

### Outcome classification
| Pattern | retention_reason | b assigned? |
|---|---|---|
| V:FAIL, A:PASS | `retained` | Yes, Uniform(0.0, 2.5) |
| V:PASS, A:PASS | `retained_easy` | Yes, Uniform(-2.5, 0.0) |
| V:FAIL, A:FAIL | `discarded_too_hard` | No |
| V:PASS, A:FAIL | `discarded_anomalous` | No |

`is_retained = True` for both `retained` and `retained_easy`.

---

## 5. Important Decisions

### Rasch (1PL) not 2PL
With only 2 solver profiles (θ_vanilla ≈ 0, θ_augmented ≈ 2.5), the discrimination `a` is not identifiable — infinitely many (a, b) pairs fit any binary (FAIL, PASS) observation. Fixed `a = RASCH_A = 1.0` globally. Only `b` varies per item.

**Legacy**: Some early DB rows have `a != 1.0` from a 2PL design phase. Identifiable by `a != 1.0`.

### Two judge modes
- **Strict** (`strict=True`): requires section-level citation (e.g., "21 CFR 11.10(e)"). Used for hard synthetic and all real-world items.
- **Lenient** (`strict=False`): requires only correct conclusion + reasoning. Used for easy synthetic items so Vanilla can pass them.

Without the lenient judge, easy items would still get Vanilla FAIL (it never cites sections), defeating the purpose.

### `is_easy` not persisted
Added in `main.py` before calibration, not stored in DB. Easy-ness is inferred from `retention_reason = 'retained_easy'` or domain key prefix `easy_`.

### Real-world track rationale
Synthetic items are generated by the same LLM that solves them — a closed loop. Real FDA warning letters represent actual enforcement actions. Key finding: real items cluster in `b ∈ [0.5, 2.5]` with zero items below b=0, confirming FDA enforcement doesn't produce easy questions. Mean b gap (real − synthetic) = **+1.14**, statistically meaningful at N=54 real items.

### Theta MLE
Newton-Raphson on the Rasch score equation. Triggers once `--theta-min-items` (default 10) items with known b are in the DB. Current estimates:
- Vanilla θ: **-0.74** (converging from 0.0 prior — makes sense, vanilla fails most hard items)
- Augmented θ: **+6.0** (boundary — passes all retained items by construction)

Augmented θ boundary is expected: retained items are items augmented passes. It will converge once too-hard items get b values (not yet implemented).

### Single persistent SQLite connection
`Database` opens one connection in `__init__` and reuses it. The `with self._connect()` pattern returns the same connection — does NOT open/close per call.

### Schema migration
`Database._migrate()` uses `PRAGMA table_info` to add columns to existing DBs without dropping data. Used to add `source_type` and `source_ref` columns.

---

## 6. What to Avoid

### Don't use system Python
macOS Python 3.12 (Homebrew) is externally managed. Always `source venv/bin/activate` first.

### Don't generate easy items with strict judge
Easy domain items + strict judge = Vanilla always fails = no easy retained items = theta MLE can't converge for vanilla.

### Don't confuse `is_retained` with difficulty band
`is_retained = True` for both hard and easy items. Check `retention_reason` to distinguish.

### Don't use 2PL with 2 solvers
`a` is not identifiable from 2 binary observations. Old DB rows with `a != 1.0` are pre-Rasch legacy.

### Experiment history (from git log)
- Exp 1: Lenient judge → Vanilla passed everything → no hard items
- Exp 2: Counterintuitive hints → improved hard item yield
- Exp 3: Strict citation requirement in judge → current design
- Exp 4: Section-level citation required + simplified promo hints
- Exp 5: Regulatory reference toolkit in augmented prompt → reduced too-hard rate

---

## 7. Current State

### Working
- Full pipeline: generate → calibrate → filter → Rasch b assignment → persist
- Hard domains (5): counterintuitive scenarios, strict judge
- Easy domains (5): obvious scenarios, lenient judge
- Real-world track: `fda_importer.py` fetches FDA warning letters, extracts items, calibrates
- `source_type` tagging: `synthetic` vs `real` in DB
- Real vs synthetic b-distribution comparison (`print_comparison`)
- Theta MLE via Newton-Raphson (vanilla converging, augmented at boundary)
- CLI: `--n`, `--easy-n`, `--db`, `--quiet`, `--theta-min-items`, `--compare`
- Background build scripts: `build_bank.sh`, `build_real.sh`

### Known issues
- Augmented θ hits +6.0 boundary — too-hard items have no b, can't contribute to MLE
- Legacy DB rows with `a != 1.0` from 2PL era
- Promo_review domain underrepresented in real items (OPDP letters use different URL structure)
- `b` values are sampled from Uniform, not empirically estimated — only band membership is data-driven

### Gaps
- No promotional material real items retained yet (0 in real bank)
- No human expert validation of item quality
- No adaptive item selection loop
- No export to standard IRT formats (mirt, py-irt)

---

## 8. Next Steps

### Immediate
1. **Fix augmented theta boundary** — assign b values to too-hard items (`b ~ Uniform(2.5, 5.0)`) so they contribute to MLE response vector
2. **Feed MLE theta back into b assignment** — use updated θ estimates as bounds for `Uniform(theta_van, theta_aug)` instead of hardcoded `(0.0, 2.5)`
3. **Promo_review real items** — OPDP warning letters are at a different URL pattern; need targeted search

### Longer term
- Human expert panel: have real compliance professionals answer a subset → empirical b calibration
- Mann-Whitney U test: formally test real vs synthetic b distributions
- Adaptive item selection: given solver θ, pick item with b closest to θ (maximum information)
- Multi-solver support: arbitrary solver profiles with configurable prompts
- Export to R `mirt` or Python `py-irt` for proper psychometric analysis
