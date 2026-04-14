# CLAUDE.md — autoresearch_IRT

Persistent context for Claude Code sessions. Update this file when significant decisions change.

---
**CORE DIRECTIVE: SELF-DOCUMENTATION**
Whenever you are asked to fix a bug, change a mathematical formula, or alter the architecture of this project, you MUST independently update this `CLAUDE.md` file to reflect the new state of the project before you commit the changes to git. Do not wait for the user to explicitly ask you to update the documentation.
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
| Pattern | retention_reason | b assigned? | is_retained |
|---|---|---|---|
| V:FAIL, A:PASS | `retained` | Yes, soft est. or Uniform(0.0, 2.5) | True |
| V:PASS, A:PASS | `retained_easy` | Yes, soft est. or Uniform(-2.5, 0.0) | True |
| V:FAIL, A:FAIL | `discarded_too_hard` | Yes, soft est. or Uniform(2.5, 5.0) ← Option 2 | False |
| V:PASS, A:FAIL | `discarded_anomalous` | No | False |

`is_retained = True` for both `retained` and `retained_easy` only.
Too-hard items get a b value for theta MLE purposes but remain `is_retained = False` and are excluded from the retained item bank.

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
Newton-Raphson on the Rasch score equation. Triggers once `--theta-min-items` (default 10) items with known b are in the DB.

- Vanilla θ: **-0.74** (converging — vanilla fails most hard items, consistent)
- Augmented θ: previously at +6.0 boundary due to selection bias (all retained items are augmented-pass by definition). **Fixed by Option 2**: too-hard items now get a b value and enter the response vector as augmented failures at high difficulty, giving the MLE finite negative terms. Augmented θ will now converge to a value above 2.5 rather than diverging.

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
- Theta MLE via Newton-Raphson (vanilla converging; augmented now finite via Option 2)
- Soft b estimation via judge logprobs (Option 4): `evaluate()` returns `(passed, reason, p_correct)`; b estimated by Fisher-information-weighted logit inversion across both solvers
- Too-hard items get `b ~ Uniform(2.5, 5.0)` (or soft estimate if logprobs are available); included in theta MLE response vectors
- `p_vanilla`, `p_augmented` columns in `calibration_results` (migrated on existing DB)
- CLI: `--n`, `--easy-n`, `--db`, `--quiet`, `--theta-min-items`, `--compare`
- Background build scripts: `build_bank.sh`, `build_real.sh`

### Known issues
- Legacy DB rows with `a != 1.0` from 2PL era
- Promo_review domain underrepresented in real items (OPDP letters use different URL structure)
- Judge logprobs reflect judge confidence in its verdict, not solver's true P(correct) — the soft b estimation assumes these are equivalent, which is an approximation
- b values still primarily informed by band membership, not a true psychometric estimation

### Gaps
- No promotional material real items retained yet (0 in real bank)
- No human expert validation of item quality or gold standards
- No adaptive item selection loop
- No export to standard IRT formats (mirt, py-irt)
- No statistical test (Mann-Whitney U) on real vs. synthetic b-distribution gap

---

## 8. Honest Assessment (Updated 2026-04-09)

This section documents an objective, critical evaluation of what has been built and what it is not.

### Industry value: 3 / 10 as-is

**Real strengths:**
- Regulatory domains (21 CFR Part 11, GCP, GMP, informed consent) are genuine industry pain points
- FDA warning letter track grounds items in actual enforcement actions
- The pipeline architecture (generate → calibrate → filter → persist) is a reasonable production skeleton

**Critical gaps for industry use:**
- No SME validation: gold standards are LLM-generated and LLM-graded. Any incorrect gold standard poisons the calibration. Regulatory nuance (exceptions, carve-outs, jurisdiction differences) requires expert review.
- The "IRT-calibrated" claim cannot be made to compliance stakeholders — b values lack standard errors, item fit statistics, or cross-validation. This is not psychometric certification.
- 130 items across 5 domains is too sparse. Real compliance exams have 200–500 items per domain.
- No production engineering: local bash scripts, no API, no audit trail, no item security.

### Academic value: 4 / 10 as-is

**What is publishable:**
- The dual-track design (synthetic LLM-generated + real FDA enforcement letters) as a method for grounding automated benchmarks is a legitimate research idea
- The real-vs-synthetic b-distribution gap (+1.14 mean) is worth a paragraph in a paper if supported by a Mann-Whitney U test
- Strict/lenient judge design and its effect on vanilla pass rate is a practical engineering insight

**Critical methodological problems:**
- **Two-solver IRT is not IRT.** IRT calibration requires many test-takers (typically 200–1000+) to estimate item parameters. With two binary observations per item, b is assigned to a Uniform band — this is a classification scheme, not psychometric estimation.
- **b values have no statistical meaning within their band.** Option 4 (logprobs) improves this but introduces a different problem: judge logprob ≠ P(solver correct). `b = θ − logit(P_judge)` is only valid if the judge is a perfectly calibrated Rasch evaluator, which it is not.
- **Closed evaluation loop with no external validity.** Same model family generates, solves, and judges. No measurement of whether the difficulty ranking matches human expert judgments, real exam difficulty, or regulatory enforcement complexity.
- **Selection bias is structural.** Retained items are by construction items the augmented solver passes. Option 2 moves the augmented θ from ±∞ to finite, but the band-sampled b values are not empirical.

**Realistic publication target:**
- Workshop/short paper (not main venue) on automated benchmark construction in regulated domains, framed around the pipeline design and FDA grounding mechanism, with limitations section that explicitly names the above. A full venue paper requires ≥5 solver profiles, human expert grading of a stratified subset, and item fit statistics.

---

## 9. Next Steps (priority-ordered)

### To make academic contribution credible
1. **Add solver profiles** — need ≥5 distinct profiles (different models: GPT-4o, Claude Sonnet, Mistral; different prompting: zero-shot, few-shot, RAG) to have enough calibration data points per item for meaningful b estimation
2. **Human expert review** — have 2–3 compliance professionals label a random sample (50 items) correct/incorrect to validate LLM gold standards and provide external anchor for the difficulty scale
3. **Mann-Whitney U test** — formally test the real vs. synthetic b-distribution difference; currently only a descriptive comparison

### To improve current pipeline
4. **Promo_review real items** — OPDP warning letters are at a different URL pattern; need targeted search
5. **Feed estimated θ back into band bounds** — use MLE θ estimates as Uniform interval endpoints instead of hardcoded (0.0, 2.5)

### Longer term
- Adaptive item selection: given solver θ, pick item with b closest to θ (maximum Fisher information)
- Export to R `mirt` or Python `py-irt` for proper psychometric analysis
- SME item review workflow before any claim of industry use
