# CLAUDE.md — autoresearch_IRT

Persistent context for Claude Code sessions. Update this file when significant decisions change.

---
**CORE DIRECTIVE: SELF-DOCUMENTATION**
Whenever you are asked to fix a bug, change a mathematical formula, or alter the architecture of this project, you MUST independently update this `CLAUDE.md` file to reflect the new state of the project before you commit the changes to git. Do not wait for the user to explicitly ask you to update the documentation.
## 1. Project Overview

This project builds an **automated bio-compliance Item Response Theory (IRT) question bank** for evaluating AI systems (and potentially human experts) on biopharma regulatory knowledge.

The core loop:
1. An LLM **generates** compliance scenarios across 5 regulatory domains (FDA 21 CFR Part 11, GCP deviations, promotional review, GMP, informed consent)
2. **15 synthetic AI solver profiles** attempt each scenario — spanning a spectrum from "general layperson" (temp 0.1–0.7) to "senior FDA auditor" (temp 0.1–0.7) across 5 expertise tiers
3. An LLM **judge** grades each response PASS or FAIL
4. Item difficulty `b` is estimated directly from the empirical **pass rate P** using `b = ln((1−P)/P)` (logit formula, P clipped to [0.01, 0.99])
5. Items are classified by pass-rate thresholds: P < 0.10 → too hard, P > 0.90 → easy, else retained
6. Results persist to **SQLite** for cumulative calibration across runs
7. Once enough items accumulate (configurable threshold), solver **ability (θ) is estimated via MLE**

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
  calibrator.py                 ← Stage 2: runs 15 solver profiles, grades them, computes pass rate → b
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

### Outcome classification (15-profile pass rate)
| Pass rate P | retention_reason | b formula | is_retained |
|---|---|---|---|
| 0.10 ≤ P ≤ 0.90 | `retained` | `ln((1−P)/P)` | True |
| P > 0.90 | `retained_easy` | `ln((1−P)/P)` (negative) | True |
| P < 0.10 | `discarded_too_hard` | `ln((1−P)/P)` (large positive) | False |

`b = ln((1−P_clipped)/P_clipped)` with `P_clipped = clip(P, 0.01, 0.99)`.
`is_retained = True` for both `retained` and `retained_easy` only.
Too-hard items (P < 0.10) still get a b value and enter the θ MLE response vector.

**DB backward compat**: `vanilla_pass`/`vanilla_response`/`p_vanilla` = profile index 0 (layperson_t0.1);
`augmented_pass`/`augmented_response`/`p_augmented` = profile index 14 (fda_auditor_t0.7).

---

## 5. Important Decisions

### Rasch (1PL) with 15-profile population
`a = RASCH_A = 1.0` fixed for all items. `b` is estimated from the empirical pass rate across the 15 synthetic profiles using the population-logit formula:

    b = ln((1 − P) / P)    where P = pass_count / 15, clipped to [0.01, 0.99]

This assumes the 15 profiles are centered at θ = 0 (THETA_POPULATION_MEAN). The formula is derived from the Rasch model: at θ = 0, P(correct) = 1/(1 + exp(b)), so b = −logit(P).

Range: P=0.01 → b ≈ +4.60 (extremely hard); P=0.99 → b ≈ −4.60 (trivially easy).

**Legacy**: Some early DB rows have `a != 1.0` from a 2PL design phase. Identifiable by `a != 1.0`.
**Legacy**: DB rows from the 2-profile era have `vanilla_pass` and `augmented_pass` from the original Vanilla/Augmented solvers. In the 15-profile architecture, these columns are populated from profile index 0 (layperson) and index 14 (expert) respectively for backward compatibility.

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

In the 15-profile architecture, θ MLE is still computed for the layperson (profile 0, stored as "vanilla") and expert (profile 14, stored as "augmented") using the same DB response vectors. Too-hard items (P < 0.10) have large positive b values and enter the MLE vector as failures for both — this prevents augmented θ from diverging to +∞.

- Vanilla θ (profile 0): estimated at ≈ −0.74 (layperson fails most regulatory items)
- Augmented θ (profile 14): converges above 0 now that too-hard items provide finite negative MLE terms

### Single persistent SQLite connection
`Database` opens one connection in `__init__` and reuses it. The `with self._connect()` pattern returns the same connection — does NOT open/close per call.

### Schema migration
`Database._migrate()` uses `PRAGMA table_info` to add columns to existing DBs without dropping data. Used to add `source_type` and `source_ref` columns.

---

## 6. What to Avoid

### Don't use system Python
macOS Python 3.12 (Homebrew) is externally managed. Always `source venv/bin/activate` first.

### Don't generate easy items with strict judge
Easy domain items + strict judge = layperson always fails = no easy retained items = θ MLE can't converge for the low-ability profiles.

### Don't confuse `is_retained` with difficulty band
`is_retained = True` for both hard and easy items. Check `retention_reason` to distinguish.

### Don't mix profile-population b values with legacy band-sampled b values
Pre-refactor DB rows have `b` values sampled from Uniform bands. Post-refactor rows have `b = ln((1−P)/P)` from pass rate. Identifiable: old rows also have `a != 1.0` (2PL era) or may have `b` falling exactly at Uniform boundaries. Do not average them without acknowledging the different estimation methods.

### Experiment history (from git log)
- Exp 1: Lenient judge → Vanilla passed everything → no hard items
- Exp 2: Counterintuitive hints → improved hard item yield
- Exp 3: Strict citation requirement in judge → current design
- Exp 4: Section-level citation required + simplified promo hints
- Exp 5: Regulatory reference toolkit in augmented prompt → reduced too-hard rate
- Exp 6 (this refactor): 15-profile synthetic population + logit pass-rate b formula

---

## 7. Current State

### Working
- Full pipeline: generate → calibrate (15 profiles) → filter by pass rate → logit b → persist
- Hard domains (5): counterintuitive scenarios, strict judge
- Easy domains (5): obvious scenarios, lenient judge
- 15-profile synthetic population: 5 expertise tiers × 3 temperatures (0.1–0.8)
- b estimation: `b = ln((1−P)/P)` from empirical pass rate P, P clipped to [0.01, 0.99]
- Retention thresholds: P < 0.10 → discarded_too_hard, P > 0.90 → retained_easy, else retained
- DB backward compat: profile 0 (layperson) → vanilla_*, profile 14 (expert) → augmented_*
- Real-world track: `fda_importer.py` fetches FDA warning letters, extracts items, calibrates
- `source_type` tagging: `synthetic` vs `real` in DB
- Real vs synthetic b-distribution comparison (`print_comparison`)
- Theta MLE via Newton-Raphson for layperson (≈ −0.74) and expert profiles
- CLI: `--n`, `--easy-n`, `--db`, `--quiet`, `--theta-min-items`, `--compare`
- Background build scripts: `build_bank.sh`, `build_real.sh`

### Known issues
- Legacy DB rows with `a != 1.0` from 2PL era; legacy rows also have band-sampled b, not logit b
- Promo_review domain underrepresented in real items (OPDP letters use different URL structure)
- 15 solver calls + 15 judge calls = 30 API calls per task — significantly slower than the 4-call 2-profile design; mitigated by Groq's fast inference
- fda_importer.py still uses the 2-profile Calibrator interface (backward compat shim in place)

### Gaps
- No promotional material real items retained yet (0 in real bank)
- No human expert validation of item quality or gold standards
- No adaptive item selection loop
- No export to standard IRT formats (mirt, py-irt)
- No statistical test (Mann-Whitney U) on real vs. synthetic b-distribution gap

---

## 8. Honest Assessment (Updated 2026-04-13)

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
- **15-profile IRT is better but still not psychometric IRT.** IRT calibration requires many independent test-takers (typically 200–1000+). 15 profiles all running the same base model with different system prompts and temperatures are not independent — they are highly correlated. b values from this procedure are more informative than 2-profile band sampling but lack standard errors or item fit statistics.
- **The logit b formula assumes profiles are centered at θ=0.** This is an untested assumption. If the 15 profiles systematically lean expert (mean θ > 0), b will be biased downward (items appear easier than they are).
- **Closed evaluation loop with no external validity.** Same model family generates, solves, and judges. No measurement of whether the difficulty ranking matches human expert judgments, real exam difficulty, or regulatory enforcement complexity.
- **Selection bias is reduced but not eliminated.** With 15 profiles, too-hard items (P < 0.10) are retained in the θ MLE vector, reducing augmented θ divergence. But items still must have P > 0 to enter the calibrated bank.

**Realistic publication target:**
- Workshop/short paper (not main venue) on automated benchmark construction in regulated domains, framed around the pipeline design and FDA grounding mechanism, with limitations section that explicitly names the above. A full venue paper now requires human expert grading of a stratified subset and item fit statistics (profile-to-profile consistency).

---

## 9. Next Steps (priority-ordered)

### To make academic contribution credible
1. ~~**Add solver profiles**~~ ✓ Done — 15 synthetic profiles spanning layperson → senior FDA auditor
2. **Human expert review** — have 2–3 compliance professionals label a random sample (50 items) correct/incorrect to validate LLM gold standards and provide external anchor for the difficulty scale
3. **Mann-Whitney U test** — formally test the real vs. synthetic b-distribution difference; currently only a descriptive comparison
4. **Cross-model profiles** — add profiles using different base models (not just llama-3.3-70b) to reduce within-population correlation and improve b estimation reliability

### To improve current pipeline
5. **Promo_review real items** — OPDP warning letters are at a different URL pattern; need targeted search
6. **Re-calibrate existing DB items** — legacy items have band-sampled b values; re-run calibration with 15-profile logit formula for consistency

### Longer term
- Adaptive item selection: given solver θ, pick item with b closest to θ (maximum Fisher information)
- Export to R `mirt` or Python `py-irt` for proper psychometric analysis
- SME item review workflow before any claim of industry use
