# autoresearch_IRT

Autonomous bio-compliance IRT item bank builder. An AI agent iterates on prompt engineering and pipeline parameters to maximize the quality and yield of a calibrated compliance item bank.

## What it does

The pipeline generates biopharma regulatory compliance scenarios, runs two solver profiles against each item, and retains only items that discriminate between them — hard enough that zero-shot fails, but solvable with structured chain-of-thought reasoning. Retained items receive 2PL IRT parameters (discrimination *a*, difficulty *b*) and are stored in a SQLite item bank.

The autoresearch loop lets an AI agent experiment autonomously: modify prompts, domain hints, or solver scaffolding; run the pipeline; measure the retention rate; keep changes that improve it.

## How it works

- **`task_generator.py`** — generates structured compliance scenarios via Claude API across five domains: FDA 21 CFR Part 11, GCP deviation, promotional material review, GMP deviation, and informed consent.
- **`calibrator.py`** — runs Vanilla (zero-shot) and Augmented (chain-of-thought) solvers, grades both with the Evaluator, applies 2PL filtration: retain only Vanilla-FAIL / Augmented-PASS items.
- **`evaluator.py`** — LLM-as-judge binary grader (PASS / FAIL with rationale).
- **`irt_parameters.py`** — assigns mock 2PL *a* and *b* parameters to retained items.
- **`database.py`** — SQLite persistence; the item bank accumulates across runs.
- **`main.py`** — end-to-end runner. The agent runs this as the experiment entry point.

The ground-truth metric is **retention rate** (retained items / total generated) — higher is better. Each experiment uses a fixed `--n` task count so runs are directly comparable.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), Anthropic API key.

```bash
# Install dependencies
uv sync

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run a single experiment (generates 5 tasks)
uv run python main.py --n 5
```

## Running the agent

Point your agent at `program.md`:

```
Have a look at program.md and kick off a new autoresearch experiment.
```

## Project structure

```
main.py           — pipeline runner (fixed harness, do not modify)
task_generator.py — compliance task generation prompts & domain registry
calibrator.py     — vanilla / augmented solver profiles + filtration
evaluator.py      — LLM-as-judge binary grader
irt_parameters.py — 2PL parameter assignment & IRT math
database.py       — SQLite persistence (fixed harness, do not modify)
program.md        — agent instructions
analysis.ipynb    — experiment result visualization
```

## Design choices

- **Single metric.** Retention rate is the ground truth. Higher = the agent is crafting prompts that land items in the discrimination band between zero-shot and augmented ability.
- **Fixed N.** Each experiment generates the same number of tasks, making runs directly comparable regardless of prompt changes.
- **Item bank accumulates.** The SQLite database persists across runs; retained items from all experiments are preserved.
- **Agent scope.** The agent modifies `task_generator.py`, `calibrator.py`, `evaluator.py`, and `irt_parameters.py`. It does not modify `main.py`, `database.py`, or `program.md`.

## License

MIT
