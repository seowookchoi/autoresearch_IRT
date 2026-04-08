# autoresearch_IRT

This is an experiment to have an LLM agent autonomously improve a bio-compliance IRT item bank by iterating on prompt engineering and pipeline parameters.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr8`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — project overview.
   - `main.py` — fixed pipeline runner. Do not modify.
   - `task_generator.py` — task generation prompts and domain registry. Fair game.
   - `calibrator.py` — vanilla / augmented solver prompts and filtration. Fair game.
   - `evaluator.py` — judge prompts and grading logic. Fair game.
   - `irt_parameters.py` — 2PL parameter assignment. Fair game.
4. **Verify API key**: Check that `GROQ_API_KEY` is set. If not, tell the human to add it to Replit Secrets.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs `main.py` with a fixed task count. Use `--n 5` as the default (adjust if the human has specified otherwise).

**What you CAN do:**
- Modify `task_generator.py` — domain hints, scenario prompt, required JSON structure, domain registry.
- Modify `calibrator.py` — vanilla system prompt, augmented step-by-step prompt, solver model.
- Modify `evaluator.py` — judge system prompt, evaluation criteria, verdict parsing.
- Modify `irt_parameters.py` — 2PL parameter distributions, retention classification logic.

**What you CANNOT do:**
- Modify `main.py` or `database.py`. These are the fixed harness.
- Modify `program.md`.
- Install new packages. Use only what is in `pyproject.toml`.

**The goal: maximize retention rate** (retained / total tasks generated). Higher is better — it means the pipeline is generating items that land in the discrimination band between zero-shot and augmented ability.

**Run command:**
```
uv run python main.py --n 5 > run.log 2>&1
```

**Check results:**
```
grep -E "Tasks generated|Retained|Discarded" run.log
```

This will show the summary:
```
  Tasks generated    : 5
  Calibrations run   : 5
  Retained (hard)    : 2
  Discarded too-easy : 1
  Discarded too-hard : 2
```

Compute `retention_rate = retained / tasks_generated` (e.g. 2/5 = 0.400).

If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	retained	retention_rate	status	description
```

1. git commit hash (short, 7 chars)
2. retained item count (integer, e.g. 2)
3. retention_rate as a decimal (e.g. 0.400)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:
```
commit	retained	retention_rate	status	description
a1b2c3d	2	0.400	keep	baseline
b2c3d4e	3	0.600	keep	stronger augmented CoT prompt
c3d4e5f	1	0.200	discard	simplified vanilla prompt (hurt discrimination)
d4e5f6g	0	0.000	crash	changed domain registry (KeyError)
```

Do not commit `results.tsv` — leave it untracked by git.

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit.
2. Pick an experiment idea (see suggestions below). Edit one or more of the in-scope files.
3. `git commit`
4. Run the experiment: `uv run python main.py --n 5 > run.log 2>&1`
5. Check results: `grep -E "Tasks generated|Retained|Discarded" run.log`
6. If the grep output is empty, the run crashed — read `tail -n 50 run.log`, fix, re-run.
7. Record results in `results.tsv`.
8. If `retention_rate` improved (higher), keep the git commit.
9. If `retention_rate` is equal or worse, `git reset --hard HEAD~1` to revert.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. Keep going until manually interrupted.

## Experiment ideas

- **Domain hints**: Tighten or expand the scenario guidance in `task_generator.py` to produce harder items.
- **Augmented solver**: Add or remove steps in the chain-of-thought prompt in `calibrator.py`.
- **Vanilla solver**: Simplify or complicate the zero-shot framing to widen the discrimination gap.
- **Judge strictness**: Adjust the evaluation criteria in `evaluator.py` — stricter judge → more FAILs.
- **New domains**: Add a new regulatory domain to the domain registry in `task_generator.py`.
- **IRT parameters**: Adjust the difficulty distribution in `irt_parameters.py` to target different θ bands.
