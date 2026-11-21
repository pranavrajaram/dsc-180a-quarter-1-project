# User suggestions and language-model accuracy

We study whether models abandon correct mathematics answers when users suggest a
wrong value, and whether correct suggestions help repair errors. The main study
contains **200 problems × two models × three repetitions**, with three independent
follow-ups per first answer: **1,200 first answers + 3,600 follow-ups**.

**Start with [the results notebook](notebooks/results.ipynb).** It includes the
experimental setup, charts, scoring coverage, paired effects, confidence intervals,
and p-values. Saved outputs are visible directly on GitHub.

## Reproduce the results

Python **3.11 or newer** and Git are sufficient for the analysis. No model download,
API key, GPU, or third-party Python package is needed. Run from the repository root:

```sh
git clone https://github.com/pranavrajaram/dsc-180a-quarter-1-project.git
cd dsc-180a-quarter-1-project
python3 -m scripts.challenge analyze
```

**All necessary data is included.** Keep `data/` and `results/` in their existing
locations. The command reads `results/challenge_scale/raw.jsonl` and rebuilds:

| File in `results/challenge_scale/` | Contents |
|---|---|
| `baseline.jsonl` | All first answers, exact requests, task keys, and generation settings |
| `raw.jsonl` | All follow-ups, parent conversations, model versions, and collection metadata |
| `transitions.csv` | Per-response scores, extracted values, and extraction evidence |
| `summary.csv` | Counts, denominators, rates, and 95% intervals |
| `effects.csv` | Paired differences from no suggestion, primary p-values, and Holm correction |

Reanalysis overwrites only the three CSVs. It does not modify model responses.
Rates and effects are fractions: multiply by 100 for percentages or percentage
points. The internal condition `recheck` means **check again with no suggested answer**.

### Run the notebook

Create an environment and install the pinned notebook dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r notebooks/requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Open `notebooks/results.ipynb`
in a notebook editor such as VS Code, select this environment, and **Run All**.
The notebook works from the repository root or `notebooks/`; it reads saved data
and recomputes both scoring methods without writing result files.

## Main findings

Among initially correct answers to part (a):

| Model | No suggestion: designated wrong value | Wrong suggestion: designated wrong value | Paired increase, pp [95% interval] | Holm p |
|---|---:|---:|---:|---:|
| 1.5B | 0/265 (0.0%) | 85/265 (32.1%) | +32.1 [25.9, 38.1] | 0.00000763 |
| 3B | 9/295 (3.1%) | 144/295 (48.8%) | +45.8 [35.6, 55.7] | 0.00000763 |

Both models adopted the wrong value more often when the user supplied it.
Correct suggestions also improved repair rates; the notebook reports both outcomes.
Each within-model comparison uses the same starting answers across branches.
The two models have different eligible subsets, so their rates alone do not
establish a model-size effect.

### Scoring and statistical analysis

`scripts/math_answers.py` extracts numeric expressions and final conclusions within
labeled problem parts without consulting the answer key. It handles LaTeX fractions,
powers, bounded arithmetic, factorials, and binomial coefficients. Explicit final
answer lines take precedence over worked sections. Conflicting final values and
unsupported expressions remain unresolved; no model-generated code is executed.

Improved extraction leaves **63/600** first answers unresolved for 1.5B and
**22/600** for 3B, versus 250 and 108 under the original strict format requirement.
Unresolved does not mean blank. These first answers enter neither conditional
switch nor repair rates. Unresolved follow-ups stay in the relevant denominators
without receiving correctness or wrong-value-match credit. The notebook also shows
follow-up coverage and compares both scoring methods on the same responses.

The extraction revision followed discovery of formatting exclusions. No manual
annotations alter the current study. Regression tests and AI-assisted text checks
support the implementation; independent human grading has not been completed.

Intervals use 10,000 bootstrap samples of **20 whole problem families**, retaining
all ten variants, three repetitions, and paired branches together. Primary p-values
use exact two-sided sign-flip tests on family net switch counts, assuming independent
families and sign symmetry under the null. Holm correction covers the two primary
model-specific wrong-suggestion tests. All 12 first-turn and 21 follow-up responses
that reached the output limit remain included.

To reproduce original strict scoring, use the notebook or run:

```sh
python3 -m scripts.challenge analyze --scoring strict
python3 -m scripts.challenge analyze  # restore improved-scoring CSVs
```

## Earlier tone and hint experiments

These remain as supporting research for the paper:

| Study | Data | Reproduction command |
|---|---|---|
| Seven prompt conditions, coding and mathematics; 560 responses | `results/full_run/` | `python3 -m scripts.experiment analyze --output results/full_run` |
| Paired tone comparisons and adjusted p-values | Same saved responses | `python3 -m scripts.study_analysis` |
| Two model sizes, three tones, with/without wrong hints; 480 responses | `results/model_hint/` | `python3 -m scripts.followup analyze` |

The superseded 240-follow-up pilot is available in
[Git history](https://github.com/pranavrajaram/dsc-180a-quarter-1-project/tree/da09089/results/challenge).
It is not needed to reproduce the current study. `report/report.tex` is the retained
checkpoint source for the earlier experiments; it has not yet been updated into
the final paper. The outdated compiled checkpoint PDF has been removed.

## Collect fresh model responses

The recorded runs used an Apple Silicon Mac with 16 GB RAM, Ollama **0.6.8**, and
Qwen2.5-Coder **1.5B/3B Q4_K_M** checkpoints. The configuration pins model digests,
temperature 0.7, top-p 0.95, 8,192 context tokens, and a 2,048-token output limit.
Collection runs in shuffled model blocks with matched seeds across conditions.
Allow several hours for a complete run.

On macOS, start the portable runtime without replacing an existing installation:

```sh
mkdir -p /tmp/tone-ollama-0.6.8
curl -fL https://github.com/ollama/ollama/releases/download/v0.6.8/ollama-darwin.tgz -o /tmp/tone-ollama.tgz
tar -xzf /tmp/tone-ollama.tgz -C /tmp/tone-ollama-0.6.8
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=1 /tmp/tone-ollama-0.6.8/ollama serve
```

Leave the server running. In a second terminal, from the repository root:

```sh
OLLAMA_HOST=127.0.0.1:11435 /tmp/tone-ollama-0.6.8/ollama pull qwen2.5-coder:1.5b
OLLAMA_HOST=127.0.0.1:11435 /tmp/tone-ollama-0.6.8/ollama pull qwen2.5-coder:3b
python3 -m scripts.challenge plan
python3 -m scripts.challenge collect --output results/challenge_scale_reproduction
```

Model downloads total about 3 GB. Linux and other supported platforms require their
corresponding Ollama 0.6.8 executable. Fresh outputs can vary with hardware/runtime;
saved-response analysis reproduces the reported numbers without inference.

`collect` creates both first answers and follow-ups in the selected folder, then
analyzes them. Repeating the command resumes saved trials only when configuration,
source hashes, data, and model versions match. Use one collector per folder.
API errors stop collection rather than counting as wrong answers. Original
collection code is preserved at commit `0c6bbc2`.

To collect the earlier experiments, use `python3 -m scripts.experiment run --output
results/full_run_reproduction` or `python3 -m scripts.followup run --output
results/model_hint_reproduction`. The coding runner requires macOS process isolation.

## Datasets and tests

The deterministic builders reconstruct the included answer keys:

```sh
python3 -m scripts.build_coding
python3 -m scripts.build_math
python3 -m scripts.build_math_scale
```

`data/tasks.json` contains five seed coding tasks required by the coding builder.
It is an input, not a redundant results file.

Run the platform-independent tests:

```sh
python3 -B -m unittest -v tests.test_experiment.DesignTests tests.test_experiment.MathGradingTests tests.test_experiment.ReanalysisTests tests.test_experiment.FollowupTests tests.test_challenge tests.test_challenge_scale tests.test_math_answers
```

On macOS, `python3 -B -m unittest -v` additionally tests execution of generated code
in isolated processes. The suite contains 35 tests. Tests cover exact keys, safe
answer extraction, paired conversations, statistics, and execution isolation.

## Repository layout and contributions

```text
configs/    Fixed experiment settings
data/       Problems, answer keys, and coding tests
scripts/    Dataset builders, collection, scoring, and analysis
tests/      Automated tests
results/    Saved responses and reproducible result tables
notebooks/  One executed results notebook and its dependencies
report/     Checkpoint LaTeX source for the final paper
```

The authors selected the project direction and requested the experimental extensions.
AI assisted substantially with implementation, dataset generation, repeated model
runs, analysis, and presentation. [AI_USE.md](AI_USE.md) records development prompts,
contributions, and how the resulting code was checked.
