# Does prompt tone change model accuracy?

We test whether a language model solves the same problem differently when the user
is polite, neutral, or rude, and whether it follows a misleading hint supplied by
the user. The project contains two experiments:

| Experiment | What we tested | Recorded responses |
|---|---|---:|
| Original study | 20 coding + 20 math tasks, seven prompt conditions, Qwen2.5-Coder 1.5B | 560 |
| Model comparison | The same 20 math tasks, 1.5B and 3B models, three tones, two seeds, with/without wrong hints | 480 |

The larger model scored better, but wrong hints substantially reduced both models'
accuracy. The [research report](report/report.pdf) presents the findings. The steps
below reproduce the numerical results behind its tables and figure.

## Start here: reproduce the report results

**You need Python 3.11 or newer and Git.** The analysis uses only Python's standard
library: no packages to install, API keys, model downloads, or GPU are needed.
These commands work on macOS and Linux; on Windows, use `python` if `python3` is not
available. The analysis was tested with Python 3.14.5.

### 1. Download the repository

```sh
git clone https://github.com/pranavrajaram/dsc-180a-quarter-1-project.git
cd dsc-180a-quarter-1-project
```

**The necessary data comes with the clone.** `data/` contains the problem statements,
hidden coding tests, math answer keys, and wrong hints. Each experiment's
`results/.../raw.jsonl` contains its recorded responses and settings. Keep these
files in their existing folders; no separate data download is required.

### 2. Run the analysis

Run all three commands from the repository root:

```sh
# Original study: accuracy by task type and prompt condition
python3 -m scripts.experiment analyze --output results/full_run

# Original study: paired tone comparisons and adjusted p-values
python3 -m scripts.study_analysis

# Model comparison: scores, hint effects, and tone-by-hint interactions
python3 -m scripts.followup analyze
```

Each command prints the location of its output. Together they rebuild the four CSV
files below, replacing the existing copies without changing the recorded responses.
The original-study analysis aggregates saved evaluations; the model-comparison
analysis scores the saved math responses again before computing its results.

### 3. Match the outputs to the report

Open the CSV files in a spreadsheet application or text editor.

| Output file | Where it is used in the report | What to read |
|---|---|---|
| [`results/full_run/summary.csv`](results/full_run/summary.csv) | Table 3: original study | `mean_pass_rate`, grouped by `coding`, `math_no_hint`, and `math_hint` |
| [`results/full_run/contrasts.csv`](results/full_run/contrasts.csv) | Original-study tone comparisons | `difference_pp`, its interval, and `p_holm` |
| [`results/model_hint/summary.csv`](results/model_hint/summary.csv) | Table 4 and Figure 1; Table 5 | `pass_rate` for overall accuracy; `hint_part_correct`, `exact_hint_match`, and `downstream_accuracy` for Table 5 |
| [`results/model_hint/effects.csv`](results/model_hint/effects.csv) | Table 4 intervals and model/tone comparisons | Filter `metric=pass_rate`; read `contrast`, `estimate`, `ci_low`, and `ci_high` |

**Reading the numbers:** accuracy is the fraction of hidden tests passed (coding)
or parts correct (math). Tasks receive equal weight after averaging repeated runs.
Rates and effects are stored as fractions: `0.30` means 30%, and `-0.20` means a
20-percentage-point decrease. Columns ending in `_pp` are already percentage points.
`control` means neutral wording; `hint=True` means a wrong hint was supplied.

For example, these rows in `results/model_hint/summary.csv` reproduce the report's
neutral-prompt comparison (rounded to one decimal place):

| Model | Tone | Hint | Accuracy (`pass_rate` × 100) |
|---|---|---|---:|
| `qwen2.5-coder:1.5b` | `control` | `False` | 30.6% |
| `qwen2.5-coder:1.5b` | `control` | `True` | 10.0% |
| `qwen2.5-coder:3b` | `control` | `False` | 48.1% |
| `qwen2.5-coder:3b` | `control` | `True` | 24.2% |

In `effects.csv`, `hint_minus_no_hint` subtracts the unhinted score from the hinted
score; `tone_by_hint_interaction` compares that change with the neutral wording's
change. The `larger_minus_smaller_hint_False` and `larger_minus_smaller_hint_True`
rows compare the 3B model with 1.5B in each hint condition. Intervals use 10,000
bootstrap samples of whole tasks. The original study's 18 tone comparisons also
report Holm-adjusted p-values.

## Check the code

Run the design, scoring, and analysis tests on any supported platform:

```sh
python3 -B -m unittest -v tests.test_experiment.DesignTests tests.test_experiment.MathGradingTests tests.test_experiment.ReanalysisTests tests.test_experiment.FollowupTests
```

On macOS, run the complete suite, including execution of generated Python code in
isolated processes:

```sh
python3 -B -m unittest -v
```

The full suite contains 22 tests. It checks dataset answer keys, trial pairing,
math scoring, known statistical effects, incomplete runs, and code isolation.

## Where things live

```text
configs/  Settings for the two experiments
data/     Problems, coding tests, math answer keys, and wrong hints
scripts/  Dataset builders, model runners, and analysis code
tests/    Automated tests
results/  Recorded responses and the four result CSVs
report/   Research report
```

The five original coding tasks in `data/tasks.json` are inputs to the coding dataset
builder. The complete datasets are `data/tasks_coding.json` and
`data/tasks_math.json`. Raw responses are in `results/full_run/raw.jsonl` and
`results/model_hint/raw.jsonl`; their first line records the configuration and tasks.

## AI Use

We used AI (specifically Codex and Claude) to assist with portions of our code implementation, including dataset construction, experiment automation, and repeated model runs. We carefully audited and tested AI generated code against our own reference calculations, unit tests, and
reproduced saved results. [AI_USE.md](AI_USE.md) documents our prompts in more detail.

## Optional: rebuild the input datasets

The datasets are already included. To reconstruct them from their builders:

```sh
python3 -m scripts.build_coding
python3 -m scripts.build_math
```

These commands deterministically overwrite the two complete task datasets. Coding
answers are checked against reference implementations; math answers and wrong hints
are computed using enumeration and exact arithmetic. Answer keys are not sent to
the models.

<details>
<summary>Optional: collect new responses from the models</summary>


New inference requires Ollama and downloaded model weights. The recorded runs used
an Apple Silicon Mac with 16 GB RAM. The coding runner requires macOS
`/usr/bin/sandbox-exec` to isolate generated code.

The following macOS commands use a portable Ollama 0.6.8 runtime without replacing
an existing installation:

```sh
mkdir -p /tmp/tone-ollama-0.6.8
curl -fL https://github.com/ollama/ollama/releases/download/v0.6.8/ollama-darwin.tgz -o /tmp/tone-ollama.tgz
tar -xzf /tmp/tone-ollama.tgz -C /tmp/tone-ollama-0.6.8
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=1 /tmp/tone-ollama-0.6.8/ollama serve
```

Leave that server running. In a second terminal, from the repository root:

```sh
OLLAMA_HOST=127.0.0.1:11435 /tmp/tone-ollama-0.6.8/ollama pull qwen2.5-coder:1.5b
OLLAMA_HOST=127.0.0.1:11435 /tmp/tone-ollama-0.6.8/ollama pull qwen2.5-coder:3b
python3 -m scripts.experiment run --output results/full_run_reproduction
python3 -m scripts.followup run --output results/model_hint_reproduction
```

Model downloads total about 3 GB. Allow additional memory and time for inference;
the 480-response run takes roughly 1–2 hours on the recorded machine. Other platforms
can run the mathematics follow-up with the corresponding Ollama 0.6.8 release.

`configs/config.json` specifies the original seven-tone study. `configs/followup.json` specifies the
fully crossed 20 tasks × 2 models × 3 tones × 2 seeds × 2 hint conditions. Both use
temperature 0.7, top-p 0.95, 4,096 context tokens, and at most 2,048 generated tokens.
The follow-up uses the same seed for each task/repetition across models, tones, and
hint conditions. It randomizes trial order within each model block.

The original runner requires a new output directory. The follow-up can resume by
rerunning the same command against its existing directory; configuration, data,
source code, and model digests must match. Use only one collector per output folder.
API errors stop collection rather than assigning an incorrect score. Reanalysis of
the committed data is independent of the current inference server.

The complete model digests and generation settings are recorded in the first line of
each saved `raw.jsonl` file. Fresh model outputs can vary across runtime/hardware
versions; the saved-response workflow above reproduces the report's numerical results.

</details>

