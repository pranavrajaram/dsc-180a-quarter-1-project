# Prompt tone, misleading hints, and model size

This project measures coding and mathematics accuracy under different prompt tones.
It includes a 560-response experiment with Qwen2.5-Coder 1.5B and a 480-response
mathematics experiment comparing the 1.5B and 3B checkpoints under neutral, polite,
and rude wording, with and without wrong hints.

Read the updated [report.pdf](report.pdf), or edit its [LaTeX source](report.tex). The main finding is that the larger
model performs better, but misleading hints still reduce accuracy substantially.

## 1. Get the code and data

```sh
git clone https://github.com/pranavrajaram/llm-tone-experiment.git
cd llm-tone-experiment
```

All data needed to reproduce the reported results is included in the repository:

| Path | Contents |
|---|---|
| `tasks.json` | Five original coding tasks used by the coding dataset builder |
| `tasks_coding.json` | Twenty coding tasks and 162 hidden tests |
| `tasks_math.json` | Twenty mathematics tasks, 63 answer-key parts, and wrong hints |
| `results/full_run/raw.jsonl` | The 560 original prompts, responses, and evaluations |
| `results/model_hint/raw.jsonl` | The 480 model-comparison prompts and responses |

Keep these files in their existing locations. No external dataset, account, API key,
or model download is required to reanalyze the saved responses. Each raw file begins
with metadata containing the complete configuration, task set, model identifiers,
and trial order. Subsequent lines contain individual trial records.

## 2. Software dependencies

- **Reanalysis:** Python 3.11 or newer; standard library only. Works on macOS, Linux,
  and Windows. No `pip install`, environment manager, or GPU is needed.
- **New model responses:** Ollama 0.6.8 and the specified Qwen model files. Both saved
  experiments ran locally on Apple Silicon with 16 GB RAM. Full coding collection
  also requires macOS `/usr/bin/sandbox-exec`, used to isolate generated code.
- **Tests:** Python only. The full test suite requires macOS for the code-isolation
  tests; a portable subset is listed below.
- **Report PDF:** a LaTeX environment with `geometry`, `fontenc`, `mathptmx`, `amsmath`,
  `booktabs`, `tabularx`, `pgfplots`, and `hyperref`, or Overleaf. All report figures
  and references are embedded in `report.tex`; there are no external figure files.

The analysis was tested with Python 3.14.5. The recorded inference runtime is Ollama
0.6.8. No Python package requirements file is needed because the experiment and
analysis code have no third-party Python dependencies.

## 3. Reproduce results from the saved responses

Run these commands from the repository root:

```sh
python3 experiment.py analyze --output results/full_run
python3 study_analysis.py
python3 followup.py analyze
```

They overwrite only the derived CSV files:

| Output | Meaning |
|---|---|
| `results/full_run/summary.csv` | Original accuracy by domain, hint condition, and tone |
| `results/full_run/contrasts.csv` | Paired tone effects, bootstrap intervals, and adjusted p-values |
| `results/model_hint/summary.csv` | Accuracy, exact hint matching, later-part accuracy, and truncation rates |
| `results/model_hint/effects.csv` | Paired model effects, hint effects, and tone-by-hint interactions |

Raw responses are not modified. Scores average repetitions within a task and then
weight tasks equally. Bootstrap intervals resample task clusters 10,000 times.
`summary.csv` and `effects.csv` store rates/effects on a 0–1 scale; multiply by 100
for percentages or percentage points. Columns ending in `_pp` in `contrasts.csv`
already use percentage points. The original study's 18 paired comparisons use Holm
adjustment; the follow-up reports pointwise bootstrap intervals.

To build the report PDF locally:

```sh
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Alternatively, upload `report.tex` to Overleaf and compile with pdfLaTeX. The source
also opens directly in Codex's LaTeX editor with its PDF preview. The report's tables
are rounded from the CSV outputs; its figure uses the same condition means.

## 4. Rebuild the task datasets

```sh
python3 build_coding.py
python3 build_math.py
```

These overwrite `tasks_coding.json` and `tasks_math.json` deterministically.
`build_coding.py` reads the five seed tasks from `tasks.json` and adds fifteen tasks
with reference and brute-force solutions. `build_math.py` derives answers and wrong
hints using enumeration and exact arithmetic. Answer keys and tests are used for
scoring, not included in model prompts.

## 5. Collect new model responses

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
python3 experiment.py run --output results/full_run_reproduction
python3 followup.py run --output results/model_hint_reproduction
```

Model downloads total about 3 GB. Allow additional memory and time for inference;
the 480-response run takes roughly 1–2 hours on the recorded machine. Other platforms
can run the mathematics follow-up with the corresponding Ollama 0.6.8 release.

`config.json` specifies the original seven-tone study. `followup.json` specifies the
fully crossed 20 tasks × 2 models × 3 tones × 2 seeds × 2 hint conditions. Both use
temperature 0.7, top-p 0.95, 4,096 context tokens, and at most 2,048 generated tokens.
The follow-up uses the same seed for each task/repetition across models, tones, and
hint conditions. It randomizes trial order within each model block.

The original runner requires a new output directory. The follow-up can resume by
rerunning the same command against its existing directory; configuration, data,
source code, and model digests must match. Use only one collector per output folder.
API errors stop collection rather than assigning an incorrect score. Reanalysis of
the committed data is independent of the current inference server.

Recorded model digests (Q4_K_M):

```text
qwen2.5-coder:1.5b  d7372fd828518a4d38b1eb196c673c31a85f2ed302b3d1e406c4c2d1b64a0668
qwen2.5-coder:3b    f72c60cabf6237b07f6e632b2c48d533cef25eda2efbd34bed21c5e9c01e6225
```

Compare these with `/api/tags` before rerunning inference. Model tags can change,
and identical seeds need not produce identical outputs across hardware/runtime
versions. The exact collection sources are preserved at commits `df1eeb7` (original
study) and `d169b60` (follow-up); current code removes presentation output while
retaining the experiment design and scoring.

## 6. Run tests

```sh
# Full suite, including macOS generated-code isolation:
python3 -B -m unittest -v
# Portable tests for design, math grading, and analysis:
python3 -B -m unittest -v test_experiment.DesignTests test_experiment.MathGradingTests test_experiment.ReanalysisTests test_experiment.FollowupTests
```

Tests cover dataset answer keys, paired trial designs, math grading, known statistical
effects, incomplete-run detection, and generated-code evaluation. The full suite
also verifies filesystem/network restrictions and timeouts for candidate Python code.
