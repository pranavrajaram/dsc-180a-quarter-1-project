# Does user tone affect coding accuracy?

A small, paired experiment following `llm_interaction_style_experiment_handoff.md`.
Five custom Python tasks × neutral/friendly/adversarial × two repetitions = **30 generations**.
Only the tone prefix changes; task text, system instructions, model, sampling settings,
and hidden tests are fixed. No tools, retries, or solution repairs are given to the model.

## Pilot result (30 generations)

| Tone | Mean hidden-test pass rate | Fully correct | Difference vs neutral, pp [95% CI] |
|---|---:|---:|---:|
| Neutral | 28.75% | 10% | — |
| Friendly | 42.50% | 10% | +13.75 [−3.75, +31.25] |
| Adversarial | 32.50% | 10% | +3.75 [−7.50, +21.25] |

**Report-ready finding:** In a 30-generation pilot using Qwen2.5-Coder 1.5B on five
coding tasks, friendly prompts had the highest observed hidden-test pass rate
(42.5%, versus 28.75% neutral and 32.5% adversarial). Both paired differences from
neutral had task-bootstrap intervals spanning zero, and full correctness was 10%
in every condition. The pilot validates the pipeline but does not establish a reliable
tone effect; more tasks and repetitions are needed.

See [the report](results/pilot/report.html) for the chart and per-task breakdown,
[the CSV](results/pilot/summary.csv) for exact statistics, or
[the presentation](results/pilot/pilot_slides_hypotheses.pptx) for the slides. All 30 responses were
retained; none hit the output-token limit. Nine focused tests passed, including
all reference solutions, isolation checks, and a known-effect analysis fixture.

## Run

Requires **macOS** and Python 3.11+. The runner uses only Python's standard library;
there is nothing to pip-install. The pilot uses **Ollama 0.6.8** on port 11435 and
[Qwen2.5-Coder 1.5B](https://ollama.com/library/qwen2.5-coder:1.5b) (roughly 986 MB).
The pre-existing Ollama 0.1.38 cannot load this model. To reproduce the pilot runtime,
start this standalone version in one terminal (it does not replace the installed app):

```sh
curl -fL https://github.com/ollama/ollama/releases/download/v0.6.8/ollama-darwin.tgz -o /tmp/tone-ollama-0.6.8.tgz
mkdir -p /tmp/tone-ollama-0.6.8
tar -xzf /tmp/tone-ollama-0.6.8.tgz -C /tmp/tone-ollama-0.6.8
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=1 /tmp/tone-ollama-0.6.8/ollama serve
```

In a second terminal, from this folder:

```sh
OLLAMA_HOST=127.0.0.1:11435 /tmp/tone-ollama-0.6.8/ollama pull qwen2.5-coder:1.5b
python3 -B -m unittest -v
python3 -B experiment.py run --output results/replication
open results/replication/report.html
```

The completed pilot lives in `results/pilot/`; a fresh checkout without results can use
`python3 -B experiment.py run`. Stop the standalone server with Ctrl-C when finished.
If using a different running Ollama server, change `base_url` in the configuration;
its actual version is recorded, but results may differ from this pilot.

Change model, repetitions, tone prefixes, or sampling parameters in `config.json`.
Use a fresh output directory for a new run:

```sh
python3 -B experiment.py run --output results/replication
```

Rebuild the summary and report from saved responses, **without model calls**:

```sh
python3 -B experiment.py analyze --output results/pilot
```

## Files

- `experiment.py`: generation, isolated evaluation, analysis, and report rendering.
- `config.json`: all experimental settings and tone prefixes.
- `tasks.json`: five task statements and 40 hidden test cases; tests never go to the model.
- `test_experiment.py`: independent task-answer oracles, design checks, evaluator and isolation checks.
- `results/pilot/`: raw records (`raw.jsonl`), summary (`summary.csv`), report (`report.html`), and the latest nine-slide presentation (`pilot_slides_hypotheses.pptx`).

The existing handoff is preserved. No framework, package scaffolding, notebooks, dependency lockfile,
or separate plotting assets are needed.

## Reading the results

The primary metric is the proportion of hidden tests passed by a generation. A syntax error scores zero;
a wrong answer, exception, or timeout fails the affected test. All planned generations must be present
before analysis produces a report. API failures stop the run, preserving earlier records, instead of
silently becoming coding failures or disappearing from the denominator.

Average repetitions within each task, then give each task equal weight. Differences from neutral are
paired by task. The 95% percentile bootstrap intervals resample whole tasks (10,000 draws), preserving
cross-tone pairing. With only five tasks these intervals are exploratory; the pilot does not support
claims of a general tone effect. Full correctness means passing all eight tests, not formal correctness.
Difficulty labels are informal. These hand-authored tasks are a convenience sample, not a benchmark.

The model receives a fresh system/user pair for each generation. Trial order is seeded and shuffled;
generation seeds are shared across tones for a given task/repetition. Equal seeds do not make generations
identical or remove sampling uncertainty. Different runtime/hardware versions can change results.

## Reproducibility and isolation

`raw.jsonl` is the canonical record: one metadata record, then one generation and one evaluation record
per trial (61 records for the pilot). Metadata contains the complete config and dataset, model digest and
metadata, Ollama/Python/OS versions, trial order, and source/dataset SHA-256 hashes. Generation records
include the exact API request, full API response, extracted code, seed, token counts, and timing. Raw
responses are flushed to disk **before** evaluation. Evaluation records include individual test outcomes.
Keep these files together with the corresponding runner revision. Generation replay is best-effort;
analysis of a saved run is deterministic.

The evaluator requires macOS `sandbox-exec` and refuses an unsandboxed fallback. Each test executes in a
fresh temporary directory/process with networking denied, user-file reads denied, writes limited to that
directory, and process, CPU, output-file, and wall-time limits. Python runtime/system libraries remain
readable. Expected answers are compared in the parent process and never sent to candidate code. These
controls are for this local pilot, not a hardened multi-tenant arbitrary-code service. No cross-platform
fallback is included. On a restricted agent host, execution may need permission to establish the OS sandbox.

A run refuses to overwrite existing raw data. Interrupted runs remain available for inspection but are
not automatically resumed; choose a new output directory to repeat a run. Latency includes warm-up and
is recorded for diagnostics, not interpreted as a tone effect.

API reference: [Ollama chat endpoint](https://docs.ollama.com/api/chat).
