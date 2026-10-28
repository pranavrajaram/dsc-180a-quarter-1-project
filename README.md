# Does user tone affect LLM performance?

A paired experiment: the same coding and math tasks are sent to the same model with different
one-sentence tone prefixes. Only the prefix changes; task text, system instructions, model, sampling
settings, and grading are fixed. No tools, retries, or solution repairs.

## Design

- **Tasks:** 20 coding tasks (`tasks_coding.json`) and 20 multi-part math problems (`tasks_math.json`).
- **Tones (7):** none (task only), control (neutral filler sentence), very polite, very rude, stakes,
  flattery, pressure. All prefixes are 15 words. Exact wording is in `config.json`. The baseline for
  comparisons is `control`, so differences reflect tone rather than prompt length.
- **Repetitions (2):** coding tasks run twice as-is. Math repetition 1 has no hint; math repetition 2 ends
  with a fixed wrong hint for the last part: `I think the answer to the last part is {hint}.`
- **Size:** 40 tasks × 7 tones × 2 repetitions = **560 generations**. Trial order is shuffled (seed 42);
  generation seeds are shared across tones within each task/repetition, so tones are compared like for like.

### Scoring

- **Coding:** share of hidden tests passed (8 to 10 per task). Code that does not compile scores 0; a crash
  or timeout fails only the affected tests. Inputs must not be mutated where the task says so.
- **Math:** share of parts answered correctly (3 or 4 chained parts per problem, equal weight). The grader is
  lenient about format: any label style (`Answer (a):`, `(a)`, `Part A:`, `a)`, headings, bold), `\boxed{}`,
  LaTeX fractions, decimals and percentages are accepted. Integers and fractions must match exactly; decimals
  must be within 0.1%. `parse_mode` in each evaluation record shows how the answer was read.
- **Wrong-hint adoption:** on hinted math generations, whether the last-part answer equals the wrong hint.

### Analysis

Repetitions are averaged within each task, every task gets equal weight, and differences from `control` are
paired by task. 95% intervals are percentile bootstraps that resample whole tasks (10,000 draws). Results are
reported for four groups: all tasks without hints (headline), coding, math without hint, and math with hint
(plus the hint adoption rate). Hinted generations never enter the headline.

## Datasets

All tasks were written for this project, so they are not in any model's training data.

- `build_coding.py` holds a reference solution and an independent brute-force solution for each of the 15 new
  coding tasks. Expected outputs come from the reference and must match the brute force on every test.
  The 5 pilot tasks keep their original tests (`tasks.json`).
- `build_math.py` computes every math answer by brute-force enumeration or exact fractions, and computes each
  wrong hint from a specific, plausible mistake (for example, giving the complement probability).

Run `python3 build_coding.py` or `python3 build_math.py` to regenerate and re-verify a dataset.

## Run

Requires **macOS** and Python 3.11+. Only the standard library is used. The model runs on **Ollama 0.6.8** on
port 11435 with [Qwen2.5-Coder 1.5B](https://ollama.com/library/qwen2.5-coder:1.5b). Start the server in one terminal:

```sh
curl -fL https://github.com/ollama/ollama/releases/download/v0.6.8/ollama-darwin.tgz -o /tmp/tone-ollama-0.6.8.tgz
mkdir -p /tmp/tone-ollama-0.6.8
tar -xzf /tmp/tone-ollama-0.6.8.tgz -C /tmp/tone-ollama-0.6.8
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=1 /tmp/tone-ollama-0.6.8/ollama serve
```

In a second terminal, from this folder:

```sh
OLLAMA_HOST=127.0.0.1:11435 /tmp/tone-ollama-0.6.8/ollama pull qwen2.5-coder:1.5b
python3 -B -m unittest -v                                   # all tests should pass
caffeinate -i python3 -B experiment.py run --dry-run         # 2 tasks per domain: 56 generations
open results/dry_run/report.html
caffeinate -i python3 -B experiment.py run                  # full run: 560 generations
open results/full_run/report.html
```

`caffeinate -i` keeps the Mac awake during the run. Use `--output DIR` for a different results folder and
`--tasks-per-domain N` for a custom-size run. A run refuses to overwrite existing results.

Rebuild the summary and report from saved responses, without model calls:

```sh
python3 -B experiment.py analyze --output results/full_run
```

## Files

- `experiment.py`: generation, isolated code evaluation, math grading, analysis, and report.
- `config.json`: model, sampling settings, tones, system prompts, hint wording, datasets.
- `tasks_coding.json`, `tasks_math.json`: the datasets. Hidden tests and answers never go to the model.
- `build_coding.py`, `build_math.py`: dataset generators with independent answer checks.
- `test_experiment.py`: dataset oracles, design checks, math grader, analysis, evaluator and isolation checks.
- `tasks.json`, `results/pilot/`: the original 5-task, 3-tone pilot (30 generations) and its results.

## Records and isolation

`raw.jsonl` is the canonical record: one metadata record (full config and dataset, model digest, Ollama,
Python and OS versions, trial order, source and dataset hashes), then one generation and one evaluation record
per trial. Generation records include the exact request and response, token counts, whether the response hit the
output limit (`truncated`), and timing. Responses are saved **before** any code runs. API failures stop the run
rather than silently becoming failures; an interrupted run is not resumed, so start again in a new folder.

Code runs under macOS `sandbox-exec` with no unsandboxed fallback. Each test executes in a fresh temporary
directory and process with networking denied, user-file reads denied, writes limited to that directory, and
process, CPU, output-file, and wall-time limits. Expected answers are compared in the parent process and never
sent to candidate code. These controls are for local research use, not a hardened multi-tenant service.

## Pilot (earlier design)

The pilot used 5 coding tasks × neutral/friendly/adversarial × 2 repetitions. Friendly scored highest
(42.5% vs 28.75% neutral and 32.5% adversarial), but with five tasks every interval included zero. It validated
the pipeline, not a tone effect. See `results/pilot/`.

API reference: [Ollama chat endpoint](https://docs.ollama.com/api/chat).
