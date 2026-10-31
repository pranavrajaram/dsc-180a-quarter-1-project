# Does user tone affect LLM performance?

A paired experiment: the same coding and math tasks are sent to the same model with different
one-sentence tone prefixes. Only the prefix changes; task text, system instructions, model, sampling
settings, and grading are fixed. No tools, retries, or solution repairs.

## Reproduce the checkpoint analysis (no model required)

Clone the repository, enter its directory, and use Python 3.11+ on macOS, Linux, or Windows:

```sh
python3 -B study_analysis.py --run results/full_run --output results/checkpoint
python3 -B -m unittest -v test_experiment.DesignTests test_experiment.MathGradingTests test_experiment.ReanalysisTests
```

All required data is already tracked in `results/full_run/raw.jsonl`. No API key, Ollama,
network, downloaded dataset, or third-party Python package is needed for this analysis.
It never executes generated code. The original raw records and summaries remain unchanged.
`results/checkpoint/` contains just `contrasts.csv`, `audit.json`, and `report.html`.
These deterministic outputs include raw-data and analysis-source hashes. Re-run the same
command to regenerate them. The saved generation source is commit `df1eeb7` on
`alex/full-run-prep`; the new analysis is on `study/reproducibility-analysis`.

### Checkpoint findings and statistical scope

- Exact paired sign-flip tests compare each of six nonbaseline conditions with control in
  coding, math without hints, and math with hints. Holm adjustment covers **all 18 comparisons**
  as one family. The overlapping all-domain headline is descriptive, not another test family.
- **None survives Holm correction at 0.05.** The smallest adjusted p-value is 0.140625
  (stakes, math without hints). Original task-bootstrap intervals remain unadjusted and
  should not be presented as multiplicity-adjusted significance tests.
- This analysis was chosen after seeing the data. It is exploratory. The sign-flip test assumes
  independent task clusters and sign exchangeability under the null. Its exact distribution
  enumerates task-level sign assignments using rational sums, including ties and zeros.
  It is not the inversion of the percentile bootstrap, so their significance decisions can differ.
- Exact wrong-hint matching is **84/140 (60%)**, also 84/140 under the original approximate-copy
  measure. Without hints, **22/140 (15.7%)** answers already match the same wrong values.
  Part (a) correctness is **67/140 (47.9%)** without hints versus **35/140 (25.0%)** with hints.
  Task-cluster bootstrap intervals for these descriptive contrasts are in `audit.json`.
- Hint status changes with repetition and generation seed. Each math task/tone has only **one**
  response per hint condition. Do not interpret this as two independent repeats per condition,
  or as a clean causal estimate of adding a hint. Next, cross hint/no-hint with the same set of
  task/tone/seed combinations and collect several seeds per cell.
- All **280 math scores** reproduce under the current grader. A separate format sensitivity
  requires answer content on the requested final-answer lines and gives missing lines zero
  credit. It intentionally mixes format compliance with accuracy and is not a replacement score.
- Equal word counts do not isolate tone perfectly. Multiple phrases per condition, a larger
  independently chosen task sample, stronger hidden tests, and another model are needed.

The audit rejects duplicate/missing trials, inconsistent recorded scores, incorrect hint labels,
and requests that disagree with the recorded prompts, model or sampling settings. It does not
claim that the model obeyed the settings, or that the grading heuristic is a semantic proof.

## Design

- **Tasks:** 20 coding tasks (`tasks_coding.json`) and 20 multi-part math problems (`tasks_math.json`).
- **Tones (7):** none (task only), control (neutral filler sentence), very polite, very rude, stakes,
  flattery, pressure. All six nonempty prefixes are 15 whitespace-delimited words. Exact wording is in `config.json`. The baseline for
  comparisons is `control`, which controls word count but does not equalize token count or separate tone from the particular wording.
- **Repetitions (2):** coding tasks run twice as-is. Math repetition 1 has no hint; math repetition 2 adds a
  fixed wrong hint for part (a) after the problem: `I think the answer to part (a) is {hint}.` Part (a) is the
  easiest part and every later part builds on it, so deference shows up in the score as well.
- **Answer format:** every math prompt ends with the same instruction to finish with `Answer (a): ...` lines.
  It is identical across tones; whether the model followed it is recorded but not scored.
- **Size:** 40 tasks × 7 tones × 2 repetitions = **560 generations**. Trial order is shuffled (seed 42);
  generation seeds are shared across tones within each task/repetition, so tones are compared like for like.

### Scoring

- **Coding:** share of hidden tests passed (8 to 10 per task). Code that does not compile scores 0; a crash
  or timeout fails only the affected tests. Inputs must not be mutated where the task says so.
- **Math:** share of parts answered correctly (3 or 4 chained parts per problem, equal weight). The grader is
  lenient about format: any label style (`Answer (a):`, `(a)`, `Part A:`, `a)`, headings, bold), `\boxed{}`,
  LaTeX fractions, decimals and percentages are accepted. Integers and fractions must match exactly; decimals
  must be within 0.1%. `parse_mode` in each evaluation record shows how the answer was read.
- **Wrong-hint effect:** two measures. *Part (a) correct* is recorded on every math generation, so the hint and
  no-hint repetitions can be compared directly. *Adopted wrong hint* is whether a hinted part (a) answer is the
  hint; for hints of 5+ digits a copy with a one-digit slip also counts (the dry run model wrote 31,187,520 for
  the hint 311,875,200).

### Analysis

Repetitions are averaged within each task, every task gets equal weight, and differences from `control` are
paired by task. 95% intervals are percentile bootstraps that resample whole tasks (10,000 draws). Results are
reported for four groups: all tasks without hints (headline), coding, math without hint, and math with hint
(plus the hint adoption rate). Hinted generations never enter the headline.

## Datasets

These task instances were written for this project. Related algorithms and problem templates may have appeared in training; contamination has not been measured.

- `build_coding.py` holds a reference solution and an independent brute-force solution for each of the 15 new
  coding tasks. Expected outputs come from the reference and must match the brute force on every test.
  The 5 pilot tasks keep their original tests (`tasks.json`).
- `build_math.py` computes every math answer by brute-force enumeration or exact fractions, and computes each
  part (a) wrong hint from a specific, common mistake (for example, counting ordered draws: 380 instead of 190).

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
caffeinate -i python3 -B experiment.py run --output results/replication  # 560 new generations
open results/replication/report.html
```

`caffeinate -i` keeps the Mac awake during the run. Use `--output DIR` for a different results folder and
`--tasks-per-domain N` for a custom-size run. A run refuses to overwrite existing results.

Rebuild the summary and report from saved responses, without model calls:

```sh
python3 -B experiment.py analyze --output results/full_run
```

## Files

- `experiment.py`: generation, isolated code evaluation, math grading, analysis, and report.
- `study_analysis.py`: saved-run validation, exact paired tests, Holm adjustment, and scoring audit.
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

## Report updates and submission

Use `results/checkpoint/contrasts.csv` for statistical comparisons and `audit.json` for the
scoring checks. In the checkpoint report:

- **Abstract, second paragraph:** replace “the neutral sentence itself had no effect” with
  “the neutral-prefix and no-prefix scores were similar (46.7% and 48.2%); the data do not
  establish equivalence.” Describe the math declines as observed, not proven tone effects.
  Add that none of the 18 domain/hint contrasts survived exploratory Holm correction.
- **Data, opening paragraph (page 2):** remove the claim that new problems cannot have
  appeared in training. State that task instances are project-authored, with unmeasured
  overlap in underlying algorithms and templates.
- **Data table (page 2):** specify that 15 added coding tasks have paired reference/brute-force
  checks. The five pilot tasks retain their original tests with separate reference checks.
- **Prompt conditions, wrong-hint paragraph (page 3):** clarify that there is one response
  per task/tone/hint condition and hint status is tied to the repetition/seed.
- **Generated data (page 3):** add the exact-match baseline (84/140 hinted versus 22/140
  unhinted), the 280/280 unchanged math scores, and the reproducible reanalysis command.
- **Results by condition and table caption (pages 3–4):** replace “tone mattered for math
  but not coding” with a descriptive statement about larger observed math differences.
  Keep existing stars only if explicitly labeled unadjusted bootstrap intervals. Add adjusted
  p-values, the 18-comparison family, and the assumptions/post-data status of the paired test.
  Do not interpret nonsignificance as no effect. The polite coding difference is +11.25 points,
  so “coding barely changed” is also too broad.
- **After the results table:** add a short limitations paragraph covering one small model,
  one wording per condition, 20 convenience-sampled tasks per domain, hint/seed confounding,
  format-sensitive math extraction, and unmeasured contamination. Add a full references list
  for the papers already linked in Prior Work.

The [Q1 guidelines](https://dsc-capstone.org/2026-27/assignments/projects/q1/)
require a public repository for submission and a LaTeX report for the final submission.
This repository remains private as requested by its owner; resolve visibility before submission.
The checkpoint calls for title, abstract, and introduction (including prior work and data),
plus mentor-specific requirements. The analysis above is additional exploratory evidence.
