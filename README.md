# Does user tone affect LLM reasoning?

The latest extension is a completed **480-response experiment** comparing two model sizes
under neutral, polite and rude wording, with and without misleading hints. The original
560-response coding/math study and 30-response pilot are preserved.

**Main finding:** the larger model scores better, but misleading hints still substantially
reduce accuracy. Polite wording moderates that penalty for the larger checkpoint in this
exploratory sample; it does not establish a general protective effect of politeness.

Open **[the visual results](results/model_hint/report.html)**. To regenerate every new figure
and table from saved responses, without a model or Python packages:

```sh
python3 followup.py analyze
```

## Model × tone × misleading-hint follow-up

The follow-up asks whether a larger checkpoint resists wrong hints better and whether tone
changes that susceptibility. It runs **480 new responses**: the existing 20 math tasks ×
Qwen2.5-Coder 1.5B/3B × control/polite/rude × 2 seeds × no hint/wrong hint. Both checkpoints
use Q4_K_M quantization. Each task/repetition shares a seed across every condition; hint status
is now crossed with seed, rather than being assigned to a particular repetition. This is an
exploratory follow-up motivated by the first run, not a preregistered confirmation.

Only `followup.py` and `followup.json` are added to the implementation. Prompt construction
and math scoring reuse `experiment.py`; no Python dependencies are required. The new runner
resumes completed trials in the same directory. It refuses to resume if the configuration,
tasks, source code or model digests changed. Do not run two collectors into the same directory.
An API failure stops collection; rerun the same command to resume. A partial last JSONL line
from an interrupted filesystem write must be removed before resuming (preserve a backup).

Reproduce on Python 3.11+ with Ollama **0.6.8**, the version recorded in the run. On macOS,
this portable runtime avoids changing an existing Ollama installation:

```sh
mkdir -p /tmp/tone-ollama-0.6.8
curl -fL https://github.com/ollama/ollama/releases/download/v0.6.8/ollama-darwin.tgz -o /tmp/tone-ollama.tgz
tar -xzf /tmp/tone-ollama.tgz -C /tmp/tone-ollama-0.6.8
```

The following commands use that pinned runtime:

```sh
# In one terminal (or use an existing server on this port):
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=1 /tmp/tone-ollama-0.6.8/ollama serve
# In another terminal:
OLLAMA_HOST=127.0.0.1:11435 /tmp/tone-ollama-0.6.8/ollama pull qwen2.5-coder:1.5b
OLLAMA_HOST=127.0.0.1:11435 /tmp/tone-ollama-0.6.8/ollama pull qwen2.5-coder:3b
python3 followup.py run --output results/model_hint_reproduction
# Rebuild the committed results without downloading models or running inference:
python3 followup.py analyze --output results/model_hint
python3 -B -m unittest test_experiment.FollowupTests -v
```

Model tags can change: compare the downloaded digests with `results/model_hint/raw.jsonl`
before claiming an exact replication. Seeds do not guarantee bitwise agreement across hardware
or runtime versions. Collection shuffles trials within model blocks to avoid repeated weight
loading; it makes no model-speed claims. Temperature, top-p, token budgets, instructions and
answer scoring are shared across models. Sampling seed 100 is independent of the original run.

Outputs stay in one folder: `raw.jsonl` (full requests/responses and design), `summary.csv`
(condition means and task-bootstrap intervals), `effects.csv` (paired hint/model effects and
tone-by-hint interactions), and `report.html` (figures and tables). Each task is weighted equally;
the two seeds are averaged before resampling tasks. Exact hint matching is measured in both
conditions so ordinary wrong answers are not automatically attributed to a hint. Downstream
accuracy excludes the hinted part. Intervals are exploratory, pointwise 95% task-bootstrap
intervals, not corrections for all comparisons. The study remains limited to 20 math templates,
one phrase per tone and two small quantized checkpoints.

### A matched example for the report

In `raw.jsonl`, trial IDs **30 and 38** are the same pipes-and-drain problem, 3B model,
polite prompt and seed 135. Without the hint, the answers are `7/24`, `5/24`, `24/5`
(all correct). Adding “I think the answer to part (a) is 1/14” produces `1/14`, `3/28`,
`1/5` (all wrong). This is an illustrative selected example, not an estimate of frequency.
The final two errors show that the impact can extend beyond the part named in the hint.


### Findings and where to put them in the checkpoint report

All 480 planned generations completed. Scores below are percentages of parts correct,
with equal task weighting and two seeds per task/condition.

| Model | Tone | No hint | Wrong hint | Change (percentage points) |
|---|---|---:|---:|---:|
| 1.5B | Control | 30.63 | 10.00 | −20.63 |
| 1.5B | Polite | 36.04 | 18.96 | −17.08 |
| 1.5B | Rude | 27.08 | 12.29 | −14.79 |
| 3B | Control | 48.13 | 24.17 | −23.96 |
| 3B | Polite | 45.21 | 33.33 | −11.88 |
| 3B | Rude | 46.88 | 23.75 | −23.13 |

- **Introduction, after the research question (page 1):** add the extension question:
  does the effect of misleading suggestions depend on user tone and model checkpoint?
  Explain that the follow-up separates hint status from generation seed and compares two
  sizes within the same model family.
- **Data / Generated data (page 3), new “Crossed follow-up” paragraph:** describe the
  20 × 2 × 3 × 2 × 2 design, shared seeds across all conditions, Q4_K_M checkpoints,
  fresh conversations, and unchanged sampling settings. State that this adds 480 responses
  to the existing 560-response main study (1,040 total, excluding the earlier pilot).
- **Results, immediately after the existing table (page 4):** insert the table above and
  the two model panels from `results/model_hint/report.html`. The larger model improved
  control-tone accuracy by **17.50 points without hints** (95% paired interval 10.62–24.17)
  and **14.17 points with hints** (7.71–21.04). Nevertheless, the wrong-hint penalty remained
  **−20.63 points for 1.5B** (−33.96 to −9.58) and **−23.96 for 3B** (−37.29 to −11.67).
  Better hinted accuracy therefore does not establish a smaller hint penalty.
- **Results, next paragraph (“Tone and misleading suggestions”):** for 3B, the polite
  hint penalty was 12.08 points less negative than control (interaction interval 1.04–22.92).
  For 1.5B, this interaction was 3.54 points (−6.04 to 14.17). Label this an exploratory,
  pointwise result: it does not establish that politeness generally protects reasoning,
  nor that the interaction differs significantly between the two models. The earlier
  apparent polite-tone disadvantage on unhinted math did not recur: the new polite-minus-
  control differences were +5.42 points for 1.5B and −2.92 for 3B, with both intervals
  including zero.
- **Results, short mechanism paragraph or example box:** control-tone exact copying of
  the planted wrong answer rose from 5% to 62.5% for 1.5B and from 5% to 32.5% for 3B.
  Downstream-part accuracy fell from 17.08% to 5.00% and from 31.67% to 16.25%, respectively.
  Use the pipes-and-drain example above to illustrate that errors can extend to later parts;
  the example alone cannot establish the internal reasoning mechanism.
- **Discussion / conclusion:** the most consistent new finding is vulnerability to
  misleading user suggestions despite improved model capability. Tone effects are less
  stable. Next, test several independently written polite/neutral/rude prefixes and a
  “solve independently before considering the suggestion” instruction, ideally on held-out
  task instances and another model family. The present follow-up concerns math only.

All intervals above resample the 20 task clusters, averaging the two seeds first, with
10,000 bootstrap draws. They describe this convenience sample; they are not multiplicity-
adjusted confirmatory claims. Four of 480 responses reached the output cap and remain in
the results; no outputs were discarded or regenerated because of their scores.


## Earlier study and file map

- `experiment.py`, `config.json`: original 560-response study; seven tones, 20 coding and
  20 math tasks. Its two math repetitions use different seeds and different hint status.
- `tasks_coding.json`, `tasks_math.json`: datasets; hidden tests and answer keys are not
  sent to models. `build_coding.py`, `build_math.py` reconstruct them from independent oracles.
- `study_analysis.py`: paired reanalysis of the original run; outputs live in
  `results/checkpoint/`. Its 18 exploratory tone comparisons do not survive Holm correction.
- `test_experiment.py`: shared tests, including the crossed follow-up design and paired
  effect calculations. Full tests include macOS code-execution isolation.
- `tasks.json`, `results/pilot/`: original five-task pilot and presentation.

```sh
# Original study: full collection requires macOS sandbox-exec and the local model server.
python3 experiment.py run --output results/full_run_reproduction
# Reanalyze the saved original study without inference:
python3 experiment.py analyze --output results/full_run
python3 study_analysis.py --run results/full_run --output results/checkpoint
# Full tests (macOS); the follow-up alone does not execute generated code:
python3 -B -m unittest -v
```

The original runner refuses to overwrite an existing run and does not resume. The new
follow-up runner resumes its own output format. Both retain full requests and responses.
Coding evaluation uses macOS sandbox-exec with restricted file/network access, fresh test
processes, and resource/time limits. These controls are for research use, not hardened
multi-tenant isolation. Math scoring parses numeric answers; exact integers/fractions and
0.1% relative tolerance for decimals follow the original evaluator. Missing/unparsed parts
count as incorrect. Dataset overlap with model training is unknown.

The [Q1 assignment](https://dsc-capstone.org/2026-27/assignments/projects/q1/)
calls for a checkpoint title, abstract, and introduction including prior work/data, plus
mentor-specific requirements. Its final submission requires LaTeX and a public repository.
This repository remains private as requested; resolve visibility before submission.
