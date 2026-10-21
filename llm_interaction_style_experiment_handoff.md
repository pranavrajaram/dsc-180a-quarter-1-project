# LLM Interaction Style Experiment
## Codex Handoff / Implementation Spec

## 1. Project Goal

Build an experiment testing whether the **way a user communicates with an LLM** changes the quality of the model's output.

The underlying task should remain identical. Only the interpersonal tone of the user's prompt should change.

Primary research question:

> Does prompt interaction style affect LLM performance, even when task content and technical instructions are held constant?

Initial focus should be **coding tasks**, since they allow objective evaluation through hidden unit tests. The system should be designed so we can later add other domains such as math, instruction following, or information extraction.

---

## 2. Experimental Conditions

Start with three primary conditions:

### Neutral
Simple and professional.

Example:

> Solve the following coding problem.

### Friendly
Collaborative and appreciative without giving any additional technical information.

Example:

> Hey, could you help me solve this coding problem? I'd really appreciate it.

### Adversarial
Confrontational and skeptical without changing the actual task requirements.

Example:

> I don't think you can solve this coding problem correctly. Prove me wrong.

Potential fourth condition for later:

### High Pressure

> This answer needs to be completely correct. Do not make any mistakes.

For the first version, prioritize **Neutral, Friendly, and Adversarial** so the experiment stays simple.

---

## 3. Independent Variable

**Prompt interaction style**

Categorical variable:

```text
neutral
friendly
adversarial
```

The manipulation should occur only in a short wrapper around the task.

Example structure:

```text
{tone_prefix}

{exact_task_text}
```

The `{exact_task_text}` must remain byte for byte identical across experimental conditions whenever possible.

---

## 4. Primary Dependent Variables

For coding tasks, collect:

### Primary Metric
**Hidden test pass rate**

```text
tests_passed / total_tests
```

This should be the main outcome variable.

### Secondary Metrics

- Whether the solution passes all hidden tests
- Number of tests passed
- Number of tests failed
- Whether the code compiles / executes
- Runtime error indicator
- Timeout indicator
- Execution time if available
- Output length
- Code length

Possible later metrics:

- Cyclomatic complexity
- Linting errors
- Human or LLM judged code quality
- Number of retries required to reach a correct answer

Do not make subjective code quality the primary metric.

---

## 5. Experimental Unit

The natural experimental unit is:

```text
task × tone × repetition
```

Example:

```text
Task 17
Tone: adversarial
Repetition: 3
Model: GPT-X
```

Each coding task should ideally be run under every tone condition.

This gives a paired design where we can compare performance on the **same task** across tones.

---

## 6. Repeated Trials

LLM responses are stochastic, so do not rely on one generation per task.

Initial target:

```text
3 tones
30 to 50 tasks
3 to 5 repetitions per task/tone
```

Example with 40 tasks and 5 repetitions:

```text
40 × 3 × 5 = 600 model calls
```

Make repetitions configurable.

---

## 7. Coding Task Dataset

Use coding problems with:

- Clear problem statements
- Objective outputs
- Existing unit tests or easily generated tests
- A range of difficulty
- Minimal ambiguity
- No need for web access

Potential sources could later include HumanEval, MBPP, or a custom task set.

For the first implementation, make the dataset interface generic rather than tightly coupling the code to one benchmark.

Suggested task representation:

```json
{
  "task_id": "task_001",
  "title": "Example Problem",
  "prompt": "Write a function that...",
  "language": "python",
  "entry_point": "solve",
  "difficulty": "easy",
  "tests": [...]
}
```

If using an existing benchmark, preserve its original test structure where practical.

---

## 8. Important Controls

The experiment needs to isolate **tone**, so keep the following constant:

### Prompt Content
The technical problem statement must be identical across tone conditions.

### Prompt Length
Tone wrappers should be reasonably similar in length. Do not give the friendly condition extra context or the adversarial condition extra instructions.

### Model
Use the exact same:

- Model
- Model version / snapshot if available
- System prompt

### Model Parameters
Keep constant:

- Temperature
- Top p
- Max output tokens
- Reasoning effort if configurable
- Tool permissions

### Context
Each trial should use a fresh conversation.

Do not allow previous trials to remain in context.

### Tool Access
For the initial experiment:

```text
No web
No external documentation
No code execution by the model unless every condition receives identical access
```

### Evaluation
All conditions must use exactly the same hidden tests.

---

## 9. Prompt Construction

Create one centralized prompt builder.

Something like:

```python
TONE_PREFIXES = {
    "neutral": "Solve the following coding problem.",
    "friendly": "Hey, could you help me solve this coding problem? I'd really appreciate it.",
    "adversarial": "I don't think you can solve this coding problem correctly. Prove me wrong."
}
```

Then:

```python
def build_prompt(task_text, tone):
    return f"{TONE_PREFIXES[tone]}\n\n{task_text}"
```

Do not manually construct prompts throughout the codebase.

We want to be able to edit the tone manipulation from one configuration file.

---

## 10. Randomization

Although every task should eventually appear under every condition, randomize execution order.

Do not run:

```text
all friendly
then all neutral
then all adversarial
```

Instead shuffle the complete set of trials.

Example:

```text
task_18 / friendly
task_02 / adversarial
task_31 / neutral
task_18 / neutral
...
```

Save the random seed.

---

## 11. Raw Result Schema

Store every individual generation.

Suggested structure:

```json
{
  "run_id": "uuid",
  "task_id": "task_001",
  "tone": "friendly",
  "repetition": 2,

  "model": "MODEL_NAME",
  "temperature": 0.7,

  "prompt": "...",
  "raw_response": "...",
  "extracted_code": "...",

  "tests_passed": 8,
  "tests_total": 10,
  "pass_rate": 0.8,
  "fully_correct": false,

  "compile_success": true,
  "runtime_error": false,
  "timeout": false,

  "response_tokens": 412,
  "latency_ms": 2500,

  "timestamp": "..."
}
```

Store raw responses permanently so we can calculate new metrics later without rerunning the model.

CSV or Parquet can be used for analysis, but keep raw JSONL output as the canonical experiment record.

---

## 12. Code Extraction

LLM responses may contain Markdown or explanation around the solution.

Build a consistent parser that:

1. Looks for fenced code blocks
2. Preferentially extracts a Python block
3. Falls back to the largest code block
4. Falls back to the raw response if necessary

Store both:

```text
raw_response
extracted_code
```

Also record whether extraction succeeded.

---

## 13. Code Execution / Evaluation

Run generated code in a controlled environment.

Requirements:

- Strict timeout
- No network access
- Restricted filesystem access if possible
- Capture stdout / stderr
- Prevent one bad generation from crashing the full experiment

For each solution, return:

```python
{
    "compile_success": True,
    "runtime_error": False,
    "timeout": False,
    "tests_passed": 9,
    "tests_total": 10
}
```

A failed compile should produce a pass rate of 0 rather than dropping the observation.

---

## 14. Analysis

At minimum, generate a summary table like:

| Tone | Mean Test Pass Rate | Fully Correct % | Runtime Error % | N |
| --- | ---: | ---: | ---: | ---: |
| Friendly | | | | |
| Neutral | | | | |
| Adversarial | | | | |

Also compute results **within each task** rather than only comparing raw averages.

Since every task receives every treatment, task difficulty should be accounted for.

Initial statistical analysis can include:

### Simple Analysis
- Mean pass rate by tone
- Median pass rate by tone
- Fully correct rate by tone
- Bootstrap confidence intervals

### Stronger Analysis

Regression such as:

```text
pass_rate ~ tone + task_fixed_effects
```

or a mixed model:

```text
pass_rate ~ tone + (1 | task)
```

For binary full correctness:

```text
fully_correct ~ tone + task controls
```

The important quantity is the estimated performance difference between tone conditions.

---

## 15. Visualizations

Generate a small set of useful plots.

### Primary
Mean hidden test pass rate by tone with confidence intervals.

### Also Useful
- Distribution of pass rates by tone
- Difference from neutral for each task
- Fully correct percentage by tone
- Effect of tone separated by task difficulty

Avoid producing a huge dashboard for the initial version.

---

## 16. Potential Non Coding Extensions

Design the framework so `evaluator` can later be swapped out.

Potential future task types:

### Math / Reasoning

Metrics:

```text
exact answer accuracy
reasoning error count
```

### Instruction Following

Give prompts containing several explicit constraints.

Metric:

```text
constraints_satisfied / total_constraints
```

### Information Extraction

Extract known fields from documents.

Metrics:

```text
precision
recall
F1
hallucinated fields
```

### Factual QA

Metrics:

```text
accuracy
hallucination rate
```

The same tone manipulation should be reusable across all task types.

---

## 17. Behavioral Outcomes

In addition to correctness, collect behavioral properties of responses where easy.

Examples:

```text
response length
code length
number of explanatory words
presence of apology
presence of uncertainty language
presence of confidence language
```

Later we may investigate whether tone affects **how the model behaves even when accuracy does not change**.

For example:

> Does an adversarial user make the model more verbose or more confident without actually improving correctness?

These should remain secondary analyses.

---

## 18. Suggested Repository Structure

```text
llm-tone-experiment/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── config/
│   ├── experiment.yaml
│   └── tones.yaml
│
├── data/
│   ├── tasks/
│   ├── raw_results/
│   └── processed/
│
├── src/
│   ├── prompt_builder.py
│   ├── model_client.py
│   ├── experiment_runner.py
│   ├── code_extractor.py
│   ├── evaluator.py
│   └── utils.py
│
├── analysis/
│   ├── analyze.py
│   └── plots.py
│
└── tests/
```

---

## 19. Configuration

Avoid hard coding experimental choices.

Example:

```yaml
model: MODEL_NAME

temperature: 0.7
max_tokens: 2000

tones:
  - neutral
  - friendly
  - adversarial

repetitions: 5

random_seed: 42

task_language: python

execution_timeout_seconds: 10
```

The experiment should be runnable from a single command.

Something like:

```bash
python -m src.experiment_runner --config config/experiment.yaml
```

Analysis:

```bash
python -m analysis.analyze
```

---

## 20. Reproducibility

For every experiment run, save:

- Config
- Tone templates
- Model identifier
- Random seed
- Date
- Dataset version
- Number of tasks
- Number of repetitions

Ideally create an `experiment_id` and put all outputs under:

```text
results/{experiment_id}/
```

---

## 21. Pilot Experiment

Before running hundreds of API calls, build and run a small pilot.

Suggested:

```text
5 coding tasks
3 tones
2 repetitions
30 total calls
```

Use the pilot to confirm:

- Prompts are being generated correctly
- Code extraction works
- Tests execute reliably
- Results are saved correctly
- Tone is the only meaningful prompt difference
- Analysis scripts run end to end

Do not optimize statistical conclusions from the pilot.

---

## 22. Immediate Implementation Goal

For the first pass, build a working pipeline that can:

1. Load coding tasks.
2. Generate friendly, neutral, and adversarial prompt variants.
3. Randomize trial order.
4. Query the configured LLM.
5. Save every raw response.
6. Extract generated code.
7. Execute hidden tests safely.
8. Record test pass rate and execution failures.
9. Aggregate results by tone.
10. Produce a simple CSV summary and basic plots.

Prioritize experimental correctness, reproducibility, and clean data collection over frontend/UI work.

Do not build a web app unless explicitly requested later.

---

## 23. Open Decisions

Leave the following easy to configure because we have not finalized them yet:

- Exact LLM model
- Exact benchmark / coding dataset
- Number of tasks
- Number of repetitions
- Temperature
- Whether to include the high pressure condition
- Exact wording of each tone prompt
- Whether the model returns only code or can include explanation

The infrastructure should make these changes straightforward without rewriting the experiment.

---

## 24. Core Principle

The central experimental constraint is:

> **Change the interpersonal tone, not the information available to the model.**

Any difference in performance should be plausibly attributable to the tone manipulation rather than differences in task wording, context, hints, model settings, or evaluation.
