"""Standard-library-only paired tone experiment (Python 3.11+, macOS).

Coding tasks are scored by hidden tests (tests passed / total). Math tasks have
chained parts and are scored by parts correct / total. On the hint repetition,
math prompts include a fixed wrong hint for part (a)."""
import argparse
import csv
import hashlib
import html
import itertools
import json
import os
from fractions import Fraction
from pathlib import Path
import platform
import random
import re
import signal
import statistics as stats
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def api(base, path, payload=None):
    request = Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=300) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"Ollama {error.code}: {error.read().decode()}") from error


# ---------------------------------------------------------------- datasets / design

def load_tasks(config, root=ROOT):
    tasks = []
    for name in config["datasets"]:
        tasks += read_json(Path(root) / name)
    return tasks


def subset(tasks, per_domain):
    """First `per_domain` tasks of each domain, in file order (used for dry runs)."""
    if per_domain is None:
        return tasks
    counts, out = {}, []
    for task in tasks:
        domain = task.get("domain", "coding")
        if counts.get(domain, 0) < per_domain:
            out.append(task)
            counts[domain] = counts.get(domain, 0) + 1
    return out


def uses_hint(task, repetition, config):
    return task.get("domain") == "math" and repetition == config.get("hint_repetition")


def build_prompt(task, tone, config, hint=False):
    # The tone prefix is the only thing that changes between tones. Math prompts end with a
    # fixed answer-format instruction; on the hint repetition a fixed wrong hint comes before it.
    pieces = [config["tones"][tone]] if config["tones"][tone] else []
    pieces.append(task["prompt"])
    if hint:
        pieces.append(config["hint_template"].format(part=task.get("hint_part", "a"), hint=task["hint"]))
    if task.get("domain") == "math" and "format_template" in config:
        lines = "\n".join(config["format_line"].format(label=p["label"]) for p in task["parts"])
        pieces.append(config["format_template"].format(lines=lines))
    return "\n\n".join(pieces)


def system_prompt(task, config):
    if "system_prompts" in config:
        return config["system_prompts"][task.get("domain", "coding")]
    return config["system_prompt"]  # pilot-format config


def validate(config, tasks):
    assert config.get("baseline_tone", "neutral") in config["tones"] and len(config["tones"]) >= 2
    assert config["repetitions"] >= 1 and config["bootstrap_samples"] >= 100
    assert config["timeout_seconds"] > 0 and config["max_tokens"] > 0
    assert len({task["task_id"] for task in tasks}) == len(tasks) > 0
    for task in tasks:
        domain = task.get("domain", "coding")
        assert task["prompt"]
        if "system_prompts" in config:
            assert domain in config["system_prompts"], domain
        if domain == "coding":
            assert task["entry_point"].isidentifier() and task["tests"]
            for case in task["tests"]:
                assert isinstance(case["args"], list) and "expected" in case
        elif domain == "math":
            assert task["parts"] and "hint" in task
            hinted = next(p for p in task["parts"] if p["label"] == task.get("hint_part", "a"))
            assert parse_value(task["hint"]) != parse_value(hinted["answer"])
            for part in task["parts"]:
                assert f'({part["label"]})' in task["prompt"] and parse_value(part["answer"]) is not None
        else:
            raise AssertionError(f"Unknown domain {domain}")


def plan_trials(config, tasks):
    trials = list(itertools.product(range(len(tasks)), config["tones"], range(config["repetitions"])))
    random.Random(config["random_seed"]).shuffle(trials)
    return trials


# ---------------------------------------------------------------- coding evaluation

def extract_code(response):
    blocks = re.findall(r"```([^\n`]*)\n(.*?)```", response, re.S)
    python = [code for lang, code in blocks if lang.strip().lower() in ("python", "py", "python3")]
    choices = python or [code for _, code in blocks]
    return (max(choices, key=len).strip(), "fenced") if choices else (response.strip(), "raw")


# Candidate code never receives expected answers. Each test gets a fresh process.
# Compare the returned JSON in the trusted parent, outside the candidate namespace.
WORKER = '''import contextlib, io, json, resource, runpy, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
data = json.loads(sys.stdin.read())
try:
    with contextlib.redirect_stdout(sys.stderr):
        namespace = runpy.run_path("candidate.py")
        result = namespace[data["entry_point"]](*data["args"])
    print(json.dumps({"value": result, "args_after": data["args"]}))
except BaseException as error:
    print(json.dumps({"error": type(error).__name__ + ": " + str(error)[:500]}))
'''


def sandbox_command(directory):
    if sys.platform != "darwin":
        raise RuntimeError("This evaluator requires macOS sandbox-exec; no unsandboxed fallback.")
    # Runtime libraries are readable; project/home files are not. Only the temporary
    # evaluation directory is writable. Network and child execution are denied.
    runtime = {"/System", "/usr/lib", "/usr/share", "/Library/Apple", "/opt/homebrew", str(Path(sys.base_prefix).resolve())}
    executable = str(Path(sys.base_prefix).resolve() / "Resources/Python.app/Contents/MacOS/Python")
    if not Path(executable).exists():
        executable = str(Path(sys.executable).resolve())
    reads = " ".join(f'(subpath {json.dumps(p)})' for p in sorted(runtime))
    profile = f'''(version 1)(deny default)
(allow process-exec (literal {json.dumps(executable)}))
(allow sysctl-read)(allow mach-lookup)(allow file-read-metadata)
(allow file-read* (literal "/"))
(allow file-read* {reads} (subpath {json.dumps(directory)}) (literal "/dev/null") (literal "/dev/urandom"))
(allow file-write* (subpath {json.dumps(directory)}) (literal "/dev/null"))'''
    return ["/usr/bin/sandbox-exec", "-p", profile, executable, "-I", "-B", "worker.py"]


def execute_case(code, task, case, timeout):
    with tempfile.TemporaryDirectory(prefix="tone-eval-") as temporary:
        directory = str(Path(temporary).resolve())
        Path(directory, "candidate.py").write_text(code)
        Path(directory, "worker.py").write_text(WORKER)
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(sandbox_command(directory), cwd=directory, stdin=subprocess.PIPE,
                                       stdout=stdout, stderr=stderr, start_new_session=True,
                                       env={"PATH": "/usr/bin:/bin", "HOME": directory, "TMPDIR": directory})
            timed_out = False
            try:
                process.communicate(json.dumps({"entry_point": task["entry_point"], "args": case["args"]}).encode(),
                                    timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stdout.seek(0)
            stderr.seek(0)
            output, errors = stdout.read(65536).decode(errors="replace"), stderr.read(65536).decode(errors="replace")
        if "sandbox_apply:" in errors:
            raise RuntimeError("OS sandbox unavailable: " + errors.strip())
        try:
            result = json.loads(output)
        except (ValueError, TypeError):
            result = {"error": "Missing/invalid JSON result"}
        if not isinstance(result, dict):
            result = {"error": "Invalid result type"}
        timed_out |= process.returncode in (-signal.SIGXCPU, -signal.SIGKILL)
        runtime_error = bool(result.get("error")) or process.returncode != 0
        unchanged = not task.get("check_no_mutation", False) or result.get("args_after") == case["args"]
        passed = not timed_out and not runtime_error and unchanged and result.get("value") == case["expected"]
        return {"passed": passed, "timeout": timed_out, "runtime_error": runtime_error,
                "result": result, "stderr": errors[:2000], "returncode": process.returncode}


def evaluate(code, task, timeout):
    started = time.monotonic()
    try:
        compile(code, "candidate.py", "exec")
        compiled = True
    except (SyntaxError, ValueError):
        compiled = False
    cases = [execute_case(code, task, case, timeout) for case in task["tests"]] if compiled else []
    passed = sum(case["passed"] for case in cases)
    return {"compile_success": compiled, "tests_passed": passed, "tests_total": len(task["tests"]),
            "pass_rate": passed / len(task["tests"]), "fully_correct": passed == len(task["tests"]),
            "runtime_error": any(case["runtime_error"] for case in cases),
            "timeout": any(case["timeout"] for case in cases), "case_results": cases,
            "evaluation_ms": round((time.monotonic() - started) * 1000)}


# ---------------------------------------------------------------- math evaluation
# Lenient by design: any reasonable label style and number format is accepted.

_INT = r"\d{1,3}(?:,\d{3})+(?!\d)|\d+"
NUMBER = re.compile(
    rf"\\[dt]?frac\s*\{{\s*(-?\s*(?:{_INT}))\s*\}}\s*\{{\s*(-?\s*(?:{_INT}))\s*\}}"  # \frac{a}{b}
    rf"|(-?(?:{_INT})(?:\.\d+)?)\s*/\s*({_INT})(?!\d|\.\d)"                    # a/b
    rf"|(-?(?:(?:{_INT})(?:\.\d+)?|\.\d+))\s*(%)?")                             # integer, decimal, percent
BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
# Explicit answer labels anywhere: "Answer (a): ...", "Final answer for part b = ..."
EXPLICIT_LABEL = re.compile(r"(?i)(?:\banswer\s*(?:to\s+|for\s+)?(?:part\s*)?|\\boxed\s*\{\s*(?:\\text\{\s*)?)"
                            r"\(?\s*([a-d])\s*\)?\s*\}?\s*\**\s*[:=]")
# The exact format the prompt asks for, at the start of a line: "Answer (a): ..."
FORMAT_LINE = re.compile(r"(?im)^[ \t*_]*answer\s*\(([a-d])\)[ \t*_]*:")
# Section labels at the start of a line: "(a)", "**(b)**", "### Part (c)", "a)", "Part d:", "c."
LINE_LABEL = re.compile(r"(?im)^[ \t>*#_\-]*(?:part\s*)?(?:\(\s*([a-d])\s*\)|([a-d])\s*[).:](?=\s|\*|$))")


def answer_span(text):
    """Narrow a part's text to where its final answer is. Take the last line containing a number,
    then the text after its last '=', or failing that after its last 'is'/'are'/'equals'/':'.
    The first number there is the answer. Handles worked steps ("10/190 = 1/19 ≈ 0.0526"),
    sentences ("the probability that 2 socks match is 2/9") and summaries ("190 ways to draw 2 socks")."""
    lines = [line for line in str(text).splitlines() if NUMBER.search(line)]
    if not lines:
        return ""
    line = lines[-1]
    for splitter in (r"=", r"\b(?:is|are|equals)\b|:"):
        pieces = re.split(splitter, line)
        if len(pieces) > 1 and NUMBER.search(pieces[-1]):
            return pieces[-1]
    return line


def parse_value(text, first=False):
    """Return (Fraction, exact) for the last (or first) number in text, or None."""
    matches = list(NUMBER.finditer(str(text)))
    if not matches:
        return None
    last = matches[0] if first else matches[-1]
    clean = lambda s: s.replace(",", "").replace(" ", "")
    if last.group(1) is not None:
        if int(clean(last.group(2))) == 0:
            return None
        return Fraction(int(clean(last.group(1))), int(clean(last.group(2)))), True
    if last.group(3) is not None:
        denominator = Fraction(clean(last.group(4)))
        if denominator == 0:
            return None
        return Fraction(clean(last.group(3))) / denominator, "." not in last.group(3)
    value = Fraction(clean(last.group(5)))
    exact = "." not in last.group(5)
    if last.group(6):
        return value / 100, False
    return value, exact


def same_value(parsed, truth):
    """Exact match for integers/fractions; decimals within 0.1% relative error."""
    if parsed is None:
        return False
    value, exact = parsed
    truth = Fraction(truth)
    if value == truth:
        return True
    return not exact and abs(value - truth) <= abs(truth) / 1000


def near_copy(parsed, target):
    """True if a whole-number answer is the target copied with at most one digit dropped,
    added or changed (e.g. 31187520 for 311875200). Only for targets of 5+ digits, where
    small models often slip when copying a long number."""
    if parsed is None or parsed[0].denominator != 1 or Fraction(target).denominator != 1:
        return False
    a, b = str(abs(parsed[0].numerator)), str(abs(Fraction(target).numerator))
    if len(b) < 5 or abs(len(a) - len(b)) > 1:
        return False
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1] <= 1


def extract_part_answers(response, labels):
    """Map each part label to the text of its final answer segment."""
    marks = []  # (position, end, label, explicit)
    for m in EXPLICIT_LABEL.finditer(response):
        marks.append((m.start(), m.end(), m.group(1).lower(), True))
    for m in LINE_LABEL.finditer(response):
        marks.append((m.start(), m.end(), (m.group(1) or m.group(2)).lower(), False))
    marks.sort()
    segments = {}
    for label in labels:
        own = [m for m in marks if m[2] == label]
        if not own:
            continue
        explicit = [m for m in own if m[3]]
        start, end, _, _ = (explicit or own)[-1]
        following = [m[0] for m in marks if m[0] > start and m[2] != label]
        stop = min(following) if following else len(response)
        segment = response[end:stop]
        boxed = BOXED.findall(segment)
        segments[label] = boxed[-1] if boxed else segment
    return segments


def last_numbers(text, count):
    """Fallback for answers with no part labels: the last `count` numbers, in order."""
    found = [m.group(0) for m in NUMBER.finditer(text)]
    return found[-count:] if len(found) >= count else []


def evaluate_math(response, task, hint):
    labels = [part["label"] for part in task["parts"]]
    segments = extract_part_answers(response, labels)
    mode = "labeled"
    if not segments:
        segments = dict(zip(labels, last_numbers(response, len(labels))))
        mode = "unlabeled_fallback" if segments else "unparsed"
    parsed = {label: parse_value(answer_span(segments[label]), first=True) if label in segments else None
              for label in labels}
    correct = [same_value(parsed[p["label"]], Fraction(p["answer"])) for p in task["parts"]]
    hint_label = task.get("hint_part", "a")
    hinted = parsed.get(hint_label)
    hinted_correct = correct[labels.index(hint_label)]
    total = len(task["parts"])
    # Did the response end with the requested "Answer (x): ..." lines? Recorded, not scored.
    format_labels = {m.group(1).lower() for m in FORMAT_LINE.finditer(response)}
    return {"compile_success": True, "tests_passed": sum(correct), "tests_total": total,
            "pass_rate": sum(correct) / total, "fully_correct": all(correct),
            "runtime_error": False, "timeout": False,
            "parse_mode": mode, "parts_found": sum(v is not None for v in parsed.values()),
            "format_followed": set(labels) <= format_labels,
            "part_results": [{"label": label, "parsed": None if parsed[label] is None else str(parsed[label][0]),
                              "correct": ok} for label, ok in zip(labels, correct)],
            # Accuracy on the hinted part, recorded on every math generation so the hint and
            # no-hint repetitions can be compared directly.
            "hint_part_correct": hinted_correct,
            "hint_adopted": (not hinted_correct and (same_value(hinted, Fraction(task["hint"]))
                                                     or near_copy(hinted, task["hint"]))) if hint else None}


def evaluate_response(response, task, config, hint):
    if task.get("domain", "coding") == "math":
        return evaluate_math(response, task, hint), {"extracted_code": None, "extraction_method": "math"}
    code, method = extract_code(response)
    return evaluate(code, task, config["timeout_seconds"]), {"extracted_code": code, "extraction_method": method}


# ---------------------------------------------------------------- running

def append_record(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(args):
    config = read_json(args.config)
    all_tasks = load_tasks(config, Path(args.config).resolve().parent)
    tasks = subset(all_tasks, args.tasks_per_domain)
    validate(config, tasks)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / "raw.jsonl"
    if raw.exists():
        raise ValueError(f"Refusing to overwrite {raw}; use a new --output directory.")
    if any(t.get("domain", "coding") == "coding" for t in tasks):
        # Fail before generation if isolation is unavailable.
        probe = execute_case("def probe(): return 1", {"entry_point": "probe"}, {"args": [], "expected": 1}, 3)
        if not probe["passed"]:
            raise RuntimeError(f"Evaluator preflight failed: {probe}")
    base = config["base_url"].rstrip("/")
    available = api(base, "/api/tags")["models"]
    model = next((m for m in available if m["name"] == config["model"]), None)
    if model is None:
        raise ValueError(f"Download model first: ollama pull {config['model']}")
    api(base, "/api/generate", {"model": config["model"], "prompt": "", "stream": False})
    trials = plan_trials(config, tasks)
    dataset_bytes = b"".join((Path(args.config).resolve().parent / name).read_bytes() for name in config["datasets"])
    metadata = {"kind": "metadata", "schema_version": 2, "started_at": datetime.now(timezone.utc).isoformat(),
                "config": config, "tasks": tasks, "tasks_per_domain": args.tasks_per_domain, "model": model,
                "model_details": api(base, "/api/show", {"model": config["model"]}),
                "ollama": api(base, "/api/version"), "python": sys.version, "platform": platform.platform(),
                "source_sha256": digest(Path(__file__).read_bytes()),
                "dataset_sha256": digest(dataset_bytes),
                "trial_order": trials, "planned_generations": len(trials)}
    append_record(raw, metadata)
    for index, (task_index, tone, repetition) in enumerate(trials):
        task = tasks[task_index]
        hint = uses_hint(task, repetition, config)
        # Same seed for every tone of a given task/repetition, so tones are paired.
        seed = config["random_seed"] + task_index * config["repetitions"] + repetition
        prompt = build_prompt(task, tone, config, hint)
        request = {"model": config["model"], "stream": False,
                   "messages": [{"role": "system", "content": system_prompt(task, config)},
                                {"role": "user", "content": prompt}],
                   "options": {"temperature": config["temperature"], "top_p": config["top_p"],
                               "num_predict": config["max_tokens"], "num_ctx": config["context_tokens"], "seed": seed}}
        started = time.monotonic()
        response = api(base, "/api/chat", request)  # No hidden retries or conversation carryover.
        content = response["message"]["content"]
        generation = {"kind": "generation", "trial_id": index, "task_id": task["task_id"],
                      "domain": task.get("domain", "coding"), "tone": tone, "repetition": repetition,
                      "hint": hint, "seed": seed, "prompt": prompt, "request": request,
                      "api_response": response, "raw_response": content,
                      "response_chars": len(content), "response_tokens": response.get("eval_count"),
                      "prompt_tokens": response.get("prompt_eval_count"),
                      "truncated": response.get("done_reason") == "length",
                      "latency_ms": round((time.monotonic() - started) * 1000),
                      "timestamp": datetime.now(timezone.utc).isoformat()}
        append_record(raw, generation)  # Persist before running any candidate code.
        evaluation, extraction = evaluate_response(content, task, config, hint)
        append_record(raw, {"kind": "evaluation", "trial_id": index, **extraction, **evaluation})
        print(f"{index + 1}/{len(trials)} {task['task_id']} {tone}{' +hint' if hint else ''} "
              f"rep={repetition + 1}: {evaluation['tests_passed']}/{evaluation['tests_total']}", flush=True)
    analyze(directory)


# ---------------------------------------------------------------- analysis

def bootstrap(values, samples, seed):
    rng = random.Random(seed)
    means = sorted(stats.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


GROUPS = [  # (name, description, filter)
    ("all_no_hint", "All tasks, no hint (headline)", lambda r: not r["hint"]),
    ("coding", "Coding tasks", lambda r: r["domain"] == "coding"),
    ("math_no_hint", "Math tasks, no hint", lambda r: r["domain"] == "math" and not r["hint"]),
    ("math_hint", "Math tasks, with wrong hint", lambda r: r["domain"] == "math" and r["hint"]),
]


def summarize(rows, config, tones, baseline):
    """Per-tone stats. Repetitions are averaged within each task; tasks get equal weight;
    differences from the baseline tone are paired by task."""
    task_ids = sorted({r["task_id"] for r in rows})
    def task_means(key):
        return {tone: [stats.mean(float(r[key]) for r in rows if r["tone"] == tone and r["task_id"] == task)
                       for task in task_ids] for tone in tones}
    means = task_means("pass_rate")
    summary = []
    for tone in tones:
        group = [r for r in rows if r["tone"] == tone]
        values = means[tone]
        low, high = bootstrap(values, config["bootstrap_samples"], config["random_seed"])
        differences = [a - b for a, b in zip(values, means[baseline])]
        delta_low, delta_high = bootstrap(differences, config["bootstrap_samples"], config["random_seed"])
        tokens = [r["response_tokens"] for r in group if r.get("response_tokens") is not None]
        hinted = [r["hint_adopted"] for r in group if r.get("hint_adopted") is not None]
        formatted = [r["format_followed"] for r in group if r.get("format_followed") is not None]
        part_a = [r["hint_part_correct"] for r in group if r.get("hint_part_correct") is not None]
        summary.append({"tone": tone, "n_generations": len(group), "n_tasks": len(task_ids),
                        "mean_pass_rate": stats.mean(values), "ci_low": low, "ci_high": high,
                        "difference_from_baseline": stats.mean(differences),
                        "difference_ci_low": delta_low, "difference_ci_high": delta_high,
                        "fully_correct_rate": stats.mean(r["fully_correct"] for r in group),
                        "runtime_error_rate": stats.mean(r["runtime_error"] for r in group),
                        "timeout_rate": stats.mean(r["timeout"] for r in group),
                        "compile_failure_rate": stats.mean(not r["compile_success"] for r in group),
                        "truncated_rate": stats.mean(bool(r.get("truncated")) for r in group),
                        "mean_response_tokens": stats.mean(tokens) if tokens else "",
                        "format_followed_rate": stats.mean(formatted) if formatted else "",
                        "part_a_correct_rate": stats.mean(part_a) if part_a else "",
                        "hint_adoption_rate": stats.mean(hinted) if hinted else ""})
    return summary, task_ids, means


def analyze(directory):
    directory = Path(directory)
    records = [json.loads(line) for line in (directory / "raw.jsonl").read_text().splitlines()]
    metadata = records[0]
    config = metadata["config"]
    generations = {r["trial_id"]: r for r in records if r["kind"] == "generation"}
    evaluations = {r["trial_id"]: r for r in records if r["kind"] == "evaluation"}
    if len(generations) != metadata["planned_generations"] or generations.keys() != evaluations.keys():
        raise ValueError("Incomplete run: raw responses preserved; finish/re-evaluate before reporting.")
    rows = []
    for key, g in generations.items():
        row = {**g, **evaluations[key]}
        row.setdefault("domain", "coding")  # pilot records predate domains and hints
        row.setdefault("hint", False)
        rows.append(row)
    tones = list(config["tones"])
    baseline = config.get("baseline_tone", "neutral")
    results = []
    for name, description, keep in GROUPS:
        group_rows = [r for r in rows if keep(r)]
        if not group_rows:
            continue
        summary, task_ids, means = summarize(group_rows, config, tones, baseline)
        results.append({"name": name, "description": description, "summary": summary,
                        "task_ids": task_ids, "means": means})
    with (directory / "summary.csv").open("w", newline="") as stream:
        fields = ["group"] + list(results[0]["summary"][0].keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerows({"group": result["name"], **row} for row in result["summary"])
    render_report(directory, metadata, results, baseline)
    print(f"Report: {directory / 'report.html'}", flush=True)


def render_report(directory, metadata, results, baseline):
    esc = html.escape
    config = metadata["config"]

    def chart(summary):
        height = 40 + 46 * len(summary)
        svg = f'<svg viewBox="0 0 780 {height}" role="img" aria-label="Mean score by tone with 95 percent intervals">'
        for i, s in enumerate(summary):
            y = 30 + i * 46
            svg += (f'<text x="0" y="{y+5}">{esc(s["tone"])}</text>'
                    f'<rect x="140" y="{y-13}" width="{500*s["mean_pass_rate"]:.2f}" height="26" fill="#bdd9eb"/>'
                    f'<path d="M {140+500*s["ci_low"]:.2f} {y} H {140+500*s["ci_high"]:.2f}" stroke="#153f59" stroke-width="3"/>'
                    f'<circle cx="{140+500*s["mean_pass_rate"]:.2f}" cy="{y}" r="5" fill="#153f59"/>'
                    f'<text x="660" y="{y+5}">{s["mean_pass_rate"]:.1%}</text>')
        return svg + (f'<text x="140" y="{height-6}">0%</text><text x="380" y="{height-6}">50%</text>'
                      f'<text x="620" y="{height-6}">100%</text></svg>')

    def table(summary, hint, math):
        head = ('<tr><th>Tone</th><th>Mean score</th><th>Fully correct</th><th>Runtime error</th><th>Truncated</th>'
                f'<th>N</th><th>Δ vs {esc(baseline)}, pp [95% CI]</th>'
                + ('<th>Part (a) correct</th><th>Format followed</th>' if math else '')
                + ('<th>Adopted wrong hint</th>' if hint else '') + '</tr>')
        def rate(value):
            return f'<td>{value:.1%}</td>' if value != "" else '<td>n/a</td>'
        body = "".join(
            f'<tr><td>{esc(s["tone"])}</td><td>{s["mean_pass_rate"]:.1%}</td><td>{s["fully_correct_rate"]:.1%}</td>'
            f'<td>{s["runtime_error_rate"]:.1%}</td><td>{s["truncated_rate"]:.1%}</td><td>{s["n_generations"]}</td>'
            f'<td>{100*s["difference_from_baseline"]:+.1f} [{100*s["difference_ci_low"]:+.1f}, {100*s["difference_ci_high"]:+.1f}]</td>'
            + (rate(s["part_a_correct_rate"]) + rate(s["format_followed_rate"]) if math else '')
            + (rate(s["hint_adoption_rate"]) if hint else '') + '</tr>'
            for s in summary)
        return f'<table>{head}{body}</table>'

    def task_table(result):
        tones = list(result["means"])
        rows = "".join(f'<tr><td>{esc(task)}</td>' + "".join(f'<td>{result["means"][t][i]:.0%}</td>' for t in tones) + '</tr>'
                       for i, task in enumerate(result["task_ids"]))
        return f'<table><tr><th>Task</th>{"".join("<th>"+esc(t)+"</th>" for t in tones)}</tr>{rows}</table>'

    sections = ""
    for result in results:
        hint = result["name"] == "math_hint"
        sections += (f'<h2>{esc(result["description"])}</h2>{chart(result["summary"])}'
                     f'{table(result["summary"], hint, result["name"].startswith("math"))}')
    headline = results[0]
    n_tasks = len(metadata["tasks"])
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Tone experiment</title>
<style>body{{font:17px/1.55 system-ui;max-width:980px;margin:48px auto;padding:0 24px;color:#183042}}
h1{{line-height:1.2}}h2{{margin-top:36px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin-top:12px}}
th,td{{padding:8px;text-align:left;border-bottom:1px solid #dbe3e8}}.note{{background:#eff5f8;padding:18px}}
code{{overflow-wrap:anywhere}}svg{{width:100%}}</style>
<h1>Does user tone change LLM performance?</h1>
<p>{n_tasks} tasks × {len(config["tones"])} tones × {config["repetitions"]} repetitions
= {metadata["planned_generations"]} generations. Model: <strong>{esc(config["model"])}</strong>.</p>
<p class="note"><strong>How to read this:</strong> Score is the share of hidden tests passed (coding) or parts answered
correctly (math). Repetitions are averaged within each task and every task gets equal weight. Bars show the mean;
lines show 95% bootstrap intervals that resample whole tasks. Differences are paired by task against the
<strong>{esc(baseline)}</strong> tone. On math repetition {config.get("hint_repetition", 0) + 1}, the prompt includes a fixed
wrong hint for part (a); those generations are reported separately and never mixed into the headline.
"Adopted wrong hint" is the share whose part (a) answer is the hint (or the hint copied with a one-digit slip);
comparing "Part (a) correct" between the two math tables shows how much the hint pulls the model off the right answer.</p>
{sections}
<h2>Within-task results: {esc(headline["description"])}</h2>{task_table(headline)}
<h2>What was controlled</h2><p>Task text, hint wording, system instructions, model and sampling settings are identical
across tones; only the prefix changes. Every generation is a fresh two-message conversation without tools. Trial order
is shuffled with seed {config["random_seed"]}; generation seeds are shared across tones within each task/repetition.
Temperature {config["temperature"]}, top-p {config["top_p"]}, output limit {config["max_tokens"]} tokens. No retries.</p>
<ul>{"".join("<li><strong>"+esc(t)+":</strong> "+(esc(p) or "<em>(no prefix)</em>")+"</li>" for t, p in config["tones"].items())}</ul>
<h2>Audit trail</h2><p><a href="summary.csv">Summary CSV</a> · <a href="raw.jsonl">Raw generations, evaluations and run metadata</a></p>
<p>Model digest: <code>{esc(str(metadata["model"].get("digest")))}</code><br>
Dataset SHA-256: <code>{esc(str(metadata["dataset_sha256"]))}</code><br>Runner SHA-256: <code>{esc(str(metadata["source_sha256"]))}</code><br>
Ollama: {esc(str(metadata["ollama"]))}<br>Python: {esc(str(metadata["python"]))}<br>Platform: {esc(str(metadata["platform"]))}</p></html>'''
    (directory / "report.html").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "analyze"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tasks-per-domain", type=int, default=None,
                        help="Use only the first N tasks of each domain.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Shortcut for --tasks-per-domain 2 with output results/dry_run.")
    args = parser.parse_args()
    if args.dry_run and args.tasks_per_domain is None:
        args.tasks_per_domain = 2
    if args.output is None:
        args.output = ROOT / "results" / ("dry_run" if args.dry_run else "full_run")
    if args.command == "run":
        run(args)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
