"""Small, standard-library-only paired tone experiment (Python 3.11+, macOS)."""
import argparse
import csv
import hashlib
import html
import itertools
import json
import os
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


def build_prompt(task, tone, config):
    return config["tones"][tone] + "\n\n" + task["prompt"]


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


def validate(config, tasks):
    assert "neutral" in config["tones"] and len(config["tones"]) >= 2
    assert config["repetitions"] >= 1 and config["bootstrap_samples"] >= 100
    assert config["timeout_seconds"] > 0 and config["max_tokens"] > 0
    assert len({task["task_id"] for task in tasks}) == len(tasks) > 0
    for task in tasks:
        assert task["prompt"] and task["entry_point"].isidentifier() and task["tests"]
        for case in task["tests"]:
            assert isinstance(case["args"], list) and "expected" in case


def plan_trials(config, tasks):
    trials = list(itertools.product(range(len(tasks)), config["tones"], range(config["repetitions"])))
    random.Random(config["random_seed"]).shuffle(trials)
    return trials


def append_record(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(args):
    config, tasks = read_json(args.config), read_json(args.tasks)
    validate(config, tasks)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / "raw.jsonl"
    if raw.exists():
        raise ValueError(f"Refusing to overwrite {raw}; use a new --output directory.")
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
    metadata = {"kind": "metadata", "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
                "config": config, "tasks": tasks, "model": model,
                "model_details": api(base, "/api/show", {"model": config["model"]}),
                "ollama": api(base, "/api/version"), "python": sys.version, "platform": platform.platform(),
                "source_sha256": digest(Path(__file__).read_bytes()),
                "dataset_sha256": digest(Path(args.tasks).read_bytes()),
                "trial_order": trials, "planned_generations": len(trials)}
    append_record(raw, metadata)
    for index, (task_index, tone, repetition) in enumerate(trials):
        task = tasks[task_index]
        seed = config["random_seed"] + task_index * config["repetitions"] + repetition
        prompt = build_prompt(task, tone, config)
        request = {"model": config["model"], "stream": False,
                   "messages": [{"role": "system", "content": config["system_prompt"]},
                                {"role": "user", "content": prompt}],
                   "options": {"temperature": config["temperature"], "top_p": config["top_p"],
                               "num_predict": config["max_tokens"], "num_ctx": config["context_tokens"], "seed": seed}}
        started = time.monotonic()
        response = api(base, "/api/chat", request)  # No hidden retries or conversation carryover.
        code, extraction = extract_code(response["message"]["content"])
        generation = {"kind": "generation", "trial_id": index, "task_id": task["task_id"], "tone": tone,
                      "repetition": repetition, "seed": seed, "prompt": prompt, "request": request,
                      "api_response": response, "raw_response": response["message"]["content"],
                      "extracted_code": code, "extraction_method": extraction,
                      "response_chars": len(response["message"]["content"]), "code_chars": len(code),
                      "response_tokens": response.get("eval_count"), "prompt_tokens": response.get("prompt_eval_count"),
                      "latency_ms": round((time.monotonic() - started) * 1000),
                      "timestamp": datetime.now(timezone.utc).isoformat()}
        append_record(raw, generation)  # Persist before running any candidate code.
        evaluation = evaluate(code, task, config["timeout_seconds"])
        append_record(raw, {"kind": "evaluation", "trial_id": index, **evaluation})
        print(f"{index + 1}/{len(trials)} {task['task_id']} {tone} rep={repetition + 1}: "
              f"{evaluation['tests_passed']}/{evaluation['tests_total']}", flush=True)
    analyze(directory)


def bootstrap(values, samples, seed):
    rng = random.Random(seed)
    means = sorted(stats.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def analyze(directory):
    records = [json.loads(line) for line in (directory / "raw.jsonl").read_text().splitlines()]
    metadata = records[0]
    config = metadata["config"]
    generations = {r["trial_id"]: r for r in records if r["kind"] == "generation"}
    evaluations = {r["trial_id"]: r for r in records if r["kind"] == "evaluation"}
    if len(generations) != metadata["planned_generations"] or generations.keys() != evaluations.keys():
        raise ValueError("Incomplete run: raw responses preserved; finish/re-evaluate before reporting.")
    rows = [{**g, **evaluations[key]} for key, g in generations.items()]
    task_ids = [t["task_id"] for t in metadata["tasks"]]
    means = {tone: [stats.mean(r["pass_rate"] for r in rows if r["tone"] == tone and r["task_id"] == task)
                    for task in task_ids] for tone in config["tones"]}
    summary = []
    for tone, values in means.items():
        group = [r for r in rows if r["tone"] == tone]
        low, high = bootstrap(values, config["bootstrap_samples"], config["random_seed"])
        differences = [a - b for a, b in zip(values, means["neutral"])]
        delta_low, delta_high = bootstrap(differences, config["bootstrap_samples"], config["random_seed"])
        summary.append({"tone": tone, "n": len(group), "mean_pass_rate": stats.mean(values),
                        "ci_low": low, "ci_high": high, "median_pass_rate": stats.median(r["pass_rate"] for r in group),
                        "fully_correct_rate": stats.mean(r["fully_correct"] for r in group),
                        "runtime_error_rate": stats.mean(r["runtime_error"] for r in group),
                        "timeout_rate": stats.mean(r["timeout"] for r in group),
                        "compile_failure_rate": stats.mean(not r["compile_success"] for r in group),
                        "mean_response_tokens": stats.mean(r["response_tokens"] for r in group),
                        "difference_from_neutral": stats.mean(differences),
                        "difference_ci_low": delta_low, "difference_ci_high": delta_high})
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    render_report(directory, metadata, summary, task_ids, means)
    print(f"Report: {directory / 'report.html'}", flush=True)


def render_report(directory, metadata, summary, task_ids, means):
    esc = html.escape
    table = "".join(f'<tr><td>{esc(s["tone"])}</td><td>{s["mean_pass_rate"]:.1%}</td>'
                    f'<td>{s["fully_correct_rate"]:.1%}</td><td>{s["runtime_error_rate"]:.1%}</td>'
                    f'<td>{s["n"]}</td><td>{100*s["difference_from_neutral"]:+.1f} '
                    f'[{100*s["difference_ci_low"]:+.1f}, {100*s["difference_ci_high"]:+.1f}]</td></tr>' for s in summary)
    task_table = "".join(f'<tr><td>{esc(task)}</td>' + ''.join(f'<td>{values[i]:.1%}</td>' for values in means.values())
                         + '</tr>' for i, task in enumerate(task_ids))
    # Embedded SVG keeps the shareable report self-contained and dependency-free.
    chart = '<svg viewBox="0 0 780 240" role="img" aria-label="Mean pass rates and task-bootstrap 95 percent intervals">'
    for i, s in enumerate(summary):
        y = 35 + i * 60
        chart += (f'<text x="0" y="{y+5}">{esc(s["tone"])}</text>'
                  f'<rect x="140" y="{y-14}" width="{500*s["mean_pass_rate"]}" height="28" fill="#bdd9eb"/>'
                  f'<path d="M {140+500*s["ci_low"]} {y} H {140+500*s["ci_high"]}" stroke="#153f59" stroke-width="3"/>'
                  f'<circle cx="{140+500*s["mean_pass_rate"]}" cy="{y}" r="5" fill="#153f59"/>'
                  f'<text x="660" y="{y+5}">{s["mean_pass_rate"]:.1%}</text>')
    chart += '<text x="140" y="220">0%</text><text x="380" y="220">50%</text><text x="620" y="220">100%</text></svg>'
    config = metadata["config"]
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Tone experiment pilot</title>
<style>body{{font:17px/1.55 system-ui;max-width:950px;margin:48px auto;padding:0 24px;color:#183042}}
h1{{line-height:1.2}}h2{{margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:15px}}
th,td{{padding:10px;text-align:left;border-bottom:1px solid #dbe3e8}}.note{{background:#eff5f8;padding:18px}}
code{{overflow-wrap:anywhere}}svg{{width:100%}}</style>
<h1>Does user tone change coding performance?</h1>
<p>Preliminary local-model pilot · {len(task_ids)} tasks × {config['repetitions']} repetitions × {len(summary)} tones
= {metadata['planned_generations']} generations. Model: <strong>{esc(config['model'])}</strong>.</p>
<p class="note"><strong>Interpretation:</strong> This is a pipeline pilot, not evidence that one tone is generally better.
Only {len(task_ids)} hand-authored tasks were sampled; confidence intervals are exploratory and may be unstable.
The report includes every planned generation, including invalid code and execution failures.</p>
<h2>Overall results</h2>{chart}<p>Bars: mean hidden-test pass rate. Lines: 95% percentile bootstrap intervals,
resampling tasks after averaging repetitions (10,000 draws by default).</p>
<table><tr><th>Tone</th><th>Test pass rate</th><th>Fully correct</th><th>Runtime error</th><th>N</th>
<th>Δ vs neutral, pp [95% CI]</th></tr>{table}</table>
<h2>Within-task results</h2><p>Each cell averages the repeated generations. Each task has eight hidden tests.</p>
<table><tr><th>Task</th>{''.join('<th>'+esc(t)+'</th>' for t in means)}</tr>{task_table}</table>
<h2>What was controlled</h2><p>Task text and system instructions were identical across conditions. Only the prefix changed.
Every generation used a fresh two-message conversation, without tools. Trial order was shuffled with seed {config['random_seed']}.
Sampling temperature {config['temperature']}, top-p {config['top_p']}, output limit {config['max_tokens']} tokens.
Generation seeds were paired across tones within each task/repetition. No retries or solution repairs.</p>
<ul>{''.join('<li><strong>'+esc(t)+':</strong> '+esc(p)+'</li>' for t,p in config['tones'].items())}</ul>
<h2>Limits and next experiment</h2><p>One small quantized model, fixed tone phrases, two repeats, and five custom tasks cannot establish a general tone effect.
Difficulty labels are informal. Training overlap is unknown. Hidden tests measure correctness only on covered inputs,
not a proof of correctness. Wording and token length still differ between prefixes. Seeds improve traceability but do not
promise bit-for-bit reproducibility across hardware/runtime versions. Latency includes model warm-up and is not a treatment outcome.</p>
<p>Freeze this pilot; next use 30–50 independently selected tasks and 3–5 repeats, and repeat on another model.
Keep task-level paired comparisons as the primary analysis; do not treat individual test cases as independent observations.</p>
<h2>Audit trail</h2><p><a href="summary.csv">Download summary CSV</a> · <a href="raw.jsonl">Raw generations, evaluations and full run metadata</a></p>
<p>Model digest: <code>{esc(metadata['model']['digest'])}</code><br>
Dataset SHA-256: <code>{metadata['dataset_sha256']}</code><br>Runner SHA-256: <code>{metadata['source_sha256']}</code><br>
Ollama: {esc(str(metadata['ollama']))}<br>Python: {esc(metadata['python'])}<br>Platform: {esc(metadata['platform'])}</p></html>'''
    (directory / "report.html").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "analyze"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "pilot")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
