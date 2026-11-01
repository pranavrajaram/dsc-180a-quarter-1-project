"""Crossed model × tone × misleading-hint experiment; standard library only."""
import argparse
import csv
from fractions import Fraction
import itertools
import json
from pathlib import Path
import platform
import random
import statistics as stats
import sys
import time

from experiment import (ROOT, api, append_record, build_prompt, digest,
                        evaluate_math, read_json, system_prompt)


def design(config, tasks):
    """Block by model to avoid reloading weights; shuffle all trials within blocks."""
    rng = random.Random(config['random_seed'])
    models = list(config['models'])
    rng.shuffle(models)
    trials = []
    for model in models:
        block = [dict(model=model, task_id=task['task_id'], tone=tone,
                      repetition=rep, hint=hint,
                      seed=config['random_seed'] + i * config['repetitions'] + rep)
                 for i, task in enumerate(tasks)
                 for tone, rep, hint in itertools.product(
                     config['tones'], range(config['repetitions']), (False, True))]
        rng.shuffle(block)
        trials.extend(block)
    return [dict(trial_id=i, **trial) for i, trial in enumerate(trials)]


def score(response, task, hint):
    result = evaluate_math(response, task, hint)
    first = next(p for p in result['part_results'] if p['label'] == task['hint_part'])
    result['exact_hint_match'] = (first['parsed'] is not None and
                                  Fraction(first['parsed']) == Fraction(task['hint']))
    later = [p['correct'] for p in result['part_results'] if p['label'] != task['hint_part']]
    result['downstream_accuracy'] = stats.mean(later)
    return result


def read_run(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records or records[0]['kind'] != 'metadata':
        raise ValueError('Missing metadata')
    rows = records[1:]
    ids = [r['trial_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate trial IDs')
    return records[0], rows


def run(args):
    config = read_json(args.config)
    tasks = read_json(args.config.resolve().parent / config['dataset'])
    plan = design(config, tasks)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'raw.jsonl'
    sources = {name: digest((ROOT / name).read_bytes())
               for name in ('followup.py', 'experiment.py')}
    tags = api(config['base_url'], '/api/tags')['models']
    models = {name: next(m for m in tags if m['name'] == name) for name in config['models']}
    if path.exists():
        metadata, rows = read_run(path)
        if any(metadata[k] != v for k, v in
               [('config', config), ('tasks', tasks), ('plan', plan), ('sources', sources)]):
            raise ValueError('Config, tasks or source changed; choose a new output directory')
        if any(metadata['models'][m]['digest'] != models[m]['digest'] for m in models):
            raise ValueError('Model digest changed')
    else:
        metadata = dict(kind='metadata', config=config, tasks=tasks, plan=plan,
                        sources=sources, models=models,
                        model_details={m: api(config['base_url'], '/api/show', {'model': m}) for m in models},
                        ollama=api(config['base_url'], '/api/version'),
                        python=sys.version, platform=platform.platform())
        append_record(path, metadata)
        rows = []
    done = {r['trial_id'] for r in rows}
    task_map = {t['task_id']: t for t in tasks}
    for trial in plan:
        if trial['trial_id'] in done:
            continue
        task = task_map[trial['task_id']]
        request = dict(model=trial['model'], stream=False, keep_alive='30m',
                       messages=[dict(role='system', content=system_prompt(task, config)),
                                 dict(role='user', content=build_prompt(task, trial['tone'], config, trial['hint']))],
                       options=dict(temperature=config['temperature'], top_p=config['top_p'],
                                    num_predict=config['max_tokens'], num_ctx=config['context_tokens'],
                                    seed=trial['seed']))
        start = time.monotonic()
        response = api(config['base_url'], '/api/chat', request)
        # Save each response before scoring; analysis can regrade it later.
        append_record(path, dict(kind='generation', **trial, request=request,
                                api_response=response, elapsed_seconds=time.monotonic() - start))
        result = score(response['message']['content'], task, trial['hint'])
        print(f"{trial['trial_id']+1}/{len(plan)} {trial['model']} {trial['task_id']} "
              f"{trial['tone']} hint={trial['hint']} score={result['pass_rate']:.2f}", flush=True)
    analyze(args.output)


def interval(values, samples, seed):
    rng = random.Random(seed)
    boot = sorted(stats.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return stats.mean(values), boot[int(.025 * samples)], boot[int(.975 * samples)]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(directory):
    metadata, generations = read_run(directory / 'raw.jsonl')
    config, tasks = metadata['config'], metadata['tasks']
    expected = {p['trial_id']: p for p in metadata['plan']}
    if set(expected) != {g['trial_id'] for g in generations}:
        raise ValueError(f"Incomplete run: {len(generations)}/{len(expected)}; resume with run")
    task_map = {t['task_id']: t for t in tasks}
    rows = []
    for g in generations:
        if any(g[k] != v for k, v in expected[g['trial_id']].items()):
            raise ValueError('Trial identity differs from saved plan')
        response = g['api_response']
        rows.append(dict(g, **score(response['message']['content'], task_map[g['task_id']], g['hint']),
                         truncated=response.get('done_reason') == 'length'))
    metrics = ['pass_rate', 'hint_part_correct', 'downstream_accuracy', 'exact_hint_match', 'truncated']
    cells = {}
    summary = []
    for model, tone, hint in itertools.product(config['models'], config['tones'], (False, True)):
        selected = [r for r in rows if (r['model'], r['tone'], r['hint']) == (model, tone, hint)]
        entry = dict(model=model, tone=tone, hint=hint, generations=len(selected))
        for metric in metrics:
            values = [stats.mean(r[metric] for r in selected if r['task_id'] == t['task_id']) for t in tasks]
            cells[model, tone, hint, metric] = values
            mean, low, high = interval(values, config['bootstrap_samples'], config['random_seed'])
            entry.update({metric: mean, metric+'_low': low, metric+'_high': high})
        summary.append(entry)
    effects = []
    def contrast(name, model, tone, metric, values):
        mean, low, high = interval(values, config['bootstrap_samples'], config['random_seed'])
        effects.append(dict(contrast=name, model=model, tone=tone, metric=metric,
                            estimate=mean, ci_low=low, ci_high=high))
    baseline = config['baseline_tone']
    for model in config['models']:
        for tone in config['tones']:
            for metric in metrics[:4]:
                delta = [b-a for a,b in zip(cells[model,tone,False,metric], cells[model,tone,True,metric])]
                contrast('hint_minus_no_hint', model, tone, metric, delta)
                if tone != baseline:
                    base = [b-a for a,b in zip(cells[model,baseline,False,metric], cells[model,baseline,True,metric])]
                    contrast('tone_by_hint_interaction', model, tone, metric, [a-b for a,b in zip(delta,base)])
                    contrast('tone_minus_control_no_hint', model, tone, metric,
                             [a-b for a,b in zip(cells[model,tone,False,metric], cells[model,baseline,False,metric])])
    small, large = config['models']
    for tone in config['tones']:
        for hint in (False, True):
            for metric in metrics[:4]:
                contrast(f'larger_minus_smaller_hint_{hint}', large, tone, metric,
                         [b-a for a,b in zip(cells[small,tone,hint,metric], cells[large,tone,hint,metric])])
    write_csv(directory / 'summary.csv', summary)
    write_csv(directory / 'effects.csv', effects)
    print(f"Saved summary.csv and effects.csv to {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'analyze'])
    parser.add_argument('--config', type=Path, default=ROOT/'followup.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/model_hint')
    args = parser.parse_args()
    run(args) if args.command == 'run' else analyze(args.output)


if __name__ == '__main__':
    main()
