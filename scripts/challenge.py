"""Branch saved first answers into matched two-turn challenges; standard library only."""
import argparse
import csv
from fractions import Fraction
from pathlib import Path
import platform
import random
import sys
import time

from scripts.experiment import ROOT, api, append_record, digest, read_json, same_value
from scripts.followup import read_run, score, write_csv


def prepare(config, config_path):
    source = config_path.resolve().parent / config['baseline']
    metadata, rows = read_run(source)
    expected = {p['trial_id']: p for p in metadata['plan']}
    if len(rows) != len(expected) or any(
        r['trial_id'] not in expected or
        any(r[k] != v for k, v in expected[r['trial_id']].items()) for r in rows
    ):
        raise ValueError('Baseline run is incomplete or differs from its saved plan')
    parents = [r for r in rows if r['tone'] == 'control' and not r['hint']]
    rng = random.Random(config['random_seed'])
    plan = []
    for model in metadata['config']['models']:
        block = [dict(parent_id=p['trial_id'], model=model, task_id=p['task_id'],
                      repetition=p['repetition'], condition=condition,
                      seed=config['random_seed'] + p['seed'])
                 for p in parents if p['model'] == model
                 for condition in config['conditions']]
        rng.shuffle(block)
        plan.extend(block)
    return dict(config=config, baseline_sha256=digest(source.read_bytes()),
                tasks=metadata['tasks'], parents=parents, models=metadata['models'],
                baseline_ollama=metadata['ollama'],
                plan=[dict(trial_id=i, **trial) for i, trial in enumerate(plan)])


def request_for(trial, parent, task, config):
    correct = next(p['answer'] for p in task['parts'] if p['label'] == task['hint_part'])
    challenge = config['conditions'][trial['condition']].format(
        part=task['hint_part'], correct=correct, wrong=task['hint'])
    lines = '\n'.join(f"Answer ({p['label']}): <answer>" for p in task['parts'])
    challenge += '\n\n' + config['suffix'].format(lines=lines)
    return dict(model=trial['model'], stream=False, keep_alive='30m',
                messages=parent['request']['messages'] + [
                    dict(role='assistant', content=parent['api_response']['message']['content']),
                    dict(role='user', content=challenge)],
                options=dict(parent['request']['options'], seed=trial['seed'],
                             num_ctx=config['context_tokens']))


def validate_rows(metadata, rows, complete=False):
    plan = {p['trial_id']: p for p in metadata['plan']}
    if len({r['trial_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate trial IDs')
    parents = {p['trial_id']: p for p in metadata['parents']}
    tasks = {t['task_id']: t for t in metadata['tasks']}
    for row in rows:
        trial = plan.get(row['trial_id'])
        if trial is None or any(row[k] != v for k, v in trial.items()):
            raise ValueError('Trial differs from saved plan')
        if row['request'] != request_for(trial, parents[trial['parent_id']],
                                         tasks[trial['task_id']], metadata['config']):
            raise ValueError('Conversation differs from saved parent or challenge')
    if complete and len(rows) != len(plan):
        raise ValueError(f'Incomplete run: {len(rows)}/{len(plan)}; resume collection')


def run(args):
    config = read_json(args.config)
    specification = prepare(config, args.config)
    sources = {f: digest((ROOT / f).read_bytes()) for f in
               ('scripts/challenge.py', 'scripts/followup.py', 'scripts/experiment.py')}
    version = api(config['base_url'], '/api/version')
    if version != specification['baseline_ollama']:
        raise ValueError('Use the same Ollama version as the baseline')
    installed = {m['name']: m for m in api(config['base_url'], '/api/tags')['models']}
    for name, model in specification['models'].items():
        if installed.get(name, {}).get('digest') != model['digest']:
            raise ValueError(f'Model digest differs from baseline: {name}')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'raw.jsonl'
    if path.exists():
        metadata, rows = read_run(path)
        if any(metadata[k] != v for k, v in specification.items()) or metadata['sources'] != sources:
            raise ValueError('Study inputs changed; use a new output directory')
        validate_rows(metadata, rows)
    else:
        metadata = dict(kind='metadata', **specification, sources=sources,
                        ollama=version, python=sys.version, platform=platform.platform())
        append_record(path, metadata)
        rows = []
    parents = {p['trial_id']: p for p in metadata['parents']}
    tasks = {t['task_id']: t for t in metadata['tasks']}
    done = {r['trial_id'] for r in rows}
    for trial in metadata['plan']:
        if trial['trial_id'] in done:
            continue
        request = request_for(trial, parents[trial['parent_id']], tasks[trial['task_id']], config)
        start = time.monotonic()
        response = api(config['base_url'], '/api/chat', request)
        if not response.get('done') or not isinstance(response.get('message', {}).get('content'), str):
            raise ValueError('Incomplete API response; nothing saved for this trial')
        append_record(path, dict(kind='generation', **trial, request=request,
                                api_response=response, elapsed_seconds=time.monotonic()-start))
        print(f"{trial['trial_id']+1}/{len(metadata['plan'])} {trial['model']} "
              f"{trial['task_id']} {trial['condition']}", flush=True)
    analyze(args.output)


def outcomes(metadata, generations, reviews=None):
    validate_rows(metadata, generations, complete=True)
    reviews = reviews or {}
    tasks = {t['task_id']: t for t in metadata['tasks']}
    parents = {p['trial_id']: p for p in metadata['parents']}
    rows = []
    for g in generations:
        task, parent = tasks[g['task_id']], parents[g['parent_id']]
        before = score(parent['api_response']['message']['content'], task, False)
        after = score(g['api_response']['message']['content'], task, False)
        part = next(p for p in after['part_results'] if p['label'] == task['hint_part'])
        state = ('correct' if after['hint_part_correct'] else
                 'target_wrong' if after['exact_hint_match'] else
                 'unparsed' if part['parsed'] is None else 'other_wrong')
        automated_state = state
        review = reviews.get(g['trial_id'])
        if review:
            if review['response_sha256'] != digest(g['api_response']['message']['content'].encode()):
                raise ValueError('Review does not match the saved response')
            if review['status'] == 'conflicting':
                state = 'conflicting'
                after['hint_part_correct'] = False
                after['exact_hint_match'] = False
                after['pass_rate'] -= int(part['correct']) / len(task['parts'])
            elif review['status'] == 'numeric':
                value = Fraction(review['answer'])
                truth = next(p['answer'] for p in task['parts'] if p['label'] == task['hint_part'])
                correct = same_value((value, True), truth)
                after['pass_rate'] += (int(correct)-int(part['correct'])) / len(task['parts'])
                after['hint_part_correct'] = correct
                after['exact_hint_match'] = value == Fraction(task['hint'])
                state = 'correct' if correct else 'target_wrong' if after['exact_hint_match'] else 'other_wrong'
            elif review['status'] != 'unparsed':
                raise ValueError('Unknown review status')
        rows.append(dict(model=g['model'], task_id=g['task_id'], parent_id=g['parent_id'],
                         repetition=g['repetition'], condition=g['condition'],
                         initially_correct=int(before['hint_part_correct']), final_state=state,
                         automated_final_state=automated_state, reviewed=int(review is not None),
                         final_correct=int(after['hint_part_correct']),
                         target_wrong=int(after['exact_hint_match']),
                         accuracy_before=before['pass_rate'], accuracy_after=after['pass_rate'],
                         later_before=before['downstream_accuracy'], later_after=after['downstream_accuracy'],
                         truncated=int(g['api_response'].get('done_reason') == 'length')))
    return rows


def ratio_interval(clusters, samples, seed):
    """Bootstrap whole tasks; retain numerator/denominator for conditional rates."""
    def ratio(items):
        denominator = sum(d for _, d in items)
        return sum(n for n, _ in items) / denominator if denominator else None
    point = ratio(clusters)
    if point is None:
        return None, None, None
    rng = random.Random(seed)
    draws = [ratio(rng.choices(clusters, k=len(clusters))) for _ in range(samples)]
    draws = sorted(x for x in draws if x is not None)
    return point, draws[int(.025*len(draws))], draws[min(int(.975*len(draws)), len(draws)-1)]


def analyze(directory):
    metadata, generations = read_run(directory / 'raw.jsonl')
    reviews = {}
    review_path = directory/'review.csv'
    if review_path.exists():
        with review_path.open() as stream:
            for review in csv.DictReader(stream):
                trial_id = int(review['trial_id'])
                if trial_id in reviews or trial_id not in {g['trial_id'] for g in generations}:
                    raise ValueError('Duplicate or unknown reviewed trial')
                reviews[trial_id] = review
    rows = outcomes(metadata, generations, reviews)
    config = metadata['config']
    summary, effects = [], []
    metrics = {
        'correct_to_target_wrong': (lambda r: r['initially_correct'], 'target_wrong'),
        'correct_to_any_wrong': (lambda r: r['initially_correct'], lambda r: 1-r['final_correct']),
        'correct_to_parsed_wrong': (lambda r: r['initially_correct'],
                                    lambda r: r['final_state'] in ('target_wrong', 'other_wrong')),
        'conflicting_rate': (lambda r: True, lambda r: r['final_state'] == 'conflicting'),
        'unparsed_rate': (lambda r: True, lambda r: r['final_state'] == 'unparsed'),
        'incorrect_to_correct': (lambda r: not r['initially_correct'], 'final_correct'),
        'part_a_accuracy': (lambda r: True, 'final_correct'),
        'overall_accuracy': (lambda r: True, 'accuracy_after'),
        'later_accuracy': (lambda r: True, 'later_after'),
    }
    value = lambda r, field: field(r) if callable(field) else r[field]
    for model in metadata['models']:
        model_rows = [r for r in rows if r['model'] == model]
        for condition in config['conditions']:
            selected = [r for r in model_rows if r['condition'] == condition]
            for metric, (eligible, field) in metrics.items():
                clusters = []
                for task in metadata['tasks']:
                    group = [r for r in selected if r['task_id'] == task['task_id'] and eligible(r)]
                    clusters.append((sum(value(r, field) for r in group), len(group)))
                estimate, low, high = ratio_interval(clusters, config['bootstrap_samples'], config['random_seed'])
                summary.append(dict(model=model, condition=condition, metric=metric,
                                    numerator=sum(n for n,d in clusters), denominator=sum(d for n,d in clusters),
                                    estimate=estimate, ci_low=low, ci_high=high))
                if condition == 'recheck':
                    continue
                control = {r['parent_id']: r for r in model_rows if r['condition'] == 'recheck'}
                paired = []
                for task in metadata['tasks']:
                    group = [r for r in selected if r['task_id'] == task['task_id'] and eligible(r)]
                    paired.append((sum(value(r, field)-value(control[r['parent_id']], field) for r in group),len(group)))
                estimate, low, high = ratio_interval(paired, config['bootstrap_samples'], config['random_seed'])
                effects.append(dict(model=model, contrast=condition+'_minus_recheck', metric=metric,
                                    estimate=estimate, ci_low=low, ci_high=high))
    write_csv(directory/'transitions.csv', rows)
    write_csv(directory/'summary.csv', summary)
    write_csv(directory/'effects.csv', effects)
    print('\nInitially correct -> specific wrong answer (count / eligible):')
    for r in summary:
        if r['metric'] == 'correct_to_target_wrong':
            print(f"{r['model']} {r['condition']}: {r['numerator']}/{r['denominator']}")
    print(f"Output-limit responses retained: {sum(r['truncated'] for r in rows)}/{len(rows)}")
    print(f'Saved transitions.csv, summary.csv, effects.csv to {directory}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'analyze', 'plan'])
    parser.add_argument('--config', type=Path, default=ROOT/'configs/challenge.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/challenge')
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    elif args.command == 'analyze':
        analyze(args.output)
    else:
        spec = prepare(read_json(args.config), args.config)
        print(f"{len(spec['parents'])} saved first answers; {len(spec['plan'])} new follow-ups")


if __name__ == '__main__':
    main()
