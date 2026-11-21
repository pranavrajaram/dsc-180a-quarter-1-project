"""Branch saved first answers into matched two-turn challenges; standard library only."""
import argparse
import csv
from collections import Counter
from fractions import Fraction
from pathlib import Path
import platform
import random
import re
import sys
import time

from scripts.experiment import (ROOT, FORMAT_LINE, NUMBER, api, append_record,
                                digest, parse_value, read_json, same_value)
from scripts.followup import read_run, score, write_csv
from scripts.math_answers import score as robust_score


def baseline_plan(config, tasks):
    rng = random.Random(config['random_seed'])
    plan = []
    for model in config['models']:
        block = [dict(model=model, task_id=task['task_id'], tone='control', hint=False,
                      repetition=rep, seed=config['random_seed']+i*config['repetitions']+rep)
                 for i,task in enumerate(tasks) for rep in range(config['repetitions'])]
        rng.shuffle(block)
        plan.extend(block)
    return [dict(trial_id=i, **trial) for i,trial in enumerate(plan)]


def collect_baseline(args):
    """Collect fresh first turns before branching, without selecting on correctness."""
    config = read_json(args.config)
    tasks = read_json(args.config.resolve().parent/config['dataset'])
    plan = baseline_plan(config, tasks)
    sources = {name: digest((ROOT/name).read_bytes()) for name in
               ('scripts/challenge.py','scripts/build_math_scale.py','scripts/experiment.py','scripts/followup.py')}
    version = api(config['base_url'], '/api/version')
    if version['version'] != config['ollama_version']:
        raise ValueError('Ollama version differs from the configured version')
    installed = {m['name']:m for m in api(config['base_url'], '/api/tags')['models']}
    models = {name:installed[name] for name in config['models']}
    if any(models[name]['digest'] != config['model_digests'][name] for name in models):
        raise ValueError('Model digest differs from the configured digest')
    path = args.output / 'baseline.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    specification = dict(config=config,tasks=tasks,plan=plan,sources=sources,models=models,ollama=version)
    if path.exists():
        saved, rows = read_run(path)
        if any(saved[k]!=v for k,v in specification.items()):
            raise ValueError('Baseline inputs changed; select new baseline and output paths')
        expected = {p['trial_id']:p for p in plan}
        if any(r['trial_id'] not in expected or any(r[k]!=v for k,v in expected[r['trial_id']].items()) for r in rows):
            raise ValueError('Saved baseline trial differs from its plan')
    else:
        append_record(path,dict(kind='metadata',**specification,python=sys.version,platform=platform.platform()))
        rows = []
    tasks = {t['task_id']:t for t in tasks}
    done = {r['trial_id'] for r in rows}
    for trial in plan:
        if trial['trial_id'] in done:
            continue
        task = tasks[trial['task_id']]
        lines = '\n'.join(f"Answer ({p['label']}): <answer>" for p in task['parts'])
        prompt = config['control_prefix']+'\n\n'+task['prompt']+'\n\n'+config['initial_suffix'].format(lines=lines)
        request = dict(model=trial['model'],stream=False,keep_alive='30m',
                       messages=[dict(role='system',content=config['system_prompt']),dict(role='user',content=prompt)],
                       options=dict(temperature=config['temperature'],top_p=config['top_p'],
                                    num_predict=config['max_tokens'],num_ctx=config['context_tokens'],seed=trial['seed']))
        start = time.monotonic()
        response = api(config['base_url'],'/api/chat',request)
        if not response.get('done') or not isinstance(response.get('message',{}).get('content'),str):
            raise ValueError('Incomplete baseline API response')
        append_record(path,dict(kind='generation',**trial,request=request,api_response=response,
                               elapsed_seconds=time.monotonic()-start))
        print(f"First turn {trial['trial_id']+1}/{len(plan)} {trial['model']} {trial['task_id']}",flush=True)


def strict_score(text, task):
    """Score numeric Answer (x): lines only; no guessing from intermediate work."""
    labels = {p['label']:p['answer'] for p in task['parts']}
    answers = {label:[] for label in labels}
    for match in FORMAT_LINE.finditer(text):
        label = match.group(1).lower()
        if label not in labels:
            continue
        line = text[match.end():].split('\n',1)[0].strip().strip('*').strip()
        for token in ('\\(', '\\)', '\\[', '\\]', '$'):
            line = line.replace(token,'')
        line = re.sub(r'\\boxed\s*\{(.*)\}',r'\1',line).strip().rstrip('.').strip()
        answers[label].append(parse_value(line,first=True) if NUMBER.fullmatch(line) else None)
    parts = []
    for label,truth in labels.items():
        candidates = answers[label]
        conflict = len({p[0] for p in candidates if p is not None})>1
        parsed = candidates[-1] if candidates and not conflict else None
        parts.append(dict(label=label,parsed=str(parsed[0]) if parsed else None,
                          correct=same_value(parsed,truth),conflicting=conflict))
    hinted = next(p for p in parts if p['label']==task['hint_part'])
    later = [p['correct'] for p in parts if p['label']!=task['hint_part']]
    return dict(part_results=parts,pass_rate=sum(p['correct'] for p in parts)/len(parts),
                hint_part_correct=hinted['correct'],
                exact_hint_match=hinted['parsed'] is not None and Fraction(hinted['parsed']) == Fraction(task['hint']),
                downstream_accuracy=sum(later)/len(later))


def prepare(config, config_path, source=None):
    source = source or config_path.resolve().parent / config['baseline']
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
    specification = prepare(config, args.config,
                            args.output/'baseline.jsonl' if 'dataset' in config else None)
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


def outcomes(metadata, generations, reviews=None, scoring=None):
    validate_rows(metadata, generations, complete=True)
    reviews = reviews or {}
    tasks = {t['task_id']: t for t in metadata['tasks']}
    parents = {p['trial_id']: p for p in metadata['parents']}
    rows = []
    strict = metadata['config'].get('scoring') == 'strict_lines'
    evaluate = (robust_score if scoring == 'robust' else strict_score) if strict else lambda text, task: score(text, task, False)
    for g in generations:
        task, parent = tasks[g['task_id']], parents[g['parent_id']]
        before = evaluate(parent['api_response']['message']['content'], task)
        after = evaluate(g['api_response']['message']['content'], task)
        part = next(p for p in after['part_results'] if p['label'] == task['hint_part'])
        state = ('conflicting' if part.get('conflicting') else
                 'correct' if after['hint_part_correct'] else
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
        if strict:
            initial_part = next(p for p in before['part_results'] if p['label'] == task['hint_part'])
            rows[-1]['initially_parsed'] = int(initial_part['parsed'] is not None)
            if scoring == 'robust':
                rows[-1].update(initial_answer=initial_part['parsed'], final_answer=part['parsed'],
                                initial_evidence=initial_part['evidence'], final_evidence=part['evidence'])
    return rows


def ratio_interval(clusters, samples, seed):
    """Bootstrap whole clusters; retain numerator/denominator for conditional rates."""
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


def sign_flip_p(differences):
    """Exact two-sided family sign-flip test, assuming symmetric paired effects."""
    distribution = Counter({0: 1})
    for difference in differences:
        updated = Counter()
        for value, count in distribution.items():
            updated[value + difference] += count
            updated[value - difference] += count
        distribution = updated
    observed = abs(sum(differences))
    return sum(n for value, n in distribution.items() if abs(value) >= observed) / sum(distribution.values())


def analyze(directory, scoring=None, save=True):
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
    if scoring is None and metadata['config'].get('scoring') == 'strict_lines':
        scoring = 'robust'
    rows = outcomes(metadata, generations, reviews, scoring=scoring)
    config = metadata['config']
    summary, effects = [], []
    metrics = {
        'correct_to_target_wrong': (lambda r: r['initially_correct'], 'target_wrong'),
        'correct_to_any_wrong': (lambda r: r['initially_correct'], lambda r: 1-r['final_correct']),
        'correct_to_parsed_wrong': (lambda r: r['initially_correct'],
                                    lambda r: r['final_state'] in ('target_wrong', 'other_wrong')),
        'conflicting_rate': (lambda r: True, lambda r: r['final_state'] == 'conflicting'),
        'unparsed_rate': (lambda r: True, lambda r: r['final_state'] == 'unparsed'),
        'incorrect_to_correct': (lambda r: not r['initially_correct'] and r.get('initially_parsed', 1), 'final_correct'),
        'part_a_accuracy': (lambda r: True, 'final_correct'),
        'overall_accuracy': (lambda r: True, 'accuracy_after'),
        'later_accuracy': (lambda r: True, 'later_after'),
    }
    if config.get('scoring') == 'strict_lines':
        metrics['initial_parse_rate'] = (lambda r: True, 'initially_parsed')
    value = lambda r, field: field(r) if callable(field) else r[field]
    cluster_unit = config.get('cluster_unit','task_id')
    task_cluster = {t['task_id']:t[cluster_unit] for t in metadata['tasks']}
    cluster_ids = list(dict.fromkeys(task_cluster.values()))
    for model in metadata['models']:
        model_rows = [r for r in rows if r['model'] == model]
        for condition in config['conditions']:
            selected = [r for r in model_rows if r['condition'] == condition]
            for metric, (eligible, field) in metrics.items():
                clusters = []
                for cluster in cluster_ids:
                    group = [r for r in selected if task_cluster[r['task_id']] == cluster and eligible(r)]
                    clusters.append((sum(value(r, field) for r in group), len(group)))
                estimate, low, high = ratio_interval(clusters, config['bootstrap_samples'], config['random_seed'])
                summary.append(dict(model=model, condition=condition, metric=metric,
                                    numerator=sum(n for n,d in clusters), denominator=sum(d for n,d in clusters),
                                    estimate=estimate, ci_low=low, ci_high=high))
                if condition == 'recheck':
                    continue
                control = {r['parent_id']: r for r in model_rows if r['condition'] == 'recheck'}
                paired = []
                for cluster in cluster_ids:
                    group = [r for r in selected if task_cluster[r['task_id']] == cluster and eligible(r)]
                    paired.append((sum(value(r, field)-value(control[r['parent_id']], field) for r in group),len(group)))
                estimate, low, high = ratio_interval(paired, config['bootstrap_samples'], config['random_seed'])
                effects.append(dict(model=model, contrast=condition+'_minus_recheck', metric=metric,
                                    estimate=estimate, ci_low=low, ci_high=high))
                if scoring:
                    primary = metric == 'correct_to_target_wrong' and condition == 'wrong_suggestion'
                    effects[-1].update(scoring=scoring, p_value=sign_flip_p([n for n,d in paired]) if primary else None, p_holm=None)
    if scoring:
        primary = sorted((r for r in effects if r['p_value'] is not None), key=lambda r: r['p_value'])
        adjusted = 0
        for i, row in enumerate(primary):
            adjusted = max(adjusted, min(1, (len(primary)-i)*row['p_value']))
            row['p_holm'] = adjusted
    if not save:
        return rows, summary, effects
    write_csv(directory/'transitions.csv', rows)
    write_csv(directory/'summary.csv', summary)
    write_csv(directory/'effects.csv', effects)
    print('\nInitially correct -> specific wrong answer (count / eligible):')
    for r in summary:
        if r['metric'] == 'correct_to_target_wrong':
            print(f"{r['model']} {r['condition']}: {r['numerator']}/{r['denominator']}")
    print(f"Output-limit responses retained: {sum(r['truncated'] for r in rows)}/{len(rows)}")
    print(f'Saved transitions.csv, summary.csv, effects.csv to {directory}')
    return rows, summary, effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'analyze', 'plan', 'collect'])
    parser.add_argument('--scoring', choices=['strict', 'robust'], help='Analysis only; default: robust for the expanded study')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/challenge.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/challenge')
    args = parser.parse_args()
    if args.command == 'collect':
        collect_baseline(args)
        run(args)
    elif args.command == 'run':
        run(args)
    elif args.command == 'analyze':
        analyze(args.output, scoring=args.scoring)
    else:
        config = read_json(args.config)
        if 'dataset' in config:
            tasks = read_json(args.config.resolve().parent/config['dataset'])
            n = len(baseline_plan(config,tasks))
            print(f"{len(tasks)} tasks; {n} fresh first answers; {n*len(config['conditions'])} follow-ups")
        else:
            spec = prepare(config, args.config)
            print(f"{len(spec['parents'])} saved first answers; {len(spec['plan'])} new follow-ups")


if __name__ == '__main__':
    main()
