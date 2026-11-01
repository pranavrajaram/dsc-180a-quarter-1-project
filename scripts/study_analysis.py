"""Compute paired tone contrasts from saved responses.

Only the Python standard library is required. No candidate code is executed.
"""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import statistics as stats


def load_run(directory):
    """Reject duplicates, missing cells, and inconsistent scores before aggregation."""
    records = [json.loads(line) for line in (Path(directory) / 'raw.jsonl').read_text().splitlines()]
    if not records or records[0].get('kind') != 'metadata':
        raise ValueError('The first record must contain run metadata.')
    metadata = records[0]
    by_kind = {'generation': {}, 'evaluation': {}}
    for record in records[1:]:
        kind = record.get('kind')
        if kind not in by_kind:
            raise ValueError(f'Unexpected record kind: {kind}')
        trial_id = record['trial_id']
        if trial_id in by_kind[kind]:
            raise ValueError(f'Duplicate {kind} for trial {trial_id}')
        by_kind[kind][trial_id] = record
    generations, evaluations = by_kind.values()
    expected_ids = set(range(metadata['planned_generations']))
    if set(generations) != expected_ids or set(evaluations) != expected_ids:
        raise ValueError('Incomplete run: every planned trial needs one generation and one evaluation.')
    config, tasks = metadata['config'], metadata['tasks']
    expected = set(itertools.product([t['task_id'] for t in tasks], config['tones'], range(config['repetitions'])))
    identities = [(g['task_id'], g['tone'], g['repetition']) for g in generations.values()]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError('Trial cells do not match the complete task/tone/repetition design.')
    task_map = {t['task_id']: t for t in tasks}
    rows = []
    for trial_id in sorted(generations):
        g, e = generations[trial_id], evaluations[trial_id]
        task = task_map[g['task_id']]
        domain = task.get('domain', 'coding')
        hint = domain == 'math' and g['repetition'] == config.get('hint_repetition')
        if g.get('domain', domain) != domain or g.get('hint', False) != hint:
            raise ValueError(f'Trial {trial_id}: domain/hint does not match design.')
        if 'trial_order' in metadata:
            i, tone, rep = metadata['trial_order'][trial_id]
            if (tasks[i]['task_id'], tone, rep) != (g['task_id'], g['tone'], g['repetition']):
                raise ValueError(f'Trial {trial_id}: saved order does not match generation.')
        total = len(task['parts'] if domain == 'math' else task['tests'])
        passed = e.get('tests_passed')
        if (type(passed) is not int or not 0 <= passed <= total or e.get('tests_total') != total
                or not isinstance(e.get('pass_rate'), (int, float))
                or not math.isclose(e['pass_rate'], passed / total, abs_tol=1e-12)
                or e.get('fully_correct') != (passed == total)):
            raise ValueError(f'Trial {trial_id}: inconsistent evaluation score.')
        rows.append({**g, **e, 'domain': domain, 'hint': hint})
    return metadata, rows


def sign_flip_pvalue(differences):
    """Exact two-sided paired sign-flip test, using rational sums and count DP.

Assumes independent task clusters and sign exchangeability under the null.
Zeros have no effect; ties are included.
"""
    values = [Fraction(str(x)).limit_denominator(1_000_000) for x in differences]
    if not values:
        raise ValueError('At least one paired difference is required.')
    denominator = math.lcm(*(x.denominator for x in values))
    integers = [int(x * denominator) for x in values if x]
    observed = abs(sum(integers))
    counts = Counter({0: 1})
    for value in integers:
        updated = Counter()
        for total, count in counts.items():
            updated[total + value] += count
            updated[total - value] += count
        counts = updated
    return sum(n for total, n in counts.items() if abs(total) >= observed) / (2 ** len(integers))


def holm_adjust(pvalues):
    """Holm family-wise adjustment, preserving the input order."""
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def contrasts(rows, config, bootstrap):
    """One correction family: six baseline contrasts × three domain/hint strata."""
    results = []
    baseline = config.get('baseline_tone', 'neutral')
    for domain, hinted in [('coding', False), ('math', False), ('math', True)]:
        grouped = defaultdict(list)
        for row in rows:
            if (row['domain'], row['hint']) == (domain, hinted):
                grouped[row['task_id'], row['tone']].append(row['pass_rate'])
        tasks = sorted({task for task, _ in grouped})
        if not tasks:
            continue
        for tone in config['tones']:
            if tone == baseline:
                continue
            differences = [stats.mean(grouped[task, tone]) - stats.mean(grouped[task, baseline]) for task in tasks]
            low, high = bootstrap(differences, config['bootstrap_samples'], config['random_seed'])
            results.append({'group': 'coding' if domain == 'coding' else 'math_hint' if hinted else 'math_no_hint',
                            'tone': tone, 'baseline': baseline, 'n_tasks': len(tasks),
                            'difference_pp': 100 * stats.mean(differences),
                            'ci_low_pp': 100 * low, 'ci_high_pp': 100 * high,
                            'p_sign_flip': sign_flip_pvalue(differences)})
    for result, adjusted in zip(results, holm_adjust([r['p_sign_flip'] for r in results])):
        result.update(p_holm=adjusted, reject_holm_05=adjusted < 0.05)
    return results


def main():
    from scripts import experiment as ex
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ex.ROOT / 'results/full_run')
    parser.add_argument('--output', type=Path, default=ex.ROOT / 'results/full_run')
    args = parser.parse_args()
    metadata, rows = load_run(args.run)
    results = contrasts(rows, metadata['config'], ex.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'contrasts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} contrasts to {args.output / 'contrasts.csv'}")


if __name__ == '__main__':
    main()
