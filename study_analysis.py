"""Audit saved trials and run exploratory paired inference without model calls.

Only the Python standard library is required. No candidate code is executed.
"""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import html
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
Zeros have no effect; ties are included. This is exploratory, not preregistered.
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


def audit(metadata, rows, ex):
    """Regrade math, verify exact saved requests, and measure scoring sensitivity."""
    tasks = {t['task_id']: t for t in metadata['tasks']}
    config = metadata['config']
    summary = {'n_generations': len(rows), 'truncated': sum(bool(r.get('truncated')) for r in rows),
               'math_generations': 0, 'math_score_changes_on_regrade': 0,
               'hinted_generations': 0, 'hint_adopted_exact': 0, 'hint_adopted_legacy': 0,
               'no_hint_matches_wrong_hint': 0, 'format_followed': 0, 'parse_modes': {},
               'hint_part_correct_no_hint': 0, 'hint_part_correct_hint': 0,
               'hint_seed_confound': True}
    modes = Counter()
    sensitivity = []
    for row in rows:
        task = tasks[row['task_id']]
        prompt = ex.build_prompt(task, row['tone'], config, row['hint'])
        messages = [{'role': 'system', 'content': ex.system_prompt(task, config)}, {'role': 'user', 'content': prompt}]
        index = next(i for i, t in enumerate(metadata['tasks']) if t['task_id'] == row['task_id'])
        seed = config['random_seed'] + index * config['repetitions'] + row['repetition']
        options = {'temperature': config['temperature'], 'top_p': config['top_p'],
                   'num_predict': config['max_tokens'], 'num_ctx': config['context_tokens'], 'seed': seed}
        if (row['prompt'] != prompt or row['request']['messages'] != messages or row['seed'] != seed
                or row['request']['options'] != options or row['request']['model'] != config['model']
                or row['raw_response'] != row['api_response']['message']['content']):
            raise ValueError(f"Trial {row['trial_id']}: saved request/response differs from recorded design.")
        if row['domain'] != 'math':
            continue
        rescored = ex.evaluate_math(row['raw_response'], task, row['hint'])
        summary['math_generations'] += 1
        summary['math_score_changes_on_regrade'] += rescored['pass_rate'] != row['pass_rate']
        modes[rescored['parse_mode']] += 1
        summary['format_followed'] += rescored['format_followed']
        summary['hint_part_correct_hint' if row['hint'] else 'hint_part_correct_no_hint'] += rescored['hint_part_correct']
        part = next(p for p in rescored['part_results'] if p['label'] == task.get('hint_part', 'a'))
        exact = part['parsed'] is not None and Fraction(part['parsed']) == Fraction(task['hint'])
        if row['hint']:
            summary['hinted_generations'] += 1
            summary['hint_adopted_exact'] += exact
            summary['hint_adopted_legacy'] += bool(row['hint_adopted'])
        else:
            summary['no_hint_matches_wrong_hint'] += exact
        # Sensitivity only: require the requested final-answer lines; no prose fallback.
        lines = '\n'.join(line for line in row['raw_response'].splitlines() if ex.FORMAT_LINE.match(line))
        strict_score = ex.evaluate_math(lines, task, row['hint'])['pass_rate'] if lines else 0.0
        sensitivity.append({'tone': row['tone'], 'hint': row['hint'], 'score': row['pass_rate'],
                            'strict_format_score': strict_score, 'format_followed': rescored['format_followed']})
    # Average tones within each task before bootstrapping hint contrasts. The
    # 140 responses per condition are not 140 independent task clusters.
    paired = defaultdict(lambda: {False: [], True: []})
    for row in rows:
        if row['domain'] == 'math':
            task = tasks[row['task_id']]
            part = next(p for p in row['part_results'] if p['label'] == task.get('hint_part', 'a'))
            exact = part['parsed'] is not None and Fraction(part['parsed']) == Fraction(task['hint'])
            paired[row['task_id']][row['hint']].append(
                (row['pass_rate'], float(row['hint_part_correct']), float(exact)))
    summary['hint_contrasts'] = []
    for index, metric in enumerate(('score', 'part_a_accuracy', 'exact_hint_match')):
        differences = [stats.mean(v[index] for v in conditions[True])
                       - stats.mean(v[index] for v in conditions[False])
                       for _, conditions in sorted(paired.items())]
        if differences:
            low, high = ex.bootstrap(differences, config['bootstrap_samples'], config['random_seed'])
            summary['hint_contrasts'].append({'metric': metric, 'n_tasks': len(differences),
                'difference_pp': 100 * stats.mean(differences),
                'ci_low_pp': 100 * low, 'ci_high_pp': 100 * high})
    summary['parse_modes'] = dict(modes)
    summary['format_sensitivity'] = [
        {'tone': tone, 'hint': hinted, 'n': len(group),
         'recorded_score': stats.mean(r['score'] for r in group),
         'strict_format_score': stats.mean(r['strict_format_score'] for r in group),
         'format_followed_rate': stats.mean(r['format_followed'] for r in group)}
        for tone in config['tones'] for hinted in (False, True)
        if (group := [r for r in sensitivity if r['tone'] == tone and r['hint'] == hinted])]
    return summary


def main():
    import experiment as ex
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ex.ROOT / 'results/full_run')
    parser.add_argument('--output', type=Path, default=ex.ROOT / 'results/checkpoint')
    args = parser.parse_args()
    if args.run.resolve() == args.output.resolve():
        parser.error('Use a separate output directory to preserve the original analysis.')
    metadata, rows = load_run(args.run)
    details = audit(metadata, rows, ex)
    results = contrasts(rows, metadata['config'], ex.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    details.update(raw_sha256=hashlib.sha256((args.run / 'raw.jsonl').read_bytes()).hexdigest(),
                   analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   evaluator_sha256=hashlib.sha256(Path(ex.__file__).read_bytes()).hexdigest(),
                   correction_family_size=len(results),
                   inference='Two-sided exact paired sign-flip tests; Holm adjustment over all domain/hint contrasts.')
    (args.output / 'audit.json').write_text(json.dumps(details, indent=2) + '\n')
    with (args.output / 'contrasts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    table = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in
                    ('group', 'tone', 'n_tasks')) +
                    f"<td>{r['difference_pp']:+.2f}</td><td>[{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]</td>"
                    f"<td>{r['p_sign_flip']:.4f}</td><td>{r['p_holm']:.4f}</td></tr>" for r in results)
    d = details
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Checkpoint reanalysis</title>
<style>body{{font:17px/1.5 Georgia;max-width:1100px;margin:40px auto;padding:20px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:8px;border-bottom:1px solid #ccc;text-align:left}}</style>
<h1>Checkpoint reanalysis</h1><p>{len(rows)} saved generations. No new model calls or candidate-code execution.</p>
<h2>Paired differences from control</h2><p><strong>{sum(r["reject_holm_05"] for r in results)} of {len(results)} contrasts survive Holm correction at 0.05.</strong></p><p>Differences and intervals use percentage points. Intervals remain unadjusted
95% task-bootstrap intervals. Exact two-sided sign-flip tests assume independent tasks and sign exchangeability under the null.
Holm adjustment covers all {len(results)} contrasts below, including the no-prefix comparison. This is an exploratory,
post-data analysis, not a preregistered confirmatory test. A nonsignificant result does not establish equivalence.</p>
<table><tr><th>Group</th><th>Tone</th><th>Tasks</th><th>Difference</th><th>95% CI</th><th>Raw p</th><th>Holm p</th></tr>{table}</table>
<h2>Wrong-hint measurement</h2><p>Exact numerical matches: {d['hint_adopted_exact']}/{d['hinted_generations']}.
The original approximate-copy measure: {d['hint_adopted_legacy']}/{d['hinted_generations']}.
Without a hint, {d['no_hint_matches_wrong_hint']} responses already matched that wrong answer.
Part (a) correct: {d['hint_part_correct_no_hint']} without hints and {d['hint_part_correct_hint']} with hints
(out of {d['hinted_generations']} each for this two-repetition design).</p>
<p>Hint status is fixed by repetition and changes the generation seed. Therefore, the observed hint/no-hint difference
also includes sampling differences. A future study should cross hint status with each task/tone/seed and use multiple seeds.
Matching a user-supplied wrong answer is an operational proxy, not proof of a psychological mechanism.</p>
<h2>Audit and sensitivity</h2><p>All saved prompts, request settings and trial cells match their recorded design.
Math regrading changed {d['math_score_changes_on_regrade']}/{d['math_generations']} scores.
{d['truncated']} responses hit the token limit. The separate strict-format sensitivity in audit.json requires the requested
answer lines and therefore mixes format compliance with accuracy; it does not replace the original grader.</p>
<p><a href="contrasts.csv">Contrasts CSV</a> · <a href="audit.json">Audit and format sensitivity</a></p>
<p>Raw SHA-256: {d['raw_sha256']}</p></html>'''
    (args.output / 'report.html').write_text(report)
    print(f"Saved {len(results)} contrasts and audit to {args.output}")


if __name__ == '__main__':
    main()
