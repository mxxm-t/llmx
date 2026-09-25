import collections
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

art = Path(__file__).parent
prefix = 'cli-' if '--cli' in sys.argv else ''
plan = json.loads((art / (prefix + 'plan.json')).read_text())
root = art / (prefix + 'matrix')
rounds = len(plan['orders'])
complete = json.loads((root / 'complete.json').read_text())
rows = json.loads((root / 'results.json').read_text())
expected = {(r, m, a) for r, order in enumerate(plan['orders']) for m in plan['models'] for a in order}
actual = {(r['round'], r['model'], r['arm']) for r in rows}
assert len(rows) == len(actual) == len(expected) == complete['rows'] and actual == expected
assert all(r['exit'] == 0 and 'protocol_error' not in r for r in rows)

def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values))

summary = {'scope': plan['scope'], 'rows': len(rows), 'cells': [], 'monitor': [], 'limitations': [
    'All planned blocks retained, including activity-flagged and slow samples.',
    'Activity flags cover the whole invocation, including model loading, warmup and measurement.',
    'Disk/GPU counters are system-wide; disk activity includes benchmark loading.',
    'Inaccessible/new or short-lived processes prevent a claim of complete background CPU accounting.',
    'The unused-helper layout control is one perturbation, not a bound on every possible layout effect.',
    plan['scope'],
]}
for model in plan['models']:
    for phase, tokens in [('pp', 215), ('tg', 32)]:
        values = {a: [next(r for r in rows if r['round'] == n and r['model'] == model and r['arm'] == a)
                      for n in range(rounds)] for a in plan['arms']}
        rates = {a: [tokens * 1000 / r['measured'][phase + '_ms'] for r in seq] for a, seq in values.items()}
        cell = {'model': model, 'phase': phase, 'tokens': tokens, 'arms': {}, 'paired': {}}
        for arm, numbers in rates.items():
            cell['arms'][arm] = {'median_tok_s': statistics.median(numbers), 'mean_tok_s': statistics.mean(numbers),
                                 'min_tok_s': min(numbers), 'max_tok_s': max(numbers), 'all_tok_s': numbers}
        pairs = [('candidate', 'base'), ('layout', 'base')]
        if 'mx' in plan['arms']:
            pairs += [('candidate', 'mx'), ('base', 'mx')]
        for numerator, denominator in pairs:
            ratios = [a / b for a, b in zip(rates[numerator], rates[denominator])]
            cell['paired'][numerator + '/' + denominator] = {'geomean_percent': 100 * (gm(ratios) - 1),
                'min_percent': 100 * (min(ratios) - 1), 'max_percent': 100 * (max(ratios) - 1), 'ratios': ratios}
        summary['cells'].append(cell)

for row in rows:
    path = Path(row['monitor_file'])
    kinds = collections.Counter()
    monitor_end = None
    parse_errors = 0
    first_collection = None
    last_collection = None
    process_errors = 0
    collect_errors = 0
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            try:
                item = json.loads(raw)
            except ValueError:
                parse_errors += 1
                continue
            kinds[item.get('kind')] += 1
            if item.get('kind') == 'metadata':
                first_collection = item.get('initial_counter_collection_bracket')
            if item.get('kind') == 'sample':
                last_collection = item.get('counter_interval_end_bracket')
                process_errors += bool(item.get('processes', {}).get('error'))
                collect_errors += bool(item.get('collect_status'))
            if item.get('kind') == 'end':
                monitor_end = item
    known = row['monitor_exit'] == 0 and kinds['metadata'] == 1 and kinds['sample'] > 0 and kinds['end'] == 1 and parse_errors == 0
    summary['monitor'].append({'round': row['round'], 'model': row['model'], 'arm': row['arm'],
        'structurally_complete': known, 'kinds': dict(kinds), 'parse_errors': parse_errors,
        'covers_invocation': bool(first_collection and last_collection and first_collection[1] <= row['monotonic_start'] and last_collection[0] >= row['monotonic_end']),
        'process_enumeration_errors': process_errors, 'counter_collection_errors': collect_errors,
        'sha256': digest.hexdigest(), 'end': monitor_end, 'activity': row['activity']})

(art / (prefix + 'summary.json')).write_text(json.dumps(summary, indent=2) + '\n')
blocks = []
for model in plan['models']:
    for number in range(rounds):
        block = {'model': model, 'round': number}
        chosen = {r['arm']: r for r in rows if r['model'] == model and r['round'] == number}
        for arm, row in chosen.items():
            for metric in ['flag_cpu', 'flag_disk', 'flag_gpu', 'unrelated_cpu_peak', 'unknown_process_samples', 'missing_counter_samples']:
                block[arm + '_' + metric] = row['activity'][metric]
        for phase in ['pp', 'tg']:
            for arm in [a for a in plan['arms'] if a != 'base']:
                block[arm + '_vs_base_' + phase + '_percent'] = 100 * (chosen['base']['measured'][phase + '_ms'] / chosen[arm]['measured'][phase + '_ms'] - 1)
        blocks.append(block)
with (art / (prefix + 'blocks.csv')).open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(blocks[0]))
    writer.writeheader()
    writer.writerows(blocks)
arms = ['base', 'candidate', 'layout'] + (['mx'] if 'mx' in plan['arms'] else [])
lines = ['| Model | Phase | Before tok/s | Candidate tok/s | Layout tok/s | ' + ('mx tok/s | ' if 'mx' in arms else '') + 'Paired candidate/before | Paired layout/before |',
         '|---|---|' + '---:|' * (len(arms) + 2)]
for c in summary['cells']:
    numbers = [c['arms'][a]['median_tok_s'] for a in arms]
    delta = c['paired']['candidate/base']['geomean_percent']
    layout = c['paired']['layout/base']['geomean_percent']
    lines.append(f"| {c['model']} | {c['phase']}{c['tokens']} | " + ' | '.join(f'{x:.2f}' for x in numbers) + f' | {delta:+.2f}% | {layout:+.2f}% |')
lines += ['', f'Rates are medians of {rounds} measured runs. Paired deltas are geometric means across all {rounds} matched blocks.', '']
lines += [f'Every per-model matched block, paired delta and activity flag is retained in {prefix}blocks.csv; complete monitor audits are in {prefix}summary.json.', '']
for arm in plan['arms']:
    subset = [m for m in summary['monitor'] if m['arm'] == arm]
    lines.append(f"{arm}: {len(subset)} rows, structurally complete monitors {sum(m['structurally_complete'] for m in subset)}, "
                 f"invocations covered {sum(m['covers_invocation'] for m in subset)}, "
                 f"CPU flags {sum(m['activity']['flag_cpu'] for m in subset)}, disk flags {sum(m['activity']['flag_disk'] for m in subset)}, "
                 f"GPU flags {sum(m['activity']['flag_gpu'] for m in subset)}, "
                 f"rows with unknown process CPU {sum(m['activity']['unknown_process_samples'] > 0 for m in subset)}")
lines += ['', *summary['limitations']]
(art / (prefix + 'summary.md')).write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
