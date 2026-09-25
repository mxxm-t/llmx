"""Read-only arithmetic audit of the retained CLI performance evidence."""
import csv
import json
import math
from pathlib import Path
import re
import statistics

root = Path(__file__).parent
plan = json.loads((root / 'cli-plan.json').read_text())
rows = json.loads((root / 'cli-matrix/results.json').read_text())
summary = json.loads((root / 'cli-summary.json').read_text())
complete = json.loads((root / 'cli-matrix/complete.json').read_text())
with (root / 'cli-blocks.csv').open(newline='') as stream:
    blocks = list(csv.DictReader(stream))
checks, errors = 0, []

def check(got, expected, label):
    global checks
    checks += 1
    same = math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-10) if isinstance(expected, float) else got == expected
    if not same:
        errors.append({'label': label, 'got': got, 'expected': expected})

index = {(r['model'], r['round'], r['arm']): r for r in rows}
expected_order = []
for number, order in enumerate(plan['orders']):
    models = list(plan['models'])
    models = models[number % len(models):] + models[:number % len(models)]
    expected_order.extend((model, number, arm) for model in models for arm in order)
check(list(index), expected_order, 'full prospective row order')
check(len(rows), 72, 'raw rows')
check(len(index), 72, 'unique rows')
check(complete['rows'], 72, 'terminal rows')
check(complete['exit'], 0, 'terminal exit')
check(summary['rows'], 72, 'summary rows')
check(len(summary['cells']), 8, 'summary cells')
check(len(summary['monitor']), 72, 'monitor entries')
check(len(blocks), 24, 'CSV rows')
check(len({(b['model'], int(b['round'])) for b in blocks}), 24, 'unique CSV blocks')

for row in rows:
    tag = f"r{row['round']}-{row['model'][:-5]}-{row['arm']}"
    raw = (root / 'cli-matrix' / (tag + '.stdout')).read_text()
    matches = re.findall(r'bench: (pp215|tg32)\s+([0-9.]+) \+- ([0-9.]+) tok/s\s+\(1 runs\)', raw)
    check(len(matches), 2, tag + ' output phases')
    check(row['exit'], 0, tag + ' exit')
    check('protocol_error' in row, False, tag + ' protocol')
    for phase, rate, deviation in matches:
        check(row['reported_rates'][phase], float(rate), tag + ' ' + phase + ' raw rate')
        check(float(deviation), 0.0, tag + ' single-run deviation')
        tokens = 215 if phase == 'pp215' else 32
        check(row['measured'][phase[:2] + '_ms'], 1000 * tokens / float(rate), tag + ' reconstructed duration')

findings = []
for cell in summary['cells']:
    model, phase = cell['model'], cell['phase']
    key = phase + str(cell['tokens'])
    rates = {arm: [index[model, number, arm]['reported_rates'][key] for number in range(6)] for arm in plan['arms']}
    for arm, values in rates.items():
        reported = cell['arms'][arm]
        for number, value in enumerate(values):
            check(reported['all_tok_s'][number], value, f'{model}/{phase}/{arm}/{number}')
        for field, value in [('median_tok_s', statistics.median(values)), ('mean_tok_s', statistics.mean(values)), ('min_tok_s', min(values)), ('max_tok_s', max(values))]:
            check(reported[field], value, f'{model}/{phase}/{arm}/{field}')
    for pair, reported in cell['paired'].items():
        numerator, denominator = pair.split('/')
        ratios = [a / b for a, b in zip(rates[numerator], rates[denominator])]
        for number, ratio in enumerate(ratios):
            check(reported['ratios'][number], ratio, f'{model}/{phase}/{pair}/{number}')
        for field, value in [('geomean_percent', 100 * (math.prod(ratios) ** (1 / 6) - 1)), ('min_percent', 100 * (min(ratios) - 1)), ('max_percent', 100 * (max(ratios) - 1))]:
            check(reported[field], value, f'{model}/{phase}/{pair}/{field}')
    ratios = [a / b for a, b in zip(rates['candidate'], rates['base'])]
    findings.append({'model': model, 'phase': key,
        'median_tok_s': {arm: statistics.median(values) for arm, values in rates.items()},
        'candidate_base_paired_percent': 100 * (math.prod(ratios) ** (1 / 6) - 1),
        'layout_base_paired_percent': cell['paired']['layout/base']['geomean_percent'],
        'candidate_faster_blocks': sum(r > 1 for r in ratios),
        'candidate_slower_blocks': sum(r < 1 for r in ratios),
        'candidate_base_block_percent': [100 * (r - 1) for r in ratios]})

for block in blocks:
    model, number = block['model'], int(block['round'])
    for arm in plan['arms']:
        activity = index[model, number, arm]['activity']
        for field in ['flag_cpu', 'flag_disk', 'flag_gpu', 'unrelated_cpu_peak', 'unknown_process_samples', 'missing_counter_samples']:
            check(block[arm + '_' + field], str(activity[field]), f'CSV {model}/{number}/{arm}/{field}')
        if arm != 'base':
            for phase in ['pp215', 'tg32']:
                ratio = index[model, number, arm]['reported_rates'][phase] / index[model, number, 'base']['reported_rates'][phase]
                check(float(block[arm + '_vs_base_' + phase[:2] + '_percent']), 100 * (ratio - 1), f'CSV {model}/{number}/{arm}/{phase}')

for monitor in summary['monitor']:
    row = index[monitor['model'], monitor['round'], monitor['arm']]
    check(monitor['activity'], row['activity'], 'monitor activity copy')
    check(monitor['kinds']['sample'], row['activity']['samples'], 'monitor sample count')

markdown_rows = [line for line in (root / 'cli-summary.md').read_text().splitlines() if line.startswith('| Qwen')]
check(len(markdown_rows), 8, 'Markdown cells')
for cell, line in zip(summary['cells'], markdown_rows):
    columns = [part.strip() for part in line.strip('|').split('|')]
    check(columns[0], cell['model'], 'Markdown model')
    check(columns[1], cell['phase'] + str(cell['tokens']), 'Markdown phase')
    for number, arm in enumerate(['base', 'candidate', 'layout']):
        check(columns[2 + number], f'{cell["arms"][arm]["median_tok_s"]:.2f}', 'Markdown median')
    for number, pair in enumerate(['candidate/base', 'layout/base']):
        check(columns[5 + number], f'{cell["paired"][pair]["geomean_percent"]:+.2f}%', 'Markdown paired delta')

activity = {arm: {field: sum(row['activity'][field] for row in rows if row['arm'] == arm) for field in ['flag_cpu', 'flag_disk', 'flag_gpu', 'unknown_process_samples', 'missing_counter_samples']} for arm in plan['arms']}
monitor = {field: sum(item[field] for item in summary['monitor']) for field in ['structurally_complete', 'covers_invocation', 'parse_errors', 'process_enumeration_errors', 'counter_collection_errors']}
print(json.dumps({'checks': checks, 'errors': errors, 'cells': findings, 'activity': activity, 'monitor_summary': monitor,
    'limits': ['Read-only arithmetic and evidence-consistency audit, not inference or performance rerun.', 'CLI rates are printed to two decimal places; reconstructed milliseconds are derived values.', 'Raw monitor logs and frozen binaries/models were not rehashed in this review.', 'All flags describe whole invocations; inaccessible processes remain unknown.']}, indent=2))
raise SystemExit(bool(errors))
