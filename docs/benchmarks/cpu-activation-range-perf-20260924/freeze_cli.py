import datetime
import hashlib
import json
import subprocess
from pathlib import Path

art = Path(__file__).parent
assert (art / 'matrix/complete.json').exists()
original = json.loads((art / 'plan.json').read_text())
plan = {'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Normal CLI translation-unit layout check: synthetic LCG prompt pp215, then independent decode32 from empty history. Default ubatch512 and model context; F16 KV, one sequence and six CPU threads. One warmup per phase followed by one measured run per invocation. This is distinct from the common harness and has no matched mx arm.',
    'orders': [['base', 'candidate', 'layout'], ['candidate', 'layout', 'base'], ['layout', 'base', 'candidate'],
               ['layout', 'candidate', 'base'], ['base', 'layout', 'candidate'], ['candidate', 'base', 'layout']],
    'models': original['models'], 'hardware': original['hardware'], 'monitor': original['monitor'], 'arms': {},
    'retention': original['monitor']['retention'], 'analysis': 'Retain every planned block and report phase/model deltas with same-file unused-helper control and whole-invocation activity. No sample replacement based on activity or timing.'}
for arm in ['base', 'candidate', 'layout']:
    tree = art.parent / f'llmx-activation-perf-{arm}-20260924'
    exe = tree / 'build/Release/llmx.exe'
    git = ['git', '-c', f'safe.directory={tree.as_posix()}', '-C', str(tree)]
    assert not subprocess.check_output(git + ['status', '--porcelain'], text=True).strip()
    version = subprocess.check_output([str(exe), '--version'], text=True).strip()
    assert '.dirty' not in version
    with exe.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    plan['arms'][arm] = {'exe': str(exe), 'sha256': digest, 'version': version,
        'revision': subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()}
for arm in plan['arms']:
    assert all(sum(order[position] == arm for order in plan['orders']) == 2 for position in range(3))
with (art / 'cli-plan.json').open('x') as stream:
    json.dump(plan, stream, indent=2)
    stream.write('\n')
print('Frozen normal-CLI plan: six balanced blocks, four models, 72 planned rows.')
