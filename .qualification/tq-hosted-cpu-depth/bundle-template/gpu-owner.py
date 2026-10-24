"""Matched TQ Vulkan timing preparation; refuses missing actual build pins before any results or workloads."""
import argparse
import copy
import hashlib
import os
import signal
import json
import tarfile
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback

try:
    import resource
except ImportError:
    resource = None


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def planned_calls(m, bundle):
    calls = json.loads((bundle / 'planned-tq-calls.json').read_text())
    assert len(calls) == 40 and sum(c['warmup'] for c in calls) == 8
    assert all(call['index'] == index and call['status'] == 'not-run'
               for index, call in enumerate(calls))
    assert {call['arm'] for call in calls} == {'candidate', 'mx'}
    assert {call['model'] for call in calls} == {'TQ1_0', 'TQ2_0'}
    assert sum(call['warmup'] for call in calls) == 8
    for call in calls:
        assert call['model_sha256'] == next(x['sha256'] for x in m['models'] if x['label'] == call['model'])
        assert call['command'][0] == ('/b/llmx' if call['arm'] == 'candidate' else '/mx/mx-cli-bench-vulkan')
        if call['arm'] == 'candidate':
            args = call['command']
            assert '--ctx-size' not in args and '--kv-tokens' not in args
            assert args[args.index('--cache-type-k') + 1] == 'f16'
            assert args[args.index('--cache-type-v') + 1] == 'f16'
            assert args[args.index('--dtype') + 1] == 'auto'
            assert args[args.index('--load-mode') + 1] == 'auto'
    assert [c['block'] for c in calls if not c['warmup']] == [0, 0, 1, 1, 2, 2, 3, 3] * 4
    return calls


def validate_manifest(m):
    context = m['context_capacity']
    assert context['gguf_metadata_key'] == 'qwen3.context_length'
    assert context['models'] == {'TQ1_0': 40960, 'TQ2_0': 40960}
    wrapper_text = Path(__file__).with_name('wrapper.cpp').read_text()
    assert 'cp.n_ctx = 40960;' in wrapper_text
    hosted = m['hosted_build']
    required = ['artifact_root', 'run_id', 'artifact_id', 'artifact_name', 'artifact_receipt', 'artifact_receipt_sha256',
                'download_archive_file', 'official_artifact_digest', 'download_archive_sha256', 'build_output_sha256',
                'image_archive_sha256', 'image_id_file_sha256', 'image_inspect_sha256', 'image_id',
                'execution_image_id', 'image_equivalence_receipt', 'image_equivalence_receipt_sha256', 'image_proof']
    missing = [key for key in required if not hosted.get(key)]
    assert not missing, ('hosted build artifact pins not filled from the completed run', missing)
    for key in ('artifact_receipt_sha256', 'download_archive_sha256', 'build_output_sha256', 'image_archive_sha256', 'image_id_file_sha256', 'image_inspect_sha256'):
        assert re.fullmatch('[0-9a-f]{64}', hosted[key]), ('hosted_build', key)
    assert re.fullmatch('sha256:[0-9a-f]{64}', hosted['official_artifact_digest'])
    assert re.fullmatch('sha256:[0-9a-f]{64}', hosted['image_id'])
    assert re.fullmatch('sha256:[0-9a-f]{64}', hosted['execution_image_id'])
    assert hosted['image_equivalence_receipt_sha256'] == 'e2a61e2e8ba2cf3b22a825620516154dd2dbe52a5c3ca3ab8e1b5a0d7ee793f3'
    assert hosted['image_proof']['loaded_manifest_sha256'] == hosted['execution_image_id'].split(':', 1)[1]
    assert hosted['image_proof']['hosted_config_sha256'] == hosted['image_id'].split(':', 1)[1]
    assert hosted['workflow_head'] == '5d07a89dbdd7c632566f3633e510b03a5fd4473f'
    assert hosted['run_id'] == 40088982736
    assert hosted['main_revision'] == 'eaea90857b2ff1bb47037ea840a421f954fd30a6'
    assert hosted['candidate_revision'] == '01055654d462043cae8c30c4f600ba24398c6dfe'
    assert hosted['mx_revision'] == 'c4d12e2c0094c9028318bed5bb1a30bb48f8a35b'
    assert all(re.fullmatch('[0-9a-f]{64}', hosted['arms'][arm]['result_sha256'] or '')
               for arm in ('main', 'candidate', 'mx'))
    assert all(re.fullmatch('[0-9a-f]{64}', hosted[key] or '') for key in
               ('candidate_binary_sha256', 'candidate_version_file_sha256', 'candidate_config_sha256',
                'mx_binary_sha256'))
    wrapper = m['wrapper_compile']
    for key in ('receipt', 'receipt_sha256', 'binary', 'binary_sha256', 'image_id', 'mx_revision'):
        assert wrapper.get(key), ('wrapper_compile', key, 'missing actual compile receipt pin')
    assert re.fullmatch('[0-9a-f]{64}', wrapper['receipt_sha256'])
    assert re.fullmatch('[0-9a-f]{64}', wrapper['binary_sha256'])
    assert wrapper['image_id'] == hosted['execution_image_id'] and wrapper['mx_revision'] == hosted['mx_revision']
    assert wrapper['linked_libraries'] == hosted['mx_linked_libraries']
    assert wrapper['source_sha256'] == sha(Path(__file__).with_name('wrapper.cpp'))
    for key in ('header_manifest', 'header_manifest_sha256', 'header_archive', 'header_archive_sha256', 'source_archive_sha256'):
        assert wrapper.get(key), ('wrapper_compile', key, 'missing header provenance pin')
    for key in ('header_manifest_sha256', 'header_archive_sha256', 'source_archive_sha256'):
        assert re.fullmatch('[0-9a-f]{64}', wrapper[key])
    assert wrapper['header_manifest_sha256'] == 'fe9d326207d727a5b14497c577c8f9a8711b93072138d8e7c8e1a7fbc4b673b9'
    assert wrapper['header_archive_sha256'] == '6ce2deb08d203f482f8cfdf921bdfd1d5e4d69cfa499febde6fee23202fba2f9'
    assert wrapper['source_archive_sha256'] == 'deef5ae289239d8816d8db5cba845120c13a0bf7b7594a32ccadf1498b0efefa'
    assert m['candidate']['source_revision'] == hosted['candidate_revision']
    assert m['reference']['head'] == hosted['mx_revision']
    assert len(m['models']) == 2 and {x['label'] for x in m['models']} == {'TQ1_0', 'TQ2_0'}
    for model in m['models']:
        assert re.fullmatch('[0-9a-f]{64}', model['sha256'])
        assert Path(model['path']).parent == Path(m['models_root'])
    assert m['cells'] == [[128, 256], [4096, 256]] and m['matched_blocks'] == 4
    assert m['resources']['cpuset'] == '0-3' and m['resources']['cpus'] == '4'
    assert m['resources']['memory'] == '8g'


def verify_image_equivalence(hosted, pin):
    proof_path = hosted['image_equivalence_receipt']
    proof_root = Path(proof_path).parent
    pin(proof_path, hosted['image_equivalence_receipt_sha256'])
    expected = hosted['image_proof']
    manifest_path = proof_root / 'loaded-oci-manifest.json'
    config_path = proof_root / 'original-image-config.json'
    inspect_path = proof_root / 'loaded-image-inspect.json'
    pin(manifest_path, expected['loaded_manifest_sha256'])
    pin(config_path, expected['hosted_config_sha256'])
    pin(inspect_path, expected['loaded_inspect_file_sha256'])
    equivalence = json.loads(Path(proof_path).read_text())
    image_manifest = json.loads(manifest_path.read_text())
    image_config = json.loads(config_path.read_text())
    loaded_record = json.loads(inspect_path.read_text())
    assert equivalence['status'] == 'passed'
    assert equivalence['hosted_image_id'] == hosted['image_id']
    assert equivalence['execution_image_id'] == hosted['execution_image_id']
    assert equivalence['manifest_sha256'] == hosted['execution_image_id'].split(':', 1)[1]
    assert equivalence['config_sha256'] == hosted['image_id'].split(':', 1)[1]
    assert image_manifest['config']['digest'] == hosted['image_id']
    assert image_manifest['config']['size'] == config_path.stat().st_size
    layer_digests = [item['digest'] for item in image_manifest['layers']]
    assert layer_digests == equivalence['rootfs_layers']
    assert image_config['rootfs']['diff_ids'] == equivalence['rootfs_layers']
    assert loaded_record['Id'] == hosted['execution_image_id']
    assert loaded_record['Descriptor']['digest'] == hosted['execution_image_id']
    assert {key: value for key, value in loaded_record['Config'].items() if value not in (None, '', False)} == image_config['config']
    assert loaded_record['Architecture'] == image_config['architecture'] == 'amd64'
    assert loaded_record['Os'] == image_config['os'] == 'linux'
    assert loaded_record['Created'] == image_config['created']
    assert loaded_record['RootFS']['Layers'] == image_config['rootfs']['diff_ids'] == equivalence['rootfs_layers']
    return image_config, loaded_record


def verify_wrapper_provenance(wrapper, hosted, pin):
    pin(wrapper['receipt'], wrapper['receipt_sha256'])
    receipt = json.loads(Path(wrapper['receipt']).read_text())
    pin(wrapper['header_manifest'], wrapper['header_manifest_sha256'])
    pin(wrapper['header_archive'], wrapper['header_archive_sha256'])
    header_manifest = json.loads(Path(wrapper['header_manifest']).read_text())
    assert header_manifest['revision'] == wrapper['mx_revision'] == hosted['mx_revision']
    assert header_manifest['source_archive_sha256'] == wrapper['source_archive_sha256']
    assert header_manifest['header_archive_sha256'] == wrapper['header_archive_sha256']
    headers = header_manifest['headers']
    assert len(headers) == 24 and all(re.fullmatch('[0-9a-f]{64}', value) for value in headers.values())
    with tarfile.open(wrapper['header_archive'], 'r:') as archive:
        members = {item.name: item for item in archive.getmembers() if item.isfile()}
        assert set(members) == set(headers)
        for name, digest in headers.items():
            data = archive.extractfile(members[name]).read()
            assert hashlib.sha256(data).hexdigest() == digest, name
    assert receipt['status'] == 'passed' and receipt['exit_code'] == 0 and receipt['removed'] is True
    assert receipt['image_id'] == hosted['execution_image_id']
    assert receipt['hosted_image_id'] == hosted['image_id']
    assert receipt['image_equivalence_sha256'] == hosted['image_equivalence_receipt_sha256']
    assert receipt['mx_revision'] == wrapper['mx_revision'] and receipt['mx_headers_verified'] is True
    assert receipt['header_manifest_sha256'] == wrapper['header_manifest_sha256']
    assert receipt['wrapper_source_sha256'] == wrapper['source_sha256']
    assert receipt['binary_sha256'] == wrapper['binary_sha256']
    assert receipt['mx_binary_sha256'] == hosted['mx_binary_sha256']
    assert receipt['linked_libraries'] == hosted['mx_linked_libraries']
    assert receipt['compiler_version'] and isinstance(receipt['compile_command'], list)
    pin(wrapper['binary'], wrapper['binary_sha256'])
    return receipt


def verify_pins(m):
    hosted = m['hosted_build']
    root = Path(hosted['artifact_root'])
    pins = {}

    def pin(path, expected):
        path = Path(path)
        got = sha(path)
        assert got == expected, (str(path), got, expected)
        pins[str(path)] = expected

    pin(hosted['artifact_receipt'], hosted['artifact_receipt_sha256'])
    receipt = json.loads(Path(hosted['artifact_receipt']).read_text())
    artifact = receipt['artifact']
    assert artifact['workflow_run']['id'] == hosted['run_id']
    assert artifact['workflow_run']['head_sha'] == hosted['workflow_head']
    assert artifact['digest'] == hosted['official_artifact_digest']
    assert receipt['archive_sha256'] == hosted['download_archive_sha256']
    assert receipt['files']['build-output.tar.gz']['sha256'] == hosted['build_output_sha256']
    assert receipt['files']['build-image.tar.gz']['sha256'] == hosted['image_archive_sha256']
    assert artifact['id'] == hosted['artifact_id']
    assert artifact['name'] == hosted['artifact_name'] == 'tq-matched-linux-builds'
    assert Path(hosted['download_archive_file']).name == 'tq-matched-linux-builds.zip'
    pin(hosted['download_archive_file'], hosted['download_archive_sha256'])
    build_archive = root / hosted['build_output_file']
    image_archive = root / hosted['image_archive_file']
    pin(build_archive, hosted['build_output_sha256'])
    pin(image_archive, hosted['image_archive_sha256'])
    pin(root / 'image-id.txt', hosted['image_id_file_sha256'])
    assert (root / 'image-id.txt').read_text().strip() == hosted['image_id']
    pin(root / 'image-inspect.json', hosted['image_inspect_sha256'])
    loaded = json.loads(command(['docker', 'image', 'inspect', hosted['execution_image_id']]))[0]
    assert loaded['Id'] == hosted['execution_image_id']
    saved_image = json.loads((root / 'image-inspect.json').read_text())
    assert isinstance(saved_image, list) and len(saved_image) == 1 and saved_image[0]['Id'] == hosted['image_id']
    image_config, loaded_record = verify_image_equivalence(hosted, pin)
    saved_hosted = saved_image[0]
    assert {key: value for key, value in saved_hosted['Config'].items() if value not in (None, '', False)} == image_config['config']
    assert saved_hosted['Architecture'] == image_config['architecture'] and saved_hosted['Os'] == image_config['os']
    assert saved_hosted['RootFS']['Layers'] == image_config['rootfs']['diff_ids']
    assert loaded['RootFS']['Layers'] == loaded_record['RootFS']['Layers']
    assert {key: value for key, value in loaded['Config'].items() if value not in (None, '', False)} == image_config['config']

    arms = {}
    for arm, revision, kind in (('main', hosted['main_revision'], 'llmx'),
                                ('candidate', hosted['candidate_revision'], 'llmx'),
                                ('mx', hosted['mx_revision'], 'mx')):
        folder = root / 'arms' / arm
        result_path = folder / 'build-result.json'
        result_hash = hosted['arms'][arm]['result_sha256']
        pin(result_path, result_hash)
        result = json.loads(result_path.read_text())
        pin(folder / 'build.log', result['log_sha256'])
        assert result['status'] == 'passed' and result['removed'] is True
        assert result['revision'] == revision and result['kind'] == kind
        assert result['image_id'] == hosted['image_id']
        assert result['attach_exit_code'] == 0 and result['container_state']['ExitCode'] == 0
        assert not result['container_state']['OOMKilled'] and not result['container_state']['Running']
        assert result['source_head_after'] == revision and not result['source_status_after'].strip()
        build_root = root / 'arms' / (arm + '-build')
        for artifact in result['build_artifacts']:
            artifact_path = build_root / artifact['path']
            pin(artifact_path, artifact['sha256'])
            assert artifact_path.stat().st_size == artifact['bytes']
        arms[arm] = result
    assert arms['main']['configure'] == arms['candidate']['configure']
    assert arms['main']['image_id'] == arms['candidate']['image_id'] == arms['mx']['image_id']
    assert arms['mx']['configure'] == hosted['mx_configure']
    assert arms['candidate']['configure'] == hosted['llmx_configure']
    assert arms['main']['configure'] == hosted['llmx_configure']
    assert arms['candidate']['targets'] == hosted['candidate_targets']
    assert arms['main']['targets'] == hosted['main_targets']
    assert arms['mx']['targets'] == hosted['mx_targets']
    mx_artifacts = {item['path']: item['sha256'] for item in arms['mx']['build_artifacts']}
    assert hosted['mx_linked_libraries']
    assert all(mx_artifacts.get(path) == digest for path, digest in hosted['mx_linked_libraries'].items())

    cbuild = root / 'arms' / 'candidate-build'
    mbuild = root / 'arms' / 'mx-build'
    candidate_binary = cbuild / 'llmx'
    candidate_config = cbuild / 'generated' / 'config.hpp'
    mx_binary = mbuild / 'bin' / 'llama-bench'
    for path in (candidate_binary, candidate_config, mx_binary):
        assert path.is_file(), str(path)
    assert os.access(candidate_binary, os.X_OK) and os.access(mx_binary, os.X_OK)
    candidate_version = (root / 'arms' / 'candidate' / 'version.txt').read_text().strip()
    assert hosted['candidate_version'] == candidate_version
    assert hosted['candidate_revision'][:12] in candidate_version and not re.search(r'dirty|unknown', candidate_version)
    pin(root / 'arms' / 'candidate' / 'version.txt', hosted['candidate_version_file_sha256'])
    pin(candidate_binary, hosted['candidate_binary_sha256'])
    pin(candidate_config, hosted['candidate_config_sha256'])
    pin(mx_binary, hosted['mx_binary_sha256'])
    wrapper = m['wrapper_compile']
    compile_receipt = verify_wrapper_provenance(wrapper, hosted, pin)
    assert compile_receipt['wrapper_source_sha256'] == wrapper['source_sha256'] == sha(Path(__file__).with_name('wrapper.cpp'))
    assert hosted['mx_linked_libraries'] and all(re.fullmatch('[0-9a-f]{64}', value)
                                                  for value in hosted['mx_linked_libraries'].values())
    assert os.access(wrapper['binary'], os.X_OK)

    for model in m['models']:
        assert Path(model['path']).stat().st_size == model['bytes']
        pin(model['path'], model['sha256'])
    pin(m['monitor']['path'], m['monitor']['sha256'])
    return pins


def peer_reservations(peers):
    unused = 0
    details = {}
    for name, peer in peers.items():
        cap = peer.get('memory_cap_bytes', 0)
        if peer['idle'] or not cap:
            continue
        used = 0
        try:
            line = next((v[3:] for v in Path('/proc', str(peer['pid']), 'cgroup').read_text().splitlines() if v.startswith('0::')))
            group = Path('/sys/fs/cgroup') / line.lstrip('/')
            used = min(max(0, int((group / 'memory.current').read_text())), cap)
            details[name] = {'memory_cap_bytes': cap, 'used_bytes': used}
        except (OSError, StopIteration, ValueError) as error:
            details[name] = {'memory_cap_bytes': cap, 'used_bytes': 0, 'unavailable': str(error)}
        details[name]['unused_reserved_bytes'] = cap - used
        unused += cap - used
    return (unused, details)


def preflight(m):
    device = m['device']
    node = Path('/sys/class/drm') / Path(device['node']).name / 'device'
    guard = dict(pci=node.resolve().name, busy=int((node / 'gpu_busy_percent').read_text()),
                 vram=int((node / 'mem_info_vram_used').read_text()), power=(node / 'power_dpm_force_performance_level').read_text().strip())
    assert guard['pci'] == device['pci'] and guard['busy'] == 0 and guard['vram'] < 64 * 1024**2 and guard['power'] == 'auto', guard
    assert subprocess.run(['fuser', device['node']], capture_output=True, timeout=30).returncode == 1
    def cpu_set(text):
        result = set()
        for span in text.split(','):
            if span:
                ends = [int(value) for value in span.split('-')]
                result.update(range(ends[0], ends[-1] + 1))
        return result
    requested = cpu_set(m['resources']['cpuset'])
    active = {}
    for cid in command(['docker', 'ps', '-q']).splitlines():
        info = json.loads(command(['docker', 'inspect', cid]))[0]
        nodes = [d['PathOnHost'] for d in info['HostConfig'].get('Devices') or []]
        assert device['node'] not in nodes, ('device mapped by running container', info['Name'], nodes)
        processes = [line.split(None, 2) for line in command(['docker', 'top', cid, '-eo', 'pid,comm,args']).splitlines()[1:]]
        idle = bool(processes) and {row[1] for row in processes} <= {'docker-init', 'sleep'} and all(
            row[1] != 'sleep' or row[2] == 'sleep infinity' for row in processes)
        cpuset = info['HostConfig']['CpusetCpus']
        assert idle or not cpu_set(cpuset).intersection(requested), ('explicit CPU overlap with active container', info['Name'], cpuset, processes)
        active[info['Name']] = dict(nodes=nodes, cpuset=cpuset, idle=idle, processes=processes,
                                  memory_cap_bytes=info['HostConfig'].get('Memory', 0), pid=info['State']['Pid'])
        if info['Name'].lstrip('/').startswith(('llmx-xdev-', 'llmx-iq3', 'llmx-iq1', 'llmx-iq2', 'llmx-dashboard-')):
            assert idle, ('competing owned workload', info['Name'], processes)
    assert requested <= os.sched_getaffinity(0), ('host CPU affinity', requested)
    # Same owned-host-work check used by the existing final timing owner; skip this launcher and its ancestors.
    ancestors, pid = set(), os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid = int(next(line.split()[1] for line in Path('/proc', str(pid), 'status').read_text().splitlines() if line.startswith('PPid:')))
        except OSError:
            break
    host = []
    for row in command(['ps', '-eo', 'pid=,comm=,args=']).splitlines():
        fields = row.split(None, 2)
        if len(fields) < 3 or int(fields[0]) in ancestors:
            continue
        if any(root in fields[2] for root in ('/zpool1/llmx-xdev-quant-20261020/', '/zpool1/llmx-p2-dashboard-landing-20261023/')):
            assert fields[1] in ('sleep', 'docker-init'), ('active owned host work', row)
            host.append(row)
    peer_unused, reservations = peer_reservations(active)
    active['host_dormant'] = host
    active['peer_memory_reservations'] = reservations
    memory = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
    assert memory > 10 * 1024**3 and shutil.disk_usage(m['candidate']['root']).free > 3 * 1024**3
    required = 4 * 1024**3 + 8 * 1024**3 + peer_unused
    active['memory_admission'] = dict(host_available_bytes=memory, required_bytes=required, own_cap_bytes=8 * 1024**3)
    assert memory >= required, ('host reserve plus own and unused peer budgets', active['memory_admission'])
    assert subprocess.run(['docker', 'inspect', m['container']], capture_output=True, timeout=30).returncode != 0
    return guard, active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--resources-released', action='store_true')
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    bundle_pins = json.loads((bundle / 'bundle-pins.json').read_text())
    assert all(sha(bundle / name) == digest for name, digest in bundle_pins.items())
    manifest_path = Path(__file__).with_name('manifest.json')
    m = json.loads(manifest_path.read_text())
    if args.self_test:
        calls = planned_calls(m, bundle)
        assert len(calls) == 40 and sum(c['warmup'] for c in calls) == 8
        assert calls[0]['command'][calls[0]['command'].index('--cache-type-k') + 1] == 'f16'
        assert any(c['command'][0] == '/mx/mx-cli-bench-vulkan' for c in calls)
        assert calls[-1]['command'][0] == '/b/llmx'
        validate_manifest(m)
        missing = copy.deepcopy(m)
        missing['wrapper_compile']['receipt_sha256'] = None
        try:
            validate_manifest(missing)
        except AssertionError as error:
            assert 'wrapper_compile' in str(error), str(error)
        else:
            raise AssertionError('missing wrapper receipt must refuse execution')
        altered = copy.deepcopy(m)
        altered['hosted_build']['execution_image_id'] = 'sha256:' + ('0' * 64)
        try:
            validate_manifest(altered)
        except AssertionError:
            pass
        else:
            raise AssertionError('unproven execution image ID must refuse execution')
        print(json.dumps({'self_test': 'pass', 'planned_calls': len(calls),
                          'warmups': sum(c['warmup'] for c in calls),
                          'wrapper_receipt': 'pinned',
                          'refusal_tests': ['missing wrapper receipt', 'altered image identity'],
                          'execution': 'fail-closed until fresh resource admission'}))
        return 0
    assert args.resources_released, 'Root must finish competing owned work and release resources; numerical gate completion is separate'
    if resource is None:
        raise RuntimeError('the timing executor requires POSIX core-limit support')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    validate_manifest(m)
    out = Path(m['output'])
    out.mkdir(exist_ok=False)
    record = dict(start=time.time(), manifest=m, manifest_sha256=sha(manifest_path), calls=planned_calls(m, bundle),
                  thresholds=m['thresholds'], passed=False, removed=False,
                  scope=m['scope'], phase_time_scope='PP and TG are independent empty-history phases; P/PP+G/TG excludes load/startup and is not request or server latency.')
    def save():
        (out / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    cid, mon, pins = None, None, {}
    try:
        save()
        pins = verify_pins(m)
        record['guard'], record['active_devices'] = preflight(m)
        (out / 'host-processes-before.txt').write_text(command(['ps', '-eo', 'pid,ppid,pcpu,comm,args']))
        args = ['docker', 'create', '--init', '--runtime', 'runc', '--name', m['container'], '--label', m['label'] + '=' + m['container'],
                '--cidfile', str(out / 'container.cid'), '--network', 'none', '--cpuset-cpus', m['resources']['cpuset'], '--cpus', m['resources']['cpus'],
                '--memory', m['resources']['memory'], '--memory-swap', m['resources']['memory'], '--pids-limit', str(m['resources']['pids']),
                '--ulimit', 'core=0', '--device', m['device']['node'], '--read-only', '--tmpfs', '/tmp:rw,size=512m',
                '-e', 'XDG_CACHE_HOME=/tmp/llmx-test-cache', '-e', 'LD_LIBRARY_PATH=/ref/bin',
                '-v', str(Path(m['candidate']['root']) / m['candidate']['build']) + ':/b:ro',
                '-v', m['reference']['root'] + ':/ref:ro',
                '-v', m['wrapper_compile']['binary'] + ':/mx/mx-cli-bench-vulkan:ro',
                '-v', str(Path(m['models'][0]['path']).parent) + ':/models:ro',
                m['hosted_build']['execution_image_id'], 'sleep', 'infinity']
        record['command'] = args
        cid = command(args)
        assert re.fullmatch('[0-9a-f]{64}', cid)
        command(['docker', 'start', cid])
        def run(tag, argv):
            with (out / (tag + '.stdout')).open('xb') as stdout, (out / (tag + '.stderr')).open('xb') as stderr:
                result = subprocess.run(['docker', 'exec', cid] + argv, stdout=stdout, stderr=stderr,
                                        timeout=m['resources']['child_timeout_seconds'])
            assert result.returncode == 0, (tag, result.returncode)
            return '\n'.join((out / (tag + '.' + suffix)).read_text(encoding='utf-8') for suffix in ('stdout', 'stderr'))
        record['version'] = run('version', ['/b/llmx', '--version']).strip()
        assert record['version'] == m['hosted_build']['candidate_version']
        device = run('device', ['vulkaninfo', '--summary'])
        assert all(m['device'][key] in device for key in ('uuid', 'driver', 'name')), device
        with (out / 'monitor.log').open('xb') as log:
            mon = subprocess.Popen(['python3', m['monitor']['path'], '--output', str(out / 'activity.jsonl'), '--stop-file', str(out / 'monitor.stop'),
                                    '--benchmark-cgroup', cid, '--interval', '1', '--seconds', str(m['resources']['monitor_seconds'])], stdout=log, stderr=subprocess.STDOUT)
        time.sleep(2)
        for call in record['calls']:
            assert mon.poll() is None
            call.update(start=time.time(), status='running')
            save()
            try:
                text = run(str(call['index']) + '-' + call['arm'], call['command'])
                rates = re.findall(r'\b(pp' + str(call['p']) + r'|tg' + str(call['g']) + r')\s+([\d.]+)(?:\s|$)', text)
                assert len(rates) == 2 and {phase for phase, _ in rates} == {f"pp{call['p']}", f"tg{call['g']}"}, text
                values = {phase: float(value) for phase, value in rates}
                assert all(value > 0 for value in values.values())
                if call['arm'] == 'candidate':
                    assert 'dtype: auto -> f16' in text, text
                    paths = [json.loads(line.split('matrix-paths: ', 1)[1]) for line in text.splitlines() if line.startswith('matrix-paths: ')]
                    assert len(paths) == 1 and paths[0]['dtype'] == 'f16' and 'block-int16' in str(paths[0]) and 'block-int8' not in str(paths[0]), paths
                    call['matrix_paths'] = paths[0]
                else:
                    assert ('reference_commit=' + m['reference']['head'] in text and
                            'kv=f16' in text and 'load=direct-io' in text), text
                call.update(end=time.time(), status='passed', rates=values,
                            pp_phase_seconds=call['p'] / values[f"pp{call['p']}"], tg_phase_seconds=call['g'] / values[f"tg{call['g']}"])
                call['pp_plus_tg_phase_seconds'] = call['pp_phase_seconds'] + call['tg_phase_seconds']
            except BaseException:
                call.update(end=time.time(), status='failed', error=traceback.format_exc())
                save()
                raise
            save()
            print(json.dumps(call), flush=True)
        time.sleep(2)
        record['passed'] = True
    except BaseException:
        record['error'] = traceback.format_exc()
    finally:
        try:
            if mon:
                (out / 'monitor.stop').write_text('stop\n')
                try:
                    mon.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    mon.terminate()
                    mon.wait(timeout=10)
                record['monitor_exit'] = mon.returncode
                record['passed'] &= mon.returncode == 0
        except BaseException:
            record['monitor_cleanup_error'] = traceback.format_exc()
            record['passed'] = False
        try:
            if cid is None and (out / 'container.cid').is_file():
                cid = (out / 'container.cid').read_text().strip()
            if cid:
                assert re.fullmatch('[0-9a-f]{64}', cid)
                info = json.loads(command(['docker', 'inspect', cid]))[0]
                assert info['Id'] == cid and info['Config']['Labels'][m['label']] == m['container'] and info['Image'] == m['hosted_build']['execution_image_id']
                record['state'] = info['State']
                command(['docker', 'rm', '-f', cid])
                record['removed'] = subprocess.run(['docker', 'inspect', cid], capture_output=True, timeout=30).returncode != 0
                record['passed'] &= record['removed']
                record['passed'] &= not info['State']['OOMKilled']
        except BaseException:
            record['container_cleanup_error'] = traceback.format_exc()
            record['passed'] = False
        try:
            record['pins_unchanged'] = bool(pins) and pins == verify_pins(m) and sha(manifest_path) == record['manifest_sha256']
            assert all(sha(bundle / name) == digest for name, digest in bundle_pins.items())
            record['passed'] &= record['pins_unchanged']
        except BaseException:
            record['pin_recheck_error'] = traceback.format_exc()
            record['passed'] = False
        record['end'] = time.time()
        record['artifacts'] = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'execution.json'}
        save()
    print(json.dumps(dict(passed=record['passed'], removed=record['removed'])), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
