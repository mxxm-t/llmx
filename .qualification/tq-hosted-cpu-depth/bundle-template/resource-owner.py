import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback


LABEL = 'llmx.tqpair.f16kvdepth'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def command(argv, timeout=30):
    return subprocess.check_output(argv, text=True, timeout=timeout)


def safe_relative(root, name):
    rel = PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts, name
    path = Path(root).joinpath(*rel.parts).resolve()
    assert path.is_relative_to(Path(root).resolve()), name
    return path


def check_hashes(root, pins):
    for name, digest in pins.items():
        assert sha(safe_relative(root, name)) == digest, name


def verify_bundle(bundle):
    manifest = json.loads((bundle / 'bundle-pins.json').read_text())
    assert manifest['schema'] == 'tq-pair-f16kv-depth-reference-bundle-v1'
    check_hashes(bundle, manifest['files'])
    return {'bundle_pin_files': len(manifest['files']),
            'bundle_pins_sha256': sha(bundle / 'bundle-pins.json'),
            'runner_sha256': sha(__file__)}


def check_source_v2(plan, bundle):
    archive = Path(plan['source_v2_archive'])
    manifest = bundle / 'source-v2.json'
    assert sha(archive) == plan['source_v2_archive_sha256']
    assert sha(manifest) == plan['source_v2_manifest_sha256']
    data = json.loads(manifest.read_text())
    assert data['revision'] == plan['source_v2_revision']
    files = data['files']
    assert len(files) == plan['source_v2_file_count']
    with tarfile.open(archive, 'r:*') as tar:
        members = tar.getmembers()
        assert all(member.isreg() or member.isdir() for member in members)
        assert all(not PurePosixPath(member.name).is_absolute() and '..' not in PurePosixPath(member.name).parts for member in members)
        by_name = {member.name: member for member in members if member.isreg()}
        assert len({member.name for member in members}) == len(members), 'duplicate source-v2 paths'
        assert set(by_name) == set(files), (set(by_name) - set(files), set(files) - set(by_name))
        for name, digest in files.items():
            member = by_name[name]
            assert member.isreg(), (name, member.type)
            stream = tar.extractfile(member)
            assert stream is not None
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
    return {'archive_sha256': sha(archive), 'manifest_sha256': sha(manifest),
            'files': len(files), 'revision': data['revision']}


def extract_source(plan, bundle, destination):
    identity = check_source_v2(plan, bundle)
    manifest = json.loads((bundle / 'source-v2.json').read_text())['files']
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(plan['source_v2_archive'], 'r:*') as tar:
        for name in sorted(manifest):
            member = tar.getmember(name)
            path = safe_relative(destination, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target)
    check_hashes(destination, manifest)
    assert (destination / 'tests' / 'baseline.py').is_file()
    return {'execution_source': identity}


def available_bytes():
    line = next(row for row in Path('/proc/meminfo').read_text().splitlines()
                if row.startswith('MemAvailable:'))
    return int(line.split()[1]) * 1024


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


def resources(plan):
    limits = plan['resources']
    info = {'at': time.time(), 'host_available_bytes': available_bytes(),
            'disk_free_bytes': shutil.disk_usage(plan['root']).free,
            'cpu_affinity': sorted(os.sched_getaffinity(0))}
    first, last = map(int, limits['cpu_set'].split('-'))
    assert set(range(first, last + 1)).issubset(os.sched_getaffinity(0)), info
    assert info['host_available_bytes'] >= limits['minimum_host_available_bytes'], info
    assert info['disk_free_bytes'] >= limits['minimum_disk_free_bytes'], info
    requested = set(range(first, last + 1))
    peers = {}
    for cid in command(['docker', 'ps', '-q']).splitlines():
        peer = json.loads(command(['docker', 'inspect', cid]))[0]
        explicit = set()
        for span in peer['HostConfig']['CpusetCpus'].split(','):
            if span:
                ends = list(map(int, span.split('-')))
                explicit.update(range(ends[0], ends[-1] + 1))
        processes = [line.split(None, 2) for line in command(['docker', 'top', cid, '-eo', 'pid,comm,args']).splitlines()[1:]]
        idle = bool(processes) and {row[1] for row in processes} <= {'docker-init', 'sleep'} and all(
            row[1] != 'sleep' or row[2] == 'sleep infinity' for row in processes)
        peers[peer['Name']] = {'cpuset': peer['HostConfig']['CpusetCpus'], 'idle': idle, 'processes': processes, 'memory_cap_bytes': peer['HostConfig'].get('Memory', 0), 'pid': peer['State']['Pid']}
        assert idle or not explicit.intersection(requested), ('CPU set overlaps active container', peer['Name'], peers[peer['Name']])
    info['active_container_cpus'] = peers
    reserved, info['peer_memory_reservations'] = peer_reservations(peers)
    info['required_host_available_bytes'] = limits['minimum_host_available_bytes'] + reserved
    assert info['host_available_bytes'] >= info['required_host_available_bytes'], info
    return info


def verify_cache(plan):
    snapshot = Path(plan['hf_assets'], plan['hf_cache_snapshot'])
    assert snapshot.is_dir(), snapshot
    actual_names = {path.name for path in snapshot.iterdir() if path.is_file()}
    assert actual_names == set(plan['hf_snapshot_files']), actual_names
    actual = {name: sha(snapshot / name) for name in sorted(actual_names)}
    assert actual == plan['hf_snapshot_files'], actual
    return {'snapshot': str(snapshot), 'files': actual}


def verify(plan, bundle, source=None, model_ids=None, full_cache=False):
    bundle_identity = verify_bundle(bundle)
    assert sha(bundle / 'plan.json') == plan['_plan_sha256']
    assert sha(bundle / 'commands.json') == plan['_commands_sha256']
    assert sha(bundle / 'quant-depth-reference.py') == plan['producer_sha256']
    assert sha(bundle / 'model-header-inventory.json') == plan['model_header_inventory_sha256']
    source_v2 = check_source_v2(plan, bundle)
    models = plan['models'] if model_ids is None else [m for m in plan['models'] if m['id'] in model_ids]
    model_ids_present = {model['id'] for model in models}
    assert model_ids_present == (set(model_ids) if model_ids is not None else
                                 {model['id'] for model in plan['models']})
    model_identity = {}
    for model in models:
        path = Path(model['path'])
        assert path.stat().st_size == model['bytes']
        model_identity[model['id']] = sha(path)
        assert model_identity[model['id']] == model['sha256']
    if source is not None:
        check_hashes(source, json.loads((bundle / 'source-v2.json').read_text())['files'])
        assert (source / 'tests' / 'baseline.py').is_file()
    inputs = Path(plan['inputs'])
    assert sha(inputs / 'complete.json') == plan['input_manifest_sha256']
    input_pins = json.loads((inputs / 'complete.json').read_text())['artifacts']
    check_hashes(inputs, {name: item['sha256'] for name, item in input_pins.items()})
    interpreter = Path(plan['hf_environment'], 'venv/bin/python')
    assert interpreter.is_symlink() and os.readlink(interpreter) == 'python3'
    assert os.readlink(interpreter.parent / 'python3') == '/usr/local/bin/python3'
    image = json.loads(command(['docker', 'image', 'inspect', plan['hf_image']]))[0]
    assert image['Id'] == plan['hf_image']
    identity = {'bundle': bundle_identity, 'source_v2': source_v2,
                'models': model_identity, 'input_manifest_sha256': sha(inputs / 'complete.json'),
                'input_artifacts': len(input_pins),
                'producer_sha256': plan['producer_sha256'],
                'image_id': image['Id'], 'versions_expected': plan['versions']}
    if full_cache:
        identity['hf_cache'] = verify_cache(plan)
    return identity


def inspect_owned(cid, name, image):
    assert re.fullmatch(r'[0-9a-f]{64}', cid), cid
    info = json.loads(command(['docker', 'inspect', cid]))[0]
    assert info['Id'] == cid and info['Name'] == '/' + name
    assert info['Config']['Labels'].get(LABEL) == name and info['Image'] == image
    return info


def assert_container(info, plan, output):
    host = info['HostConfig']
    limits = plan['resources']
    assert host['NetworkMode'] == 'none' and host['Runtime'] == limits['runtime']
    assert host['CpusetCpus'] == limits['cpu_set'] and host['NanoCpus'] == limits['cpu_quota'] * 1000000000
    assert host['Memory'] == limits['memory_bytes']
    assert host['MemorySwap'] == limits['memory_swap_bytes']
    assert host['PidsLimit'] == limits['pids_limit']
    assert any(item['Name'] == 'core' and item['Soft'] == item['Hard'] == 0
               for item in host['Ulimits'])
    assert host['ReadonlyRootfs'] and not host['Privileged']
    assert not host.get('Devices') and not host.get('DeviceRequests')
    binds = [mount for mount in info['Mounts'] if mount['Type'] == 'bind']
    assert len([mount for mount in binds if mount['RW']]) == 1
    assert next(mount for mount in binds if mount['RW'])['Destination'] == '/output'
    assert host.get('Tmpfs', {}).get('/tmp') == limits['tmpfs'].split(':', 1)[1]
    assert Path(output).resolve() == Path(next(mount for mount in binds if mount['RW'])['Source']).resolve()



def live_host_floor(plan, sample):
    limits = plan['resources']
    used = max(0, min(int(sample.get('memory.current', 0)), limits['memory_bytes']))
    return limits['minimum_host_available_bytes'] - used


def monitor(info, plan):
    row = {'at': time.time(), 'host_available_bytes': available_bytes()}
    free = shutil.disk_usage(plan['root']).free
    row['disk_free_bytes'] = free
    pid = info['State']['Pid']
    if pid:
        try:
            cgroup = next(line[3:].strip('/') for line in Path('/proc', str(pid), 'cgroup').read_text().splitlines()
                          if line.startswith('0::'))
            root = Path('/sys/fs/cgroup') / cgroup
            for key in ('memory.current', 'memory.peak', 'memory.events', 'cpu.stat'):
                row[key] = (root / key).read_text().strip()
        except (OSError, StopIteration) as error:
            row['cgroup_unavailable'] = str(error)
    return row


def phase(plan, commands, bundle, source, runroot, item):
    model = next(entry for entry in plan['models'] if entry['id'] == item['model_id'])
    phase_root = runroot / model['id'] / item['name']
    phase_root.mkdir(parents=True, exist_ok=False)
    name = 'llmx-tqpair-f16kvdepth-' + runroot.name + '-' + model['id'] + '-' + item['name']
    record = {'model_id': model['id'], 'phase': item['name'], 'start': time.time(),
              'timeout_seconds': item['timeout_seconds'], 'passed': False, 'removed': False}
    container_id = None
    child = None
    try:
        record['resources'] = resources(plan)
        record['before'] = verify(plan, bundle, source=source, model_ids=[model['id']], full_cache=False)
        assert subprocess.run(['docker', 'inspect', name], capture_output=True, timeout=30).returncode != 0
        argv = ['docker', 'create', '--init', '--name', name, '--cidfile', str(phase_root / 'container.cid'),
                '--label', LABEL + '=' + name, '--network', 'none', '--runtime', 'runc',
                '--cpuset-cpus', plan['resources']['cpu_set'], '--cpus', str(plan['resources']['cpu_quota']),
                '--memory', str(plan['resources']['memory_bytes']), '--memory-swap', str(plan['resources']['memory_swap_bytes']),
                '--pids-limit', str(plan['resources']['pids_limit']), '--ulimit', 'core=0',
                '--read-only', '--tmpfs', plan['resources']['tmpfs'], '--security-opt', 'no-new-privileges']
        for key, value in plan['environment'].items():
            argv += ['-e', key + '=' + value]
        mounts = [
            (str(source), '/source', True),
            (plan['inputs'], '/inputs', True),
            (str(bundle / 'quant-depth-reference.py'), '/producer.py', True),
            (model['path'], commands['models'][model['id']]['model'], True),
            (plan['hf_environment'], '/hf', True),
            (plan['hf_assets'], '/assets', True),
            (str(bundle / 'source-v2.json'), '/source-v2.json', True),
            (str(bundle / 'plan.json'), '/plan.json', True),
            (str(bundle / 'commands.json'), '/commands.json', True),
            (str(bundle / 'payload.py'), '/payload.py', True),
            (str(phase_root), '/output', False),
        ]
        for src, dst, readonly in mounts:
            assert Path(src).exists(), src
            argv += ['--mount', 'type=bind,src=' + src + ',dst=' + dst + (',readonly' if readonly else '')]
        argv += [plan['hf_image'], '/hf/venv/bin/python', '-B', '-u', '/payload.py', model['id'], item['name']]
        record['container_create_argv'] = argv
        save(phase_root / 'execution.json', record)
        command(argv)
        container_id = (phase_root / 'container.cid').read_text().strip()
        info = inspect_owned(container_id, name, plan['hf_image'])
        assert_container(info, plan, phase_root)
        resources(plan)
        with (phase_root / 'run.log').open('xb') as log, (phase_root / 'activity.jsonl').open('x') as activity:
            child = subprocess.Popen(['docker', 'start', '-a', container_id], stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + item['timeout_seconds']
            while child.poll() is None:
                info = inspect_owned(container_id, name, plan['hf_image'])
                sample = monitor(info, plan)
                reserved, sample['peer_memory_reservations'] = peer_reservations(record['resources']['active_container_cpus'])
                sample['required_host_available_bytes'] = live_host_floor(plan, sample) + reserved
                activity.write(json.dumps(sample) + '\n')
                activity.flush()
                assert sample['host_available_bytes'] >= sample['required_host_available_bytes'], sample
                assert sample['disk_free_bytes'] >= plan['resources']['minimum_disk_free_bytes'], sample
                if time.monotonic() >= deadline:
                    record['timed_out'] = True
                    inspect_owned(container_id, name, plan['hf_image'])
                    subprocess.run(['docker', 'kill', container_id], check=False, timeout=30)
                    raise TimeoutError('phase exceeded ' + str(item['timeout_seconds']) + ' seconds')
                time.sleep(1)
        record['process_exit'] = child.returncode
        record['after'] = verify(plan, bundle, source=source, model_ids=[model['id']], full_cache=False)
        phase_result = json.loads((phase_root / 'payload-result.json').read_text())
        record['payload_result'] = phase_result
        assert child.returncode == 0 and phase_result['passed'] is True
        record['passed'] = True
    except BaseException:
        record['error'] = traceback.format_exc()
    finally:
        previous_handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            if not container_id and (phase_root / 'container.cid').is_file():
                container_id = (phase_root / 'container.cid').read_text().strip()
            if not container_id:
                recovery = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=30)
                if recovery.returncode == 0:
                    recovered = json.loads(recovery.stdout)[0]
                    container_id = recovered['Id']
                    inspect_owned(container_id, name, plan['hf_image'])
            if container_id:
                info = inspect_owned(container_id, name, plan['hf_image'])
                record['container_state'] = info['State']
                try:
                    save(phase_root / 'container.json', info)
                finally:
                    command(['docker', 'rm', '-f', container_id])
                record['removed'] = subprocess.run(['docker', 'inspect', container_id],
                                                   capture_output=True, timeout=30).returncode != 0
            else:
                record['removed'] = True
            if child is not None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=15)
        except BaseException:
            record['cleanup_error'] = traceback.format_exc()
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
        state = record.get('container_state', {})
        record['passed'] = bool(record.get('passed', False) and record.get('removed')
                                and state.get('ExitCode') == 0 and not state.get('OOMKilled', False))
        record['end'] = time.time()
        record['artifacts'] = {str(path.relative_to(phase_root)): sha(path)
                               for path in sorted(phase_root.rglob('*'))
                               if path.is_file() and path.name != 'execution.json'}
        save(phase_root / 'execution.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description='Prepare or run four offline CPU-only TQ F32-matrix/F16KV depth reference phases.')
    parser.add_argument('mode', choices=('preflight', 'run'))
    args = parser.parse_args()
    assert sys.version_info >= (3, 11), sys.version
    bundle = Path(__file__).resolve().parent
    plan = json.loads((bundle / 'plan.json').read_text())
    plan['_plan_sha256'] = sha(bundle / 'plan.json')
    plan['_commands_sha256'] = sha(bundle / 'commands.json')
    commands = json.loads((bundle / 'commands.json').read_text())
    if args.mode == 'preflight':
        resource_identity = resources(plan)
        print(json.dumps({'identity': verify(plan, bundle, full_cache=True),
                          'resources': resource_identity,
                          'host_python': sys.version}, indent=2), flush=True)
        return 0

    assert sys.platform.startswith('linux'), sys.platform
    import fcntl
    root = Path(plan['root'])
    lock = (root / '.tq-pair-f16kv-depth-reference.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    runroot = root / ('released-reference-' + stamp + '-' + str(os.getpid()))
    runroot.mkdir(exist_ok=False)
    report = {'start': time.time(), 'scope': plan['scope'], 'head': plan['feature_base'],
              'phases': [], 'passed': False}
    save(runroot / 'summary.json', report)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    signal.signal(signal.SIGINT, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt('SIGINT')))
    try:
        report['resources_before'] = resources(plan)
        report['preflight'] = verify(plan, bundle, full_cache=True)
        source = runroot / 'source'
        report['extracted_source'] = extract_source(plan, bundle, source)
        for item in plan['phases']:
            outcome = phase(plan, commands, bundle, source, runroot, item)
            report['phases'].append({'model_id': item['model_id'], 'name': item['name'],
                                     'passed': outcome['passed'],
                                     'execution_sha256': sha(runroot / item['model_id'] / item['name'] / 'execution.json')})
            save(runroot / 'summary.json', report)
            if not outcome['passed']:
                break
        report['hf_cache_after'] = verify_cache(plan)
        report['passed'] = len(report['phases']) == len(plan['phases']) and all(item['passed'] for item in report['phases'])
    except BaseException:
        report['error'] = traceback.format_exc()
    finally:
        report['end'] = time.time()
        completed = {(item['model_id'], item['name']) for item in report['phases']}
        report['not_started'] = [{'model_id': item['model_id'], 'name': item['name']}
                                 for item in plan['phases']
                                 if (item['model_id'], item['name']) not in completed]
        save(runroot / 'summary.json', report)
        print(json.dumps({'output': str(runroot), 'passed': report['passed'],
                          'phases': report['phases'], 'not_started': report['not_started']}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
