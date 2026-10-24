import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S = ROOT / 'build-quant-resume-20261020'
COLLECTED = S / 'tq-hosted-probes-collected-20261024' / 'tq-probes-40106666694'
WORKFLOW = ROOT / 'worktrees' / 'tq-probe-build-ci-20261024'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))
pins = json.loads((HERE / 'bundle-pins.json').read_text(encoding='utf-8'))['files']
assert set(pins) == {p.name for p in HERE.iterdir() if p.is_file() and p.name != 'bundle-pins.json'}
assert all(sha(HERE / name) == digest for name, digest in pins.items())
assert plan['schema'] == 'tq-clean-depth-capture-prep-v2'

build_receipt_path = HERE / 'a0fa-build-result.json'
build_receipt = json.loads(build_receipt_path.read_text(encoding='utf-8'))
build_audit_path = S / 'tq-final-matched-build-prep-20261024' / 'run-40101812090' / 'build-audit.json'
build_audit = json.loads(build_audit_path.read_text(encoding='utf-8'))
collection_path = S / 'tq-final-matched-build-prep-20261024' / 'run-40101812090' / 'collection.json'
collection = json.loads(collection_path.read_text(encoding='utf-8'))
assert sha(build_receipt_path) == plan['candidate_source']['build_receipt_sha256']
assert build_receipt['revision'] == plan['candidate_source']['revision']
assert build_receipt['status'] == 'passed' and build_receipt['removed']
assert build_receipt['attach_exit_code'] == 0
assert build_receipt['container_state']['ExitCode'] == 0 and not build_receipt['container_state']['OOMKilled']
assert build_receipt['source_head_after'] == build_receipt['revision'] and build_receipt['source_status_after'] == ''
assert build_receipt['image_id'] == plan['image_identity']['id']
assert build_audit['status'] == 'passed' and build_audit['run'] == 40101812090
assert build_audit['artifact_sha256'] == collection['archive_sha256']
assert build_audit['image_id'] == build_receipt['image_id']
assert build_audit['build_output_sha256'] == plan['hosted_build_run']['output_archive_sha256']
assert build_audit['image_archive_sha256'] == plan['hosted_build_run']['image_archive_sha256']
tree = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', plan['candidate_source']['revision'] + '^{tree}'], text=True).strip()
assert tree == plan['candidate_source']['source_tree_git_sha1']
recorded = {row['path']: row for row in build_receipt['build_artifacts']}
for item in plan['candidate_source']['build_artifacts']:
    assert item['path'] in recorded
    assert item['bytes'] == recorded[item['path']]['bytes'] and item['sha256'] == recorded[item['path']]['sha256']
assert recorded['llmx']['sha256'] == plan['candidate_source']['cli_sha256']
assert recorded['libllmx-vulkan.a']['sha256'] == plan['candidate_source']['library_sha256']
assert recorded['generated/config.hpp']['sha256'] == plan['candidate_source']['generated_config_sha256']
assert recorded['generated/llmx-build-info.hpp']['sha256'] == plan['candidate_source']['generated_build_info_sha256']

metadata = json.loads((COLLECTED / 'artifact-metadata.json').read_text(encoding='utf-8'))
execution_path = COLLECTED / 'probe-build' / 'execution.json'
execution = json.loads(execution_path.read_text(encoding='utf-8'))
probe_root = COLLECTED / 'probe-build' / 'probes'
probe_receipt_path = probe_root / 'probe-build.json'
probe_receipt = json.loads(probe_receipt_path.read_text(encoding='utf-8'))
assert metadata['id'] == plan['probe_owner']['hosted_probe_artifact_id']
assert metadata['digest'] == 'sha256:' + sha(COLLECTED / 'artifact.zip')
assert metadata['digest'].split(':', 1)[1] == plan['probe_owner']['hosted_probe_artifact_sha256']
assert metadata['workflow_run']['id'] == plan['probe_owner']['hosted_probe_run_id']
assert metadata['workflow_run']['head_sha'] == plan['probe_owner']['hosted_probe_artifact_head']
head = subprocess.check_output(['git', '-C', str(WORKFLOW), 'rev-parse', 'HEAD'], text=True).strip()
parent = subprocess.check_output(['git', '-C', str(WORKFLOW), 'rev-parse', 'HEAD^'], text=True).strip()
assert head == plan['probe_owner']['hosted_probe_artifact_head']
assert parent == plan['candidate_source']['revision']
assert execution['status'] == 'passed' and execution['candidate_revision'] == plan['candidate_source']['revision']
assert execution['candidate_tree'] == plan['candidate_source']['source_tree_git_sha1']
assert execution['source']['git_archive_sha256'] == plan['candidate_source']['git_archive_sha256']
assert execution['source']['file_count'] == plan['candidate_source']['source_file_count']
assert execution['hosted_build_run_id'] == 40101812090
assert execution['hosted_artifact_id'] == 12563587325
assert execution['hosted_artifact_digest'] == plan['hosted_build_run']['artifact_sha256']
assert execution['hosted_image_id'] == plan['image_identity']['id']
assert execution['loaded_image_id'] == plan['image_identity']['id']
assert execution['docker_load_exit_code'] == 0
assert execution['container_attach_exit_code'] == 0
assert execution['container_state']['ExitCode'] == 0 and not execution['container_state']['OOMKilled']
assert execution['container_state_final']['ExitCode'] == 0 and not execution['container_state_final']['OOMKilled']
assert execution['container_remove_exit_code'] == 0 and execution['container_absent_after_cleanup']
assert sha(execution_path) == plan['probe_owner']['hosted_probe_execution_sha256']
assert sha(probe_receipt_path) == plan['probe_owner']['probe_build_receipt_sha256']
assert execution['probe_build_receipt_sha256'] == sha(probe_receipt_path)
assert probe_receipt['sources'] == plan['probe_source_sha256']
assert execution['probe_source_hashes']['build-probes.py'] == pins['build-probes.py']
assert probe_receipt['library_sha256'] == plan['candidate_source']['library_sha256']
assert execution['probe_build_receipt']['library_sha256'] == probe_receipt['library_sha256']
assert {key: value['sha256'] for key, value in execution['probes'].items()} == probe_receipt['probes']
for policy, filename in [('default', 'quant-depth-default-probe.cpp'), ('f32', 'quant-depth-f32-probe.cpp')]:
    expected = plan['probe_owner'][policy + '_probe_sha256']
    binary = probe_root / (policy + '-probe')
    assert sha(binary) == expected == probe_receipt['probes'][policy]
    assert sha(HERE / filename) == plan['probe_source_sha256'][filename]
    assert sha(WORKFLOW / '.qualification' / 'tq-probes' / filename) == plan['probe_source_sha256'][filename]
assert sha(WORKFLOW / '.qualification' / 'tq-probes' / 'build-probes.py') == pins['build-probes.py']

with zipfile.ZipFile(COLLECTED / 'artifact.zip') as archive:
    contents = {item.filename: archive.read(item.filename) for item in archive.infolist() if not item.is_dir()}
expected_zip_files = {
    'probe-build/compile.log': COLLECTED / 'probe-build' / 'compile.log',
    'probe-build/docker-load.log': COLLECTED / 'probe-build' / 'docker-load.log',
    'probe-build/cleanup-check.json': COLLECTED / 'probe-build' / 'cleanup-check.json',
    'probe-build/execution.json': execution_path,
    'probe-build/probes/default-probe': probe_root / 'default-probe',
    'probe-build/probes/f32-probe': probe_root / 'f32-probe',
    'probe-build/probes/probe-build.json': probe_receipt_path,
    'source-pin-self-test.json': COLLECTED / 'source-pin-self-test.json',
}
assert set(contents) == set(expected_zip_files)
assert all(contents[name] == path.read_bytes() for name, path in expected_zip_files.items())
print(json.dumps({
    'passed': True,
    'scope': 'local source/build/probe artifact and bundle binding only; no transfer, probe execution, capture or inference',
    'candidate_revision': build_receipt['revision'],
    'candidate_tree': tree,
    'build_receipt_sha256': sha(build_receipt_path),
    'workflow_head': head,
    'hosted_probe_artifact_sha256': sha(COLLECTED / 'artifact.zip'),
    'probe_build_receipt_sha256': sha(probe_receipt_path),
    'probe_hashes': probe_receipt['probes'],
    'verified_bundle_files': len(pins),
    'artifact_files_reconciled': len(contents),
    'container_removed_no_oom': True,
}, indent=2), flush=True)
