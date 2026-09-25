"""Verify preserved performance evidence without extracting files or running workloads."""
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'package-manifest.json').read_text())

def hash_stream(stream):
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1048576), b''):
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()

def check_file(path, entry):
    with path.open('rb') as stream:
        size, digest = hash_stream(stream)
    if size != entry['bytes'] or digest != entry['sha256']:
        raise ValueError('File mismatch: ' + str(path))

def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
        raise ValueError('Unsafe relative path: ' + name)
    return path

copied = 0
members = 0
for name, entry in manifest['copied_files'].items():
    check_file(root / safe_name(name), entry)
    copied += 1
for name, entry in manifest['archives'].items():
    archive = root / safe_name(name)
    check_file(archive, entry)
    seen = set()
    with tarfile.open(archive, 'r:xz') as stream:
        for item in stream:
            safe_name(item.name)
            if not item.isfile() or item.name in seen or item.name not in entry['members']:
                raise ValueError('Unexpected archive member: ' + item.name)
            seen.add(item.name)
            with stream.extractfile(item) as content:
                size, digest = hash_stream(content)
            expected = entry['members'][item.name]
            if size != expected['bytes'] or digest != expected['sha256']:
                raise ValueError('Member mismatch: ' + item.name)
    if seen != set(entry['members']):
        raise ValueError('Missing archive members: ' + name)
    members += len(seen)
print(json.dumps({'status': 'passed', 'copied_files': copied, 'archive_members': members,
                  'archives': len(manifest['archives']), 'scope': 'Integrity only; no workload execution.'}))
