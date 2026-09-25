import datetime
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

art = Path(__file__).parent
plan = json.loads((art / 'plan.json').read_text())
repo = Path(r'C:/Users/Marko/AppData/Local/Temp/llmx-cpu-activation-range-20260924')
trees = {arm: art.parent / f'llmx-activation-perf-{arm}-20260924' for arm in ['base', 'candidate', 'layout']}
out = art / 'correctness'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def run(command, stem, timeout=3600):
    record = {'command': command, 'timeout_seconds': timeout}
    try:
        with (out / (stem + '.stdout')).open('wb') as stdout, (out / (stem + '.stderr')).open('wb') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout)
        record['exit'] = result.returncode
    except BaseException as error:
        record['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        (out / (stem + '.command.json')).write_text(json.dumps(record, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(f'{stem}: exit {result.returncode}')
    return (out / (stem + '.stdout')).read_text(encoding='utf-8')

def build(source, arm, name):
    tree = trees[arm]
    script = out / (name + '.cmd')
    script.write_text('@call "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat" >nul\n'
        + '@cl /nologo /std:c++17 /O2 /Ob2 /DNDEBUG /EHsc /W4 /arch:AVX2 '
        + f'/I "{tree / "build/generated"}" /I "{tree / "src"}" "{art / source}" '
        + f'/Fo:"{out / (name + ".obj")}" /Fe:"{out / (name + ".exe")}"\n@exit /b %errorlevel%\n')
    run(['cmd', '/d', '/c', str(script)], 'build-' + name, 300)

def get_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]

def long_run(exe, model, tag, prompt, count):
    port = get_port()
    command = [str(exe), 'serve', model, '--host', '127.0.0.1', '--port', str(port),
        '--ctx-size', '32768', '--threads', '6', '--ubatch', '128', '--device', 'cpu',
        '--cache-type-k', 'f16', '--cache-type-v', 'f16']
    record = {'tag': tag, 'command': command, 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (out / (tag + '.server.log')).open('wb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(tag + ': server exited before health')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/health', timeout=2) as response:
                        healthy = json.load(response).get('status') == 'ok'
                    if healthy:
                        break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError(tag + ': health timeout')
                time.sleep(0.25)
            request_body = {'prompt': prompt, 'temperature': 0, 'seed': 0, 'max_tokens': -1}
            request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/generate',
                data=json.dumps(request_body).encode('utf-8'), headers={'Content-Type': 'application/json'})
            start = time.monotonic()
            with urllib.request.urlopen(request, timeout=7200) as response:
                raw = response.read()
            (out / (tag + '.response.json')).write_bytes(raw)
            record['wall_seconds'] = time.monotonic() - start
            got = json.loads(raw)
            record['observed'] = {key: got.get(key) for key in ['finish', 'prompt_tokens', 'reused_tokens', 'tokens']}
            record['response_sha256'] = hashlib.sha256(raw).hexdigest()
            assert got['finish'] == 'eos', f"{tag}: incomplete finish {got.get('finish')}"
            assert got['prompt_tokens'] == count >= 16384
            assert got['reused_tokens'] == 0
            assert got['ids'] and got['tokens'] == len(got['ids'])
            record.update({key: got[key] for key in ['finish', 'prompt_tokens', 'reused_tokens', 'tokens']})
            record['text_sha256'] = hashlib.sha256(got['text'].encode('utf-8')).hexdigest()
            record['ids_sha256'] = hashlib.sha256(json.dumps(got['ids'], separators=(',', ':')).encode('ascii')).hexdigest()
            record['response_sha256'] = hashlib.sha256(raw).hexdigest()
            return record, got
        except BaseException as error:
            if isinstance(error, urllib.error.HTTPError):
                (out / (tag + '.http-error')).write_bytes(error.read())
            record['error'] = type(error).__name__ + ': ' + str(error)
            raise
        finally:
            (out / (tag + '.record.json')).write_text(json.dumps(record, indent=2) + '\n')
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)

def main():
    assert (art / 'matrix/complete.json').exists(), 'Finish timing before correctness work.'
    assert (art / 'cli-matrix/complete.json').exists(), 'Finish CLI timing before correctness work.'
    out.mkdir(exist_ok=False)
    manifest = {'scope': 'Two 512-token windows scored through both batched and per-token paths; fresh CPU servers with a fixed >=16k prompt and greedy generation to EOS. This supplements the independent short HF gate; it is not a full-corpus or external long-context correctness claim.',
        'models': {m: v for m, v in plan['models'].items() if '0.6B' in m}, 'arms': {}, 'ppl': [], 'long': []}
    for arm in trees:
        build('exact_ppl.cpp', arm, 'exact-ppl-' + arm)
        cli = trees[arm] / 'build/Release/llmx.exe'
        manifest['arms'][arm] = {'cli': str(cli), 'cli_sha256': sha(cli),
            'version': subprocess.check_output([str(cli), '--version'], text=True).strip(),
            'ppl_exe_sha256': sha(out / ('exact-ppl-' + arm + '.exe'))}
    build('tokenize_file.cpp', 'base', 'tokenize-file')
    corpus = repo / 'tests/data/wiki.test.raw'
    manifest['corpus_sha256'] = sha(corpus)
    raw = corpus.read_text(encoding='utf-8').lstrip('\n ')
    prompt = raw[:80000] + '\n\nSummarize the key topics covered in the text above in one paragraph.\n'
    prompt_file = out / 'long-prompt.txt'
    prompt_file.write_text(prompt, encoding='utf-8', newline='')
    manifest['prompt_sha256'] = sha(prompt_file)
    (out / 'plan.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for model, info in manifest['models'].items():
        ids_text = run([str(out / 'tokenize-file.exe'), info['path'], str(prompt_file)], model + '-tokenize', 120)
        ids = [int(x) for x in ids_text.split()]
        assert 16384 <= len(ids) < 24576, f'unexpected prompt length {len(ids)}'
        info['prompt_tokens'] = len(ids)
        info['prompt_ids_sha256'] = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()
    (out / 'plan.json').write_text(json.dumps(manifest, indent=2) + '\n')
    cpu_command = [sys.executable, '-u', '-X', 'utf8', str(repo / 'tests/run_tests.py'),
        '--exe', str(trees['candidate'] / 'build/Release/llmx.exe'), '--device', 'cpu', '--require-baseline']
    run(cpu_command, 'required-hf-candidate', 3600)
    manifest['required_hf'] = {'command': cpu_command, 'exit': 0, 'stdout_sha256': sha(out / 'required-hf-candidate.stdout')}
    (out / 'results.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Clean candidate required-HF CPU suite passed.', flush=True)
    hf8b_command = [sys.executable, '-X', 'utf8', str(repo / 'tests/baseline_8b.py'),
        '--exe', str(trees['candidate'] / 'build/Release/llmx.exe'),
        '--model', plan['models']['Qwen3-8B-Q8_0.gguf']['path'], '--device', 'cpu',
        '--output-dir', str(out / 'hf-8b')]
    run(hf8b_command, 'hf-8b-candidate', 3600)
    manifest['hf_8b'] = json.loads((out / 'hf-8b/report.json').read_text())
    (out / 'results.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Candidate 8B independent HF gate passed.', flush=True)
    for model, info in manifest['models'].items():
        reference = None
        for tag, arm in [('base', 'base'), ('repeat', 'base'), ('layout', 'layout'), ('candidate', 'candidate')]:
            stem = model + '-ppl-' + tag
            command = [str(out / ('exact-ppl-' + arm + '.exe')), info['path'], str(corpus)]
            result = run(command, stem)
            scoring = result.splitlines()[1:]
            assert len(scoring) == 2 and scoring[0].startswith('batched ') and scoring[1].startswith('decode ')
            for line in scoring:
                for value in line.split()[-2:]:
                    assert math.isfinite(float.fromhex(value))
            if reference is None:
                reference = scoring
            manifest['ppl'].append({'model': model, 'tag': tag, 'arm': arm, 'command': command, 'result': scoring, 'matches': scoring == reference})
            (out / 'results.json').write_text(json.dumps(manifest, indent=2) + '\n')
            assert scoring == reference, f'{stem}: NLL differs'
            print(stem + ': exact NLL match', flush=True)
    for model, info in manifest['models'].items():
        reference = None
        for tag, arm in [('base', 'base'), ('repeat', 'base'), ('candidate', 'candidate')]:
            label = model + '-long-' + tag
            try:
                record, got = long_run(trees[arm] / 'build/Release/llmx.exe', info['path'], label, prompt, info['prompt_tokens'])
            except BaseException:
                record_path = out / (label + '.record.json')
                manifest['long'].append(json.loads(record_path.read_text()) if record_path.exists() else {'tag': label, 'error': 'launch failed'})
                (out / 'results.json').write_text(json.dumps(manifest, indent=2) + '\n')
                raise
            comparable = {key: got[key] for key in ['ids', 'text', 'finish', 'prompt_tokens', 'tokens', 'reused_tokens']}
            if reference is None:
                reference = comparable
            record['matches'] = comparable == reference
            manifest['long'].append(record)
            (out / 'results.json').write_text(json.dumps(manifest, indent=2) + '\n')
            assert record['matches'], label + ': exact output differs'
            print(f"{label}: {got['prompt_tokens']} prompt tokens, {got['tokens']} reply tokens, exact ids/text/finish match", flush=True)
    (out / 'complete.json').write_text(json.dumps({'ppl': len(manifest['ppl']), 'long': len(manifest['long']), 'exit': 0}) + '\n')

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if out.exists():
            (out / 'failure.json').write_text(json.dumps({'error': type(error).__name__ + ': ' + str(error),
                'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'completed': False}, indent=2) + '\n')
        raise
