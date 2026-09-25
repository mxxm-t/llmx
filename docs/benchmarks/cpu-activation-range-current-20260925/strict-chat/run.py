import datetime,hashlib,json,socket,subprocess,time,urllib.request,urllib.error,sys
from pathlib import Path
art=Path(__file__).parent
plan=json.loads((art/'plan.json').read_text())
out=art/'runs'
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
        record['pid'] = process.pid
        (art / 'active.json').write_text(json.dumps(record, indent=2) + '\n')
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
            request_body = plan['request']
            request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat',
                data=json.dumps(request_body).encode('utf-8'), headers={'Content-Type': 'application/json'})
            start = time.monotonic()
            record['request_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            (art / 'active.json').write_text(json.dumps(record, indent=2) + '\n')
            with urllib.request.urlopen(request, timeout=plan['request_timeout_seconds']) as response:
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
            record['server_exit'] = process.returncode
            record['terminal_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            (out / (tag + '.record.json')).write_text(json.dumps(record, indent=2) + '\n')
            (art / 'last-terminal.json').write_text(json.dumps(record, indent=2) + '\n')


def main():
    assert hashlib.sha256((art/'plan.json').read_bytes()).hexdigest()=='7b269b13a140398e97efbab605db555dc9f74177ac1e50aee8cded143c78c170'
    try:
        for arm, info in plan['arms'].items():
            assert hashlib.sha256(Path(info['cli']).read_bytes()).hexdigest()==info['cli_sha256'],arm
        for model, info in plan['models'].items():
            with Path(info['path']).open('rb') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==info['sha256'],model
        (art/'preflight.json').write_text(json.dumps({'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arms':2,'models':3})+'\n')
    except BaseException as error:
        (art/'failure.json').write_text(json.dumps({'stage':'preflight','error':repr(error)})+'\n')
        raise
    out.mkdir(exist_ok=False)
    result={'plan_sha256':hashlib.sha256((art/'plan.json').read_bytes()).hexdigest(),'completed':[],'scope':plan['scope']}
    reference={}
    for cell in plan['order']:
        model,tag,arm=cell['model'],cell['tag'],cell['arm']
        label=model+'-'+tag
        info=plan['models'][model]
        exe=Path(plan['arms'][arm]['cli'])
        print(label+' started',flush=True)
        try:
            assert hashlib.sha256(exe.read_bytes()).hexdigest()==plan['arms'][arm]['cli_sha256']
            record,got=long_run(exe,info['path'],label,plan['request']['messages'][0]['content'],info['prompt_tokens'])
            compare={k:got[k] for k in ['ids','text','finish','prompt_tokens','tokens','reused_tokens']}
            if model not in reference:reference[model]=compare
            record['matches']=compare==reference[model]
            result['completed'].append(record)
            (art/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            assert record['matches'],label+': exact response differs'
            print(label+': prompt '+str(got['prompt_tokens'])+', generated '+str(got['tokens'])+', '+got['finish']+', exact match',flush=True)
        except BaseException as error:
            failure={'cell':cell,'error':type(error).__name__+': '+str(error),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_count':len(result['completed'])}
            (art/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
            raise
    (art/'complete.json').write_text(json.dumps({'exit':0,'calls':len(result['completed'])})+'\n')
if __name__=='__main__':main()
