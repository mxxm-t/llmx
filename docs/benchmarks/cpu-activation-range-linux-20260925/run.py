from pathlib import Path
import tarfile,subprocess,json,hashlib,os,sys
art=Path('/mnt/c/Users/Marko/AppData/Local/Temp/llmx-activation-linux-20260925')
root=Path('/tmp/llmx-activation-linux-20260925')
root.mkdir(exist_ok=False)
with tarfile.open(art/'source.tar.gz') as archive:archive.extractall(root,filter='data')
plan=json.loads((art/'plan.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in plan['files'].items())
records=[]
def run(stage,command,env=None):
 with (art/(stage+'.log')).open('wb') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
 records.append({'stage':stage,'command':command,'exit':result.returncode,'baseline_gguf':(env or {}).get('LLMX_BASELINE_GGUF')})
 (art/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 assert result.returncode==0,(stage,result.returncode)
 print(stage,'passed',flush=True)
run('configure',['cmake','-S',str(root),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release','-DLLMX_HAS_BACKEND_VULKAN=OFF'])
run('build',['cmake','--build',str(root/'build'),'--parallel','2'])
run('native',['ctest','--test-dir',str(root/'build'),'--output-on-failure'])
for name,spec in plan['models'].items():
 with open(spec['linux_path'],'rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 assert digest==spec['sha256'],name
print('All three real-model identities verified',flush=True)
exe=root/'build/llmx'
env=os.environ.copy();env['LLMX_BASELINE_GGUF']=plan['models']['Qwen3-0.6B-Q8_0.gguf']['linux_path']
run('full-cpu-q8',['python3','-u','-X','utf8',str(root/'tests/run_tests.py'),'--exe',str(exe),'--device','cpu'],env)
for name,spec in plan['models'].items():
 if 'Q8_0' in name:continue
 env=os.environ.copy();env['LLMX_BASELINE_GGUF']=spec['linux_path'];env['LLMX_DEVICE']='cpu'
 code='import sys; sys.path.insert(0,'+repr(str(root/'tests'))+'); import common,baseline; common.EXE='+repr(str(exe))+'; assert baseline.run()'
 run('hf-'+name,['python3','-u','-X','utf8','-c',code],env)
for name in plan['models']:
 stage='full-cpu-q8' if 'Q8_0' in name else 'hf-'+name
 text=(art/(stage+'.log')).read_text()
 assert 'baseline-logits['+name+']: top-1 6/6' in text,name
 rows=[line for line in text.splitlines() if line.startswith('baseline-ppl['+name+' c=')]
 assert len(rows)==8 and all('[ok]' in line for line in rows),(name,rows)
(art/'complete.json').write_text(json.dumps({'exit':0,'version':subprocess.check_output([str(exe),'--version'],text=True).strip(),'sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'scope':'Full CPU suite with required-present hash-verified Q8 fixture, followed by Q4 and Q5 HF checks with each explicit test fixture. Other fixture SKIPs within an individual invocation are not missing aggregate coverage. No Vulkan/performance claim.'})+'\n')
print('Linux CPU backend audit validation complete',flush=True)
