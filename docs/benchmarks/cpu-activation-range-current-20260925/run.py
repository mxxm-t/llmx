from pathlib import Path
import tempfile,subprocess,json,sys,hashlib
root=Path(tempfile.gettempdir())/'llmx-cpu-activation-current-20260925';art=Path(tempfile.gettempdir())/'llmx-cpu-activation-current-evidence-20260925'
records=[]
def run(name,cmd):
 print(name+' started',flush=True)
 with (art/(name+'.log')).open('wb') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root)
 records.append({'stage':name,'command':cmd,'exit':r.returncode});(art/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 if r.returncode:raise RuntimeError((name,r.returncode))
 print(name+' passed',flush=True)
try:
 run('configure',['cmake','-S',str(root),'-B',str(root/'build'),'-G','Visual Studio 18 2026','-A','x64','-DLLMX_HAS_BACKEND_VULKAN=OFF'])
 run('build',['cmake','--build',str(root/'build'),'--config','Release','--parallel','2'])
 exe=root/'build/Release/llmx.exe'
 identity={'version':subprocess.check_output([str(exe),'--version'],text=True).strip(),'sha256':hashlib.sha256(exe.read_bytes()).hexdigest()};(art/'binary.json').write_text(json.dumps(identity,indent=2)+'\n');print(identity,flush=True)
 run('native',['ctest','--test-dir',str(root/'build'),'-C','Release','--output-on-failure'])
 for device in ('cpu',):
  run(device.replace(':','-')+'-hf',[sys.executable,'-u','-X','utf8',str(root/'tests/run_tests.py'),'--exe',str(exe),'--no-perf-floor','--require-baseline','--device',device])
 (art/'complete.json').write_text(json.dumps({'exit':0,'identity':identity})+'\n')
except BaseException as e:
 (art/'failure.json').write_text(json.dumps({'error':repr(e)})+'\n');raise
