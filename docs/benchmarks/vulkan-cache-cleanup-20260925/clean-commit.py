from pathlib import Path
import subprocess,json,hashlib,sys,tempfile
root=Path(tempfile.gettempdir())/'llmx-vulkan-cache-cleanup-20260925'
art=Path(tempfile.gettempdir())/'llmx-vulkan-cache-cleanup-final-evidence-20260925'
records=[]
def run(name,cmd):
 print(name+' started',flush=True)
 with (art/(name+'.log')).open('wb') as out:r=subprocess.run(cmd,cwd=root,stdout=out,stderr=subprocess.STDOUT)
 records.append({'stage':name,'command':cmd,'exit':r.returncode})
 assert r.returncode==0,(name,r.returncode)
 print(name+' passed',flush=True)
run('clean-commit-build',['cmake','--build',str(root/'build-final'),'--config','Release','--clean-first','--parallel','2'])
run('clean-commit-native',['ctest','--test-dir',str(root/'build-final'),'-C','Release','--output-on-failure'])
exe=root/'build-final/Release/llmx.exe'
code="import os,sys;os.environ['LLMX_DEVICE']='vulkan:0';sys.path.insert(0,'tests');import common,version,f32,threads;common.EXE=sys.argv[1];assert all(f() for f in (version.run,f32.run,threads.run))"
run('clean-commit-focused',[sys.executable,'-X','utf8','-c',code,str(exe)])
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
identity={'commit':head,'version':subprocess.check_output([str(exe),'--version'],text=True).strip(),'sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}
assert identity['version']=='llmx 0.1.0+g'+head[:12],identity
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
(art/'clean-commit.json').write_text(json.dumps({'commands':records,'identity':identity},indent=2)+'\n')
print(json.dumps(identity),flush=True)
