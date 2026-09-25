from pathlib import Path
import subprocess,json,hashlib,tempfile
art=Path(tempfile.gettempdir())/'llmx-activation-final-perf-20260925'
plan=json.loads((art/'trees.json').read_text());records=json.loads((art/'build-commands.json').read_text());binaries=json.loads((art/'build-identities.json').read_text())
def run(label,cmd,root):
 print(label+' started',flush=True)
 with (art/(label+'.log')).open('wb') as out:r=subprocess.run(cmd,cwd=root,stdout=out,stderr=subprocess.STDOUT)
 records.append({'label':label,'command':cmd,'exit':r.returncode});(art/'build-commands.json').write_text(json.dumps(records,indent=2)+'\n')
 if r.returncode:raise RuntimeError(label)
def identity(exe):return {'path':str(exe),'version':subprocess.check_output([str(exe),'--version'],text=True).strip(),'sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}
for arm in ['candidate']:
 root=Path(plan['trees'][arm]);build=root/'build'
 run(arm+'-configure',['cmake','-S',str(root),'-B',str(build),'-G','Visual Studio 18 2026','-A','x64','-DBUILD_TESTING=OFF','-DLLMX_HAS_BACKEND_VULKAN=OFF'],root)
 run(arm+'-build',['cmake','--build',str(build),'--config','Release','--parallel','2','--target','llmx'],root)
 command='@echo off\ncall "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat"\nif errorlevel 1 exit /b %errorlevel%\ncl /nologo /std:c++17 /O2 /Ob2 /DNDEBUG /EHsc /W4 /arch:AVX2 /I "'+str(build/'generated')+'" /I "'+str(root/'src')+'" "'+str(art/'compare.cpp')+'" /Fo:"'+str(art/(arm+'.obj'))+'" /Fe:"'+str(art/(arm+'.exe'))+'"\n'
 path=art/(arm+'-harness.cmd');path.write_text(command)
 run(arm+'-harness',['cmd','/c',str(path)],root)
 binaries[arm]={'commit':plan['heads'][arm],'cli':identity(build/'Release/llmx.exe'),'harness':identity(art/(arm+'.exe'))}
 for kind in ['cli','harness']:assert '+g'+plan['heads'][arm][:12] in binaries[arm][kind]['version'] and '.dirty' not in binaries[arm][kind]['version'],binaries[arm]
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
 (art/'build-identities.json').write_text(json.dumps(binaries,indent=2)+'\n');print(arm+' identities verified',flush=True)

code = "import sys;sys.path.insert(0,sys.argv[1]);import common,version,f32,threads;common.EXE=sys.argv[2];assert version.run();assert f32.run();assert threads.run()"
run('candidate-focused',[__import__('sys').executable,'-X','utf8','-c',code,str(root/'tests'),str(build/'Release/llmx.exe')],root)
(art/'candidate-focused-complete.json').write_text(json.dumps({'exit':0,'identity':binaries['candidate']},indent=2)+'\n')
