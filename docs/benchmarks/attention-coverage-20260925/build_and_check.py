from pathlib import Path
import subprocess,json,os,datetime,hashlib
root=Path(__file__).resolve().parent
build=root/'build-attention'
evidence=root/'docs/benchmarks/attention-coverage-20260925'
assert not build.exists()
env=os.environ.copy()
i=int(env.get('GIT_CONFIG_COUNT','0'))
env['GIT_CONFIG_COUNT']=str(i+1)
env['GIT_CONFIG_KEY_'+str(i)]='safe.directory'
env['GIT_CONFIG_VALUE_'+str(i)]=root.as_posix()
commands=[('configure',['cmake','-S',str(root),'-B',str(build),'-G','Visual Studio 18 2026','-A','x64','-DBUILD_TESTING=ON','-DLLMX_HAS_BACKEND_VULKAN=ON','-DLLMX_VULKAN_INCLUDE_DIR=C:/VulkanSDK/1.4.357.0/Include','-DLLMX_GLSLC=C:/VulkanSDK/1.4.357.0/Bin/glslc.exe']),('build',['cmake','--build',str(build),'--config','Release','--parallel','2']),('native',['ctest','--test-dir',str(build),'-C','Release','--output-on-failure'])]
records=[]
for name,cmd in commands:
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    path=evidence/(name+'.log')
    with path.open('wb') as out:r=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
    records.append({'stage':name,'argv':cmd,'exit':r.returncode,'started':started,'finished':datetime.datetime.now(datetime.timezone.utc).isoformat(),'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (evidence/'commands.json').write_text(json.dumps(records,indent=2)+'\n',encoding='ascii')
    print(name,r.returncode,flush=True)
    if r.returncode:raise RuntimeError(name+' failed: '+str(path))
exe=build/'Release/llmx.exe'
result={'exit':0,'exe':str(exe),'version':subprocess.check_output([str(exe),'--version'],text=True).strip(),'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}
(evidence/'complete.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
print(json.dumps(result),flush=True)
