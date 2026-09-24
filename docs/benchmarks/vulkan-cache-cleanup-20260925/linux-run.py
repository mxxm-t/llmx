from pathlib import Path
import subprocess,json,hashlib
base=Path('/mnt/c/Users/Marko/AppData/Local/Temp')
r=base/'llmx-vulkan-cache-cleanup-20260925';a=base/'llmx-vulkan-cache-cleanup-final-evidence-20260925'
exe=a/'linux-vulkan-lifetime'
cmd=['g++','-std=c++17','-O3','-DNDEBUG','-mavx2','-mfma','-mf16c','-Wall','-Wextra','-I'+str(r/'src'),'-I'+str(r/'build-final/generated'),'-I/mnt/c/VulkanSDK/1.4.357.0/Include',str(r/'tests/vulkan_buffer.cpp'),'-o',str(exe),'-pthread','-ldl']
records=[]
for name,command in [('linux-build',cmd),('linux-buffer',[str(exe)]),('linux-queue',[str(exe),'--queue'])]:
 print(name+' started',flush=True)
 with (a/(name+'.log')).open('wb') as f: p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
 records.append({'stage':name,'command':command,'exit':p.returncode})
 (a/'linux-commands.json').write_text(json.dumps(records,indent=2)+'\n')
 print(name+' exit '+str(p.returncode),flush=True)
 if p.returncode and p.returncode!=77:raise SystemExit(p.returncode)
(a/'linux-binary.json').write_text(json.dumps({'sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'scope':'Local WSL GCC13.3 compile and available native cases only; not MI50 or HF numerical validation'})+'\n')
