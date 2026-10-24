import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

bundle=Path('/bundle');out=Path('/out')
assert not (out/'probe-build.json').exists()
assert not any((out/(policy+'-probe')).exists() for policy in ['default','f32'])
assert hashlib.sha256(Path('/b/libllmx-vulkan.a').read_bytes()).hexdigest()==json.loads((bundle/'plan.json').read_text())['library_sha256']
flags=['-I/b/generated','-I/source/src','-O3','-DNDEBUG','-std=c++17','-mavx2','-mfma','-mf16c','-Wall','-Wextra','-Werror=unused-function','-Werror=unused-variable','-Werror=unused-but-set-variable']
commands=[['/usr/bin/c++']+flags+[str(bundle/('quant-depth-'+policy+'-probe.cpp')),'/b/libllmx-vulkan.a','-ldl','-o',str(out/(policy+'-probe'))] for policy in ['default','f32']]
print(subprocess.check_output(['/usr/bin/c++','--version'],text=True),flush=True)
print(json.dumps({'commands':commands,'max_parallel_compilers':2}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(subprocess.run,command,check=True) for command in commands]
    for future in futures:future.result()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(out/'probe-build.json').write_text(json.dumps({'commands':commands,'probes':{policy:sha(out/(policy+'-probe')) for policy in ['default','f32']},'sources':{p.name:sha(p) for p in bundle.glob('*.cpp')},'library_sha256':sha(Path('/b/libllmx-vulkan.a'))},indent=2)+'\n')
