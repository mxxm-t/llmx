import datetime,hashlib,json,math,os,re,subprocess,sys,time
from pathlib import Path
ART=Path(__file__).parent
STAGE='common'
guarded_bytes=(ART/'guarded-driver-freeze.json').read_bytes()
guarded=json.loads(guarded_bytes)
assert hashlib.sha256((ART/'freeze.json').read_bytes()).hexdigest()==guarded['original_freeze_sha256']
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==guarded['driver_sha256'][Path(__file__).name]
plan=json.loads((ART/'plan.json').read_text())
gate=Path(guarded['strict_chat_gate'])
assert not (gate/'failure.json').exists(), 'strict chat gate failed'
assert hashlib.sha256((gate/'plan.json').read_bytes()).hexdigest()==guarded['strict_chat_plan_sha256']
assert json.loads((gate/'complete.json').read_text())=={'exit':0,'calls':9}
results=json.loads((gate/'results.json').read_text())
assert results['plan_sha256']==guarded['strict_chat_plan_sha256']
assert len(results['completed'])==9
expected_tags={model+'-'+arm for model in ['Qwen3-0.6B-Q8_0.gguf','Qwen3-0.6B-Q4_0.gguf','Qwen3-0.6B-Q5_K_M.gguf'] for arm in ['base','repeat','candidate']}
assert {r['tag'] for r in results['completed']}==expected_tags
assert all(r['matches'] and r['finish']=='eos' and r['prompt_tokens']==19820 and r['reused_tokens']==0 for r in results['completed'])
readiness=json.loads(Path(guarded['readiness_path']).read_text())
assert readiness['guarded_freeze_sha256']==hashlib.sha256(guarded_bytes).hexdigest()
assert readiness['stage']==STAGE and readiness['reviewed'] and readiness['own_competing_work_stopped']
assert isinstance(readiness['checked_utc'],str) and readiness['checked_utc']

original_freeze=json.loads((ART/'freeze.json').read_text())
for name,digest in original_freeze['sha256'].items():
    assert hashlib.sha256((ART/name).read_bytes()).hexdigest()==digest,name
all_plans=[json.loads((ART/name).read_text()) for name in ['plan.json','cli-plan.json']]
inputs={}
for frozen_plan in all_plans:
    for item in list(frozen_plan['arms'].values())+list(frozen_plan['models'].values()):
        path=item.get('exe',item.get('path'))
        assert path not in inputs or inputs[path]==item['sha256'],path
        inputs[path]=item['sha256']
    for name,digest in frozen_plan.get('reference_dlls',{}).items():
        inputs[str(ART/name)]=digest
for path,digest in inputs.items():
    with Path(path).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,path
out=ART/'matrix'
out.mkdir(exist_ok=False)
rows=[]
cells=[]
for round_id,order in enumerate(plan['orders']):
    models=list(plan['models'])
    models=models[round_id%len(models):]+models[:round_id%len(models)]
    for model in models:
        for arm in order:cells.append((round_id,model,arm))
assert len(cells)==128

def describe_error(exc):
    return type(exc).__name__+': '+str(exc)

def monitor_activity(record,monfile):
    own={os.getpid(),record.get('monitor_pid'),record.get('pid')}
    activity={'samples':0,'unknown_process_samples':0,'process_enumeration_errors':0,'missing_counter_samples':0,'missing_counter_counts':{},'unrelated_cpu_peak':None,'disk_bytes_peak':None,'gpu_engine_peak':None,'system_cpu_peak':None,'flag_cpu':False,'flag_disk':False,'flag_gpu':False,'process_peaks':{}}
    quality={'file_exists':monfile.exists(),'metadata_records':0,'end_records':0,'parse_errors':0,'schema_errors':0,'collect_errors':0,'coverage_known':False,'covers_workload':None,'limitations':[]}
    samples=[];metadata=[];ends=[]
    if monfile.exists():
        for line in monfile.read_text().splitlines():
            try:
                item=json.loads(line)
                if not isinstance(item,dict):raise ValueError('record is not an object')
            except (ValueError,TypeError):
                quality['parse_errors']+=1;continue
            kind=item.get('kind')
            if kind=='metadata':metadata.append(item)
            elif kind=='end':ends.append(item)
            elif kind=='sample':samples.append(item)
            else:quality['schema_errors']+=1
    quality['metadata_records']=len(metadata);quality['end_records']=len(ends)
    if len(metadata)==1:
        quality['unavailable_counters']=metadata[0].get('unavailable',{})
        if metadata[0].get('pid')!=record.get('monitor_pid'):quality['limitations'].append('metadata PID mismatch')
    brackets=[];times=[]
    for sample in samples:
        activity['samples']+=1
        if sample.get('collect_status')!=0:quality['collect_errors']+=1
        try:
            start=sample['counter_interval_start_bracket'];end=sample['counter_interval_end_bracket']
            if not (len(start)==len(end)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in [*start,*end]) and start[0]<=start[1]<=end[0]<=end[1]):raise ValueError('invalid collection interval')
            brackets.append((start,end));times.append(float(sample['monotonic']))
        except (KeyError,TypeError,ValueError):quality['schema_errors']+=1
        processes=sample.get('processes',{})
        if not isinstance(processes,dict) or processes.get('error') is not None or not isinstance(processes.get('items'),list):
            activity['process_enumeration_errors']+=1;activity['unknown_process_samples']+=1;items=[]
        else:items=processes['items']
        unrelated=0.0;known=0;unknown=0
        for process in items:
            if not isinstance(process,dict):quality['schema_errors']+=1;unknown+=1;continue
            if process.get('pid') in own:continue
            cpu=process.get('cpu_percent')
            if not isinstance(cpu,(int,float)) or not math.isfinite(cpu):unknown+=1;continue
            known+=1;unrelated+=cpu
            if cpu>1:
                key=str(process.get('pid'))+':'+str(process.get('name'))
                activity['process_peaks'][key]=max(activity['process_peaks'].get(key,0),cpu)
        if unknown:activity['unknown_process_samples']+=1
        if known:
            activity['unrelated_cpu_peak']=max(activity['unrelated_cpu_peak'] or 0,unrelated)
            activity['flag_cpu']|=unrelated>50
        counters=sample.get('counters',{})
        if not isinstance(counters,dict):quality['schema_errors']+=1;counters={}
        for counter,key,threshold,flag in [('disk_bytes_per_second','disk_bytes_peak',104857600,'flag_disk'),('gpu_engine_percent','gpu_engine_peak',20,'flag_gpu'),('cpu_percent','system_cpu_peak',None,None)]:
            data=counters.get(counter,{})
            entries=data.get('items',[]) if isinstance(data,dict) else []
            if not isinstance(entries,list):entries=[];quality['schema_errors']+=1
            values=[v['value'] for v in entries if isinstance(v,dict) and isinstance(v.get('value'),(int,float)) and math.isfinite(v['value'])]
            if not values:
                activity['missing_counter_samples']+=1
                activity['missing_counter_counts'][counter]=activity['missing_counter_counts'].get(counter,0)+1
                continue
            peak=max(values);activity[key]=max(activity[key] or 0,peak)
            if flag:activity[flag]|=peak>threshold
    if brackets:
        quality['first_counter_start_bracket']=brackets[0][0];quality['last_counter_end_bracket']=brackets[-1][1]
        quality['max_sample_gap_seconds']=max((b-a for a,b in zip(times,times[1:])),default=None)
        begin=record.get('monotonic_start');finish=record.get('monotonic_end')
        if begin is not None and finish is not None:
            quality['coverage_known']=True
            quality['covers_workload']=brackets[0][0][1]<=begin and brackets[-1][1][0]>=finish
            quality['overlapping_samples']=sum(end[1]>=begin and start[0]<=finish for start,end in brackets)
        if any(b<a for a,b in zip(times,times[1:])):quality['limitations'].append('nonmonotonic sample times')
        if quality['max_sample_gap_seconds'] is not None and quality['max_sample_gap_seconds']>2*plan['monitor']['interval_seconds']:
            quality['limitations'].append('sample gap exceeds two planned intervals')
    if not quality['file_exists']:quality['limitations'].append('monitor file missing')
    if len(metadata)!=1 or len(ends)!=1:quality['limitations'].append('metadata/end records incomplete')
    if record.get('monitor_exit')!=0:quality['limitations'].append('monitor did not exit successfully')
    if quality['parse_errors'] or quality['schema_errors'] or quality['collect_errors']:quality['limitations'].append('monitor parse/schema/collection errors')
    if not quality['coverage_known'] or not quality['covers_workload']:quality['limitations'].append('full workload coverage not established')
    if activity['unknown_process_samples']:quality['limitations'].append('some process CPU deltas unknown')
    if activity['missing_counter_samples']:quality['limitations'].append('some system counters missing')
    quality['interpretation']='Flags describe observed activity only; false does not establish quiet when coverage or values are unknown.'
    return activity,quality

for round_id,model,arm in cells:
    tag=f"r{round_id}-{model[:-5]}-{arm}"
    monfile=out/(tag+'.monitor.jsonl');stop=out/(tag+'.stop')
    command=[plan['arms'][arm]['exe'],plan['models'][model]['path'],plan['harness']['tokens'],'--threads','6']
    record={'round':round_id,'model':model,'arm':arm,'command':command,'runner_pid':os.getpid(),'utc_start':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stdout_file':str(out/(tag+'.stdout')),'stderr_file':str(out/(tag+'.stderr')),'monitor_file':str(monfile),'monitor_log':str(out/(tag+'.monitor.log'))}
    monitor=None;process=None;terminal_error=None
    try:
        with (out/(tag+'.monitor.log')).open('wb') as monlog:
            monitor=subprocess.Popen([sys.executable,plan['monitor']['script'],'--output',str(monfile),'--seconds','620','--interval','1','--stop-file',str(stop)],stdout=monlog,stderr=subprocess.STDOUT)
            record['monitor_pid']=monitor.pid
            time.sleep(2)
            try:
                with (out/(tag+'.stdout')).open('wb') as stdout,(out/(tag+'.stderr')).open('wb') as stderr:
                    process=subprocess.Popen(command,stdout=stdout,stderr=stderr)
                    record['pid']=process.pid;record['monotonic_start']=time.perf_counter()
                    try:record['exit']=process.wait(timeout=600)
                    except subprocess.TimeoutExpired:
                        process.kill();record['exit']=process.wait();record['timeout']=True
                    record['monotonic_end']=time.perf_counter()
                time.sleep(1)
            finally:
                if process is not None and process.poll() is None:
                    process.kill();record['exit']=process.wait(timeout=10);record['forced_cleanup']=True
                    record['monotonic_end']=time.perf_counter()
                stop.write_text('complete\n')
                try:record['monitor_exit']=monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.terminate();record['monitor_exit']=monitor.wait(timeout=10);record['monitor_timeout']=True
        output=(out/(tag+'.stdout')).read_text(errors='replace')
        samples=[json.loads(line) for line in output.splitlines() if line]
        assert record['exit']==0 and len(samples)==2 and [s['run'] for s in samples]==[0,1]
        assert all(s['pp_tokens']==215 and s['tg_tokens']==32 and s['threads']==6 and all(math.isfinite(s[k]) and s[k]>0 for k in ['pp_ms','tg_ms']) for s in samples)
        record['samples']=samples;record['measured']=samples[1]
    except BaseException as exc:
        terminal_error=exc;record['protocol_error']=describe_error(exc)
    finally:
        try:
            if 'monitor_exit' not in record:
                stop.write_text('complete\n')
            if monitor is not None and 'monitor_exit' not in record:
                try:record['monitor_exit']=monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.terminate();record['monitor_exit']=monitor.wait(timeout=10);record['monitor_timeout']=True
        except BaseException as exc:
            record['monitor_cleanup_error']=describe_error(exc)
            if terminal_error is None:terminal_error=exc;record['protocol_error']=describe_error(exc)
        record['utc_end']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:record['activity'],record['monitor_quality']=monitor_activity(record,monfile)
        except BaseException as exc:
            record['monitor_analysis_error']=describe_error(exc)
            record['monitor_quality']={'coverage_known':False,'covers_workload':None,'limitations':['monitor analysis failed; activity unknown']}
        rows.append(record)
        temporary=out/'results.json.tmp'
        temporary.write_text(json.dumps(rows,indent=2)+'\n')
        temporary.replace(out/'results.json')
    if terminal_error is not None:
        print('PROTOCOL FAILURE',tag,record['protocol_error'],flush=True)
        raise SystemExit(1)
    m=record['measured'];activity=record.get('activity',{})
    print(f"{tag}: pp {215000/m['pp_ms']:.2f} tok/s, tg {32000/m['tg_ms']:.2f} tok/s; observed activity CPU={activity.get('flag_cpu')} disk={activity.get('flag_disk')} GPU={activity.get('flag_gpu')}; monitor limitations={record['monitor_quality']['limitations']}",flush=True)
(out/'complete.json').write_text(json.dumps({'exit':0,'rows':len(rows),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
print('COMPLETE',len(rows),'planned rows',flush=True)
