import datetime,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
ART=Path(__file__).parent
plan=json.loads((ART/'plan.json').read_text())
out=ART/'matrix'
out.mkdir(exist_ok=False)
rows=[]
for round_id,order in enumerate(plan['orders']):
    models=list(plan['models'])
    models=models[round_id%len(models):]+models[:round_id%len(models)]
    for model in models:
        for arm in order:
            tag=f"r{round_id}-{model[:-5]}-{arm}"
            monfile=out/(tag+'.monitor.jsonl');stop=out/(tag+'.stop')
            command=[plan['arms'][arm]['exe'],plan['models'][model]['path'],plan['harness']['tokens'],'--threads','6']
            record={'round':round_id,'model':model,'arm':arm,'command':command,'runner_pid':os.getpid(),'utc_start':datetime.datetime.now(datetime.timezone.utc).isoformat()}
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
                    stop.write_text('complete\n')
                    try:record['monitor_exit']=monitor.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        monitor.terminate();record['monitor_exit']=monitor.wait(timeout=10);record['monitor_timeout']=True
            record['utc_end']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            output=(out/(tag+'.stdout')).read_text(errors='replace')
            try:
                samples=[json.loads(line) for line in output.splitlines() if line]
                assert record['exit']==0 and len(samples)==2 and [s['run'] for s in samples]==[0,1]
                assert all(s['pp_tokens']==215 and s['tg_tokens']==32 and s['threads']==6 and all(math.isfinite(s[k]) and s[k]>0 for k in ['pp_ms','tg_ms']) for s in samples)
                record['samples']=samples;record['measured']=samples[1]
            except Exception as exc:record['protocol_error']=str(exc) or type(exc).__name__
            own={os.getpid(),record['monitor_pid'],record.get('pid')}
            activity={'samples':0,'unknown_process_samples':0,'missing_counter_samples':0,'unrelated_cpu_peak':None,'disk_bytes_peak':None,'gpu_engine_peak':None,'system_cpu_peak':None,'flag_cpu':False,'flag_disk':False,'flag_gpu':False,'process_peaks':{}}
            if monfile.exists():
                for line in monfile.read_text().splitlines():
                    try:sample=json.loads(line)
                    except ValueError:record['monitor_parse_error']=True;continue
                    if sample.get('kind')!='sample':continue
                    activity['samples']+=1
                    unrelated=0.0;known=0;unknown=0
                    for p in sample.get('processes',{}).get('items',[]):
                        if p.get('pid') in own:continue
                        cpu=p.get('cpu_percent')
                        if cpu is None:unknown+=1;continue
                        known+=1;unrelated+=cpu
                        if cpu>1:
                            key=str(p['pid'])+':'+str(p.get('name'))
                            activity['process_peaks'][key]=max(activity['process_peaks'].get(key,0),cpu)
                    if unknown:activity['unknown_process_samples']+=1
                    if known:
                        activity['unrelated_cpu_peak']=max(activity['unrelated_cpu_peak'] or 0,unrelated)
                        activity['flag_cpu']|=unrelated>50
                    counters=sample.get('counters',{})
                    for counter,key,threshold,flag in [('disk_bytes_per_second','disk_bytes_peak',104857600,'flag_disk'),('gpu_engine_percent','gpu_engine_peak',20,'flag_gpu'),('cpu_percent','system_cpu_peak',None,None)]:
                        values=[v['value'] for v in counters.get(counter,{}).get('items',[]) if v.get('value') is not None]
                        if not values:activity['missing_counter_samples']+=1;continue
                        peak=max(values);activity[key]=max(activity[key] or 0,peak)
                        if flag:activity[flag]|=peak>threshold
            record['activity']=activity
            record['monitor_file']=str(monfile)
            rows.append(record)
            (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            if 'protocol_error' in record:
                print('PROTOCOL FAILURE',tag,record['protocol_error'],flush=True)
                raise SystemExit(1)
            m=record['measured']
            print(f"{tag}: pp {215000/m['pp_ms']:.2f} tok/s, tg {32000/m['tg_ms']:.2f} tok/s; activity CPU={activity['flag_cpu']} disk={activity['flag_disk']} GPU={activity['flag_gpu']}",flush=True)
(out/'complete.json').write_text(json.dumps({'exit':0,'rows':len(rows),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
print('COMPLETE',len(rows),'planned rows',flush=True)
