"""Offline generated fixtures for analyze_final.py; no workloads or model reads."""
import ast,copy,datetime,hashlib,importlib.util,json,math
from pathlib import Path

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('analysis',here/'analyze_final.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
root=here/('offline-fixtures-'+stamp);root.mkdir()
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def reject(name,callback):
    try:callback()
    except (a.AuditError,OSError):checks.append(name);return
    raise AssertionError(name+' accepted invalid evidence')
def plan(mode):
    arms=['base','candidate','layout']+(['mx'] if mode=='common' else [])
    return {'orders':[arms,arms[::-1]],'arms':{x:{'exe':'fake/'+x+'.exe'} for x in arms},'models':{'fixture.gguf':{'path':'fake/model.gguf'}},'harness':{'tokens':'fake/ids.txt'},'monitor':{'interval_seconds':1},'scope':'Synthetic offline evidence only.'}
def telemetry(flag=False):
    records=[{'kind':'metadata','pid':20}]
    for n in range(1,6):
        records.append({'kind':'sample','monotonic':n,'counter_interval_start_bracket':[n-1,n-0.99],'counter_interval_end_bracket':[n,n+0.01],'collect_status':0,'processes':{'items':[{'pid':40,'cpu_percent':60 if flag else 0},{'pid':30,'cpu_percent':600}]},'counters':{k:{'items':[{'name':'total','status':0,'value':0}]} for k in ['cpu_percent','disk_bytes_per_second','gpu_engine_percent']}})
    records.append({'kind':'end'})
    return records
def fixture(mode):
    p=plan(mode);directory=root/mode;directory.mkdir();rows=[]
    for cell in a.schedule(p):
        n,model,arm=cell;name=a.tag(cell)
        rate={'base':100,'candidate':200 if n==0 else 50,'layout':110,'mx':80}[arm]
        measured={'pp_tokens':215,'tg_tokens':32,'threads':6,'pp_ms':215000/rate,'tg_ms':32000/rate}
        row={'round':n,'model':model,'arm':arm,'command':a.command(p,cell,mode),'runner_pid':10,'monitor_pid':20,'pid':30,'exit':0,'monitor_exit':0,'utc_end':'2026-09-25T00:00:00Z','monotonic_start':1.5,'monotonic_end':4.5,'measured':measured,'monitor_quality':{'limitations':[]}}
        if mode=='common':
            measured['run']=1
            warmup=dict(measured,run=0,pp_ms=1,tg_ms=1)
            row['samples']=[warmup,measured]
            raw='\n'.join(json.dumps(s) for s in row['samples'])+'\n'
        else:
            row['reported_rates']={'pp215':float(rate),'tg32':float(rate)}
            raw=f'bench: pp215 {rate:.2f} +- 0.00 tok/s (1 runs)\nbench: tg32 {rate:.2f} +- 0.00 tok/s (1 runs)\n'
        (directory/(name+'.stdout')).write_text(raw,encoding='ascii')
        (directory/(name+'.monitor.jsonl')).write_text('\n'.join(json.dumps(v) for v in telemetry(n==0 and arm=='candidate'))+'\n',encoding='ascii')
        rows.append(row)
    return p,rows,directory,{'exit':0,'rows':len(rows)}

for mode in ['common','cli']:
    p,rows,d,t=fixture(mode)
    ev=a.validate(p,rows,d,mode,True,t);report=a.summarize(p,rows,ev,mode,True)
    check(mode+' complete terminal accepted',report['status']=='complete' and report['final_result_audit'])
    c=report['cells'][0]
    check(mode+' all measured retained',len(report['retained_rows'])==len(rows))
    check(mode+' measured median excludes warmup',c['arms']['candidate']['median_tok_s']==125)
    pair=c['paired']['candidate/base']['all_blocks']
    check(mode+' paired geometric ratios cancel',abs(pair['geomean_percent'])<1e-12 and pair['faster']==pair['slower']==1)
    check(mode+' layout gain',math.isclose(c['paired']['layout/base']['all_blocks']['geomean_percent'],10,abs_tol=1e-10))
    check(mode+' flagged complete results retained',c['paired']['candidate/base']['activity_flagged_blocks']['n']==1 and pair['n']==2)
    check(mode+' known telemetry not unknown',not any(e['monitor']['unknown'] for e in ev))
    check(mode+' benchmark CPU excluded',not ev[0]['monitor']['observed_flags']['cpu'])
    if mode=='common':
        check('candidate/mx paired gain',math.isclose(c['paired']['candidate/mx']['all_blocks']['geomean_percent'],25,abs_tol=1e-10))
        check('warmup retained verbatim',all(r['samples'][0]['pp_ms']==1 for r in report['retained_rows']))
    else:check('CLI has no mx claim','candidate/mx' not in c['paired'])
    partial=a.summarize(p,rows[:2],a.validate(p,rows[:2],d,mode,False,None),mode,False)
    check(mode+' partial never final',partial['status']=='partial' and not partial['final_result_audit'])
    check(mode+' report has units','tok/s' in a.markdown(report))
    reject(mode+' rejects missing rows as final',lambda:a.validate(p,rows[:-1],d,mode,True,t))
    reject(mode+' rejects terminal failure',lambda:a.validate(p,rows,d,mode,True,dict(t,exit=1)))
    reject(mode+' rejects missing terminal',lambda:a.validate(p,rows,d,mode,True,None))
    reject(mode+' rejects shuffled order',lambda:a.validate(p,[rows[1],rows[0],*rows[2:]],d,mode,True,t))
    reject(mode+' rejects duplicate row',lambda:a.validate(p,[rows[0],rows[0],*rows[2:]],d,mode,True,t))
    changed=copy.deepcopy(rows);changed[0]['exit']=1
    reject(mode+' rejects failed row',lambda:a.validate(p,changed,d,mode,True,t))
    changed=copy.deepcopy(rows);changed[0]['command']+=['--extra']
    reject(mode+' rejects changed command',lambda:a.validate(p,changed,d,mode,True,t))
    changed=copy.deepcopy(rows);changed[0]['measured']['pp_ms']+=1
    reject(mode+' rejects raw record disagreement',lambda:a.validate(p,changed,d,mode,True,t))
    missing=a.monitoring(rows[0],d/'nonexistent.jsonl',1)
    check(mode+' missing monitoring explicit unknown',missing['unknown'] and not missing['flagged'] and not missing['covers_invocation'])

# Alter only new synthetic fixture files, preserving real run artifacts.
p=plan('common');d=root/'common';t={'exit':0,'rows':8}
rows=[]
for cell in a.schedule(p):
    samples=[json.loads(s) for s in (d/(a.tag(cell)+'.stdout')).read_text().splitlines()]
    rows.append({'round':cell[0],'model':cell[1],'arm':cell[2],'command':a.command(p,cell,'common'),'runner_pid':10,'monitor_pid':20,'pid':30,'exit':0,'monitor_exit':0,'utc_end':'x','monotonic_start':1.5,'monotonic_end':4.5,'samples':samples,'measured':samples[1]})
first=d/(a.tag(a.schedule(p)[0])+'.stdout');original=first.read_text()
first.write_text('0\n1\n',encoding='ascii')
reject('rejects scalar common raw schema',lambda:a.validate(p,rows,d,'common',True,t))
first.write_text(original,encoding='ascii')
raw=copy.deepcopy(rows[0]['samples']);raw[0]['run']=1
first.write_text('\n'.join(json.dumps(x) for x in raw),encoding='ascii')
reject('rejects wrong warmup ID',lambda:a.validate(p,rows,d,'common',True,t))
first.write_text(original,encoding='ascii')
reject('rejects nonobject row schema',lambda:a.validate(p,[None],d,'common',False,None))
mon=d/'malformed.monitor.jsonl';mon.write_text('{bad json}\n',encoding='ascii')
check('malformed telemetry explicit unknown',a.monitoring(rows[0],mon,1)['unknown'])
records=telemetry();records[3]['monotonic']=20
mon.write_text('\n'.join(json.dumps(v) for v in records),encoding='ascii')
check('sampling gap explicit unknown','monitor sampling gap' in a.monitoring(rows[0],mon,1)['unknown_reasons'])
records=telemetry();records[2]['processes']['items'][0]['cpu_percent']=None
mon.write_text('\n'.join(json.dumps(v) for v in records),encoding='ascii')
check('missing process delta explicit unknown','process CPU deltas unknown' in a.monitoring(rows[0],mon,1)['unknown_reasons'])
for name,expected in [('plan.json',128),('cli-plan.json',72)]:
    frozen=json.loads((here.parent/name).read_text())
    check(name+' exact prospective schedule count',len(a.schedule(frozen))==expected)
for name in ['analyze_final.py','test_analyze_final.py']:
    data=(here/name).read_bytes();ast.parse(data)
    check(name+' ASCII and syntax',data.isascii())
report={'status':'passed','checks':len(checks),'names':checks,'fixture_directory':str(root),'analyzer_sha256':a.digest(here/'analyze_final.py'),'test_sha256':a.digest(Path(__file__)),'limits':['Generated small stdout/JSON fixtures only. No model payload reads, workloads, builds, monitoring or child processes.','Actual performance results were not read or analyzed. This does not prove future driver completion or adoption.']}
(root/'offline-checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='ascii')
print(json.dumps(report,indent=2))
