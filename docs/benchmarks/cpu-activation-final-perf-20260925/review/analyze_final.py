"""Audit frozen activation performance results; no workload execution or model reads."""
import argparse,csv,datetime,hashlib,json,math,re,statistics
from pathlib import Path

class AuditError(ValueError): pass

def require(ok,message):
    if not ok:raise AuditError(message)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def schedule(plan):
    cells=[]
    for number,order in enumerate(plan['orders']):
        models=list(plan['models']);models=models[number%len(models):]+models[:number%len(models)]
        cells.extend((number,model,arm) for model in models for arm in order)
    return cells

def tag(cell):return f'r{cell[0]}-{cell[1][:-5]}-{cell[2]}'

def command(plan,cell,mode):
    _,model,arm=cell
    if mode=='common':return [plan['arms'][arm]['exe'],plan['models'][model]['path'],plan['harness']['tokens'],'--threads','6']
    return [plan['arms'][arm]['exe'],'bench','--model',plan['models'][model]['path'],'--threads','6','--device','cpu','--cache-type-k','f16','--cache-type-v','f16','--p','215','--n','32','--r','1']

def number(value):return type(value) in (int,float) and math.isfinite(value)

def monitoring(row,path,interval):
    info={'path':str(path),'sha256':None,'metadata':0,'end':0,'samples':0,'parse_errors':0,'covers_invocation':False,'structurally_complete':False,'unknown_reasons':[],'observed_flags':{'cpu':False,'disk':False,'gpu':False}}
    records=[]
    if path.exists():
        info['sha256']=digest(path)
        for line in path.read_text().splitlines():
            try:
                item=json.loads(line)
                if not isinstance(item,dict):raise ValueError('nonobject')
                records.append(item)
            except (ValueError,TypeError):info['parse_errors']+=1
    else:info['unknown_reasons'].append('monitor file missing')
    meta=[r for r in records if r.get('kind')=='metadata'];ends=[r for r in records if r.get('kind')=='end'];samples=[r for r in records if r.get('kind')=='sample']
    info.update(metadata=len(meta),end=len(ends),samples=len(samples))
    info['structurally_complete']=len(meta)==len(ends)==1 and bool(samples) and info['parse_errors']==0 and row.get('monitor_exit')==0
    if not info['structurally_complete']:info['unknown_reasons'].append('monitor structure/exit incomplete')
    if any(r.get('kind') not in ('metadata','sample','end') for r in records):info['unknown_reasons'].append('unknown monitor record')
    if meta and meta[0].get('pid')!=row.get('monitor_pid'):info['unknown_reasons'].append('monitor PID mismatch')
    own={row.get('runner_pid'),row.get('monitor_pid'),row.get('pid')};bounds=[];times=[];unknown_process=0;missing_counter=0
    for sample in samples:
        if number(sample.get('monotonic')):times.append(sample['monotonic'])
        else:info['unknown_reasons'].append('sample timestamp missing')
        try:
            a=sample['counter_interval_start_bracket'];b=sample['counter_interval_end_bracket']
            require(len(a)==len(b)==2 and all(number(v) for v in [*a,*b]) and a[0]<=a[1]<=b[0]<=b[1],'invalid collection bracket')
            bounds.append((a,b))
        except (KeyError,TypeError,AuditError):info['unknown_reasons'].append('invalid collection bracket')
        if sample.get('collect_status')!=0:info['unknown_reasons'].append('counter collection failed')
        processes=sample.get('processes',{});items=processes.get('items') if isinstance(processes,dict) else None
        if not isinstance(items,list) or not items or processes.get('error') is not None:unknown_process+=1;items=[]
        cpu=0.0
        for process in items:
            if not isinstance(process,dict):unknown_process+=1;continue
            if process.get('pid') in own:continue
            if not number(process.get('cpu_percent')):unknown_process+=1;continue
            cpu+=process['cpu_percent']
        info['observed_flags']['cpu']|=cpu>50
        counters=sample.get('counters',{})
        if not isinstance(counters,dict):counters={}
        for key,flag,threshold in [('disk_bytes_per_second','disk',104857600),('gpu_engine_percent','gpu',20),('cpu_percent',None,None)]:
            data=counters.get(key,{});entries=data.get('items',[]) if isinstance(data,dict) else []
            if not isinstance(entries,list):entries=[]
            values=[v['value'] for v in entries if isinstance(v,dict) and v.get('status') in (0,1) and number(v.get('value'))]
            if not values or len(values)!=len(entries) or data.get('error') is not None:missing_counter+=1
            if values and flag:info['observed_flags'][flag]|=max(values)>threshold
    if bounds:
        info['covers_invocation']=bounds[0][0][1]<=row['monotonic_start'] and bounds[-1][1][0]>=row['monotonic_end']
        if any(later[0][0]<earlier[0][0] for earlier,later in zip(bounds,bounds[1:])):info['unknown_reasons'].append('monitor intervals out of order')
        if any(later[0][0]>earlier[1][1] for earlier,later in zip(bounds,bounds[1:])):info['unknown_reasons'].append('monitor interval discontinuity')
    if any(b<a for a,b in zip(times,times[1:])):info['unknown_reasons'].append('sample timestamps out of order')
    if any(b-a>2*interval for a,b in zip(times,times[1:])):info['unknown_reasons'].append('monitor sampling gap')
    if row.get('monitor_timeout'):info['unknown_reasons'].append('monitor timed out')
    if not info['covers_invocation']:info['unknown_reasons'].append('full invocation coverage unestablished')
    if unknown_process:info['unknown_reasons'].append('process CPU deltas unknown')
    if missing_counter:info['unknown_reasons'].append('system counters missing')
    if row.get('monitor_quality',{}).get('limitations'):info['unknown_reasons'].extend(row['monitor_quality']['limitations'])
    if 'monitor_analysis_error' in row:info['unknown_reasons'].append('driver monitor analysis failed')
    info['unknown_reasons']=sorted(set(info['unknown_reasons']));info['unknown']=bool(info['unknown_reasons'])
    info['flagged']=any(info['observed_flags'].values())
    info['driver_activity']=row.get('activity');info['driver_quality']=row.get('monitor_quality')
    return info

def validate(plan,rows,matrix,mode,final,terminal):
    require(isinstance(rows,list) and all(isinstance(r,dict) for r in rows),'results must be an array of objects')
    expected=schedule(plan);actual=[(r.get('round'),r.get('model'),r.get('arm')) for r in rows]
    require(actual==expected[:len(rows)] and len(rows)<=len(expected),'rows are not the exact planned prefix/order')
    if final:
        require(len(rows)==len(expected),'final audit requires every planned row')
        require(isinstance(terminal,dict) and terminal.get('exit')==0 and terminal.get('rows')==len(expected),'terminal success/count missing')
    evidence=[]
    for row,cell in zip(rows,expected):
        name=tag(cell)
        require(row.get('exit')==0 and not any(row.get(k) for k in ['protocol_error','timeout','forced_cleanup','monitor_cleanup_error']),name+': unsuccessful terminal row')
        require('utc_end' in row and number(row.get('monotonic_start')) and number(row.get('monotonic_end')) and row['monotonic_end']>=row['monotonic_start'],name+': terminal timing missing')
        require(row.get('command')==command(plan,cell,mode),name+': command mismatch')
        stdout=matrix/(name+'.stdout');text=stdout.read_text(errors='replace')
        if mode=='common':
            try:samples=[json.loads(s) for s in text.splitlines() if s]
            except ValueError as error:raise AuditError(name+': malformed stdout') from error
            require(len(samples)==2 and all(isinstance(s,dict) for s in samples) and [s.get('run') for s in samples]==[0,1],name+': warmup/measured IDs wrong')
            for s in samples:
                require(s.get('pp_tokens')==215 and s.get('tg_tokens')==32 and s.get('threads')==6 and all(number(s.get(k)) and s[k]>0 for k in ['pp_ms','tg_ms']),name+': invalid raw sample')
            require(row.get('samples')==samples and row.get('measured')==samples[1],name+': raw stdout/record mismatch')
        else:
            matches=re.findall(r'bench: (pp215|tg32)\s+([0-9.]+) \+- ([0-9.]+) tok/s\s+\(1 runs\)',text)
            require(len(matches)==2 and {m[0] for m in matches}=={'pp215','tg32'},name+': CLI phase/count mismatch')
            rates={m[0]:float(m[1]) for m in matches}
            require(all(number(v) and v>0 for v in rates.values()),name+': invalid CLI rates')
            measured={'pp_tokens':215,'tg_tokens':32,'threads':6,'pp_ms':215000/rates['pp215'],'tg_ms':32000/rates['tg32']}
            require(row.get('reported_rates')==rates and row.get('measured')==measured,name+': CLI raw stdout/record mismatch')
        monitor=monitoring(row,matrix/(name+'.monitor.jsonl'),plan['monitor']['interval_seconds'])
        evidence.append({'cell':list(cell),'stdout_sha256':digest(stdout),'monitor':monitor})
    return evidence

def pair_stats(items):
    if not items:return None
    ratios=[x['ratio'] for x in items]
    return {'n':len(ratios),'geomean_percent':100*math.expm1(statistics.mean(math.log(x) for x in ratios)),'min_percent':100*(min(ratios)-1),'max_percent':100*(max(ratios)-1),'faster':sum(x>1 for x in ratios),'slower':sum(x<1 for x in ratios),'equal':sum(x==1 for x in ratios),'blocks':items}

def summarize(plan,rows,evidence,mode,final):
    lookup={(r['round'],r['model'],r['arm']):r for r in rows};mon={tuple(e['cell']):e['monitor'] for e in evidence};cells=[]
    for model in plan['models']:
        for phase,tokens in [('pp',215),('tg',32)]:
            c={'model':model,'phase':phase,'tokens':tokens,'rate_unit':'tokens/second','arms':{},'paired':{}}
            for arm in plan['arms']:
                values=[{'round':n,'tok_s':tokens*1000/lookup[(n,model,arm)]['measured'][phase+'_ms'],'activity_flagged':mon[(n,model,arm)]['flagged'],'activity_unknown':mon[(n,model,arm)]['unknown']} for n in range(len(plan['orders'])) if (n,model,arm) in lookup]
                c['arms'][arm]={'runs':values,'n':len(values),'median_tok_s':statistics.median(v['tok_s'] for v in values) if values else None}
            pairs=[('candidate','base'),('layout','base')]+([('candidate','mx')] if mode=='common' else [])
            for numerator,denominator in pairs:
                a={v['round']:v for v in c['arms'][numerator]['runs']};b={v['round']:v for v in c['arms'][denominator]['runs']};items=[]
                for n in sorted(a.keys()&b.keys()):items.append({'round':n,'ratio':a[n]['tok_s']/b[n]['tok_s'],'percent':100*(a[n]['tok_s']/b[n]['tok_s']-1),'activity_flagged':a[n]['activity_flagged'] or b[n]['activity_flagged'],'activity_unknown':a[n]['activity_unknown'] or b[n]['activity_unknown']})
                c['paired'][numerator+'/'+denominator]={'all_blocks':pair_stats(items),'activity_flagged_blocks':pair_stats([v for v in items if v['activity_flagged']]),'activity_unknown_blocks':pair_stats([v for v in items if v['activity_unknown']])}
            cells.append(c)
    return {'status':'complete' if final else 'partial','final_result_audit':final,'mode':mode,'planned_rows':len(schedule(plan)),'observed_rows':len(rows),'cells':cells,'evidence':evidence,'retained_rows':rows,'adoption_verdict':'None. Report gains and regressions together; no automatic cutoff, parity, noise or merge verdict.','limits':['Every observed planned row is retained; flagged and unknown subsets overlap and do not replace complete results.','Common run0 is retained as prospective warmup; only run1 contributes measured rates. CLI warmup is internal and unreported.','Activity spans entire invocations, including loading and warmup; system disk/GPU flags do not establish unrelated contention. Missing/inaccessible processes and telemetry remain unknown.','CLI rates are rounded to 0.01 tokens/second; reconstructed milliseconds are approximate.','A single layout control does not define all possible layout variation. Same-model witnesses and independent HF/depth gates are separate.',plan['scope']]}

def markdown(report):
    arms=['base','candidate','layout']+(['mx'] if report['mode']=='common' else [])
    pairs=['candidate/base','layout/base']+(['candidate/mx'] if report['mode']=='common' else [])
    lines=[report['status'].upper()+': '+str(report['observed_rows'])+'/'+str(report['planned_rows'])+' rows. No adoption verdict.','', '| Model | Phase | '+' | '.join(a+' tok/s' for a in arms)+' | '+' | '.join(p+' paired %' for p in pairs)+' |','|---|---|'+('---:|'*(len(arms)+len(pairs)))]
    for c in report['cells']:
        values=[c['arms'][a]['median_tok_s'] for a in arms];deltas=[c['paired'][p]['all_blocks'] for p in pairs]
        lines.append('| '+c['model']+' | '+c['phase']+str(c['tokens'])+' | '+' | '.join('pending' if v is None else f'{v:.2f}' for v in values)+' | '+' | '.join('pending' if v is None else f"{v['geomean_percent']:+.3f}" for v in deltas)+' |')
    lines+=['','Rates are medians. Paired percentages are geometric means of same-round ratios; positive means higher throughput.','', '| Model | Phase | Pair | All blocks | Flagged blocks | Unknown-monitor blocks |','|---|---|---|---:|---:|---:|']
    for c in report['cells']:
        for p in pairs:
            entries=[c['paired'][p][k] for k in ['all_blocks','activity_flagged_blocks','activity_unknown_blocks']]
            lines.append('| '+c['model']+' | '+c['phase']+' | '+p+' | '+' | '.join('none' if v is None else f"{v['geomean_percent']:+.3f}% (n={v['n']})" for v in entries)+' |')
    lines+=['',*report['limits']]
    return '\n'.join(lines)+'\n'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cli',action='store_true');parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    art=Path(__file__).resolve().parent.parent;mode='cli' if args.cli else 'common';name='cli-plan.json' if args.cli else 'plan.json';matrix=art/('cli-matrix' if args.cli else 'matrix')
    report={'status':'failed','final_result_audit':False,'mode':mode};rows=None
    try:
        freeze=json.loads((art/'freeze.json').read_text());require(digest(art/name)==freeze['sha256'][name],'frozen plan hash mismatch')
        plan=json.loads((art/name).read_text());require(len(schedule(plan))==(72 if args.cli else 128),'frozen call count mismatch')
        rows=json.loads((matrix/'results.json').read_text());terminal=json.loads((matrix/'complete.json').read_text()) if (matrix/'complete.json').exists() else None
        evidence=validate(plan,rows,matrix,mode,not args.partial,terminal);report=summarize(plan,rows,evidence,mode,not args.partial)
        report['plan_sha256']=digest(art/name);report['results_sha256']=digest(matrix/'results.json');report['terminal']=terminal
    except (OSError,ValueError,TypeError,KeyError,OverflowError,AttributeError) as error:
        report['error']=type(error).__name__+': '+str(error)
        report['retained_rows']=rows
        if (matrix/'results.json').exists():report['results_sha256']=digest(matrix/'results.json')
    report['analyzer_sha256']=digest(Path(__file__));report['audit_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');output=Path(__file__).parent/(mode+'-analysis-'+stamp);output.mkdir()
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='ascii')
    if report['status']!='failed':
        (output/'summary.md').write_text(markdown(report),encoding='ascii')
        with (output/'blocks.csv').open('w',newline='',encoding='ascii') as f:
            writer=csv.writer(f);writer.writerow(['model','phase','pair','round','ratio','percent','activity_flagged','activity_unknown'])
            for c in report['cells']:
                for pair,groups in c['paired'].items():
                    for block in (groups['all_blocks'] or {}).get('blocks',[]):writer.writerow([c['model'],c['phase'],pair,*[block[k] for k in ['round','ratio','percent','activity_flagged','activity_unknown']]])
    print(json.dumps({'status':report['status'],'output':str(output),'error':report.get('error')}))
    return int(report['status']=='failed')

if __name__=='__main__':raise SystemExit(main())
