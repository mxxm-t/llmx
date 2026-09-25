import csv,json,math,pathlib,statistics,collections
p=pathlib.Path.home()/'AppData/Local/Temp/llmx-activation-perf-20260924'
r=json.loads((p/'matrix/results.json').read_text()); s=json.loads((p/'summary.json').read_text()); plan=json.loads((p/'plan.json').read_text()); cp=json.loads((p/'cli-plan.json').read_text()); blocks=list(csv.DictReader((p/'blocks.csv').open(newline='')))
index={(x['model'],x['round'],x['arm']):x for x in r};checks=0
errors=[]
def equal(got,want,label):
 global checks
 checks+=1
 if isinstance(want,float):good=math.isclose(got,want,rel_tol=1e-12,abs_tol=1e-10)
 else:good=got==want
 if not good:errors.append([label,got,want])
equal(len(r),128,'matrix count');equal(len(index),128,'unique keys');equal(len(blocks),32,'CSV block count')
for c in s['cells']:
 model,phase=c['model'],c['phase']; n=215 if phase=='pp' else 32
 rates={arm:[n*1000/index[model,i,arm]['measured'][phase+'_ms'] for i in range(8)] for arm in plan['arms']}
 for arm,values in rates.items():
  for j,v in enumerate(values):equal(c['arms'][arm]['all_tok_s'][j],v,f'{model}/{phase}/{arm}/sample{j}')
  for key,v in [('median_tok_s',statistics.median(values)),('mean_tok_s',statistics.mean(values)),('min_tok_s',min(values)),('max_tok_s',max(values))]:equal(c['arms'][arm][key],v,f'{model}/{phase}/{arm}/{key}')
 for pair,reported in c['paired'].items():
  a,b=pair.split('/'); ratios=[index[model,i,b]['measured'][phase+'_ms']/index[model,i,a]['measured'][phase+'_ms'] for i in range(8)]
  for j,v in enumerate(ratios):equal(reported['ratios'][j],v,f'{model}/{phase}/{pair}/{j}')
  equal(reported['geomean_percent'],100*(math.prod(ratios)**(1/8)-1),f'{model}/{phase}/{pair}/gm')
  equal(reported['min_percent'],100*(min(ratios)-1),f'{model}/{phase}/{pair}/min');equal(reported['max_percent'],100*(max(ratios)-1),f'{model}/{phase}/{pair}/max')
for b in blocks:
 model,i=b['model'],int(b['round'])
 for arm in plan['arms']:
  raw=index[model,i,arm]
  for metric in ['flag_cpu','flag_disk','flag_gpu','unrelated_cpu_peak','unknown_process_samples','missing_counter_samples']:equal(b[arm+'_'+metric],str(raw['activity'][metric]),f'CSV {model}/{i}/{arm}/{metric}')
 for phase in ['pp','tg']:
  for arm in ['candidate','layout','mx']:equal(float(b[arm+'_vs_base_'+phase+'_percent']),100*(index[model,i,'base']['measured'][phase+'_ms']/index[model,i,arm]['measured'][phase+'_ms']-1),f'CSV {model}/{i}/{phase}/{arm}')
for m in s['monitor']:
 raw=index[m['model'],m['round'],m['arm']]
 equal(m['activity'],raw['activity'],f'monitor {m["model"]}/{m["round"]}/{m["arm"]}')
 equal(m['kinds']['sample'],raw['activity']['samples'],'monitor sample count')
markdown=(p/'summary.md').read_text().splitlines()
for c,line in zip(s['cells'],[l for l in markdown if l.startswith('| Qwen')]):
 columns=[x.strip() for x in line.strip('|').split('|')]
 for j,arm in enumerate(['base','candidate','layout','mx']):equal(columns[2+j],f'{c["arms"][arm]["median_tok_s"]:.2f}','markdown rate')
 for j,pair in enumerate(['candidate/base','layout/base']):equal(columns[6+j],f'{c["paired"][pair]["geomean_percent"]:+.2f}%','markdown delta')
print(json.dumps({'checks':checks,'errors':errors,'cli_positions':{a:[sum(o[pos]==a for o in cp['orders']) for pos in range(3)] for a in cp['arms']},'cli_rows':sum(len(o)*len(cp['models']) for o in cp['orders']),'monitor_error_totals':{k:sum(m[k] for m in s['monitor']) for k in ['parse_errors','process_enumeration_errors','counter_collection_errors']},'counter_missing_observations':sum(x['activity']['missing_counter_samples'] for x in r),'cells':[{'model':c['model'],'phase':c['phase'],'candidate_mx_paired_percent':c['paired']['candidate/mx']['geomean_percent'],'candidate_base_block_range':[c['paired']['candidate/base']['min_percent'],c['paired']['candidate/base']['max_percent']],'layout_base_block_range':[c['paired']['layout/base']['min_percent'],c['paired']['layout/base']['max_percent']]} for c in s['cells']]},indent=2))
