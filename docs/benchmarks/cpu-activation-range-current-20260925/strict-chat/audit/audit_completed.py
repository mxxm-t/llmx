"""Audit retained chat responses without running inference or reading model payloads."""
import argparse,datetime,hashlib,json,math,os
from pathlib import Path

PINNED_PLAN='7b269b13a140398e97efbab605db555dc9f74177ac1e50aee8cded143c78c170'
ROOT=Path(__file__).resolve().parent.parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
ids_sha=lambda ids:sha(json.dumps(ids,separators=(',',':')).encode('ascii'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final',action='store_true')
    parser.add_argument('--expected-completed',type=int,default=8)
    args=parser.parse_args()
    expected_count=9 if args.final else args.expected_completed
    report={'audit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':'final' if args.final else 'partial','expected_completed':expected_count,'errors':[],'models':{},'responses':[], 'file_sha256':{},'limits':['This is same-model, same-backend base/repeat/candidate self-consistency at the frozen old revisions, not independent HF numerical or semantic correctness.','Only completed response files are inspected. No inference, binaries, models or large model payloads are executed/read/hashed.','HF/Jinja provenance and saved prompt identities are checked for consistency; HF tokenization and Jinja rendering are not re-executed in this stdlib audit.','Recorded commands/timestamps support separate server invocations; wire request bodies and live process identity are not independently observed. The driver source sends the pinned plan request unchanged.','A server_exit of 1 is recorded after the driver terminates each fresh server; it is not an inference completion status.']}
    def check(ok,message):
        if not ok:raise AssertionError(message)
    def read(name):
        path=ROOT/name;raw=path.read_bytes();report['file_sha256'][name]=sha(raw)
        return json.loads(raw)
    try:
        plan=read('plan.json');check(report['file_sha256']['plan.json']==PINNED_PLAN,'plan hash changed')
        check(not (ROOT/'failure.json').exists(),'failure.json exists')
        result=read('results.json');check(result['plan_sha256']==PINNED_PLAN,'results plan hash mismatch')
        done=result['completed'];check(len(done)==expected_count,'completed count differs from expected snapshot')
        check(0<expected_count<=9,'invalid expected completed count')
        order=plan['order'];check(len(order)==9,'plan must contain nine calls')
        tags=[cell['model']+'-'+cell['tag'] for cell in order]
        check([entry['tag'] for entry in done]==tags[:expected_count],'completed tags are not the exact planned prefix')
        check(len(set(tags))==9,'plan tags duplicated')
        report['completed_tags']=[entry['tag'] for entry in done]
        request=plan['request'];check(set(request)=={'messages','temperature','seed','max_tokens'},'unexpected request option')
        check(request['temperature']==0 and request['seed']==0 and request['max_tokens']==-1,'request is not uncapped greedy')
        check(len(request['messages'])==1 and request['messages'][0]['role']=='user','request is not one user message')
        check(sha(request['messages'][0]['content'].encode('utf8'))==plan['message_text_sha256'],'message hash mismatch')
        check(plan['route']=='/v1/chat','route changed')
        hf=read('hf-tokenization.json');check(hf['plan_sha256']==PINNED_PLAN,'HF evidence plan mismatch')
        check(hf['hf_revision']=='c1899de289a04d12100db370d81485cdf75e47ca','HF revision changed')
        hf_models={entry['model']:entry for entry in hf['models']}
        check(len(hf_models)==len(hf['models'])==3 and set(hf_models)==set(plan['models']),'HF model inventory mismatch')
        for model,info in plan['models'].items():
            prompt=read(model+'-prompt.json');ids=prompt['ids']
            check(len(ids)==info['prompt_tokens']>=16384,'prompt count/depth mismatch')
            check(all(type(v) is int and v>=0 for v in ids),'invalid prompt IDs')
            check(ids_sha(ids)==info['prompt_ids_sha256'],'prompt ID hash mismatch')
            check(sha(prompt['prompt'].encode('utf8'))==info['prompt_sha256'],'rendered prompt hash mismatch')
            check(sha(prompt['template'].encode('utf8'))==info['template_sha256'],'template hash mismatch')
            h=hf_models[model]
            check(h['tokens']==len(ids) and h['ids_sha256']==ids_sha(ids),'saved HF prompt identity mismatch')
            report['models'][model]={'prompt_tokens':len(ids),'prompt_ids_sha256':ids_sha(ids),'prompt_sha256':info['prompt_sha256'],'template_sha256':info['template_sha256'],'saved_hf_exact_ids_claim':h.get('exact_hf_ids'),'saved_jinja_match_claim':info.get('jinja2_matches'),'completed_calls':0}
        references={};previous_terminal=None
        for cell,entry in zip(order,done):
            model,arm=cell['model'],cell['arm'];tag=entry['tag']
            record=read('runs/'+tag+'.record.json');got=read('runs/'+tag+'.response.json')
            check(record=={k:v for k,v in entry.items() if k!='matches'},tag+': per-call record/result mismatch')
            check(set(got)=={'ids','text','finish','prompt_tokens','tokens','reused_tokens'},tag+': unexpected response fields')
            check(got['finish']=='eos',tag+': did not finish at EOS')
            check(type(got['tokens']) is int and got['tokens']==len(got['ids'])>0,tag+': token accounting')
            check(all(type(v) is int and v>=0 for v in got['ids']),tag+': invalid response IDs')
            check(type(got['text']) is str and got['text'],tag+': empty or invalid text')
            check(got['prompt_tokens']==plan['models'][model]['prompt_tokens']>=16384,tag+': prompt count')
            check(got['reused_tokens']==0,tag+': cached tokens reused')
            check(got['prompt_tokens']+got['tokens']<plan['ctx_size'],tag+': completion reached context boundary')
            for key in ['finish','prompt_tokens','tokens','reused_tokens']:
                check(record[key]==got[key] and record['observed'][key]==got[key],tag+': recorded accounting mismatch '+key)
            checks={'response_sha256':report['file_sha256']['runs/'+tag+'.response.json'],'text_sha256':sha(got['text'].encode('utf8')),'ids_sha256':ids_sha(got['ids'])}
            for key,value in checks.items():check(record[key]==value,tag+': '+key+' mismatch')
            command=record['command'];check(len(command)==19,tag+': command argument count')
            port=command[6];check(port.isdigit() and 0<int(port)<65536,tag+': port invalid')
            expected=[plan['arms'][arm]['cli'],'serve',plan['models'][model]['path'],'--host','127.0.0.1','--port',port,'--ctx-size',str(plan['ctx_size']),'--threads',str(plan['threads']),'--ubatch',str(plan['ubatch']),'--device',plan['device'],'--cache-type-k',plan['cache_type_k'],'--cache-type-v',plan['cache_type_v']]
            check(command==expected,tag+': command differs from frozen arm/flags/model')
            start=datetime.datetime.fromisoformat(record['start_utc']);requested=datetime.datetime.fromisoformat(record['request_utc']);terminal=datetime.datetime.fromisoformat(record['terminal_utc'])
            check(start<=requested<=terminal,tag+': timestamp ordering')
            check(previous_terminal is None or previous_terminal<=start,tag+': completed calls overlap')
            previous_terminal=terminal
            check(type(record['pid']) is int and record['pid']>0,tag+': invalid PID')
            check(math.isfinite(record['wall_seconds']) and 0<record['wall_seconds']<=plan['request_timeout_seconds'],tag+': request duration invalid')
            if model not in references:references[model]=got
            check(got==references[model],tag+': raw response content differs within model')
            report['models'][model]['completed_calls']+=1
            report['models'][model]['generated_tokens']=got['tokens']
            report['responses'].append({'tag':tag,'finish':got['finish'],'prompt_tokens':got['prompt_tokens'],'generated_tokens':got['tokens'],'reused_tokens':got['reused_tokens'],'raw_within_model_equal':True,**checks,'command_matches_plan':True,'server_exit':record['server_exit']})
        report['hf_identity_scope']={'saved_scope':hf['scope'],'revision':hf['hf_revision'],'tokenizer_sha256_recorded':hf['hf_tokenizer_sha256'],'fresh_tokenizer_hash_or_execution':False}
        if args.final:
            check(read('complete.json')=={'exit':0,'calls':9},'final terminal success marker missing or invalid')
            check(all(v['completed_calls']==3 for v in report['models'].values()),'final model missing arm')
        report['status']='pass' if args.final else 'partial_pass'
        report['full_gate_complete']=bool(args.final)
    except Exception as exc:
        report['status']='failed';report['full_gate_complete']=False;report['errors'].append(type(exc).__name__+': '+str(exc))
    report['script_sha256']=sha(Path(__file__).read_bytes())
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=Path(__file__).parent/(report['mode']+'-'+str(expected_count)+'-'+stamp+'.json')
    with output.open('x',encoding='ascii') as f:json.dump(report,f,indent=2,ensure_ascii=True);f.write('\n')
    print(json.dumps({'status':report['status'],'completed':len(report['responses']),'counts':{k:(v['completed_calls'],v.get('generated_tokens')) for k,v in report['models'].items()},'errors':report['errors'],'report':str(output)}))
    return 0 if not report['errors'] else 1

if __name__=='__main__':raise SystemExit(main())
