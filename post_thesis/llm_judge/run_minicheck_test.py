"""Bounded fresh MiniCheck RAGTruth TEST scores; no fitting or label metrics."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time

from .audit_test_tokens import TEST_MANIFEST_SHA256, validate_inputs
from .check_frozen_fit import load_freeze
from .check_minicheck_live import (RUN_ID as SMOKE_RUN_ID, ENGINE, VERSIONS,
                                  configure_runtime, sentence_resources)
from .minicheck_contract import SUPPORT_THRESHOLD
from .minicheck_inputs import explicit_tokenizer, prepare, prepare_item, verify_snapshot
from .minicheck_inference import MiniCheckBackend, reference, validate_result
from .prepare_test_manifest import RUN_ID as TEST_RUN_ID, save_once
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import Journal, RunConflict, atomic_json, exclusive_run

RUN_ID = 'minicheck-ragtruth-test-fresh-v1'
SMOKE_SHA256 = 'eab310d3c23fdcdb08f23217a01862f8177deb6cbf0949b440f17e23904aa797'
PLAN = {'study_stage':'post_thesis', 'run_id':RUN_ID, 'examples':2700,
        'maximum_sentence_requests':100000, 'maximum_total_input_tokens':100000000,
        'maximum_requests_per_example':512, 'output_tokens_per_request':1,
        'maximum_model_initializations':4, 'maximum_client_seconds':7200,
        'deadline':'checked before initialization and each example; active work not preempted',
        'attempts_per_example':1, 'automatic_retries':0, 'max_model_len':32768,
        'truncation':'none; stop before GPU loading if any prepared prompt exceeds limit',
        'batch_boundary':'one example per generate call, chunk-major then answer-sentence order',
        'engine':ENGINE, 'temperature':0, 'logprobs':5,
        'support_aggregation':'min over answer sentences of max over document chunks',
        'support_score':'sum exp(logprob) over returned tokens with decoded_token.lower() == yes',
        'TRAIN_threshold':{'score_space':'support', 'comparator':'<',
                           'threshold_hex':SUPPORT_THRESHOLD.hex()},
        'TEST_metrics':False, 'fitting_calls':0, 'judge_calls':0}


def verified_smoke(bundle):
    report = bundle['report']
    if (bundle.get('report_sha256') != SMOKE_SHA256 or content_hash(report) != SMOKE_SHA256
            or report['result']['status'] != 'ok'
            or report['result']['requested_generations'] != 6
            or report['result']['prompt_ids_match'] is not True):
        raise RunConflict('expected successful preserved MiniCheck v2 synthetic report')
    return report


def workload(refs):
    if not refs or len(refs)>PLAN['examples'] or len({r['sample_id'] for r in refs})!=len(refs):
        raise RunConflict('invalid MiniCheck example count or duplicate IDs')
    lengths=[]
    for index,ref in enumerate(refs):
        n=len(ref['context_chunks'])*len(ref['answer_sentences'])
        if (ref['test_index']!=index or not 0<n<=PLAN['maximum_requests_per_example']
                or len(ref['prompts'])!=n):
            raise RunConflict('MiniCheck preparation count/order outside fixed limits')
        for prompt in ref['prompts']:
            length=prompt['input_tokens']
            if type(length) is not int or not 0<length<PLAN['max_model_len']:
                raise RunConflict('MiniCheck prompt exceeds fixed length budget')
            lengths.append(length)
    if (len(lengths)>PLAN['maximum_sentence_requests']
            or sum(lengths)>PLAN['maximum_total_input_tokens']):
        raise RunConflict('MiniCheck token/request workload exceeds fixed plan; no inference')
    return {'examples':len(refs), 'sentence_requests':len(lengths), 'input_tokens':sum(lengths),
            'max_input_tokens':max(lengths), 'min_input_tokens':min(lengths),
            'output_token_allowance':len(lengths), 'all_inputs_fit':True, 'truncated_prompts':0}


def summarize(refs, records, starts, identity):
    keys=[content_hash(r) for r in refs]; expected=dict(zip(keys,refs)); by_key={}
    timing={'known_example_seconds':0., 'known_initialization_seconds':0.,
            'unknown_timing_examples':0, 'unknown_initializations':0}
    known={'input_tokens':0,'output_tokens':0}; unknown=0; predictions=[]
    for row in records:
        key=row['request_key']
        if key not in expected or key in by_key or row['ordinal']!=1 or row['state'] not in ('finished','interrupted'):
            raise RunConflict('unexpected MiniCheck journal attempt')
        by_key[key]=row
        if row['state']=='finished': validate_result(row['result'],expected[key])
    startup_failed=False
    for index,row in enumerate(starts):
        if row['request_key']!='model_initialization' or row['ordinal']!=index+1:
            raise RunConflict('unexpected MiniCheck initialization record')
        if row['state']!='finished':
            startup_failed=True; timing['unknown_initializations']+=1
        else:
            r=row['result']; seconds=r['seconds']
            if (r['status'] not in ('ok','error') or type(seconds) not in (int,float)
                    or not math.isfinite(seconds) or seconds<0):
                raise RunConflict('invalid MiniCheck initialization result')
            timing['known_initialization_seconds']+=seconds
            startup_failed |= r['status']!='ok'
    if len(starts)>PLAN['maximum_model_initializations']:
        raise RunConflict('too many MiniCheck initializations')
    for key,ref in zip(keys,refs):
        row=by_key.get(key)
        result=row['result'] if row and row['state']=='finished' else None
        status=result['status'] if result else ('interrupted' if row else 'pending')
        prediction={k:ref[k] for k in ('sample_id','test_index','input_sha256')}
        prediction.update(status=status,support_score=None)
        if status=='ok':
            prediction.update({k:result[k] for k in ('support_score','best_support_per_answer_sentence',
                               'upstream_supported_label','frozen_predicted_unsupported')})
            timing['known_example_seconds']+=result['example_seconds']
            known['input_tokens']+=sum(p['input_tokens'] for p in ref['prompts'])
            known['output_tokens']+=len(ref['prompts'])
        elif row:
            timing['unknown_timing_examples']+=1; unknown+=1
            prediction['error']=result.get('error') if result else 'interrupted; token usage unknown'
        predictions.append(prediction)
    valid=sum(p['status']=='ok' for p in predictions)
    pending=sum(p['status']=='pending' for p in predictions)
    failures=len(refs)-valid-pending
    charged=timing['known_example_seconds']+timing['known_initialization_seconds']
    halt=None
    if startup_failed or failures: halt='terminal_failure_or_interruption; no automatic retry'
    elif pending and charged>=PLAN['maximum_client_seconds']: halt='client_time_budget_exhausted'
    elif pending and len(starts)>=PLAN['maximum_model_initializations']: halt='model_initialization_budget_exhausted'
    report={'study_stage':'post_thesis','run_id':RUN_ID,'identity':identity,
            'examples_total':len(refs),'valid_scores':valid,'pending':pending,'terminal_failures':failures,
            'model_initializations':len(starts),'initialization_failure':startup_failed,
            'initialization_results':[r['result'] if r['state']=='finished' else {'status':'interrupted'} for r in starts],
            'halt_reason':halt,'timing':timing,'known_token_totals':known,
            'examples_with_unknown_usage':unknown,'known_client_seconds':charged,
            'workload':workload(refs),'predictions':predictions,
            'raw_logprobs_location':'journal.sqlite3', 'prepared_input_location':'preparation.json',
            'legacy_cache_provenance_verified':False, 'legacy_files_rewritten':False,
            'comparison_ready':False,'TEST_label_metrics_computed':False,
            'fitting_calls':0,'judge_calls':0,'HaluBench_read':False}
    return {'report_sha256':content_hash(report),'report':report}


def execute(examples, refs, *, directory, identity, backend_factory, max_new_examples=2700):
    if type(max_new_examples) is not int or not 0<=max_new_examples<=PLAN['examples']:
        raise ValueError('max-new-examples must be from 0 to 2700')
    workload(refs)
    if len(examples)!=len(refs): raise RunConflict('MiniCheck example coverage differs')
    directory=Path(directory)
    with exclusive_run(directory):
        journal=Journal(directory/'journal.sqlite3',{'identity':identity,'references':refs})
        try:
            startup=Journal(directory/'initializations.sqlite3',identity)
            try:
                journal.recover(); startup.recover()
                def report(): return summarize(refs,journal.records(),startup.records(),identity)
                summary=report(); seen={r['request_key'] for r in journal.records()}
                backend=None; new=0
                charged=summary['report']['known_client_seconds']
                invocation=time.perf_counter()
                if not summary['report']['halt_reason']:
                    for ex,ref in zip(examples,refs):
                        key=content_hash(ref)
                        if key in seen: continue
                        if new>=max_new_examples or charged+time.perf_counter()-invocation>=PLAN['maximum_client_seconds']: break
                        if backend is None:
                            starts=startup.records()
                            if len(starts)>=PLAN['maximum_model_initializations']: break
                            attempt=startup.start('model_initialization',len(starts)+1)
                            begin=time.perf_counter()
                            try:
                                backend=backend_factory()
                                result={'status':'ok','seconds':time.perf_counter()-begin}
                            except Exception as exc:
                                result={'status':'error','seconds':time.perf_counter()-begin,
                                        'error':type(exc).__name__+': '+str(exc)}
                            startup.finish(attempt,result)
                            if result['status']!='ok': break
                        if charged+time.perf_counter()-invocation>=PLAN['maximum_client_seconds']: break
                        attempt=journal.start(key,1); new+=1
                        try:
                            result=backend.score(ex,ref); validate_result(result,ref)
                        except Exception as exc:
                            result={'status':'error','error':type(exc).__name__+': '+str(exc),
                                    'token_usage':'unknown; may include partial generation'}
                        journal.finish(attempt,result)
                        if new%25==0 or result['status']!='ok':
                            summary=report(); atomic_json(directory/'summary.json',summary)
                            print(f"MiniCheck: {summary['report']['valid_scores']}/{len(refs)} valid; new examples {new}",flush=True)
                        if result['status']!='ok': break
                summary=report()
                atomic_json(directory/'summary.json',summary)
                return summary,new
            finally: startup.close()
        finally: journal.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-cache',type=Path,default=Path('/root/llm-judge-hf-cache'))
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--max-new-examples',type=int,default=2700)
    args=parser.parse_args()
    revision=code_revision(); runtime_env=configure_runtime()
    packages={p:version(p) for p in VERSIONS}
    if packages!=VERSIONS or platform.python_version_tuple()[:2]!=('3','12'):
        raise RunConflict('use the recorded judge-serving environment')
    smoke_dir=run_directory(SMOKE_RUN_ID)
    smoke=verified_smoke(json.loads((smoke_dir/'report.json').read_text(encoding='utf-8')))
    snapshot,files=verify_snapshot(args.model_cache)
    adapter=smoke_dir/'saved-tokenizer-adapter'
    if not adapter.is_dir(): raise RunConflict('preserve the successful synthetic tokenizer adapter')
    tokenizer,tok_record=explicit_tokenizer(snapshot,adapter)
    split,resources=sentence_resources()
    previous=smoke['identity']
    observed={'packages':packages,'python':platform.python_version(),'files':files,
              'tokenizer':tok_record,'punkt_english_files':resources,'engine':ENGINE,
              'runtime_environment':runtime_env,'prepared':prepare(tokenizer,split)}
    if any(previous.get(k)!=v for k,v in observed.items()):
        raise RunConflict('current MiniCheck preparation/runtime differs from successful synthetic run')
    bundle=json.loads((run_directory(TEST_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    examples=validate_inputs(bundle,load_freeze())
    refs=[]
    for i,ex in enumerate(examples):
        item=prepare_item(ex.item.answer,ex.item.context,tokenizer,split)
        refs.append(reference(ex,i,item))
        if (i+1)%250==0: print('MiniCheck CPU preparation:',i+1,'/',len(examples),flush=True)
    totals=workload(refs)
    identity={'code_revision':revision,'plan':PLAN,'packages':packages,'python':platform.python_version(),
              'smoke_report_sha256':SMOKE_SHA256,'TEST_manifest_sha256':TEST_MANIFEST_SHA256,
              'files':files,'tokenizer':tok_record,'punkt_english_files':resources,
              'runtime_environment':runtime_env,'references_sha256':content_hash(refs)}
    preparation={'identity':identity,'references':refs,'workload':totals}
    directory=run_directory(RUN_ID)
    with exclusive_run(directory):
        save_once(directory/'preparation.json',{'preparation_sha256':content_hash(preparation),'preparation':preparation})
    print('MiniCheck fixed workload:',json.dumps(totals),flush=True)
    if args.prepare_only:
        print('Preparation SHA256:',content_hash(preparation))
        print('CPU preparation only; no model loading, generation or TEST metrics.')
        return 0
    summary,new=execute(examples,refs,directory=directory,identity=identity,
                        backend_factory=lambda:MiniCheckBackend(snapshot,adapter,tokenizer,split),
                        max_new_examples=args.max_new_examples)
    r=summary['report']
    print('Post-thesis fresh MiniCheck TEST inference')
    print('Valid scores:',r['valid_scores'],'/',r['examples_total'])
    print('New attempted examples:',new)
    print('Terminal failures / pending:',r['terminal_failures'],'/',r['pending'])
    print('Halt reason:',r['halt_reason'])
    print('Known token totals:',r['known_token_totals'])
    print('Examples with unknown usage:',r['examples_with_unknown_usage'])
    print('Timing (loading recorded separately):',json.dumps(r['timing']))
    if r['initialization_failure']: print('Initialization results:',json.dumps(r['initialization_results']))
    for p in r['predictions']:
        if p.get('error'): print('Error:',p['sample_id'],p['error'])
    print('Report SHA256:',summary['report_sha256'])
    print('Private summary:',directory/'summary.json')
    print('Fresh scores only. No TEST label metrics, fitting, judge calls, HaluBench reads or legacy-file changes.')
    return 1 if r['terminal_failures'] or r['initialization_failure'] else 0


if __name__=='__main__': raise SystemExit(main())
