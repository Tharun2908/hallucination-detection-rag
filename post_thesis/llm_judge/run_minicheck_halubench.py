"""Bounded fresh MiniCheck canonical HaluBench scores; no target-domain fitting."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time

from .audit_halubench_tokens import TEST_MANIFEST_SHA256, validate_inputs
from .check_frozen_fit import FREEZE_SHA256, load_freeze
from .check_minicheck_live import (RUN_ID as SMOKE_RUN_ID, ENGINE, VERSIONS,
                                  configure_runtime, sentence_resources)
from .minicheck_contract import SUPPORT_THRESHOLD
from .minicheck_inputs import explicit_tokenizer, prepare, prepare_item, verify_snapshot
from .minicheck_inference import MiniCheckBackend, reference, validate_result
from .prepare_halubench import RUN_ID as TEST_RUN_ID
from .prepare_test_manifest import save_once
from .prepare_pilot import file_sha256
from .recover_fusion import checked_bundle
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import Journal, RunConflict, atomic_json, exclusive_run

RUN_ID = 'minicheck-halubench-test-fresh-v1'
SMOKE_SHA256 = 'eab310d3c23fdcdb08f23217a01862f8177deb6cbf0949b440f17e23904aa797'
PLAN = {'study_stage':'post_thesis', 'run_id':RUN_ID, 'examples':8000,
        'maximum_sentence_requests':9158, 'maximum_total_input_tokens':4905370,
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

REFERENCES_SHA256 = '0a31303eab49c4c8022024927d87cebf6b7a486ab42743d99e8d0792844e1a07'
EXPECTED_WORKLOAD = {'examples': 8000, 'sentence_requests': 9158, 'input_tokens': 4905370, 'max_input_tokens': 6756, 'min_input_tokens': 102, 'output_token_allowance': 9158, 'all_inputs_fit': True, 'truncated_prompts': 0}
FUSION_RUN_ID = 'halubench-fresh-metadata-free-fusion-v1'
FUSION_SHA256 = '2dbd25f11f167cc7e0ec74c1557450dcf2407cb9ea89db53e876d84f2acea415'
IMPLEMENTATION_FILES = {'minicheck_inputs.py': '93b8bbbcf18a9dec952dda196210775cdd84b55ace529a8d3e17ed56b54ea0b4', 'minicheck_inference.py': '807fa0b8b398d597420159096442b8e3f5b633c213178dbd3eccf9fe9b6e7382', 'minicheck_contract.py': '576212715e6ce8a9e46abe7ebfc40d87c77ad47fac8c318e135f9b74eb5d9859', 'check_minicheck_live.py': '0d28904317d194f08a964b31d34e136c41e013a6d88652ba78bd9cf3fb9fb156', 'minicheck_files_v1.json': '6c9477c2fe4c03bdf60b6b4d79de78a0d2459e1d36f055b8c36678fbec81d74b'}
PLAN.update(dataset='canonical_HaluBench_8k_TEST', zero_shot_transfer=True,
            adaptation_training=False, implementation_files=IMPLEMENTATION_FILES)


def verify_implementation():
    for name, expected in IMPLEMENTATION_FILES.items():
        if file_sha256(Path(__file__).with_name(name)) != expected:
            raise RunConflict('registered MiniCheck implementation changed: ' + name)
    return dict(IMPLEMENTATION_FILES)


def verify_fusion(examples):
    report=checked_bundle(run_directory(FUSION_RUN_ID)/'report.json',
                          'report','report_sha256',FUSION_SHA256)
    if (report['run_id'] != FUSION_RUN_ID or report['TEST_manifest_sha256'] != TEST_MANIFEST_SHA256
            or report['judge_freeze_sha256'] != FREEZE_SHA256
            or report['valid_scores'] != 8000 or len(report['predictions']) != len(examples)
            or len(examples) != 8000 or report['adaptation_training'] is not False):
        raise RunConflict('expected complete frozen HaluBench fusion report')
    from dataclasses import asdict
    for index,(row,ex) in enumerate(zip(report['predictions'],examples)):
        if (row['sample_id'] != ex.sample_id or row['test_index'] != index or row['status'] != 'ok'
                or row['input_sha256'] != content_hash(asdict(ex.item))):
            raise RunConflict('fusion canonical input alignment differs')
    return FUSION_SHA256


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
            'fitting_calls':0,'judge_calls':0,'HaluBench_read':True,'adaptation_training':False,'RAGTruth_inputs_read':False}
    return {'report_sha256':content_hash(report),'report':report}


def execute(examples, refs, *, directory, identity, backend_factory, max_new_examples=8000):
    if type(max_new_examples) is not int or not 0<=max_new_examples<=PLAN['examples']:
        raise ValueError('max-new-examples must be from 0 to 8000')
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
                path=directory/'summary.json'
                if not path.exists() or json.loads(path.read_text(encoding='utf-8')) != summary:
                    atomic_json(path,summary)
                return summary,new
            finally: startup.close()
        finally: journal.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-cache',type=Path,default=Path('/root/llm-judge-hf-cache'))
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--max-new-examples',type=int,default=8000)
    args=parser.parse_args()
    if not 0 <= args.max_new_examples <= PLAN['examples']:
        parser.error('--max-new-examples must be from 0 to 8000')
    revision=code_revision(); implementation=verify_implementation()
    runtime_env=configure_runtime()
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
    fusion_hash=verify_fusion(examples)
    refs=[]
    for i,ex in enumerate(examples):
        item=prepare_item(ex.item.answer,ex.item.context,tokenizer,split)
        refs.append(reference(ex,i,item))
        if (i+1)%250==0: print('MiniCheck CPU preparation:',i+1,'/',len(examples),flush=True)
    totals=workload(refs)
    if totals != EXPECTED_WORKLOAD or content_hash(refs) != REFERENCES_SHA256:
        raise RunConflict('canonical MiniCheck HaluBench workload differs; no model loading')
    identity={'code_revision':revision,'plan':PLAN,'packages':packages,'python':platform.python_version(),
              'smoke_report_sha256':SMOKE_SHA256,'TEST_manifest_sha256':TEST_MANIFEST_SHA256,
              'judge_freeze_sha256':FREEZE_SHA256,'fusion_report_sha256':fusion_hash,
              'implementation_files':implementation,
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
    print('Post-thesis fresh MiniCheck HaluBench zero-shot TEST inference')
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
    print('Fresh scores only. No TEST label metrics, fitting, judge calls, target-domain training or legacy-file changes.')
    return 1 if r['terminal_failures'] or r['initialization_failure'] else 0


if __name__=='__main__': raise SystemExit(main())
