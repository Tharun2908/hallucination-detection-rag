"""Frozen RAGTruth TEST request preparation; labels never enter model payloads."""
from dataclasses import asdict
import json
from pathlib import Path

from .audit_label_pilot import LABELS, summary as audit_summary
from .audit_test_tokens import TEST_MANIFEST_SHA256, validate_inputs
from .check_frozen_fit import FREEZE_SHA256, load_freeze
from .check_evaluation_math import CONTRACT_SHA256, load_contract
from .freeze_comparison import RUN_ID as COMPARISON_RUN_ID, SOURCES, align_sources, read_report
from .label_live import PROFILE, SOURCE_BLOBS, completion_payload
from .label_score_contract import LABEL_PROMPT, prepare_tokenized_input
from .prompts import content_hash
from .runner import run_directory
from .storage import RunConflict
from .vllm_backend import load_profile

RUN_ID = 'qwen3-ragtruth-test-label-score-v1'
VERSION = 'ragtruth-test-label-scoring-v1'
AUDIT_SHA256 = 'ee66f093aef059363151cc714cca1ee5cc318ba443dd9e1a9b3e8db51674407c'
COMPARISON_SHA256 = '3e8c25eaaeaf03aa93377f8a7b6c516e6a9f4e8c811305fb88683e78e0b615bb'
PLAN_PATH = Path(__file__).parent / 'configs' / 'ragtruth_test_label_scoring_v1.json'


def plan_descriptor():
    return {'study_stage':'post_thesis','version':VERSION,'run_id':RUN_ID,
        'scope':'RAGTruth_TEST_frozen_judge_inference_only',
        'TEST_manifest_sha256':TEST_MANIFEST_SHA256,'judge_freeze_sha256':FREEZE_SHA256,
        'comparison_manifest_sha256':COMPARISON_SHA256,'evaluation_contract_sha256':CONTRACT_SHA256,
        'audit_sha256':AUDIT_SHA256,'prompt_version':LABEL_PROMPT.version,'prompt_sha256':LABEL_PROMPT.sha256,
        'profile':load_profile(PROFILE),'source_blobs':SOURCE_BLOBS,
        'class_mapping':{'supported':32,'unsupported':33},'request_count':2700,
        'max_attempts':2700,'attempts_per_slot':1,'client_budget_seconds':7200,
        'request_timeout_seconds':60,'audited_input_tokens':3334607,
        'audited_min_input_tokens':649,'audited_max_input_tokens':2849,
        'max_output_tokens_per_request':1,'max_output_tokens_total':2700,
        'concurrency':1,'retries':0,'truncation':'none','halt_on_any_request_error':True,
        'unknown_window_policy':'charge_full_reserved_remaining_budget',
        'orientation_policy':'primary_only_no_selection_or_averaging',
        'fit_or_threshold_selection':False,'TEST_metrics':False,'HaluBench_access':False}


def load_plan():
    plan=json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    if plan!=plan_descriptor(): raise RunConflict('changed frozen TEST scoring plan')
    return plan


def validate_audit(audited):
    state=audited['audit'];identity=state['identity'];plan=load_plan()
    if audited.get('audit_sha256')!=AUDIT_SHA256 or content_hash(state)!=AUDIT_SHA256:
        raise RunConflict('expected unchanged completed TEST token audit')
    if (state['status']!='completed' or len(state['rows'])!=plan['request_count']
            or identity['TEST_manifest_sha256']!=TEST_MANIFEST_SHA256
            or identity['judge_freeze_sha256']!=FREEZE_SHA256
            or identity['prompt_sha256']!=LABEL_PROMPT.sha256
            or identity['profile']!=plan['profile'] or identity['class_mapping']!=LABELS
            or state['summary']!=audit_summary(state['rows'],plan['request_count'],plan['profile']['max_model_len'])
            or not state['summary']['all_inputs_fit']
            or state['summary']['total_input_tokens']!=plan['audited_input_tokens']):
        raise RunConflict('TEST audit coverage, budget or frozen judge differs')
    return state


def validate_comparison(bundle, targets, reports):
    manifest=bundle['manifest']
    if (bundle.get('manifest_sha256')!=COMPARISON_SHA256 or content_hash(manifest)!=COMPARISON_SHA256
            or manifest['TEST_manifest_sha256']!=TEST_MANIFEST_SHA256
            or manifest['judge_freeze_sha256']!=FREEZE_SHA256
            or manifest['evaluation_contract_sha256']!=CONTRACT_SHA256
            or manifest['targets']!=targets):
        raise RunConflict('fresh comparison manifest or population differs')
    projected=align_sources(reports,targets)
    for name,baseline in manifest['baselines'].items():
        if (baseline['comparison_ready'] is not True
                or baseline['report_sha256']!=SOURCES[name][2]
                or baseline['aligned_predictions_sha256']!=content_hash(projected[name])):
            raise RunConflict('fresh baseline comparison identity differs')
    return manifest


def load_inputs():
    from .prepare_test_manifest import RUN_ID as INPUT_RUN_ID
    from .audit_test_tokens import RUN_ID as AUDIT_RUN_ID
    manifest=json.loads((run_directory(INPUT_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    examples=validate_inputs(manifest,load_freeze())
    load_contract()
    audited=json.loads((run_directory(AUDIT_RUN_ID)/'audit.json').read_text(encoding='utf-8'))
    validate_audit(audited)
    targets=[{k:r[k] for k in ('sample_id','test_index','input_sha256','component_sha256')}
             for r in manifest['manifest']['offline_rows']]
    comparison=json.loads((run_directory(COMPARISON_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    validate_comparison(comparison,targets,{k:read_report(v) for k,v in SOURCES.items()})
    return examples,audited,targets


def make_requests(examples, state, tokenizer):
    if len(examples)!=len(state['rows']): raise RunConflict('TEST audit coverage differs')
    profile=load_profile(PROFILE);requests=[]
    for index,(ex,row) in enumerate(zip(examples,state['rows'])):
        prepared=prepare_tokenized_input(ex.item,tokenizer)
        prompt=prepared.pop('rendered_prompt')
        if row!={'sample_id':ex.sample_id,'prepared':prepared} or prepared['labels']!=LABELS:
            raise RunConflict('rendered TEST input differs from frozen token audit')
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        if content_hash(ids)!=prepared['prompt_token_ids_sha256'] or len(ids)+1>profile['max_model_len']:
            raise RunConflict('TEST token IDs differ or require forbidden truncation')
        payload=completion_payload(ids,profile)
        mapping={'supported':32,'unsupported':33};slot='test:'+ex.sample_id
        identity={'version':VERSION,'slot':slot,'test_index':index,'input_sha256':content_hash(asdict(ex.item)),
                  'TEST_manifest_sha256':TEST_MANIFEST_SHA256,'judge_freeze_sha256':FREEZE_SHA256,
                  'audit_sha256':AUDIT_SHA256,'comparison_manifest_sha256':COMPARISON_SHA256,
                  'prompt_sha256':LABEL_PROMPT.sha256,'prepared':prepared,
                  'class_mapping':mapping,'payload':payload}
        requests.append({'slot':slot,'sample_id':ex.sample_id,'orientation':'primary',
                         'class_mapping':mapping,'identity':identity,'key':content_hash(identity),'payload':payload})
    return requests


def validate_preparation(preparation,requests,plan):
    state=validate_audit(preparation['audited'])
    if (plan!=load_plan() or preparation['comparison_manifest_sha256']!=COMPARISON_SHA256
            or preparation['source']!={'version':'0.29.0','git_blob_sha1':SOURCE_BLOBS}
            or len(requests)!=plan['request_count']
            or len({r['sample_id'] for r in requests})!=len(requests)):
        raise RunConflict('TEST request count, source or plan differs')
    for key in ('versions','tokenizer_files','tokenizer_reference_sha256'):
        if preparation[key]!=state['identity'][key]: raise RunConflict('tokenizer audit environment differs')
    for i,(request,row) in enumerate(zip(requests,state['rows'])):
        identity=request['identity'];prepared=row['prepared'];payload=request['payload'];ids=payload['prompt']
        if (request['sample_id']!=row['sample_id'] or request['slot']!='test:'+row['sample_id']
                or request['orientation']!='primary' or identity['test_index']!=i
                or identity['prepared']!=prepared or identity['input_sha256']!=prepared['input_sha256']
                or identity['prompt_sha256']!=LABEL_PROMPT.sha256
                or identity['TEST_manifest_sha256']!=TEST_MANIFEST_SHA256
                or identity['judge_freeze_sha256']!=FREEZE_SHA256
                or identity['audit_sha256']!=AUDIT_SHA256
                or identity['comparison_manifest_sha256']!=COMPARISON_SHA256
                or identity['version']!=VERSION
                or request['class_mapping']!={'supported':32,'unsupported':33}
                or not isinstance(ids,list) or any(type(t) is not int or t<0 for t in ids)
                or len(ids)!=prepared['input_tokens'] or content_hash(ids)!=prepared['prompt_token_ids_sha256']
                or payload!=completion_payload(ids,plan['profile'])):
            raise RunConflict('TEST prepared request changed or unregistered payload')
