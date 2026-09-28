"""Frozen HaluBench TEST request preparation; labels never enter model payloads."""
from dataclasses import asdict
import json
from pathlib import Path

from .audit_label_pilot import LABELS, summary as audit_summary
from .audit_halubench_tokens import TEST_MANIFEST_SHA256, validate_inputs
from .check_frozen_fit import FREEZE_SHA256, load_freeze
from .check_evaluation_math import CONTRACT_SHA256, load_contract
from .halubench_comparison import COMPARISON_SHA256, verify_and_save
from .label_live import PROFILE, SOURCE_BLOBS, completion_payload
from .label_score_contract import LABEL_PROMPT, prepare_tokenized_input
from .prompts import content_hash
from .runner import run_directory
from .storage import RunConflict
from .vllm_backend import load_profile

RUN_ID = 'qwen3-halubench-test-label-score-v1'
VERSION = 'halubench-test-label-scoring-v1'
AUDIT_SHA256 = '7173c30e3a197aabb61c8ffd8c0e145cc556affa9aad20d7dcd43bc9cb24782a'
PLAN_PATH = Path(__file__).parent / 'configs' / 'halubench_test_label_scoring_v1.json'


IMPLEMENTATION_FILES = {'label_live.py': 'de9378cab4b549b05fc94e768a52435265cf98fcaaea33f05c9e56185d0d19a9', 'label_score_contract.py': '8967fa91ea784fb05f3b379b82f319390a17373905885afa86c3f3c86b96e475', 'run_development_scores.py': 'd2a7c85ff396c7b4551479f3d4c1e92f6e78b8521623b9e33253de5d04cc72be', 'label_diagnose.py': '92c6e9234883b3b232d1028d877674c97a18b67dcc0fa5b19054b508ce0b1927', 'run_pilot.py': 'd8370f9fe641b8913a52d33acdb1af00d0087555d509098433d14411ef7a9aa9'}


def verify_implementation():
    from .prepare_pilot import file_sha256
    for name,digest in IMPLEMENTATION_FILES.items():
        if file_sha256(Path(__file__).with_name(name))!=digest:
            raise RunConflict('registered judge implementation changed: '+name)


def plan_descriptor():
    return {'study_stage':'post_thesis','version':VERSION,'run_id':RUN_ID,
        'scope':'HaluBench_TEST_frozen_zero_shot_judge_inference_only',
        'TEST_manifest_sha256':TEST_MANIFEST_SHA256,'judge_freeze_sha256':FREEZE_SHA256,
        'comparison_manifest_sha256':COMPARISON_SHA256,'evaluation_contract_sha256':CONTRACT_SHA256,
        'audit_sha256':AUDIT_SHA256,'prompt_version':LABEL_PROMPT.version,'prompt_sha256':LABEL_PROMPT.sha256,
        'profile':load_profile(PROFILE),'source_blobs':SOURCE_BLOBS,'implementation_files':IMPLEMENTATION_FILES,
        'class_mapping':{'supported':32,'unsupported':33},'request_count':8000,
        'max_attempts':8000,'attempts_per_slot':1,'client_budget_seconds':14400,
        'request_timeout_seconds':60,'audited_input_tokens':7794485,
        'audited_min_input_tokens':506,'audited_max_input_tokens':7298,
        'max_output_tokens_per_request':1,'max_output_tokens_total':8000,
        'concurrency':1,'retries':0,'truncation':'none','halt_on_any_request_error':True,
        'unknown_window_policy':'charge_full_reserved_remaining_budget',
        'orientation_policy':'primary_only_no_selection_or_averaging',
        'fit_or_threshold_selection':False,'TEST_metrics':False,'HaluBench_access':True,'adaptation_training':False}


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


def load_inputs():
    from .prepare_halubench import RUN_ID as INPUT_RUN_ID
    from .audit_halubench_tokens import RUN_ID as AUDIT_RUN_ID
    verify_implementation()
    manifest=json.loads((run_directory(INPUT_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    examples=validate_inputs(manifest,load_freeze())
    load_contract()
    audited=json.loads((run_directory(AUDIT_RUN_ID)/'audit.json').read_text(encoding='utf-8'))
    validate_audit(audited)
    targets=[{k:r[k] for k in ('sample_id','test_index','input_sha256','component_sha256')}
             for r in manifest['manifest']['offline_rows']]
    verify_and_save(targets)
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
