"""CPU-only lengths for the frozen judge on the unchanged canonical HaluBench 8k."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .audit_label_pilot import LABELS, summary
from .audit_test_tokens import validate_row
from .check_frozen_fit import FIT_RUN_ID, FREEZE_SHA256, load_freeze, validate_fit
from .check_label_tokenizer import load_tokenizer, package_versions, inspect_tokenizer, verify_reference
from .judge import JudgeInput
from .label_score_contract import LABEL_PROMPT, label_messages, prepare_tokenized_input
from .prepare_halubench import RUN_ID as INPUT_RUN_ID, SPLIT_SHA256, DATA_SHA256
from .prompts import content_hash
from .runner import Example, run_directory
from .serve import code_revision
from .storage import RunConflict, atomic_json, exclusive_run
from .vllm_backend import load_profile

RUN_ID = 'halubench-test-token-audit-label-score-v1'
TEST_MANIFEST_SHA256 = '0e307781570c9cef907023d713a31e798be3289e9150d2250fd59aa27b4bcded'
INPUT_REVISION = 'aac2d9e0ce9ef4ac98ff9a60e26238daa08a46f3'
TEST_ROWS = 8000
CHECKPOINT_EVERY = 25


def validate_inputs(bundle, frozen):
    manifest = bundle['manifest']
    if (bundle.get('manifest_sha256') != TEST_MANIFEST_SHA256
            or content_hash(manifest) != TEST_MANIFEST_SHA256
            or manifest['code_revision'] != INPUT_REVISION
            or manifest['judge_freeze_sha256'] != content_hash(frozen)
            or manifest['canonical_split']['sha256'] != SPLIT_SHA256
            or manifest['canonical_split']['membership_changed'] is not False
            or manifest['dataset']['sha256'] != DATA_SHA256
            or manifest['summary']['scoring_authorized'] is not False):
        raise RunConflict('expected completed canonical HaluBench input manifest and unchanged judge')
    inputs, offline = manifest['model_inputs'], manifest['offline_rows']
    if len(inputs) != TEST_ROWS or len(offline) != TEST_ROWS:
        raise RunConflict('all canonical HaluBench TEST rows required')
    examples, seen = [], set()
    for index, (row, target) in enumerate(zip(inputs, offline)):
        if (set(row) != {'sample_id', 'answer', 'context'}
                or type(row['sample_id']) is not str or not row['sample_id']
                or row['sample_id'] in seen or row['sample_id'] != target['sample_id']
                or type(target['test_index']) is not int or target['test_index'] != index):
            raise RunConflict('HaluBench input projection, IDs or order differ')
        item = JudgeInput(answer=row['answer'], context=row['context'])
        if content_hash(asdict(item)) != target['input_sha256']:
            raise RunConflict('HaluBench answer/context text changed')
        seen.add(row['sample_id'])
        examples.append(Example(row['sample_id'], item))
    return examples


def audit(examples, tokenizer, *, directory, revision, files, versions, reference_hash):
    """CPU tokenization only. Checkpoint every 25 rows and on handled interruption."""
    if not examples or len({ex.sample_id for ex in examples}) != len(examples):
        raise RunConflict('nonempty unique TEST inputs required')
    profile = load_profile('label-score-v1')
    identity = {'study_stage': 'post_thesis', 'version': 'halubench-test-label-token-audit-v1',
                'code_revision': revision, 'TEST_manifest_sha256': TEST_MANIFEST_SHA256,
                'judge_freeze_sha256': FREEZE_SHA256, 'canonical_split_sha256': SPLIT_SHA256,
                'prompt_version': LABEL_PROMPT.version, 'prompt_sha256': LABEL_PROMPT.sha256,
                'profile': profile, 'class_mapping': LABELS, 'tokenizer_files': files,
                'versions': versions, 'tokenizer_reference_sha256': reference_hash,
                'inputs': [{'sample_id': ex.sample_id, 'input_sha256': content_hash(asdict(ex.item)),
                            'messages_sha256': content_hash(label_messages(ex.item))} for ex in examples],
                'scope': 'HaluBench_canonical_8k_TEST_token_lengths_only', 'output_allowance': 1, 'truncation': 'none',
                'checkpoint_every': CHECKPOINT_EVERY, 'prepared_storage': 'hashes_and_lengths_only'}
    directory = Path(directory)
    path = directory / 'audit.json'

    def save(state):
        state['summary'] = summary(state['rows'], len(examples), profile['max_model_len'])
        atomic_json(path, {'audit_sha256': content_hash(state), 'audit': state})

    with exclusive_run(directory):
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            state = saved['audit']
            if saved['audit_sha256'] != content_hash(state) or state['identity'] != identity:
                raise RunConflict('changed or corrupt TEST token audit; preserve it')
            if len(state['rows']) > len(examples):
                raise RunConflict('too many cached token rows')
            for index, row in enumerate(state['rows']):
                validate_row(row, identity['inputs'][index])
            expected_summary = summary(state['rows'], len(examples), profile['max_model_len'])
            if state['summary'] != expected_summary:
                raise RunConflict('cached summary differs from token rows')
            if state['status'] not in ('in_progress', 'interrupted', 'completed', 'overlength'):
                raise RunConflict('unrecognized token audit state')
            if state['status'] in ('completed', 'overlength'):
                expected_status = 'completed' if expected_summary['all_inputs_fit'] else 'overlength'
                if len(state['rows']) != len(examples) or state['status'] != expected_status:
                    raise RunConflict('terminal token audit coverage/status differs')
                return saved, 0
        else:
            state = {'identity': identity, 'rows': []}
        state['status'] = 'in_progress'
        save(state)
        new = 0
        try:
            for ex in examples[len(state['rows']):]:
                prepared = prepare_tokenized_input(ex.item, tokenizer)
                prepared.pop('rendered_prompt')
                row = {'sample_id': ex.sample_id, 'prepared': prepared}
                validate_row(row, identity['inputs'][len(state['rows'])])
                state['rows'].append(row)
                new += 1
                if len(state['rows']) % CHECKPOINT_EVERY == 0:
                    save(state)
        except BaseException:
            state['status'] = 'interrupted'
            save(state)
            raise
        state['summary'] = summary(state['rows'], len(examples), profile['max_model_len'])
        state['status'] = 'completed' if state['summary']['all_inputs_fit'] else 'overlength'
        save(state)
        return {'audit_sha256': content_hash(state), 'audit': state}, new


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    revision = code_revision()
    frozen = load_freeze()
    validate_fit(json.loads((run_directory(FIT_RUN_ID)/'fit.json').read_text(encoding='utf-8')), frozen)
    bundle = json.loads((run_directory(INPUT_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    examples = validate_inputs(bundle, frozen)
    versions = package_versions()
    tokenizer, files = load_tokenizer(run_directory('label-tokenizer-cache-v1'), download=False)
    reference_hash = verify_reference(inspect_tokenizer(tokenizer, files=files, versions=versions))
    directory = run_directory(RUN_ID)
    result, new = audit(examples, tokenizer, directory=directory, revision=revision,
                        files=files, versions=versions, reference_hash=reference_hash)
    print(json.dumps(result['audit']['summary'], indent=2))
    print('New CPU tokenizations:', new)
    print('Audit status:', result['audit']['status'])
    print('HaluBench manifest SHA256:', TEST_MANIFEST_SHA256)
    print('Judge freeze SHA256:', FREEZE_SHA256)
    print('Prompt:', LABEL_PROMPT.version, LABEL_PROMPT.sha256)
    print('Audit SHA256:', result['audit_sha256'])
    print('Audit code revision:', revision)
    print('Private audit:', directory/'audit.json')
    print('Post-thesis canonical HaluBench 8k input lengths only. No model calls, fitting or metrics.')
    print('Legacy baseline provenance remains unresolved. No scoring allowance; separate execution plan required.')
    return 0 if result['audit']['summary']['all_inputs_fit'] else 1


if __name__ == '__main__': raise SystemExit(main())
