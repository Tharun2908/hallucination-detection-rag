"""CPU-only RAGTruth TEST token audit under the frozen post-thesis judge."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from .audit_baseline_provenance import TEST_MANIFEST_SHA256
from .audit_label_pilot import LABELS, summary
from .check_frozen_fit import FIT_RUN_ID, FREEZE_SHA256, load_freeze, validate_fit
from .check_label_tokenizer import load_tokenizer, package_versions, inspect_tokenizer, verify_reference
from .judge import JudgeInput
from .label_score_contract import ASSISTANT_BOUNDARY, LABEL_PROMPT, label_messages, prepare_tokenized_input
from .prepare_test_manifest import RUN_ID as TEST_RUN_ID
from .prompts import content_hash
from .recover_fusion import RUN_ID as FUSION_RUN_ID
from .runner import Example, run_directory
from .serve import code_revision
from .storage import RunConflict, atomic_json, exclusive_run
from .vllm_backend import load_profile

RUN_ID = 'ragtruth-test-token-audit-label-score-v1'
TEST_REVISION = '977dbf4bb6af71e9a2fe2f4ad999ad813f37e055'
TEST_ROWS = 2700
FUSION_REPORT_SHA256 = '1c90a003abbda9044767bb9167b1ecc171f7a8a746b87093a143d833ba4ec0d6'
FUSION_REVISION = '538e4d6814e8193aa79a9e84e95de910acf016f3'
CHECKPOINT_EVERY = 25
PREPARED_KEYS = {'input_sha256', 'messages_sha256', 'rendered_prompt_sha256',
                 'prompt_token_ids_sha256', 'input_tokens', 'score_position',
                 'score_position_convention', 'template_sha256', 'assistant_boundary', 'labels'}


def validate_inputs(bundle, frozen):
    """Integrity includes offline labels; model-input projection never reads them."""
    manifest = bundle['manifest']
    if (bundle.get('manifest_sha256') != TEST_MANIFEST_SHA256
            or content_hash(manifest) != TEST_MANIFEST_SHA256
            or manifest['code_revision'] != TEST_REVISION
            or manifest['freeze_sha256'] != content_hash(frozen)
            or manifest['fit_report_sha256'] != frozen['source_fit_report_sha256']):
        raise RunConflict('expected unchanged completed TEST manifest and frozen judge')
    inputs, offline = manifest['model_inputs'], manifest['offline_rows']
    if len(inputs) != TEST_ROWS or len(offline) != TEST_ROWS:
        raise RunConflict('all original TEST rows are required')
    examples, seen = [], set()
    for index, (row, target) in enumerate(zip(inputs, offline)):
        if (set(row) != {'sample_id', 'answer', 'context'}
                or type(row['sample_id']) is not str or not row['sample_id']
                or row['sample_id'] in seen or row['sample_id'] != target['sample_id']
                or type(target['test_index']) is not int or target['test_index'] != index):
            raise RunConflict('TEST model-input fields, IDs or order differ')
        item = JudgeInput(answer=row['answer'], context=row['context'])
        if content_hash(asdict(item)) != target['input_sha256']:
            raise RunConflict('TEST answer/context text changed')
        examples.append(Example(row['sample_id'], item))
        seen.add(row['sample_id'])
    return examples


def validate_recovery(bundle, manifest):
    """Verify the preserved reconstruction without fitting or changing readiness."""
    report = bundle['report']
    if (bundle.get('report_sha256') != FUSION_REPORT_SHA256
            or content_hash(report) != FUSION_REPORT_SHA256):
        raise RunConflict('expected completed immutable fusion recovery report')
    identity, result = report['identity'], report['results']
    if (report['run_id'] != FUSION_RUN_ID or report['study_stage'] != 'post_thesis'
            or identity['code_revision'] != FUSION_REVISION
            or identity['TEST_manifest_sha256'] != TEST_MANIFEST_SHA256
            or identity['judge_freeze_sha256'] != FREEZE_SHA256
            or result['status'] != 'reconstructed_legacy_baseline_with_provenance_limits'
            or result['rounded_TRAIN_reference_matches'] is not True
            or result['fusion_fit_calls'] != 6
            or report['comparison_ready'] is not False
            or report['TEST_metrics_computed'] is not False):
        raise RunConflict('fusion recovery status or source identity differs')
    predictions, targets = result['TEST_predictions'], manifest['offline_rows']
    if len(predictions) != TEST_ROWS or len(targets) != TEST_ROWS:
        raise RunConflict('fusion reconstruction has incomplete TEST coverage')
    for row, target in zip(predictions, targets):
        value = row['unsupported_score']
        if (row['sample_id'] != target['sample_id'] or row['test_index'] != target['test_index']
                or row['target_manifest_input_sha256'] != target['input_sha256']
                or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
            raise RunConflict('fusion prediction alignment or score differs')
    return {'report_sha256': FUSION_REPORT_SHA256, 'preserved_TEST_predictions': len(predictions),
            'comparison_ready': False, 'new_fits': 0,
            'note': 'Legacy inference input/checkpoint/evidence provenance remains unverified.'}


def validate_row(row, expected):
    prepared = row['prepared']
    if (set(row) != {'sample_id', 'prepared'} or set(prepared) != PREPARED_KEYS
            or row['sample_id'] != expected['sample_id']
            or prepared['input_sha256'] != expected['input_sha256']
            or prepared['messages_sha256'] != expected['messages_sha256']
            or prepared['labels'] != LABELS
            or prepared['assistant_boundary'] != ASSISTANT_BOUNDARY
            or prepared['score_position_convention'] != 'zero_based_next_token_after_complete_prompt'
            or type(prepared['input_tokens']) is not int or prepared['input_tokens'] < 1
            or type(prepared['score_position']) is not int
            or prepared['score_position'] != prepared['input_tokens']):
        raise RunConflict('token row input, mapping, boundary or length differs')
    for key in ('input_sha256', 'messages_sha256', 'rendered_prompt_sha256',
                'prompt_token_ids_sha256', 'template_sha256'):
        digest = prepared[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise RunConflict('invalid prepared token fingerprint')


def audit(examples, tokenizer, *, directory, revision, files, versions, reference_hash):
    """CPU tokenization only. Checkpoint every 25 rows and on handled interruption."""
    if not examples or len({ex.sample_id for ex in examples}) != len(examples):
        raise RunConflict('nonempty unique TEST inputs required')
    profile = load_profile('label-score-v1')
    identity = {'study_stage': 'post_thesis', 'version': 'ragtruth-test-label-token-audit-v1',
                'code_revision': revision, 'TEST_manifest_sha256': TEST_MANIFEST_SHA256,
                'judge_freeze_sha256': FREEZE_SHA256, 'fusion_report_sha256': FUSION_REPORT_SHA256,
                'prompt_version': LABEL_PROMPT.version, 'prompt_sha256': LABEL_PROMPT.sha256,
                'profile': profile, 'class_mapping': LABELS, 'tokenizer_files': files,
                'versions': versions, 'tokenizer_reference_sha256': reference_hash,
                'inputs': [{'sample_id': ex.sample_id, 'input_sha256': content_hash(asdict(ex.item)),
                            'messages_sha256': content_hash(label_messages(ex.item))} for ex in examples],
                'scope': 'RAGTruth_TEST_token_lengths_only', 'output_allowance': 1, 'truncation': 'none',
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
    validate_fit(json.loads((run_directory(FIT_RUN_ID) / 'fit.json').read_text(encoding='utf-8')), frozen)
    bundle = json.loads((run_directory(TEST_RUN_ID) / 'manifest.json').read_text(encoding='utf-8'))
    examples = validate_inputs(bundle, frozen)
    recovery = validate_recovery(json.loads((run_directory(FUSION_RUN_ID) / 'report.json').read_text(encoding='utf-8')), bundle['manifest'])
    versions = package_versions()
    tokenizer, files = load_tokenizer(run_directory('label-tokenizer-cache-v1'), download=False)
    reference_hash = verify_reference(inspect_tokenizer(tokenizer, files=files, versions=versions))
    directory = run_directory(RUN_ID)
    result, new = audit(examples, tokenizer, directory=directory, revision=revision,
                        files=files, versions=versions, reference_hash=reference_hash)
    print(json.dumps(result['audit']['summary'], indent=2))
    print('New CPU tokenizations:', new)
    print('Audit status:', result['audit']['status'])
    print('Frozen fusion verification:', json.dumps(recovery))
    print('TEST manifest SHA256:', TEST_MANIFEST_SHA256)
    print('Judge freeze SHA256:', FREEZE_SHA256)
    print('Prompt:', LABEL_PROMPT.version, LABEL_PROMPT.sha256)
    print('Audit SHA256:', result['audit_sha256'])
    print('Audit code revision:', revision)
    print('Private audit:', directory / 'audit.json')
    print('Post-thesis TEST input lengths only. No generation, fitting, metrics or HaluBench access.')
    print('Legacy comparison provenance unresolved; no scoring allowance. Separate execution plan required.')
    return 0 if result['audit']['summary']['all_inputs_fit'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
