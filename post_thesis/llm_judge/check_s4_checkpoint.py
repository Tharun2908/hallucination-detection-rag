"""Post-thesis local S4 load check: four artificial pairs, CPU only."""
import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform

from .prepare_pilot import file_sha256
from .prepare_test_manifest import save_once
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID = 's4-checkpoint-compatibility-v1'
# Operator-supplied on-pod fingerprints, 2026-09-20. These identify the current
# checkpoint; they do not establish which checkpoint produced a legacy cache.
FILES = {
    'config.json': (1049, '91ccff48161f4ea197df2902ae4958ae19a387dfb54f72fb2d7f28787abae82a'),
    'model.safetensors': (737719248, 'd2a0d5b7ce847ce8c5f1fc591b63e780eacf1d548365c4129fe490a882cb8c7a'),
    'tokenizer.json': (8340140, 'ccab0d2b1c2aec656e37e1f5a3bc8a71e64186f221717059852bfc1c99829d59'),
    'tokenizer_config.json': (476, 'a44cc773b4bf4a423f3eee743630eca58709e3b84c934232ea794f7b8fdd0c2f'),
}
VERSIONS = {'transformers': '5.17.0', 'tokenizers': '0.23.2', 'torch': '2.13.0+cu130'}
PAIR_POLICY = {'order': ['answer', 'context'], 'max_length': 512,
               'truncation': 'longest_first', 'padding': 'max_length',
               'forward_fields': ['input_ids', 'attention_mask'],
               'device': 'cpu', 'dtype': 'float32', 'batch_size': 1,
               'unsupported_class_index': 1}


def fixtures():
    return [
        {'sample_id': 'supported', 'answer': 'The venue has outdoor seating.',
         'context': 'The venue has outdoor seating.'},
        {'sample_id': 'contradicted', 'answer': 'The venue has outdoor seating.',
         'context': 'The venue does not have outdoor seating.'},
        {'sample_id': 'long_answer', 'answer': 'The venue has outdoor seating. ' * 100,
         'context': 'The venue has outdoor seating.'},
        {'sample_id': 'long_context', 'answer': 'The venue has outdoor seating.',
         'context': 'The venue has outdoor seating. ' * 100},
    ]


def verify_checkpoint(directory):
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != set(FILES):
        raise RunConflict('expected the four recorded final S4 files; do not borrow fold tokenizer files')
    observed = {}
    for name, (size, digest) in FILES.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size or file_sha256(path) != digest:
            raise RunConflict('S4 checkpoint fingerprint mismatch: ' + name)
        observed[name] = {'bytes': size, 'sha256': digest}
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if (config.get('model_type') != 'deberta-v2'
            or config.get('architectures') != ['DebertaV2ForSequenceClassification']
            or config.get('id2label') != {'0': 'FAITHFUL', '1': 'HALLUCINATED'}
            or config.get('vocab_size') != 128100
            or config.get('max_position_embeddings') != 512):
        raise RunConflict('unexpected S4 architecture or class mapping')
    return observed


def validate_loading_info(info):
    required = {'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'}
    if not isinstance(info, dict) or not required.issubset(info) or any(info.values()):
        raise RunConflict('S4 weights did not load cleanly: ' + repr(info))


def validate_logits(rows):
    if (len(rows) != 1 or len(rows[0]) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in rows[0])):
        raise RunConflict('expected one finite two-class S4 logit row')


def cpu_probe(directory):
    # Applied before importing the frameworks, to this process only.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False, use_fast=True)
    saved_tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    saved_tokenizer.no_truncation()
    saved_tokenizer.no_padding()
    if (not tokenizer.is_fast or len(tokenizer) != saved_tokenizer.get_vocab_size(with_added_tokens=True)
            or not 0 < len(tokenizer) <= 128100
            or tokenizer.padding_side != 'right' or tokenizer.truncation_side != 'right'):
        raise RunConflict('unexpected S4 tokenizer vocabulary, backend or truncation/padding side')
    backend_hash = content_hash(json.loads(tokenizer.backend_tokenizer.to_str()))
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, ignore_mismatched_sizes=False, output_loading_info=True,
        dtype=torch.float32)
    validate_loading_info(loading_info)
    model.to('cpu').eval()
    if (model.config.num_labels != 2
            or model.config.id2label != {0: 'FAITHFUL', 1: 'HALLUCINATED'}
            or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters())):
        raise RunConflict('unexpected loaded S4 mapping, device or parameter dtype')
    results = []
    with torch.no_grad():
        for item in fixtures():
            full = tokenizer(item['answer'], item['context'], truncation=False, padding=False)
            saved = saved_tokenizer.encode(item['answer'], item['context'])
            if full['input_ids'] != saved.ids or full.sequence_ids() != saved.sequence_ids:
                raise RunConflict('loaded tokenizer changed the saved tokenizer pair encoding')
            encoded = tokenizer(item['answer'], item['context'], max_length=512,
                                truncation=True, padding='max_length', return_tensors='pt')
            full_ids, kept_ids = full.sequence_ids(), encoded.sequence_ids()
            fields = {key: encoded[key] for key in PAIR_POLICY['forward_fields']}
            if list(fields['input_ids'].shape) != [1, 512]:
                raise RunConflict('S4 pair encoding did not produce the required padded shape')
            if any(not 0 <= token < 128100 for token in fields['input_ids'][0].tolist()):
                raise RunConflict('S4 tokenizer produced an out-of-vocabulary model input')
            logits = model(**fields).logits
            values = logits.tolist()
            validate_logits(values)
            score = torch.softmax(logits, dim=1)[0, 1].item()
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise RunConflict('invalid S4 unsupported score')
            coverage = {name: {'full_tokens': full_ids.count(i), 'kept_tokens': kept_ids.count(i)}
                        for i, name in enumerate(('answer', 'context'))}
            if any(v['kept_tokens'] > v['full_tokens'] for v in coverage.values()):
                raise RunConflict('inconsistent pair token coverage')
            results.append({'sample_id': item['sample_id'], 'input_sha256': content_hash(item),
                            'forward_tensors_sha256': content_hash({k: v.tolist() for k, v in fields.items()}),
                            'coverage': coverage, 'logits': values[0], 'unsupported_score': score})
    for item, side in ((results[2], 'answer'), (results[3], 'context')):
        if item['coverage'][side]['kept_tokens'] >= item['coverage'][side]['full_tokens']:
            raise RunConflict('long synthetic fixture did not exercise pair truncation')
    return {'status': 'passed', 'synthetic_forward_calls': 4,
            'tokenizer_class': type(tokenizer).__name__, 'tokenizer_backend_sha256': backend_hash,
            'tokenizer_vocabulary_size': len(tokenizer), 'saved_tokenizer_pair_matches': 4,
            'model_class': type(model).__name__, 'loading_info': {k: [] for k in loading_info},
            'tokenizer_padding_side': tokenizer.padding_side,
            'tokenizer_truncation_side': tokenizer.truncation_side,
            'cpu_threads': torch.get_num_threads(), 'results': results}


def check_once(directory, checkpoint, revision, versions, probe=cpu_probe):
    observed = verify_checkpoint(checkpoint)
    identity = {'code_revision': revision, 'checkpoint_files': observed, 'versions': versions,
                'pair_policy': PAIR_POLICY, 'fixtures_sha256': content_hash(fixtures())}
    path = Path(directory) / 'report.json'
    with exclusive_run(Path(directory)):
        if path.exists():
            bundle = json.loads(path.read_text(encoding='utf-8'))
            if (bundle.get('report_sha256') != content_hash(bundle['report'])
                    or bundle['report']['identity'] != identity):
                raise RunConflict('S4 compatibility record changed; preserve the existing run')
            return bundle, False
        result = probe(Path(checkpoint).resolve(strict=True))
        # Detect changes to the checkpoint during loading before recording success.
        if verify_checkpoint(checkpoint) != observed:
            raise RunConflict('S4 files changed during the compatibility check')
        report = {'study_stage': 'post_thesis', 'run_id': RUN_ID, 'identity': identity,
                  'check': result, 'benchmark_inputs_read': False, 'gpu_inference_calls': 0,
                  'judge_calls': 0, 'fitting_calls': 0, 'comparison_ready': False,
                  'legacy_cache_to_checkpoint_link_verified': False,
                  'scope': 'current_checkpoint_CPU_compatibility_only'}
        bundle = {'report_sha256': content_hash(report), 'report': report}
        save_once(path, bundle)
        return bundle, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('/workspace/signal4_model'))
    args = parser.parse_args()
    revision = code_revision()
    versions = {'python': platform.python_version(),
                **{p: version(p) for p in (*VERSIONS, 'safetensors', 'huggingface-hub')}}
    if (platform.python_version_tuple()[:2] != ('3', '12')
            or any(versions[p] != expected for p, expected in VERSIONS.items())):
        raise RunConflict('use the existing .venv-judge-serving environment; do not install or upgrade packages')
    print('Checking final S4 checkpoint fingerprints, then four synthetic CPU forwards...', flush=True)
    bundle, created = check_once(run_directory(RUN_ID), args.checkpoint, revision, versions)
    report = bundle['report']
    print('Post-thesis S4 CPU compatibility:', report['check']['status'])
    print('New synthetic forward calls:', 4 if created else 0)
    print('Versions:', json.dumps(versions, sort_keys=True))
    print('Check:', json.dumps(report['check'], indent=2))
    print('Report SHA256:', bundle['report_sha256'])
    print('Private report:', run_directory(RUN_ID) / 'report.json')
    print('No benchmark inputs, GPU inference, judge calls or fitting. Legacy inference provenance remains unverified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
