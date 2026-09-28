"""Preserve canonical HaluBench 8k TEST inputs; CPU only, no inference or metrics."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

from .check_frozen_fit import FIT_RUN_ID, FREEZE_SHA256, load_freeze, validate_fit
from .judge import JudgeInput
from .prepare_pilot import file_sha256
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, atomic_json, exclusive_run

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'halubench-canonical-test-input-manifest-v1'
DATASET = 'PatronusAI/HaluBench'
REVISION = '5966a87929f51c204ab3cbef986b449495cc97b6'
DATA_FILE = 'data/test-00000-of-00001.parquet'
DATA_SHA256 = 'c7e9cf966085ffae88d2947744418a05a26ea94380c35c238b9fc12ecb874cdc'
SPLIT_PATH = 'results/cross_domain/halubench_groupfix/halubench_group_split.json'
SPLIT_SHA256 = '9d78af5b4623893cfc7d761dc6be4c2e0f64278faaba11b2d10e6f829e67bb4b'
CACHE_FILES = {
    'S2_S4': ('results/cross_domain/halubench_final_s2s4_scores.json',
              '8c95c2f4e6c90d80792f4d301372227c06d1fb4103fe8b704a5ba38c25b8edf9',
              ('raw_min_relevance', 's2_support', 's4_score')),
    'MiniCheck_7B': ('results/cross_domain/halubench_per_example_scores.json',
                    '86b6d86eebd04fe69a458934dcc4538f48fdc4687a985564f1a1d70bb793a5d7', ('mc_hall',)),
}


def norm(text):
    if type(text) is not str:
        raise RunConflict('expected string source/question/passage')
    return re.sub(r'\s+', ' ', text).strip()


def canonical_group(row):
    """Same case-sensitive normalization and SHA1 as the canonical split code."""
    text = ' ||| '.join(norm(row[k]) for k in ('source_ds', 'question', 'passage'))
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def checked_json(path, sha):
    if file_sha256(path) != sha:
        raise RunConflict('pinned file changed: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def validate_split(split, total=14000, test_size=8000):
    train, test = split['train_filtered_indices'], split['test_filtered_indices']
    for indices, size in ((train, total-test_size), (test, test_size)):
        if (len(indices) != size or any(type(i) is not int for i in indices)
                or len(set(indices)) != len(indices)):
            raise RunConflict('canonical split sizes, index types or uniqueness differ')
    if set(train) & set(test) or set(train) | set(test) != set(range(total)):
        raise RunConflict('canonical split must partition the filtered dataset exactly')
    return train, test


def project_parquet(path, split):
    """Adaptation answers/labels never enter Python records or the manifest.

    Arrow decodes the answer/label columns to select TEST rows, so this is not
    a claim that physical parquet I/O never touches adaptation values.
    """
    if file_sha256(path) != DATA_SHA256:
        raise RunConflict('expected pinned HaluBench parquet bytes before loading')
    import pyarrow.parquet as pq
    metadata = pq.read_table(path, columns=['id', 'source_ds', 'question', 'passage']).to_pylist()
    if len(metadata) != 14900:
        raise RunConflict('expected all 14900 upstream rows')
    filtered = [dict(row, upstream_index=i) for i, row in enumerate(metadata) if row['source_ds'] != 'RAGTruth']
    if len(filtered) != 14000:
        raise RunConflict('expected 14000 rows after excluding source_ds == RAGTruth')
    _, test = validate_split(split)
    selected = pq.read_table(path, columns=['answer', 'label']).take(
        [filtered[i]['upstream_index'] for i in test]).to_pylist()
    return filtered, {i: row for i, row in zip(test, selected)}


def group_components(metadata):
    """Canonical groups merged by exact nonempty passages, without changing split."""
    parent = list(range(len(metadata)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def merge(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    groups, passages, canonical = {}, {}, []
    for i, row in enumerate(metadata):
        key = canonical_group(row); canonical.append(key)
        if key in groups: merge(i, groups[key])
        else: groups[key] = i
        passage = row['passage']
        if passage.strip():
            if passage in passages: merge(i, passages[passage])
            else: passages[passage] = i
    members = defaultdict(list)
    for i, row in enumerate(metadata): members[find(i)].append(row['id'])
    components = {i: content_hash(sorted(members[find(i)])) for i in range(len(metadata))}
    return canonical, components


def inspect_caches(caches, offline, total):
    reports = {}
    for name, rows in caches.items():
        by_index = {}
        for row in rows:
            i = row.get('idx')
            if type(i) is not int or i in by_index:
                raise RunConflict('legacy duplicate or invalid index: ' + name)
            by_index[i] = row
        if set(by_index) != set(range(total)):
            raise RunConflict('legacy index coverage differs: ' + name)
        fields = CACHE_FILES[name][2]
        for target in offline:
            row = by_index[target['filtered_index']]
            if (type(row.get('label')) is not int or row['label'] != target['label']
                    or row.get('source') != target['metadata']['source']):
                raise RunConflict('legacy TEST label/source mismatch: ' + name)
            for key in fields:
                score = row.get(key)
                if (type(score) not in (int, float) or not math.isfinite(score)
                        or (key != 'raw_min_relevance' and not 0 <= score <= 1)):
                    raise RunConflict('invalid legacy TEST score: ' + name + '/' + key)
        reports[name] = {'rows': total, 'selected_TEST_rows': len(offline),
                         'selected_index_label_source_alignment': True,
                         'answer_context_alignment_verified': False, 'comparison_ready': False,
                         'score_fields_checked': list(fields),
                         'status': 'legacy_index_alignment_only',
                         'limitation': 'Caches have no original IDs or answer/context hashes; fresh input hashes do not attest historical inference.'}
    return reports


def build_manifest(metadata, selected, split, caches, *, revision, total=14000, test_size=8000):
    train, test = validate_split(split, total, test_size)
    if len(metadata) != total or set(selected) != set(test):
        raise RunConflict('filtered metadata or selected TEST membership differs')
    ids = [row['id'] for row in metadata]
    upstream = [row['upstream_index'] for row in metadata]
    if (any(type(s) is not str or not s for s in ids) or len(set(ids)) != total
            or any(type(i) is not int or i < 0 for i in upstream) or upstream != sorted(set(upstream))
            or any(row['source_ds'] == 'RAGTruth' for row in metadata)):
        raise RunConflict('upstream IDs, row order or filter differs')
    canonical, components = group_components(metadata)
    if {canonical[i] for i in train} & {canonical[i] for i in test}:
        raise RunConflict('canonical normalized groups cross the saved outer split')
    shared = {components[i] for i in train} & {components[i] for i in test}
    model_inputs, offline = [], []
    for position, i in enumerate(test):
        row, target = metadata[i], selected[i]
        if target['label'] not in ('PASS', 'FAIL'):
            raise RunConflict('unexpected HaluBench label')
        item = JudgeInput(answer=target['answer'], context=row['passage'])
        model_inputs.append({'sample_id': row['id'], **asdict(item)})
        offline.append({'sample_id': row['id'], 'test_index': position, 'filtered_index': i,
                        'upstream_index': row['upstream_index'], 'canonical_group_sha1': canonical[i],
                        'component_sha256': components[i], 'input_sha256': content_hash(asdict(item)),
                        'answer_sha256': content_hash(item.answer), 'context_sha256': content_hash(item.context),
                        'question_sha256': content_hash(row['question']),
                        'label': int(target['label'] == 'FAIL'), 'metadata': {'source': row['source_ds']}})
    counts = dict(sorted(Counter(r['metadata']['source']+'__'+str(r['label']) for r in offline).items()))
    if counts != split['actual_test_counts']:
        raise RunConflict('TEST source/label counts disagree with the saved split')
    if set(caches) != set(CACHE_FILES): raise RunConflict('expected both pinned legacy caches')
    inventory = inspect_caches(caches, offline, total)
    payload = {
        'study_stage': 'post_thesis', 'version': RUN_ID, 'code_revision': revision,
        'judge_freeze_sha256': FREEZE_SHA256,
        'dataset': {'repository': DATASET, 'revision': REVISION, 'file': DATA_FILE, 'sha256': DATA_SHA256},
        'canonical_split': {'path': SPLIT_PATH, 'sha256': SPLIT_SHA256, 'membership_changed': False},
        'legacy_cache_files': {k: {'path': v[0], 'sha256': v[1]} for k,v in CACHE_FILES.items()},
        'grouping': {'canonical': 'source_ds + normalized question + normalized passage (case-sensitive)',
                     'bootstrap': 'canonical groups merged by exact nonempty passage equality',
                     'strict_document_disjointness_established': False},
        'model_inputs': model_inputs, 'offline_rows': offline, 'baseline_inventory': inventory,
        'summary': {'filtered_rows': total, 'test_rows': len(offline),
                    'canonical_test_groups': len({canonical[i] for i in test}),
                    'test_bootstrap_components': len({components[i] for i in test}),
                    'canonical_outer_group_overlap': 0,
                    'exact_passage_merged_components_touching_adaptation': len(shared),
                    'test_rows_in_components_touching_adaptation': sum(components[i] in shared for i in test),
                    'source_label_counts_offline': counts,
                    'adaptation_metadata_used_for_group_check': True,
                    'adaptation_answers_or_labels_used': False,
                    'generation_calls': 0, 'http_requests': 0, 'fitting_calls': 0,
                    'metrics_computed': False, 'token_audit_completed': False, 'scoring_authorized': False,
                    'baseline_comparison_ready': False}}
    return {'manifest_sha256': content_hash(payload), 'manifest': payload}


def save_once(path, bundle):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != bundle:
            raise RunConflict('existing HaluBench manifest differs; preserve original')
        return False
    atomic_json(path, bundle)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, required=True)
    args = parser.parse_args()
    revision = code_revision()
    frozen = load_freeze()
    fit = json.loads((run_directory(FIT_RUN_ID)/'fit.json').read_text(encoding='utf-8'))
    validate_fit(fit, frozen)
    split = checked_json(ROOT/SPLIT_PATH, SPLIT_SHA256)
    caches = {k: checked_json(ROOT/v[0], v[1]) for k,v in CACHE_FILES.items()}
    metadata, selected = project_parquet(args.parquet, split)
    bundle = build_manifest(metadata, selected, split, caches, revision=revision)
    directory = run_directory(RUN_ID)
    with exclusive_run(directory):
        created = save_once(directory/'manifest.json', bundle)
    print('Created canonical HaluBench TEST manifest.' if created else 'Identical HaluBench manifest verified; no rewrite.')
    print(json.dumps(bundle['manifest']['summary'], indent=2))
    print('Baseline inventory:', json.dumps(bundle['manifest']['baseline_inventory'], indent=2))
    print('Manifest SHA256:', bundle['manifest_sha256'])
    print('Code revision:', revision)
    print('Private manifest:', directory/'manifest.json')
    print('Post-thesis canonical 8k only. No new split, model calls, fitting, metrics or scoring allowance.')
    print('Legacy baseline text/checkpoint provenance remains unresolved; fresh hashes do not prove historical inputs.')
    return 0


if __name__ == '__main__': raise SystemExit(main())
