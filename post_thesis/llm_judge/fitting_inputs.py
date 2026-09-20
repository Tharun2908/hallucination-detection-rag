"""Read-only replay of the completed reserved TRAIN scoring runs."""
import json
from contextlib import closing
import math
import sqlite3
from dataclasses import asdict
from types import SimpleNamespace

from .audit_development_tokens import ARMS, validate_inputs
from .development_scoring import load_plan, validate_preparation
from .label_diagnose import _load, _validate_server
from .prompts import content_hash
from .run_development_scores import summarize
from .runner import run_directory
from .storage import RunConflict

SCORING_REVISION = 'a341e3e805827a1122cc74dec99c66a749543572'
REPORT_HASHES = {
    'calibration': ('48a648c60f621233060e9ceb093b004c6441c60805803358bb5ec397e85f6e59',
                    'e699eaf86140f95816c63624a78cf3d506453473e2ecfa8a6096e6ac15e82de3'),
    'operating_threshold': ('2dd114803d012a963593d87ce72261dc8b805952b38ec156208168845bef72cd',
                            '8a7b86eabbffcb5b4e01a4e4043b50f754b2a8042125c944d05922809fd1f314'),
}


def read_journal(path, identity):
    # Never initialize, recover, checkpoint or write to the scoring journal.
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        stored = db.execute('SELECT manifest FROM run WHERE singleton=1').fetchone()
        if stored is None or json.loads(stored[0]) != identity:
            raise RunConflict('scoring journal identity mismatch')
        rows = [dict(row) for row in db.execute('SELECT * FROM attempts ORDER BY id')]
    for row in rows:
        if row['state'] != 'finished' or not row['finished_at']:
            raise RunConflict('unfinished scoring attempt; no recovery during fitting')
        result = json.loads(row['result'])
        if content_hash(result) != row['result_hash'] or result.get('status') != 'ok':
            raise RunConflict('corrupt or unsuccessful scoring attempt')
        row['result'] = result
    return rows


def verify_arm(arm, expected_inputs, *, artifact_root=None):
    plan = load_plan(arm)
    directory = run_directory(plan['run_id'], artifact_root)
    report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    digest = report.get('report_sha256')
    if (digest not in REPORT_HASHES[arm]
            or content_hash({k: v for k, v in report.items() if k != 'report_sha256'}) != digest):
        raise RunConflict('summary does not match either recorded completed invocation')
    ledger, requests = _load(directory / 'budget.json'), _load(directory / 'prepared.json')
    identity = ledger['identity']
    if (identity['code_revision'] != SCORING_REVISION or identity['plan'] != plan
            or identity['requests_sha256'] != content_hash(requests)):
        raise RunConflict('historical scoring identity mismatch')
    validate_preparation(identity['preparation'], requests, plan)
    if [r['sample_id'] for r in requests] != list(expected_inputs):
        raise RunConflict('request membership/order differs from reserved arm')
    for request in requests:
        if request['identity']['prepared']['input_sha256'] != expected_inputs[request['sample_id']]:
            raise RunConflict('request input differs from reserved answer/context')
    if not ledger['windows'] or ledger.get('halt_reason') is not None:
        raise RunConflict('missing or halted scoring execution')
    for window in ledger['windows']:
        if window['status'] != 'finished':
            raise RunConflict('unfinished scoring budget window')
        _validate_server(window['server_session_snapshot'], SimpleNamespace(profile=plan['profile']),
                         SCORING_REVISION, identity['preparation']['source'])
    rows = read_journal(directory / 'journal.sqlite3', identity)
    replay = summarize(requests, rows, ledger, new_attempts=report['new_attempts'])
    if replay != report:
        raise RunConflict('summary differs from read-only response replay')
    if (len(rows) != plan['request_count'] or not report['execution_complete']
            or report['valid_scores'] != plan['request_count'] or report['terminal_failures'] or report['pending']
            or report['known_token_totals'] != {'input_tokens': plan['audited_input_tokens'], 'output_tokens': plan['request_count']}
            or any(report['unknown_usage_attempts'].values())
            or not 0 <= report['charged_client_seconds'] <= plan['client_budget_seconds']):
        raise RunConflict('incomplete scoring coverage or budget/usage mismatch')
    margins = [p['score']['unsupported_log_odds'] for p in report['predictions']]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in margins):
        raise RunConflict('nonfinite or invalid score')
    return margins, {'summary_sha256': digest, 'requests_sha256': content_hash(requests),
                     'ledger_sha256': content_hash(ledger), 'journal_records_sha256': content_hash(rows),
                     'scoring_code_revision': SCORING_REVISION, 'rows': len(rows)}


def join_labels(ids, rows):
    labels = {row['sample_id']: row['label'] for row in rows}
    if (len(labels) != len(rows) or set(labels) != set(ids)
            or any(type(v) is not int or v not in (0, 1) for v in labels.values())
            or set(labels.values()) != {0, 1}):
        raise RunConflict('missing, duplicate or invalid offline labels')
    return [labels[rid] for rid in ids]


def load_arms(protocol, *, artifact_root=None):
    bundle = json.loads((run_directory('ragtruth-train-development-reservation-v1', artifact_root) /
                         'manifest.json').read_text(encoding='utf-8'))
    examples = validate_inputs(bundle, protocol)
    arms, identities = {}, {}
    for arm in ARMS:
        inputs = {ex.sample_id: content_hash(asdict(ex.item)) for ex in examples[arm]}
        margins, identities[arm] = verify_arm(arm, inputs, artifact_root=artifact_root)
        labels = join_labels(list(inputs), bundle['manifest']['labels_offline'][arm])
        arms[arm] = {'margins': margins, 'labels': labels}
        identities[arm]['aligned_scores_labels_sha256'] = content_hash({'ids': list(inputs), **arms[arm]})
    return arms, identities
