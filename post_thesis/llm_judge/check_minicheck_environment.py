"""Read-only MiniCheck runtime inventory; no downloads, GPU allocations or fits."""
import argparse
from importlib.metadata import PackageNotFoundError, distribution, version
import json
from pathlib import Path
import platform
import shutil
import subprocess

from .minicheck_contract import MODEL, MODEL_REVISION, UPSTREAM_REVISION
from .prepare_test_manifest import save_once
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import exclusive_run

PACKAGES = ('torch', 'vllm', 'transformers', 'tokenizers', 'huggingface-hub',
            'nltk', 'sentencepiece', 'numpy', 'jinja2')


def collect(cache):
    packages = {}
    for name in PACKAGES:
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    registry = {'source_file_present': False, 'InternLM2ForCausalLM_name_present': False,
                'native_runtime_compatibility_verified': False}
    if packages['vllm']:
        path = Path(distribution('vllm').locate_file('vllm/model_executor/models/registry.py'))
        if path.is_file():
            source = path.read_text(encoding='utf-8')
            registry.update(source_file_present=True,
                            InternLM2ForCausalLM_name_present='InternLM2ForCausalLM' in source)
    executable = shutil.which('nvidia-smi')
    gpu = {'status': 'nvidia-smi_unavailable', 'csv': None}
    if executable:
        result = subprocess.run([executable, '--query-gpu=index,name,memory.total,memory.used,memory.free',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
        gpu = {'status': 'ok' if result.returncode == 0 else 'query_failed',
               'csv': result.stdout.strip() if result.returncode == 0 else None}
    snapshot = cache / ('models--' + MODEL.replace('/', '--')) / 'snapshots' / MODEL_REVISION
    return {'python': platform.python_version(), 'packages': packages, 'vllm_registry': registry, 'gpu_memory_MiB': gpu,
            'root_free_GiB': round(shutil.disk_usage('/').free / 1024**3, 2),
            'workspace_free_GiB': round(shutil.disk_usage(Path.cwd()).free / 1024**3, 2),
            'model_cache': str(cache), 'pinned_snapshot_present': snapshot.is_dir(),
            'snapshot_filenames': sorted(p.name for p in snapshot.iterdir()) if snapshot.is_dir() else [],
            'cached_file_bytes_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-cache', type=Path, default=Path('/root/llm-judge-hf-cache'))
    args = parser.parse_args()
    report = {'study_stage': 'post_thesis', 'code_revision': code_revision(),
              'upstream_revision': UPSTREAM_REVISION, 'model': MODEL, 'model_revision': MODEL_REVISION,
              **collect(args.model_cache), 'model_calls': 0, 'downloads': 0, 'fitting_calls': 0,
              'GPU_allocations': 0, 'package_changes': False, 'server_stopped': False,
              'live_compatibility_verified': False}
    digest = content_hash(report)
    path = run_directory('minicheck-environment-' + digest[:16]) / 'report.json'
    with exclusive_run(path.parent):
        save_once(path, {'report_sha256': digest, 'report': report})
    print(json.dumps(report, indent=2))
    print('Report SHA256:', digest)
    print('Private environment record:', path)
    print('Environment inventory only. Tokenizer, sentence-resource and live MiniCheck compatibility remain unchecked.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
