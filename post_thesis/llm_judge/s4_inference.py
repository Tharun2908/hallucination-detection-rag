"""Fresh post-thesis S4 inference with explicit effective-input records."""
from dataclasses import asdict
import json
import math
import os
import time

from .check_s4_checkpoint import validate_loading_info, verify_checkpoint
from .prompts import content_hash
from .storage import RunConflict

TOKENIZER_SHA256 = '896dbf60bcf96848ef756a84975905a072b156ac2d17082b7ac0077593afc5fc'


def prepare_pair(tokenizer, example):
    item = example.item
    full = tokenizer(item.answer, item.context, truncation=False, padding=False)
    encoded = tokenizer(item.answer, item.context, max_length=512, truncation=True,
                        padding='max_length')
    fields = {k: encoded[k] for k in ('input_ids', 'attention_mask')}
    if (any(len(v) != 512 for v in fields.values())
            or any(type(v) is not int or not 0 <= v < 128100 for v in fields['input_ids'])
            or any(type(v) is not int or v not in (0, 1) for v in fields['attention_mask'])):
        raise RunConflict('invalid S4 forward input')
    before, after = full.sequence_ids(), encoded.sequence_ids()
    coverage = {name: {'full_tokens': before.count(i), 'kept_tokens': after.count(i)}
                for i, name in enumerate(('answer', 'context'))}
    if any(v['kept_tokens'] > v['full_tokens'] for v in coverage.values()):
        raise RunConflict('S4 token coverage is inconsistent')
    record = {'sample_id': example.sample_id, 'input_sha256': content_hash(asdict(item)),
              # Includes all 512 token IDs and mask values, including padding.
              'forward_tensors_sha256': content_hash({k: [v] for k, v in fields.items()}),
              'nonpadding_tokens': sum(fields['attention_mask']), 'coverage': coverage}
    return fields, record


class S4Backend:
    """One FP32 H200 model; no downloads, generation, retries or label input."""
    def __init__(self, checkpoint):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .serve import observe_gpu

        self.torch = torch
        if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name(0):
            raise RunConflict('this S4 inference profile requires GPU 0 to be an H200')
        gpu = observe_gpu(0)
        free, total = torch.cuda.mem_get_info(0)
        if free < 3 * 1024 ** 3:
            raise RunConflict('less than 3 GiB GPU memory free; do not change the batch size or precision')
        torch.set_num_threads(2)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=False, use_fast=True)
        if (not self.tokenizer.is_fast or len(self.tokenizer) != 128001
                or self.tokenizer.padding_side != 'right' or self.tokenizer.truncation_side != 'right'
                or content_hash(json.loads(self.tokenizer.backend_tokenizer.to_str())) != TOKENIZER_SHA256):
            raise RunConflict('S4 tokenizer differs from the completed CPU compatibility check')
        self.model, info = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=False, use_safetensors=True,
            ignore_mismatched_sizes=False, output_loading_info=True, dtype=torch.float32,
            attn_implementation='eager')
        validate_loading_info(info)
        self.model.to('cuda:0').eval()
        if (self.model.config.id2label != {0: 'FAITHFUL', 1: 'HALLUCINATED'}
                or self.model.config.num_labels != 2
                or self.model.config._attn_implementation != 'eager'
                or any(p.device.type != 'cuda' or p.dtype != torch.float32 for p in self.model.parameters())):
            raise RunConflict('unexpected S4 class mapping, device or precision')
        verify_checkpoint(checkpoint)
        self.metadata = {'gpu': gpu, 'free_gpu_bytes_before_load': free,
                         'total_gpu_bytes': total, 'dtype': 'float32', 'tf32': False,
                         'pytorch_cuda': torch.version.cuda, 'model_class': type(self.model).__name__,
                         'loading_info_clean': True,
                         'attention_implementation': self.model.config._attn_implementation,
                         'tokenizer_backend_sha256': TOKENIZER_SHA256,
                         'timing_context': 'GPU sharing is not controlled; not an isolated latency benchmark'}

    def score(self, examples):
        torch = self.torch
        started = time.perf_counter()
        prepared = [prepare_pair(self.tokenizer, example) for example in examples]
        tensors = {key: torch.tensor([p[0][key] for p in prepared], dtype=torch.long, device='cuda:0')
                   for key in ('input_ids', 'attention_mask')}
        torch.cuda.synchronize(0)
        forward_started = time.perf_counter()
        with torch.no_grad():
            logits = self.model(**tensors).logits
            scores = torch.softmax(logits, dim=1)[:, 1]
        torch.cuda.synchronize(0)
        forward_seconds = time.perf_counter() - forward_started
        values, probabilities = logits.cpu().tolist(), scores.cpu().tolist()
        if len(values) != len(examples) or any(len(row) != 2 for row in values):
            raise RunConflict('S4 returned unexpected logit shape')
        rows = []
        for (_, record), row, score in zip(prepared, values, probabilities):
            if any(not math.isfinite(v) for v in row) or not math.isfinite(score) or not 0 <= score <= 1:
                raise RunConflict('S4 returned nonfinite logits or an invalid score')
            rows.append({**record, 'logits': row, 'unsupported_score': score})
        return {'status': 'ok', 'rows': rows, 'batch_seconds': time.perf_counter() - started,
                'cuda_forward_seconds': forward_seconds, 'runtime': self.metadata}
