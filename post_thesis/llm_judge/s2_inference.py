"""Pinned post-thesis S2 raw relevance logits and sentence-pair provenance."""
from dataclasses import asdict
import json
import math
import os
import re
import time

from .check_s4_checkpoint import validate_loading_info
from .prepare_pilot import file_sha256
from .prompts import content_hash
from .storage import RunConflict

MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
FILES = {
    'config.json': (794, '380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc'),
    'model.safetensors': (90870598, '821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae'),
    'tokenizer_config.json': (1330, 'a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8'),
    'tokenizer.json': (711396, 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'),
    'special_tokens_map.json': (132, '3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6'),
    'vocab.txt': (231508, '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'),
}
FIELDS = ('input_ids', 'attention_mask', 'token_type_ids')


def split_sentences(text):
    # Same transformation and >=10-character filter as relevance_verifier_full_v2.py.
    if not text or not isinstance(text, str): return []
    text = text.replace('\n', ' ').strip()
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) >= 10]


def sentences(example):
    return split_sentences(example.item.answer), split_sentences(example.item.context)


def reference(example, index):
    answer, context = sentences(example)
    return {'sample_id': example.sample_id, 'test_index': index,
            'input_sha256': content_hash(asdict(example.item)),
            'sentence_inputs_sha256': content_hash({'answer': answer, 'context': context}),
            'n_answer_sentences': len(answer), 'n_context_sentences': len(context),
            'pairs': len(answer) * len(context), 'forward_batches': len(answer) * math.ceil(len(context) / 32)}


def verify_model(path):
    for name, (size, digest) in FILES.items():
        source = path / name
        if not source.is_file() or source.stat().st_size != size or file_sha256(source) != digest:
            raise RunConflict('pinned S2 model file differs: ' + name)
    config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    if (config.get('architectures') != ['BertForSequenceClassification']
            or config.get('num_hidden_layers') != 6 or config.get('max_position_embeddings') != 512
            or config.get('sbert_ce_default_activation_function') != 'torch.nn.modules.linear.Identity'):
        raise RunConflict('unexpected S2 architecture or score activation')


def aggregate(pair_batches, n_answer, n_context):
    """Validate complete ordered pair coverage and recover per-answer maxima."""
    expected = [(a, start, min(32, n_context - start))
                for a in range(n_answer) for start in range(0, n_context, 32)]
    if len(pair_batches) != len(expected): raise RunConflict('incomplete S2 pair batch coverage')
    best = [None] * n_answer
    for batch, (a, start, count) in zip(pair_batches, expected):
        if (batch['answer_index'] != a or batch['context_start'] != start
                or len(batch['logits']) != count
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in batch['logits'])):
            raise RunConflict('S2 pair order, cardinality or raw logits differ')
        maximum = max(batch['logits'])
        best[a] = maximum if best[a] is None else max(best[a], maximum)
    if not n_answer or not n_context:
        return {'raw_min_relevance': 0., 'raw_mean_relevance': 0., 'per_answer_best_logits': []}
    return {'raw_min_relevance': min(best), 'raw_mean_relevance': sum(best) / len(best),
            'per_answer_best_logits': best}


class S2Backend:
    def __init__(self, cache, *, download=False):
        from pathlib import Path
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=str(cache),
                                     allow_patterns=list(FILES), local_files_only=not download, token=False))
        verify_model(path)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .serve import observe_gpu
        self.torch = torch
        if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name(0):
            raise RunConflict('S2 profile requires an H200 at GPU 0')
        gpu = observe_gpu(0)
        if torch.cuda.mem_get_info(0)[0] < 3 * 1024 ** 3:
            raise RunConflict('S2 requires at least 3 GiB free GPU memory before loading')
        torch.set_num_threads(2)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True,
                                                       trust_remote_code=False)
        if (not self.tokenizer.is_fast or len(self.tokenizer) != 30522
                or self.tokenizer.model_max_length != 512 or self.tokenizer.padding_side != 'right'
                or self.tokenizer.truncation_side != 'right'):
            raise RunConflict('unexpected S2 tokenizer')
        self.model, info = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, attn_implementation='eager', output_loading_info=True,
            ignore_mismatched_sizes=False)
        validate_loading_info(info)
        self.model.to('cuda:0').eval()
        if (self.model.config.num_labels != 1 or self.model.config._attn_implementation != 'eager'
                or any(p.dtype != torch.float32 or p.device.type != 'cuda' for p in self.model.parameters())):
            raise RunConflict('unexpected S2 score head, precision or device')
        verify_model(path)
        self.runtime = {'gpu': gpu, 'pytorch_cuda': torch.version.cuda, 'dtype': 'float32', 'tf32': False,
                        'model_class': type(self.model).__name__, 'tokenizer_class': type(self.tokenizer).__name__,
                        'tokenizer_backend_sha256': content_hash(json.loads(self.tokenizer.backend_tokenizer.to_str())),
                        'activation': 'Identity; raw logits', 'loading_info_clean': True,
                        'timing_context': 'shared GPU; not a controlled latency comparison'}

    def score(self, example):
        torch = self.torch
        answer, context = sentences(example)
        started, forward_seconds, batches = time.perf_counter(), 0., []
        with torch.no_grad():
            for a, claim in enumerate(answer):
                for start in range(0, len(context), 32):
                    passages = context[start:start + 32]
                    claims = [claim] * len(passages)
                    full = self.tokenizer(claims, passages, padding=False, truncation=False)
                    encoded = self.tokenizer(claims, passages, padding=True, truncation='longest_first',
                                             max_length=512, return_tensors='pt')
                    fields = {key: encoded[key] for key in FIELDS}
                    coverage = []
                    for i in range(len(passages)):
                        before, after = full.sequence_ids(i), encoded.sequence_ids(i)
                        coverage.append({side: [before.count(j), after.count(j)]
                                         for j, side in enumerate(('answer', 'context'))})
                    batch_hash = content_hash({k: v.tolist() for k, v in fields.items()})
                    inputs = {k: v.to('cuda:0') for k, v in fields.items()}
                    torch.cuda.synchronize(0)
                    tick = time.perf_counter()
                    logits = self.model(**inputs).logits
                    torch.cuda.synchronize(0)
                    forward_seconds += time.perf_counter() - tick
                    if list(logits.shape) != [len(passages), 1]: raise RunConflict('unexpected S2 logits shape')
                    batches.append({'answer_index': a, 'context_start': start,
                                    'logits': logits[:, 0].cpu().tolist(),
                                    'forward_tensors_sha256': batch_hash, 'coverage': coverage})
        return {'status': 'ok', 'pair_batches': batches, **aggregate(batches, len(answer), len(context)),
                'example_seconds': time.perf_counter() - started, 'cuda_forward_seconds': forward_seconds,
                'runtime': self.runtime}
