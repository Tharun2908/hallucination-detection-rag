"""Fresh MiniCheck inference using the live-verified synthetic configuration."""
from dataclasses import asdict
import math
import time

from .check_minicheck_live import ENGINE
from .minicheck_contract import aggregate_support, support_from_returned_tokens
from .minicheck_inputs import prepare_item
from .prompts import content_hash
from .storage import RunConflict


def reference(example, index, prepared):
    return {'sample_id':example.sample_id, 'test_index':index,
            'input_sha256':content_hash(asdict(example.item)),
            'context_chunks':prepared['chunks'], 'answer_sentences':prepared['answer_sentences'],
            'prompts':[{'text_sha256':content_hash(text), 'token_ids_sha256':content_hash(ids),
                        'input_tokens':len(ids)}
                       for text,ids in zip(prepared['texts'],prepared['prompt_token_ids'])]}


def decode_outputs(outputs, ref):
    if len(outputs) != len(ref['prompts']): raise RunConflict('missing MiniCheck responses')
    raw = []
    for response, prompt in zip(outputs, ref['prompts']):
        if (response.prompt_token_ids is None
                or content_hash(list(response.prompt_token_ids)) != prompt['token_ids_sha256']):
            raise RunConflict('MiniCheck returned prompt IDs differ from preparation')
        if len(response.outputs) != 1: raise RunConflict('unexpected MiniCheck output count')
        output = response.outputs[0]
        if len(output.token_ids) != 1 or not output.logprobs or len(output.logprobs) != 1:
            raise RunConflict('missing MiniCheck first-position token/logprobs')
        tokens = [{'token_id':int(key), 'decoded_token':value.decoded_token,
                   'logprob':float(value.logprob)} for key,value in sorted(output.logprobs[0].items())]
        raw.append({'prompt_token_ids_sha256':prompt['token_ids_sha256'],
                    'emitted_token_id':int(output.token_ids[0]), 'text':output.text,
                    'finish_reason':output.finish_reason, 'returned_logprobs':tokens})
    return raw


def aggregate_responses(raw, ref):
    expected = len(ref['context_chunks']) * len(ref['answer_sentences'])
    if not expected or len(raw) != expected or len(ref['prompts']) != expected:
        raise RunConflict('MiniCheck chunk/sentence response coverage differs')
    values = []
    for row,prompt in zip(raw,ref['prompts']):
        tokens = row['returned_logprobs']
        ids = [t['token_id'] for t in tokens]
        if (not 5 <= len(tokens) <= 6 or len(set(ids)) != len(ids)
                or any(type(i) is not int or i < 0 for i in ids)
                or row['emitted_token_id'] not in ids
                or row['prompt_token_ids_sha256'] != prompt['token_ids_sha256']):
            raise RunConflict('invalid MiniCheck logprob or prompt coverage')
        values.append(support_from_returned_tokens(tokens))
    width = len(ref['answer_sentences'])
    matrix = [values[i:i+width] for i in range(0,len(values),width)]
    return {'support_matrix':matrix, **aggregate_support(matrix)}


def validate_result(result, ref):
    if result.get('status') == 'error':
        if not isinstance(result.get('error'),str): raise RunConflict('invalid MiniCheck error')
        return
    if result.get('status') != 'ok': raise RunConflict('invalid MiniCheck status')
    computed = aggregate_responses(result['raw_responses'],ref)
    if any(result.get(k) != v for k,v in computed.items()):
        raise RunConflict('MiniCheck scores differ from saved logprobs')
    seconds = result['example_seconds']
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds < 0:
        raise RunConflict('invalid MiniCheck timing')


class MiniCheckBackend:
    def __init__(self, snapshot, adapter, tokenizer, split):
        import torch
        from vllm import LLM, SamplingParams
        if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name(0):
            raise RunConflict('expected H200 GPU 0')
        if torch.cuda.mem_get_info(0)[0] < 64 * 1024**3:
            raise RunConflict('less than 64 GiB GPU memory free')
        self.llm = LLM(model=str(snapshot),tokenizer=str(adapter),trust_remote_code=True,**ENGINE)
        self.live_tokenizer = self.llm.get_tokenizer()
        self.tokenizer, self.split = tokenizer, split
        stops = [self.live_tokenizer.eos_token_id]
        eot = self.live_tokenizer.convert_tokens_to_ids('<|eot_id|>')
        if eot is not None: stops.append(eot)
        self.params = SamplingParams(temperature=0,max_tokens=1,logprobs=5,stop_token_ids=stops)
        self.runtime = {'gpu':torch.cuda.get_device_name(0), 'cuda':torch.version.cuda,
                        'stop_token_ids':stops, 'attention':'FLASH_ATTN', 'flashinfer_sampler':False}

    def score(self, example, ref):
        started = time.perf_counter()
        item = prepare_item(example.item.answer,example.item.context,self.tokenizer,self.split)
        if reference(example,ref['test_index'],item) != ref:
            raise RunConflict('MiniCheck inputs changed after preparation')
        for text,ids in zip(item['texts'],item['prompt_token_ids']):
            if self.live_tokenizer.encode(text) != ids:
                raise RunConflict('live MiniCheck tokenizer differs before generation')
        outputs = self.llm.generate(item['texts'],self.params,use_tqdm=False)
        raw = decode_outputs(outputs,ref)
        return {'status':'ok', 'raw_responses':raw, **aggregate_responses(raw,ref),
                'example_seconds':time.perf_counter()-started, 'runtime':self.runtime}
