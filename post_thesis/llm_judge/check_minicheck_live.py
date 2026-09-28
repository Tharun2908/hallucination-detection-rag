"""Six synthetic MiniCheck requests with verified files and saved tokenizer; no TEST."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

from .minicheck_contract import aggregate_support, support_from_returned_tokens
from .minicheck_inputs import verify_snapshot, explicit_tokenizer, prepare
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import Journal, RunConflict, atomic_json, exclusive_run

RUN_ID = 'minicheck-synthetic-compatibility-v2'
PREVIOUS_RUN_ID = 'minicheck-synthetic-compatibility-v1'
PREVIOUS_REPORT_SHA256 = '525e75192fb7b2e588a3b3e659228eaf30ece5478280e5c1c1a30da53c50ec89'
RUNTIME_ENV = {'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'CUDA_VISIBLE_DEVICES':'0',
               'VLLM_NO_USAGE_STATS':'1', 'DO_NOT_TRACK':'1',
               'VLLM_WORKER_MULTIPROC_METHOD':'spawn', 'VLLM_USE_FLASHINFER_SAMPLER':'0'}
VERSIONS = {'torch':'2.13.0+cu130', 'vllm':'0.29.0', 'transformers':'5.17.0',
            'tokenizers':'0.23.2', 'huggingface-hub':'1.32.0', 'nltk':'3.10.3',
            'sentencepiece':'0.2.2', 'numpy':'2.3.5', 'jinja2':'3.1.6'}
ENGINE = {'dtype':'bfloat16', 'tensor_parallel_size':1, 'seed':2024,
          'max_model_len':32768, 'enable_prefix_caching':False, 'enforce_eager':True,
          'gpu_memory_utilization':.35, 'max_num_seqs':16, 'max_num_batched_tokens':4096,
          'enable_chunked_prefill':True, 'logprobs_mode':'raw_logprobs',
          'model_impl':'vllm', 'attention_backend':'FLASH_ATTN', 'generation_config':'vllm'}


def configure_runtime():
    # Attention and sampling have independent backend selection. Set before
    # importing vLLM; spawned workers inherit the same sampling configuration.
    os.environ.update(RUNTIME_ENV)
    return dict(RUNTIME_ENV)


def failed_predecessor(path):
    bundle = json.loads(Path(path).read_text(encoding='utf-8'))
    report = bundle['report']
    if (bundle['report_sha256'] != PREVIOUS_REPORT_SHA256
            or content_hash(report) != PREVIOUS_REPORT_SHA256
            or report['result']['status'] != 'error'):
        raise RunConflict('expected the preserved v1 startup failure report')
    return {'run_id':PREVIOUS_RUN_ID, 'report_sha256':PREVIOUS_REPORT_SHA256,
            'reason':'FlashInfer sampler warm-up required missing nvcc; disable its sampler for v2'}


def sentence_resources():
    import nltk
    pointer = nltk.data.find('tokenizers/punkt_tab/english/')
    import hashlib
    files = {}
    for name in ('collocations.tab','sent_starters.txt','abbrev_types.txt','ortho_context.tab'):
        with pointer.join(name).open() as handle:
            data = handle.read()
        if isinstance(data, str): data = data.encode('utf-8')
        files[name] = {'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()}
    return nltk.sent_tokenize, files


def score_outputs(outputs, prepared):
    if len(outputs) != 6: raise RunConflict('expected six MiniCheck responses')
    raw, scores = [], []
    for response, expected in zip(outputs, prepared['prompt_token_ids']):
        if list(response.prompt_token_ids) != expected:
            raise RunConflict('vLLM prompt token IDs differ from the saved-tokenizer preparation')
        if len(response.outputs) != 1: raise RunConflict('unexpected MiniCheck output count')
        output = response.outputs[0]
        if len(output.token_ids) != 1 or not output.logprobs or len(output.logprobs) != 1:
            raise RunConflict('missing MiniCheck first-token output/logprobs')
        probs = output.logprobs[0]
        if not 5 <= len(probs) <= 6 or output.token_ids[0] not in probs:
            raise RunConflict('unexpected top-five plus emitted-token logprob coverage')
        tokens = [{'token_id':int(key), 'decoded_token':value.decoded_token,
                   'logprob':float(value.logprob)} for key,value in sorted(probs.items())]
        score = support_from_returned_tokens(tokens)
        raw.append({'prompt_token_ids':list(response.prompt_token_ids),
                    'emitted_token_id':int(output.token_ids[0]), 'text':output.text,
                    'finish_reason':output.finish_reason, 'returned_logprobs':tokens,
                    'support_score':score})
        scores.append(score)
    rows = []
    for row in prepared['rows']:
        values = scores[row['start']:row['end']]
        width = len(row['answer_sentences'])
        matrix = [values[i:i+width] for i in range(0,len(values),width)]
        if len(matrix) != len(row['chunks']): raise RunConflict('MiniCheck chunk/sentence coverage differs')
        scored = aggregate_support(matrix)
        rows.append({'sample_id':row['sample_id'], 'support_matrix':matrix, **scored,
                     'expected_supported':row['expected_supported'],
                     'upstream_label_matches_expectation':scored['upstream_supported_label']==row['expected_supported']})
    return {'status':'ok','raw_responses':raw,'rows':rows,'prompt_ids_match':True,
            'requested_generations':6,'output_tokens':6,
            'input_tokens':sum(len(ids) for ids in prepared['prompt_token_ids'])}


def execute_once(directory, identity, compute):
    with exclusive_run(directory):
        journal = Journal(directory / 'journal.sqlite3', identity)
        try:
            journal.recover()
            records = journal.records()
            if not records:
                attempt = journal.start(content_hash(identity), 1)
                started = time.perf_counter()
                try: result = compute()
                except Exception as exc: result = {'status':'error','error':type(exc).__name__+': '+str(exc)}
                result['client_seconds_including_loading'] = time.perf_counter()-started
                journal.finish(attempt,result)
                records = journal.records(); new = True
            else: new = False
            if len(records) != 1 or records[0]['request_key'] != content_hash(identity) or records[0]['ordinal'] != 1:
                raise RunConflict('unexpected MiniCheck synthetic attempts')
            stored = records[0]
            result = stored['result'] if stored['state']=='finished' else {'status':'interrupted','usage':'unknown'}
            report = {'study_stage':'post_thesis','identity':identity,'result':result,
                      'model_initialization_attempts':1,'automatic_retries':0,
                      'benchmark_inputs_read':False,'TEST_metrics_computed':False,'fitting_calls':0,
                      'scope':'synthetic compatibility only; not benchmark scoring or quality validation'}
            bundle = {'report_sha256':content_hash(report),'report':report}
            atomic_json(directory / 'report.json',bundle)
            return bundle,new
        finally: journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-cache',type=Path,default=Path('/root/llm-judge-hf-cache'))
    args = parser.parse_args()
    revision = code_revision()
    packages = {p:version(p) for p in VERSIONS}
    if packages != VERSIONS or platform.python_version_tuple()[:2] != ('3','12'):
        raise RunConflict('use the unchanged recorded judge-serving environment')
    runtime_env = configure_runtime()
    predecessor = failed_predecessor(run_directory(PREVIOUS_RUN_ID) / 'report.json')
    path, files = verify_snapshot(args.model_cache)
    directory = run_directory(RUN_ID)
    with exclusive_run(directory):
        tokenizer, tokenizer_record = explicit_tokenizer(path,directory / 'saved-tokenizer-adapter')
    split, resources = sentence_resources()
    prepared = prepare(tokenizer,split)
    print('Saved-tokenizer encoding checks:',tokenizer_record['probe_encodings_matched'])
    print('Synthetic examples / generation requests:',len(prepared['rows']),len(prepared['texts']),flush=True)
    identity = {'code_revision':revision,'packages':packages,'python':platform.python_version(),
                'files':files,'tokenizer':tokenizer_record,'punkt_english_files':resources,
                'engine':ENGINE,'runtime_environment':runtime_env,'predecessor':predecessor,
                'sampling':{'temperature':0,'max_tokens':1,'logprobs':5},
                'prepared':prepared,'max_attempts':1,'external_wall_limit_seconds':1200}

    def compute():
        import torch
        if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name(0):
            raise RunConflict('expected H200 GPU 0')
        if torch.cuda.mem_get_info(0)[0] < 64 * 1024**3:
            raise RunConflict('less than 64 GiB free; inspect other GPU processes before running MiniCheck')
        from vllm import LLM, SamplingParams
        llm = LLM(model=str(path),tokenizer=str(directory / 'saved-tokenizer-adapter'),
                  trust_remote_code=True,**ENGINE)
        live_tokenizer = llm.get_tokenizer()
        for text,ids in zip(prepared['texts'],prepared['prompt_token_ids']):
            if live_tokenizer.encode(text) != ids:
                raise RunConflict('vLLM tokenizer differs before generation')
        stops = [live_tokenizer.eos_token_id]
        eot = live_tokenizer.convert_tokens_to_ids('<|eot_id|>')
        if eot is not None: stops.append(eot)  # Literal upstream behavior; may include unk.
        params = SamplingParams(temperature=0,max_tokens=1,logprobs=5,stop_token_ids=stops)
        outputs = llm.generate(prepared['texts'],params,use_tqdm=False)
        result = score_outputs(outputs,prepared)
        result.update(stop_token_ids=stops,gpu=torch.cuda.get_device_name(0),
                      cuda=torch.version.cuda,engine_internal_profiling_calls='not counted as user generation requests')
        return result

    bundle,new = execute_once(directory,identity,compute)
    result = bundle['report']['result']
    print('Post-thesis MiniCheck synthetic compatibility')
    print('New model attempt:',new)
    print('Status:',result['status'])
    for row in result.get('rows',[]): print(json.dumps(row))
    if result.get('error'): print('Error:',result['error'])
    print('Report SHA256:',bundle['report_sha256'])
    print('Private report:',directory / 'report.json')
    print('No benchmark inputs, fitting or TEST metrics. Synthetic behavior is not benchmark performance.')
    return 0 if result['status']=='ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
