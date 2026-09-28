"""Pinned MiniCheck files and explicit saved-tokenizer compatibility adapter."""
import json
from pathlib import Path

from .minicheck_contract import MODEL, MODEL_REVISION, DEFAULT_CHUNK_SIZE
from .prepare_pilot import file_sha256
from .prompts import content_hash
from .storage import RunConflict

FILES_PATH = Path(__file__).with_name('minicheck_files_v1.json')
SYSTEM = ('Determine whether the provided claim is consistent with the corresponding document. '
          'Consistency in this context implies that all information presented in the claim is substantiated '
          'by the document. If not, it should be considered inconsistent. Please assess the claim\'s '
          'consistency with the document by responding with either "Yes" or "No".')
PROBES = ['Hello world.', 'The sample contains four units.', 'Café costs 3.50 euros.',
          'first\nsecond', '<s>Hello', 'A long document sentence. ' * 200]
FIXTURES = [
    {'sample_id': 'single-supported', 'context': 'The sample contains four units. Its color is blue.',
     'answer': 'The sample contains four units.', 'expected_supported': True},
    {'sample_id': 'single-contradicted', 'context': 'The sample contains four units. Its color is blue.',
     'answer': 'The sample contains five units.', 'expected_supported': False},
    {'sample_id': 'multi-supported', 'context': 'The sample contains four units. Its color is blue.',
     'answer': 'The sample contains four units. Its color is blue.', 'expected_supported': True},
    {'sample_id': 'multi-mixed', 'context': 'The sample contains four units. Its color is blue.',
     'answer': 'The sample contains four units. Its color is red.', 'expected_supported': False},
]


def verify_snapshot(cache):
    manifest = json.loads(FILES_PATH.read_text(encoding='utf-8'))
    if manifest['model'] != MODEL or manifest['revision'] != MODEL_REVISION:
        raise RunConflict('MiniCheck file manifest identity differs')
    path = Path(cache) / ('models--' + MODEL.replace('/', '--')) / 'snapshots' / MODEL_REVISION
    for name, expected in manifest['files'].items():
        source = path / name
        print('Verifying cached MiniCheck file:', name, flush=True)
        if (not source.is_file() or source.stat().st_size != expected['bytes']
                or file_sha256(source) != expected['sha256']):
            raise RunConflict('missing or changed pinned MiniCheck file: ' + name)
    index = json.loads((path / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    if set(index['weight_map'].values()) != {n for n in manifest['files'] if n.endswith('.safetensors')}:
        raise RunConflict('MiniCheck shard index differs')
    return path, manifest


def explicit_tokenizer(path, destination):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    config = json.loads((path / 'tokenizer_config.json').read_text(encoding='utf-8'))
    raw = Tokenizer.from_file(str(path / 'tokenizer.json'))
    # The serialized file retains an incidental 869-token truncation setting.
    # Ordinary HF encode(truncation=False) disables it; mirror that explicitly.
    raw.no_truncation()
    raw.no_padding()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='</s>',
        chat_template=config['chat_template'], padding_side='left', clean_up_tokenization_spaces=False)
    if len(tokenizer) != 92550 or tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
        raise RunConflict('MiniCheck saved tokenizer special IDs/vocabulary differ')
    destination = Path(destination)
    if not destination.exists():
        destination.mkdir(parents=True)
        tokenizer.save_pretrained(destination)
    loaded = AutoTokenizer.from_pretrained(str(destination), local_files_only=True, trust_remote_code=False)
    probes = PROBES + [tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': 'Document: Test.\nClaim: Test.'}],
        tokenize=False, add_generation_prompt=True)]
    for text in probes:
        for special in (False, True):
            expected = raw.encode(text, add_special_tokens=special).ids
            if (tokenizer.encode(text, add_special_tokens=special) != expected
                    or loaded.encode(text, add_special_tokens=special) != expected):
                raise RunConflict('saved tokenizer/adapter/reloaded tokens differ')
    if loaded.chat_template != config['chat_template']:
        raise RunConflict('MiniCheck chat template differs')
    if json.loads(loaded.backend_tokenizer.to_str()) != json.loads(raw.to_str()):
        raise RunConflict('MiniCheck adapter changes the saved backend beyond disabling transient truncation/padding')
    return loaded, {'class': type(loaded).__name__, 'probe_encodings_matched': len(probes)*2,
                    'vocabulary_size': len(loaded), 'chat_template_sha256': content_hash(loaded.chat_template),
                    'backend_sha256': content_hash(json.loads(loaded.backend_tokenizer.to_str())),
                    'adapter_files': {p.name: file_sha256(p) for p in sorted(destination.iterdir()) if p.is_file()}}


def document_chunks(document, tokenizer, sent_tokenize, size=DEFAULT_CHUNK_SIZE):
    # Preserve the pinned LLMCheck newline/sentence joins, including its filtering.
    sentences = []
    for block in document.split('\n'):
        sentences.extend(sent_tokenize(block)); sentences.append('\n')
    sentences = sentences[:-1] or ['']
    pieces, current, count = [], [], 0
    for sentence in sentences:
        length = len(tokenizer(sentence, add_special_tokens=False)['input_ids'])
        if count + length > size:
            pieces.append(' '.join(current)); current, count = [sentence], length
        else:
            current.append(sentence); count += length
    if current: pieces.append(' '.join(current))
    pieces = [p.replace(' \n ', '\n').strip() for p in pieces]
    return [p for p in pieces if p] or ['']


def prepare(tokenizer, split):
    rows, texts, ids = [], [], []
    for fixture in FIXTURES:
        chunks = document_chunks(fixture['context'], tokenizer, split)
        sentences = split(fixture['answer'])
        if not sentences: raise RunConflict('empty synthetic answer sentences')
        start = len(texts)
        for chunk in chunks:
            for sentence in sentences:
                # Preserve upstream sequential replacement semantics.
                user = 'Document: [DOCUMENT]\nClaim: [CLAIM]'.replace('[DOCUMENT]', chunk).replace('[CLAIM]', sentence)
                text = tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},
                    {'role':'user','content':user}], tokenize=False, add_generation_prompt=True)
                tokens = tokenizer.encode(text)  # Includes ordinary encoding BOS, as upstream string generation.
                if not tokens or len(tokens)+1 > 32768: raise RunConflict('synthetic MiniCheck prompt exceeds limit')
                texts.append(text); ids.append(tokens)
        rows.append({**fixture, 'chunks': chunks, 'answer_sentences': sentences,
                     'start': start, 'end': len(texts)})
    if len(texts) != 6: raise RunConflict('expected exactly six synthetic chunk/sentence requests')
    return {'rows': rows, 'texts': texts, 'prompt_token_ids': ids}
