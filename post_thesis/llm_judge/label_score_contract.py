"""Offline class-token score contract. No backend, model loading or generation."""

from dataclasses import asdict
import math

from .binary_contract import BINARY_PROMPT
from .judge import JudgeInput
from .prompts import PromptSpec, canonical_json, content_hash

CONTRACT_VERSION = "label-score-offline-v1"
MODEL = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
LABELS = {"supported": "A", "unsupported": "B"}
EXTRACTION = "raw_full_vocabulary_next_token_logprobs"
NUMERIC_TOLERANCE = 1e-6
ASSISTANT_BOUNDARY = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
LABEL_PROMPT = PromptSpec(
    version="faithfulness-label-score-v1",
    system_text=BINARY_PROMPT.system_text.split("\n\nReturn unsupported", 1)[0] + """

Choose exactly one class:
A = supported: all factual assertions are supported by the supplied context,
or the answer contains no factual assertions.
B = unsupported: at least one factual assertion is unsupported by or contradicts
the supplied context.
Return exactly the single uppercase character A or B, with no leading whitespace,
JSON, explanation, evidence quotes, markdown, or numeric probability.""",
)


def label_messages(item, *, prompt=LABEL_PROMPT):
    if type(item) is not JudgeInput:
        raise TypeError("expected JudgeInput with only answer and context")
    return [{"role": "system", "content": prompt.system_text},
            {"role": "user", "content": canonical_json(asdict(item))}]


def _token_id(value):
    if type(value) is not int or value < 0:
        raise ValueError("token IDs must be nonnegative integers")
    return value


def score_from_logprobs(entries, *, supported_token_id, unsupported_token_id):
    """Arithmetic only; the future adapter must attest model/position/raw mode.

    Require exactly both selected IDs, not a top-k approximation. Invalid inputs
    raise; no replacement score is generated. Retain margin even if sigmoid rounds.
    """
    ids = [_token_id(supported_token_id), _token_id(unsupported_token_id)]
    if len(set(ids)) != 2:
        raise ValueError("class token IDs must differ")
    if not isinstance(entries, (list, tuple)) or len(entries) != 2:
        raise ValueError("both selected token log probabilities are required")
    values = {}
    for token_id, value in entries:
        _token_id(token_id)
        if token_id not in ids or token_id in values:
            raise ValueError("wrong or duplicate class token")
        if type(value) not in (float, int) or not math.isfinite(value) or value > 0:
            raise ValueError("expected finite nonpositive raw log probability")
        values[token_id] = value
    supported, unsupported = (values[i] for i in ids)
    largest = max(supported, unsupported)
    log_mass = largest + math.log1p(math.exp(min(supported, unsupported) - largest))
    if log_mass > math.log1p(NUMERIC_TOLERANCE):
        raise ValueError("class probability mass exceeds one")
    margin = unsupported - supported
    if not math.isfinite(margin):
        raise ValueError("log-odds overflow")
    if margin >= 0:
        score = 1 / (1 + math.exp(-margin))
    else:
        odds = math.exp(margin)
        score = odds / (1 + odds)
    return {"supported_logprob": supported, "unsupported_logprob": unsupported,
            "unsupported_log_odds": margin, "unsupported_score": score,
            "log_class_token_mass": log_mass, "calibrated": False}


def prepare_tokenized_input(item, tokenizer, *, prompt=LABEL_PROMPT):
    """Check the actual first class-token position, without loading model weights."""
    messages = label_messages(item, prompt=prompt)
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template:
        raise ValueError("missing exact chat template")
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
    if not rendered.endswith(ASSISTANT_BOUNDARY):
        raise ValueError("unexpected Qwen non-thinking assistant boundary")
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    direct = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                    add_generation_prompt=True, enable_thinking=False)
    if not prompt_ids or direct != prompt_ids:
        raise ValueError("chat-template and explicit tokenization disagree")
    for token in prompt_ids:
        _token_id(token)
    labels = {}
    for meaning, label in LABELS.items():
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError("class label must encode to exactly one token")
        token_id = _token_id(encoded[0])
        decoded = tokenizer.decode([token_id], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if decoded != label or token_id in tokenizer.all_special_ids:
            raise ValueError("class token has wrong bytes or is a special token")
        continued = tokenizer.encode(rendered + label, add_special_tokens=False)
        if continued != prompt_ids + encoded:
            raise ValueError("class token merges across the assistant boundary")
        labels[meaning] = {"text": label, "token_id": token_id,
                           "utf8_hex": decoded.encode("utf-8").hex()}
    if labels["supported"]["token_id"] == labels["unsupported"]["token_id"]:
        raise ValueError("class token IDs collide")
    return {"input_sha256": content_hash(asdict(item)),
            "messages_sha256": content_hash(messages), "rendered_prompt": rendered,
            "rendered_prompt_sha256": content_hash(rendered),
            "prompt_token_ids_sha256": content_hash(prompt_ids),
            "input_tokens": len(prompt_ids), "score_position": len(prompt_ids),
            "score_position_convention": "zero_based_next_token_after_complete_prompt",
            "template_sha256": content_hash(template), "assistant_boundary": ASSISTANT_BOUNDARY,
            "labels": labels}


def prepared_identity(prepared, *, tokenizer_files, versions):
    """Identity for offline prepared inputs; not a live request authorization."""
    required = ("transformers", "tokenizers", "huggingface_hub", "jinja2")
    if not tokenizer_files or any(not versions.get(k) for k in required):
        raise ValueError("tokenizer files and rendering versions must be recorded")
    return {"study_stage": "post_thesis", "contract_version": CONTRACT_VERSION,
            "prompt_sha256": LABEL_PROMPT.sha256, "model": MODEL,
            "model_revision": REVISION, "tokenizer_revision": REVISION,
            "tokenizer_files": tokenizer_files, "versions": {k: versions[k] for k in required},
            "extraction": EXTRACTION, "enable_thinking": False,
            "input": {k: v for k, v in prepared.items() if k != "rendered_prompt"}}
