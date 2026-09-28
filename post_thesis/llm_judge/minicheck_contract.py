"""Post-thesis MiniCheck semantics from pinned upstream; no model loading."""
import math

from .storage import RunConflict

UPSTREAM_REVISION = 'b58b9fa69acbd1015ec970fa65dd752413a053d2'
MODEL = 'bespokelabs/Bespoke-MiniCheck-7B'
MODEL_REVISION = '1ed7786bcda3fa1dc35f7c4ed9e3f36b785d33b8'
MAX_MODEL_LEN = 32768
DEFAULT_CHUNK_SIZE = MAX_MODEL_LEN - 300
SUPPORT_THRESHOLD = float.fromhex('0x1.999999999999bp-3')


def support_from_returned_tokens(tokens):
    """Upstream sums probabilities of returned tokens whose lowercase text is yes.

    Tokens are the returned first-position top-logprob entries, not the full
    vocabulary or a renormalized yes/no pair. Missing yes in a valid returned
    list yields zero. Missing/malformed transport output must not call this
    function with a fabricated empty list.
    """
    if not tokens: raise RunConflict('missing MiniCheck first-position logprobs')
    score = 0.
    for token in tokens:
        text, value = token['decoded_token'], token['logprob']
        if (not isinstance(text, str) or type(value) not in (int, float)
                or not math.isfinite(value) or value > 0):
            raise RunConflict('invalid MiniCheck token logprob')
        if text.lower() == 'yes': score += math.exp(value)
    if not 0 <= score <= 1 + 1e-6: raise RunConflict('invalid MiniCheck support mass')
    return score


def aggregate_support(matrix):
    """Rows are document chunks, columns are answer sentences: min(max(axis=0))."""
    if not matrix or not matrix[0]: raise RunConflict('empty MiniCheck document/answer matrix')
    width = len(matrix[0])
    if any(len(row) != width for row in matrix): raise RunConflict('ragged MiniCheck support matrix')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for row in matrix for v in row):
        raise RunConflict('invalid MiniCheck chunk/sentence probability')
    maxima = [max(row[j] for row in matrix) for j in range(width)]
    score = min(maxima)
    return {'support_score': score, 'best_support_per_answer_sentence': maxima,
            'upstream_supported_label': score > .5,
            'frozen_predicted_unsupported': score < SUPPORT_THRESHOLD}
