"""Strict, provider-independent parsing. Invalid responses never yield a score."""

import json
import math


class JudgeParseError(ValueError):
    """Machine-readable failure without exposing raw benchmark/response text."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JudgeParseError("duplicate_key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise JudgeParseError("non_finite_number")


def parse_score(text: str) -> float:
    """Accept exactly {unsupported_probability: finite JSON number in [0, 1]}."""
    if not isinstance(text, str):
        raise JudgeParseError("invalid_text_type")
    try:
        payload = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except JudgeParseError:
        raise
    except (ValueError, RecursionError):
        raise JudgeParseError("invalid_json") from None
    if not isinstance(payload, dict) or set(payload) != {"unsupported_probability"}:
        raise JudgeParseError("invalid_schema")
    value = payload["unsupported_probability"]
    if type(value) not in (int, float):
        raise JudgeParseError("invalid_score_type")
    # Range comparison precedes float conversion so arbitrarily large integers
    # cannot overflow the parser. JSON exponent overflow is caught here too.
    if not 0 <= value <= 1:
        raise JudgeParseError("score_out_of_range")
    if not math.isfinite(value):
        raise JudgeParseError("non_finite_number")
    return float(value)
