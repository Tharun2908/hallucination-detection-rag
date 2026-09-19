"""Manually authored post-thesis cases; expectations are offline annotations."""

from .judge import JudgeInput
from .runner import Example
from .prompts import content_hash

CASE_VERSION = "synthetic-attributes-numbers-v1"

# The extra facts stay identical in short/embedded contexts. Only the answer
# gains supported sentences, with the target claim at the same middle position.
FACTS = (
    "The venue is named Cedar Hall.",
    "The venue is on Maple Street.",
    "Its main entrance faces east.",
    "Its lobby has a blue noticeboard.",
    "Its reception desk is beside the entrance.",
    "The building has two floors.",
    "The upper floor contains an archive room.",
    "The archive contains meeting records.",
    "A staircase connects the floors.",
    "A clock hangs above the staircase.",
    "The ground floor contains a meeting room.",
    "The meeting room has a whiteboard.",
)


def _answer(target, embedded):
    return " ".join((*FACTS[:6], target, *FACTS[6:])) if embedded else target


def case_records():
    cases = []
    for variant, value in (("supported", "True"), ("contradicted", "False"), ("unknown", "None")):
        context = "{'OutdoorSeating': " + value + "}. " + " ".join(FACTS)
        for length in ("short", "embedded"):
            cases.append({"sample_id": f"attribute-{variant}-{length}",
                          "input": {"answer": _answer("The venue offers outdoor seating.", length == "embedded"),
                                    "context": context},
                          "expected_unsupported": variant != "supported",
                          "family": "attribute", "variant": variant, "length": length})
    context = "The selected group comprises the lowest 60% of recorded scores. " + " ".join(FACTS)
    for variant, direction in (("supported", "lowest"), ("contradicted", "highest")):
        for length in ("short", "embedded"):
            cases.append({"sample_id": f"numeric-{variant}-{length}",
                          "input": {"answer": _answer(f"The selected group comprises the {direction} 60% of recorded scores.", length == "embedded"),
                                    "context": context},
                          "expected_unsupported": variant != "supported",
                          "family": "numeric", "variant": variant, "length": length})
    return tuple(cases)


def examples():
    return tuple(Example(r["sample_id"], JudgeInput(**r["input"])) for r in case_records())


def cases_sha256():
    return content_hash(case_records())


def comparisons(report):
    scores = {p["sample_id"]: p["unsupported_probability"] for p in report["predictions"]}
    pairs = []
    for family, variants in (("attribute", ("contradicted", "unknown")), ("numeric", ("contradicted",))):
        for variant in variants:
            for length in ("short", "embedded"):
                supported = f"{family}-supported-{length}"
                unsupported = f"{family}-{variant}-{length}"
                a, b = scores[supported], scores[unsupported]
                pairs.append({"supported_id": supported, "unsupported_id": unsupported,
                              "unsupported_minus_supported": None if a is None or b is None else b - a,
                              "expected_direction_observed": None if a is None or b is None else b > a})
    length_changes = []
    for row in case_records():
        if row["length"] == "short":
            sid = row["sample_id"]
            embedded = sid.removesuffix("short") + "embedded"
            a, b = scores[sid], scores[embedded]
            length_changes.append({"short_id": sid, "embedded_id": embedded,
                                   "embedded_minus_short": None if a is None or b is None else b - a})
    return {"support_pairs": pairs, "answer_length_changes": length_changes}
