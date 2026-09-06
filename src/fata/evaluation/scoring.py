"""Unified scorer used by new evaluations.

Historical raw files were produced by several looser scorers. They are not
silently rescored; provenance documents label that distinction.
"""

from __future__ import annotations

import re
from typing import Iterable

_ARTICLES = {"a", "an", "the"}
_NUMBERS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
_CONTRACTIONS = {
    "cant": "can't", "couldnt": "couldn't", "didnt": "didn't",
    "doesnt": "doesn't", "dont": "don't", "isnt": "isn't",
    "shouldnt": "shouldn't", "wasnt": "wasn't", "werent": "weren't",
    "wont": "won't", "wouldnt": "wouldn't",
}
_PUNCT = re.compile(r"[;\/\[\]\"{}()=+\\_<>@`,?!]")
_PERIOD = re.compile(r"(?<!\d)\.(?!\d)")
_COMMA_NUMBER = re.compile(r"(?<=\d),(?=\d)")


def vqa_normalize(value: object) -> str:
    text = str(value).replace("\n", " ").replace("\t", " ").strip().lower()
    text = _COMMA_NUMBER.sub("", text)
    text = _PUNCT.sub(" ", text)
    text = _PERIOD.sub(" ", text)
    words = []
    for word in text.split():
        word = _NUMBERS.get(word, word)
        if word in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def score_vqa(prediction: object, references: Iterable[object]) -> float:
    """VQA consensus score, `min(exact normalized matches / 3, 1)`.

    No substring fallback is used. Duplicate human answers intentionally count.
    """
    pred = vqa_normalize(prediction)
    refs = [vqa_normalize(item) for item in references]
    if not refs:
        raise ValueError("VQA scoring requires at least one reference answer")
    return min(refs.count(pred) / 3.0, 1.0)


def extract_option_letter(prediction: object) -> str | None:
    text = str(prediction).strip()
    match = re.match(r"^(?:option\s+)?\(?\s*([A-F])(?:\s*[).:]|\s|$)", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def score_mc(prediction: object, correct_letter: object) -> float:
    expected = str(correct_letter).strip().upper()
    if expected not in set("ABCDEF"):
        raise ValueError(f"invalid MC answer letter: {correct_letter!r}")
    return float(extract_option_letter(prediction) == expected)
