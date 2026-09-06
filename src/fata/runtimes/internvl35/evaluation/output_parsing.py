"""Deterministic parsing and normalization for open and A/B/C/D outputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Literal


TaskType = Literal["open", "multiple_choice"]
ParseStatus = Literal["ok", "empty", "invalid", "ambiguous"]

OPEN_TASK = "open"
MULTIPLE_CHOICE_TASK = "multiple_choice"
VALID_MC_ANSWERS = ("A", "B", "C", "D")


_ARTICLES = frozenset({"a", "an", "the"})
_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

# The VQA evaluator canonicalizes common un-apostrophized contractions.  The
# mapping is intentionally applied token-wise after punctuation normalization.
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "couldve": "could've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "lets": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldnt": "shouldn't",
    "shouldve": "should've",
    "thats": "that's",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "whys": "why's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "wouldve": "would've",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

_ANSWER_PREFIX_RE = re.compile(
    r"^(?:(?:the\s+)?(?:final\s+)?answer)\b(?:\s+is)?\s*[:：=\-]?\s*(.*)$",
    flags=re.IGNORECASE,
)
_MC_DIRECT_RE = re.compile(
    r"^\s*"
    r"(?:(?:the\s+)?(?:final\s+)?(?:answer|option|choice)"
    r"\s*(?:is)?\s*[:：=\-]?\s*)?"
    r"[\(\[]?\s*([A-D])\s*[\)\]]?\s*[\.!,:;]?\s*$",
    flags=re.IGNORECASE,
)
_MC_CUE_RE = re.compile(
    r"\b(?:(?:final\s+)?answer|option|choice|choose|chose|select|selected|pick|picked)\b"
    r"\s*(?:is|was)?\s*[:：=\-]?\s*[\(\[]?\s*([A-D])"
    r"(?=\s*[\)\]\.,;:!?]|\b)",
    flags=re.IGNORECASE,
)
_MC_ALTERNATIVE_RE = re.compile(
    r"(?<![A-Za-z])[\(\[]?\s*([A-D])\s*[\)\]]?"
    r"\s*(?:/|or|and|,)\s*"
    r"[\(\[]?\s*([A-D])\s*[\)\]]?(?![A-Za-z])",
    flags=re.IGNORECASE,
)
_MC_LEADING_RE = re.compile(
    r"^\s*[\(\[]?\s*([A-D])(?=\s|[\)\]\.,;:!?\-]|$)",
    flags=re.IGNORECASE,
)
_MC_STANDALONE_UPPER_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


@dataclass(frozen=True)
class ParsedOutput:
    """Stable, JSON-friendly fields emitted by every output parser."""

    task_type: TaskType
    raw_output: str
    normalized_output: str
    candidate_text: str
    parsed_answer: str | None
    status: ParseStatus
    strategy: str
    candidates: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status == "ok"

    def to_record(self) -> dict[str, object]:
        """Return fields in a fixed order suitable for CSV/JSON assembly."""

        return {
            "task_type": self.task_type,
            "raw_output": self.raw_output,
            "normalized_output": self.normalized_output,
            "candidate_text": self.candidate_text,
            "parsed_answer": self.parsed_answer,
            "parse_status": self.status,
            "parse_strategy": self.strategy,
            "parse_candidates": list(self.candidates),
        }


def _as_text(value: object) -> str:
    return "" if value is None else str(value)


def _nfkc_and_collapse_whitespace(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _vqa_punctuation_pass(text: str) -> str:
    """Remove VQA punctuation while preserving decimal points and apostrophes."""

    output: list[str] = []
    for index, character in enumerate(text):
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""

        if character in {"'", "’"}:
            output.append("'")
        elif character == "." and previous.isdigit() and following.isdigit():
            output.append(character)
        elif character == "," and previous.isdigit() and following.isdigit():
            # Thousands separators do not affect an answer's canonical form.
            continue
        elif unicodedata.category(character)[0] in {"P", "S"}:
            output.append(" ")
        else:
            output.append(character)
    return "".join(output)


def normalize_vqa_answer(answer: object) -> str:
    """Canonicalize an open answer using VQA-style normalization.

    The transform is deterministic: Unicode NFKC, lowercase, punctuation and
    whitespace normalization, number-word mapping (``none``/zero through ten),
    article removal, and common contraction canonicalization.  It does not use
    substring matching or language-model-dependent heuristics.
    """

    text = unicodedata.normalize("NFKC", _as_text(answer)).lower().strip()
    text = _vqa_punctuation_pass(text)
    normalized_tokens: list[str] = []
    for token in text.split():
        token = _NUMBER_WORDS.get(token, token)
        if token in _ARTICLES:
            continue
        normalized_tokens.append(_CONTRACTIONS.get(token, token))
    return " ".join(normalized_tokens)


def _strip_matching_wrapper(text: str) -> str:
    wrappers = (("\"", "\""), ("'", "'"), ("`", "`"))
    for opening, closing in wrappers:
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            return text[len(opening) : -len(closing)].strip()
    return text


def parse_open_answer(raw_output: object) -> ParsedOutput:
    """Extract the first requested short answer and apply VQA normalization."""

    raw = _as_text(raw_output)
    normalized_output = _nfkc_and_collapse_whitespace(raw)
    lines = [line.strip() for line in unicodedata.normalize("NFKC", raw).splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ParsedOutput(
            task_type=OPEN_TASK,
            raw_output=raw,
            normalized_output=normalized_output,
            candidate_text="",
            parsed_answer=None,
            status="empty",
            strategy="none",
        )

    candidate = lines[0]
    strategy = "first_nonempty_line" if len(lines) > 1 else "direct"
    prefix_match = _ANSWER_PREFIX_RE.match(candidate)
    if prefix_match:
        payload = prefix_match.group(1).strip()
        if payload:
            candidate = payload
            strategy = "answer_prefix"
        elif len(lines) > 1:
            candidate = lines[1]
            strategy = "answer_prefix_next_line"
        else:
            candidate = ""
            strategy = "answer_prefix_empty"

    candidate = _strip_matching_wrapper(candidate.strip())
    parsed = normalize_vqa_answer(candidate)
    if not candidate:
        return ParsedOutput(
            task_type=OPEN_TASK,
            raw_output=raw,
            normalized_output=normalized_output,
            candidate_text=candidate,
            parsed_answer=None,
            status="empty",
            strategy=strategy,
        )
    # A non-empty answer can legitimately normalize to the empty string under
    # VQA rules (notably the standalone answers "a", "an", or "the").  Preserve
    # that canonical value rather than conflating it with an absent generation.
    return ParsedOutput(
        task_type=OPEN_TASK,
        raw_output=raw,
        normalized_output=normalized_output,
        candidate_text=candidate,
        parsed_answer=parsed,
        status="ok",
        strategy=strategy,
        candidates=(parsed,),
    )


def parse_mc_answer(raw_output: object) -> ParsedOutput:
    """Parse only choices A/B/C/D with deterministic ambiguity handling."""

    raw = _as_text(raw_output)
    normalized = _nfkc_and_collapse_whitespace(raw)
    if not normalized:
        return ParsedOutput(
            task_type=MULTIPLE_CHOICE_TASK,
            raw_output=raw,
            normalized_output=normalized,
            candidate_text="",
            parsed_answer=None,
            status="empty",
            strategy="none",
        )

    direct_match = _MC_DIRECT_RE.fullmatch(normalized)
    if direct_match:
        answer = direct_match.group(1).upper()
        return ParsedOutput(
            task_type=MULTIPLE_CHOICE_TASK,
            raw_output=raw,
            normalized_output=normalized,
            candidate_text=normalized,
            parsed_answer=answer,
            status="ok",
            strategy="direct",
            candidates=(answer,),
        )

    alternative_matches = _MC_ALTERNATIVE_RE.findall(normalized)
    if alternative_matches:
        alternative_candidates = tuple(
            sorted(
                {
                    candidate.upper()
                    for pair in alternative_matches
                    for candidate in pair
                }
            )
        )
        if len(alternative_candidates) > 1:
            return ParsedOutput(
                task_type=MULTIPLE_CHOICE_TASK,
                raw_output=raw,
                normalized_output=normalized,
                candidate_text=normalized,
                parsed_answer=None,
                status="ambiguous",
                strategy="alternatives",
                candidates=alternative_candidates,
            )

    cue_candidates = tuple(
        sorted({match.upper() for match in _MC_CUE_RE.findall(normalized)})
    )
    if len(cue_candidates) == 1:
        return ParsedOutput(
            task_type=MULTIPLE_CHOICE_TASK,
            raw_output=raw,
            normalized_output=normalized,
            candidate_text=normalized,
            parsed_answer=cue_candidates[0],
            status="ok",
            strategy="explicit_cue",
            candidates=cue_candidates,
        )
    if len(cue_candidates) > 1:
        return ParsedOutput(
            task_type=MULTIPLE_CHOICE_TASK,
            raw_output=raw,
            normalized_output=normalized,
            candidate_text=normalized,
            parsed_answer=None,
            status="ambiguous",
            strategy="explicit_cue",
            candidates=cue_candidates,
        )

    leading_match = _MC_LEADING_RE.match(normalized)
    if leading_match:
        leading = leading_match.group(1).upper()
        uppercase_candidates = {
            match.upper() for match in _MC_STANDALONE_UPPER_RE.findall(normalized)
        }
        candidates = tuple(sorted(uppercase_candidates | {leading}))
        if len(candidates) > 1:
            return ParsedOutput(
                task_type=MULTIPLE_CHOICE_TASK,
                raw_output=raw,
                normalized_output=normalized,
                candidate_text=normalized,
                parsed_answer=None,
                status="ambiguous",
                strategy="leading_letter",
                candidates=candidates,
            )
        return ParsedOutput(
            task_type=MULTIPLE_CHOICE_TASK,
            raw_output=raw,
            normalized_output=normalized,
            candidate_text=normalized,
            parsed_answer=leading,
            status="ok",
            strategy="leading_letter",
            candidates=(leading,),
        )

    return ParsedOutput(
        task_type=MULTIPLE_CHOICE_TASK,
        raw_output=raw,
        normalized_output=normalized,
        candidate_text=normalized,
        parsed_answer=None,
        status="invalid",
        strategy="none",
    )


def parse_output(raw_output: object, task_type: TaskType) -> ParsedOutput:
    """Dispatch to the strict parser for a dataset mapping task type."""

    if task_type == OPEN_TASK:
        return parse_open_answer(raw_output)
    if task_type == MULTIPLE_CHOICE_TASK:
        return parse_mc_answer(raw_output)
    raise ValueError(
        f"unsupported task_type {task_type!r}; expected 'open' or 'multiple_choice'"
    )


__all__ = [
    "MULTIPLE_CHOICE_TASK",
    "OPEN_TASK",
    "ParseStatus",
    "ParsedOutput",
    "TaskType",
    "VALID_MC_ANSWERS",
    "normalize_vqa_answer",
    "parse_mc_answer",
    "parse_open_answer",
    "parse_output",
]
