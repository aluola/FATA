#!/usr/bin/env python
"""
Conservative output parsing for VQA short answers.
Does NOT use reference answers to guide extraction.
"""

import re
from typing import Tuple


def extract_final_answer(raw_output: str) -> Tuple[str, str]:
    """
    Extract final answer from model raw output.

    Returns:
        extracted_answer: cleaned short answer
        normalized_answer: VQA-normalized version
    """
    text = str(raw_output)

    # 1. Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Handle unmatched <think> or </think>
    text = re.sub(r'</?think>', '', text)

    # 2. Strip leading/trailing whitespace and newlines
    text = text.strip()

    # 3. Remove common prefixes
    prefixes = [
        r'^Answer:\s*',
        r'^The answer is\s*',
        r'^It is\s*',
        r'^This is\s*',
        r'^\d+\.\s+',  # Numbered list
        r'^\*\*.*?\*\*\s*',  # Bold markdown
    ]
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)

    # 4. Remove surrounding quotes
    text = text.strip('"\'""''').strip()

    # 5. Remove trailing period if not part of an abbreviation
    if text.endswith('.') and not re.search(r'\b\w\.\w', text):
        text = text[:-1].strip()

    # 6. Take first line/sentence as the answer
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        # Take the first non-empty line
        text = lines[0]

    # Split on sentence boundaries and take first
    # But don't split on periods that are part of numbers or abbreviations
    sentences = re.split(r'(?<=[a-z])\.(?:\s+|$)', text)
    if sentences:
        text = sentences[0].strip()

    extracted = text

    # Normalize for VQA matching
    normalized = _vqa_normalize(extracted)

    return extracted, normalized


def _vqa_normalize(text: str) -> str:
    """VQA-style normalization."""
    text = str(text).lower().replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'([^\w\s])', r' ', text)
    words = [w for w in text.split() if w not in ['a', 'an', 'the']]
    num_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    }
    return ' '.join([num_map.get(w, w) for w in words])


def is_format_compliant(raw_output: str, max_words: int = 8) -> Tuple[bool, dict]:
    """
    Check if output format is compliant (short answer).
    Returns (compliant, diagnostics).
    """
    extracted, normalized = extract_final_answer(raw_output)

    diag = {
        "nonempty": len(normalized.strip()) > 0,
        "contains_think": '<think>' in str(raw_output).lower(),
        "contains_explanation": bool(re.search(
            r'(?i)(first|step \d|scan|observe|analyze|let me|i see|the user is asking)',
            str(raw_output)
        )),
        "word_count": len(normalized.split()),
        "extracted": extracted,
        "normalized": normalized,
    }

    compliant = (
        diag["nonempty"] and
        not diag["contains_think"] and
        not diag["contains_explanation"] and
        diag["word_count"] <= max_words
    )

    return compliant, diag
