"""Fail-closed LLaVA image-placeholder and expanded hidden-span helpers."""

from __future__ import annotations

from typing import Any


def single_image_placeholder_position(
    input_ids: Any,
    *,
    config_token_id: int | None,
    tokenizer_token_id: int | None,
) -> int:
    """Locate exactly one real image placeholder; never guess a magic offset."""

    if getattr(input_ids, "ndim", None) != 2 or tuple(input_ids.shape[:1]) != (1,):
        raise RuntimeError(f"expected one batched input_ids row, got shape={input_ids.shape}")
    candidates: list[int] = []
    for raw in (config_token_id, tokenizer_token_id):
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 and raw not in candidates:
            candidates.append(raw)
    if not candidates:
        raise RuntimeError("model/tokenizer exposes no valid image token id")
    matches: list[tuple[int, list[int]]] = []
    for token_id in candidates:
        positions = (input_ids[0] == token_id).nonzero(as_tuple=False).flatten().tolist()
        if positions:
            matches.append((token_id, [int(value) for value in positions]))
    if len(matches) != 1 or len(matches[0][1]) != 1:
        raise RuntimeError(
            "expected exactly one unambiguous image placeholder; "
            f"candidate_matches={matches}"
        )
    return matches[0][1][0]


def expanded_image_and_trailing_text_span(
    input_ids: Any,
    *,
    output_sequence_length: int,
    image_token_count: int,
    config_token_id: int | None,
    tokenizer_token_id: int | None,
) -> tuple[int, int, int]:
    """Validate one-token→patch expansion and return image/text boundaries."""

    if image_token_count <= 0 or output_sequence_length <= 0:
        raise RuntimeError("image token count and output sequence length must be positive")
    start = single_image_placeholder_position(
        input_ids,
        config_token_id=config_token_id,
        tokenizer_token_id=tokenizer_token_id,
    )
    input_length = int(input_ids.shape[1])
    expected_output_length = input_length - 1 + image_token_count
    if output_sequence_length != expected_output_length:
        raise RuntimeError(
            "LLaVA hidden sequence does not match one-placeholder expansion: "
            f"output={output_sequence_length} expected={expected_output_length} "
            f"input={input_length} image_tokens={image_token_count}"
        )
    end = start + image_token_count
    if not (0 <= start < end < output_sequence_length):
        raise RuntimeError(
            "expanded image span is out of bounds or has no trailing text: "
            f"start={start} end={end} output={output_sequence_length}"
        )
    return start, end, end
