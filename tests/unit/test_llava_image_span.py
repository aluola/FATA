from __future__ import annotations

import pytest
import torch

from fata.runtimes.llava.image_span import (
    expanded_image_and_trailing_text_span,
    single_image_placeholder_position,
)


def test_exact_placeholder_and_expanded_span():
    input_ids = torch.tensor([[1, 2, 32000, 3, 4]])
    assert single_image_placeholder_position(
        input_ids, config_token_id=32000, tokenizer_token_id=None
    ) == 2
    assert expanded_image_and_trailing_text_span(
        input_ids,
        output_sequence_length=580,
        image_token_count=576,
        config_token_id=32000,
        tokenizer_token_id=7,
    ) == (2, 578, 578)


@pytest.mark.parametrize(
    "input_ids,output_length",
    [
        (torch.tensor([[1, 2, 3, 4]]), 579),
        (torch.tensor([[1, 32000, 32000, 4]]), 579),
        (torch.tensor([[1, 2, 32000, 4]]), 578),
        (torch.tensor([[1, 2, 3, 32000]]), 579),
    ],
)
def test_missing_duplicate_bad_length_or_empty_text_fails(input_ids, output_length):
    with pytest.raises(RuntimeError):
        expanded_image_and_trailing_text_span(
            input_ids,
            output_sequence_length=output_length,
            image_token_count=576,
            config_token_id=32000,
            tokenizer_token_id=None,
        )
