import pytest
import torch

from fata.compression.protocol import validate_compressor_output


def test_compressor_output_token_count():
    original = torch.zeros(2, 64, 8)
    compressed = torch.zeros(2, 16, 8)
    assert validate_compressor_output(original, compressed, 16) == 16
    with pytest.raises(ValueError):
        validate_compressor_output(original, compressed, 15)
