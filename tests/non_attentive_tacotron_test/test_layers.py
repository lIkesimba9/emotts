from typing import Tuple

import pytest
import torch

from src.models.feature_models.layers import (
    ConvNorm, Idomp, LinearWithActivation, PositionalEncoding,
)


def test_idomp_layer() -> None:
    input_tensor = torch.randn(16, 10, 2)
    layer = Idomp()
    out = layer(input_tensor)
    assert (input_tensor == out).all(), "Layer must return tensor as it is"


@pytest.mark.parametrize(
    ("kernel_size", "output_channel", "dilation", "input_tensor", "expected_shape"),
    [
        pytest.param(3, 256, 1, torch.randn(16, 128, 24), (16, 256, 24)),
        pytest.param(5, 128, 3, torch.randn(16, 256, 32), (16, 128, 32)),
    ],
)
def test_conv_norm_layer(
    kernel_size: int,
    output_channel: int,
    dilation: int,
    input_tensor: torch.Tensor,
    expected_shape: Tuple[int, int, int]
) -> None:
    layer = ConvNorm(
        input_tensor.shape[1], output_channel, kernel_size, dilation=dilation
    )
    layer_out = layer(input_tensor)
    assert (
        layer_out.shape == expected_shape
    ), f"Wrong out shape, expected: {expected_shape}, got: {layer_out.shape}"


@pytest.mark.parametrize(
    ("dimension", "input_tensor", "expected_shape"),
    [
        pytest.param(32, torch.randn(16, 24, 128), (16, 24, 160)),
        pytest.param(64, torch.randn(16, 32, 256), (16, 32, 320)),
    ],
)
def test_positional_encoding_layer(
        dimension: int, input_tensor: torch.Tensor, expected_shape: Tuple[int, int, int]
) -> None:
    layer = PositionalEncoding(dimension, torch.device("cpu"))
    layer_out = layer(input_tensor)
    assert (
        layer_out.shape == expected_shape
    ), f"Wrong out shape, expected: {expected_shape}, got: {layer_out.shape}"


@pytest.mark.parametrize(
    ("dimension", "input_tensor", "expected_shape"),
    [
        pytest.param(32, torch.randn(16, 24, 128), (16, 24, 32)),
        pytest.param(64, torch.randn(16, 32, 256), (16, 32, 64)),
    ],
)
def test_liner_norm_layer(dimension: int, input_tensor: torch.Tensor, expected_shape: Tuple[int, int, int]) -> None:
    layer = LinearWithActivation(input_tensor.shape[-1], dimension)
    layer_out = layer(input_tensor)
    assert (
        layer_out.shape == expected_shape
    ), f"Wrong out shape, expected: {expected_shape}, got: {layer_out.shape}"
