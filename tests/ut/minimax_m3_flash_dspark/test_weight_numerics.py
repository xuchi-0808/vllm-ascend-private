# SPDX-License-Identifier: Apache-2.0
"""Real CPU torch arithmetic; no model, vLLM, accelerator or checkpoint needed."""

import pytest


@pytest.fixture
def torch_module():
    # Only numeric cases require torch; schema/config/guard tests also run without it.
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("CPU torch with float8_e4m3fn is required")
    return torch


@pytest.mark.parametrize(("rows", "columns"), [(3, 64), (5, 33), (257, 33)])
def test_fp8_scales_multiply_each_32_columns_and_preserve_tail(helpers, torch_module, rows, columns):
    torch = torch_module
    raw = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns) % 19 - 9
    weight = raw.to(torch.float8_e4m3fn)
    scale = torch.tensor([[125, 129]] * rows, dtype=torch.uint8)
    actual = helpers.weights.resolve_fp8(weight, scale, torch.bfloat16)
    # Independently fixed numeric scales, not the raw bytes 125 and 129 or reciprocals.
    factors = torch.tensor([[0.25] * 32 + [4.0] * 32] * rows)[:, :columns]
    expected = (weight.float() * factors).bfloat16()
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("exponent_byte", "expected"), [(0, 2.0**-127), (126, 0.5), (127, 1.0), (128, 2.0), (254, 2.0**127)]
)
def test_e8m0_bytes_are_unsigned_biased_exponents(helpers, torch_module, exponent_byte, expected):
    torch = torch_module
    weight = torch.ones((1, 1)).to(torch.float8_e4m3fn)
    scale = torch.tensor([[exponent_byte]], dtype=torch.uint8)
    actual = helpers.weights.resolve_fp8(weight, scale, torch.float32)
    assert actual[0, 0].item() == expected


def test_nan_e8m0_scale_is_rejected_even_for_zero_weight(helpers, torch_module):
    torch = torch_module
    weight = torch.zeros((1, 32)).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="NaN"):
        helpers.weights.resolve_fp8(weight, torch.tensor([[255]], dtype=torch.uint8), torch.bfloat16)


@pytest.mark.parametrize("malformation", ["missing_tail_scale", "wrong_scale_dtype", "vector_weight"])
def test_scale_shape_and_encoding_are_checked(helpers, torch_module, malformation):
    torch = torch_module
    weight = torch.ones((1, 33)).to(torch.float8_e4m3fn)
    scale = torch.ones((1, 2), dtype=torch.uint8)
    if malformation == "missing_tail_scale":
        scale = scale[:, :1]
    elif malformation == "wrong_scale_dtype":
        scale = scale.float()
    else:
        weight = weight.reshape(-1)
    with pytest.raises(ValueError):
        helpers.weights.resolve_fp8(weight, scale, torch.bfloat16)


def test_loader_consumes_scale_tensors_and_preserves_auxiliary_weights(helpers, torch_module, tiny_config, tiny_header):
    torch = torch_module
    dtypes = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}
    tensors = {
        name: torch.full(metadata["shape"], 127 if metadata["dtype"] == "U8" else 1).to(dtypes[metadata["dtype"]])
        for name, metadata in tiny_header.items()
    }
    resolved = dict(helpers.weights.iter_resolved_weights(tensors.items(), tiny_config, torch.bfloat16))
    assert len(resolved) == 18
    assert not any(name.endswith("weight_scale_inv") for name in resolved)
    assert resolved["norm.weight"] is tensors["language_model.model.dspark.final_norm.weight"]
    assert resolved["fc.weight"] is tensors["language_model.model.dspark.fc.weight"]
    for name, value in resolved.items():
        assert value.dtype == torch.bfloat16
        torch.testing.assert_close(value, torch.ones_like(value), atol=0, rtol=0)


def test_loader_rejects_duplicate_input_instead_of_overwriting(helpers, torch_module, tiny_config, tiny_header):
    torch = torch_module
    dtypes = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}
    tensors = [
        (name, torch.full(metadata["shape"], 127 if metadata["dtype"] == "U8" else 1).to(dtypes[metadata["dtype"]]))
        for name, metadata in tiny_header.items()
    ]
    duplicate_name, duplicate_value = tensors[0]
    tensors.append((duplicate_name, torch.zeros_like(duplicate_value)))
    with pytest.raises(ValueError, match="Duplicate input tensor"):
        list(helpers.weights.iter_resolved_weights(tensors, tiny_config, torch.bfloat16))
