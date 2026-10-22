# SPDX-License-Identifier: Apache-2.0
"""Load the example's CPU helpers without importing vLLM or torch-npu.

Standalone invocation from the repository root:
pytest --confcutdir=tests/ut/minimax_m3_flash_dspark tests/ut/minimax_m3_flash_dspark
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example_dir():
    repo = Path(__file__).resolve().parents[3]
    return repo / "examples" / "spec_decode" / "minimax_m3_flash_dspark"


@pytest.fixture
def helpers(example_dir, monkeypatch):
    # The installed adapter, if any, must not shadow the source under test.
    package_name = "minimax_dspark_adapter"
    package = ModuleType(package_name)
    package.__path__ = [str(example_dir / package_name)]
    monkeypatch.setitem(sys.modules, package_name, package)
    weights = _load_module(f"{package_name}.weights", example_dir / package_name / "weights.py")
    monkeypatch.setitem(sys.modules, f"{package_name}.weights", weights)
    prepare = _load_module("_flash_dspark_prepare_test", example_dir / "prepare_draft_view.py")
    return SimpleNamespace(weights=weights, prepare=prepare)


@pytest.fixture
def tiny_config():
    return {
        "hidden_size": 8,
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "dspark_ffn_hidden_size": 12,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "dspark_markov_rank": 2,
        "dspark_target_layer_ids": [3, 17],
    }


@pytest.fixture
def tiny_header():
    """A complete one-layer synthetic checkpoint; no tensor payload required."""
    prefix = "language_model.model.dspark."
    shapes = {
        "fc.weight": [8, 16],
        "hidden_norm.weight": [8],
        "final_norm.weight": [8],
        "markov_head.markov_w1.weight": [32, 2],
        "markov_head.markov_w2.weight": [32, 2],
        "confidence_head.proj.weight": [1, 10],
        "confidence_head.proj.bias": [1],
        "layers.0.decoder_layer.input_layernorm.weight": [8],
        "layers.0.decoder_layer.post_attention_layernorm.weight": [8],
        "layers.0.decoder_layer.self_attn.q_norm.weight": [4],
        "layers.0.decoder_layer.self_attn.k_norm.weight": [4],
    }
    header = {prefix + name: {"shape": shape, "dtype": "BF16"} for name, shape in shapes.items()}
    matrices = {
        "self_attn.q_proj": [8, 8],
        "self_attn.k_proj": [4, 8],
        "self_attn.v_proj": [4, 8],
        "self_attn.o_proj": [8, 8],
        "mlp.gate_proj": [12, 8],
        "mlp.up_proj": [12, 8],
        "mlp.down_proj": [8, 12],
    }
    for name, shape in matrices.items():
        base = prefix + "layers.0.decoder_layer." + name
        header[base + ".weight"] = {"shape": shape, "dtype": "F8_E4M3"}
        header[base + ".weight_scale_inv"] = {"shape": [shape[0], 1], "dtype": "U8"}
    return header


@pytest.fixture
def source_config(tiny_config):
    text = dict(tiny_config)
    text.update(
        num_hidden_layers=5,
        model_type="source_text_schema",
        rope_theta=500000,
        rotary_dim=2,
        quantization_config={"quant_method": "mxfp8"},
        use_gemma_norm=True,
        hidden_act="swigluoai",
    )
    return {
        "architectures": ["DSparkMiniMaxDraftModel"],
        "text_config": text,
        "dspark_block_size": 7,
        "dspark_markov_rank": 2,
        "dspark_markov_head_type": "low_rank",
        "dspark_noise_token_id": 31,
        "dspark_ffn_hidden_size": 12,
        "dspark_target_layer_ids": [3, 17, 31, 45, 59],
        "dspark_hybrid_context_fusion": False,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "dflash_config": {"target_layer_ids": [3, 17, 31, 45, 59]},
        "dspark_config": {"block_size": 7},
    }
