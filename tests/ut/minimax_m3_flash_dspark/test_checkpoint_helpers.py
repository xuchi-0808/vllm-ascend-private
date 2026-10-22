# SPDX-License-Identifier: Apache-2.0
"""Synthetic checkpoint contracts for the opt-in Flash DSpark example."""

import copy
import json
import sys

import pytest


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("layers.3.decoder_layer.self_attn.k_norm.weight", "layers.3.self_attn.k_norm.weight"),
        ("layers.0.decoder_layer.mlp.down_proj.weight_scale_inv", "layers.0.mlp.down_proj.weight_scale_inv"),
        ("final_norm.weight", "norm.weight"),
        ("hidden_norm.weight", "hidden_norm.weight"),
        ("markov_head.markov_w1.weight", "markov_head.markov_w1.weight"),
    ],
)
def test_weight_name_mapping(helpers, source, expected):
    assert helpers.weights.map_weight_name("language_model.model.dspark." + source) == expected


@pytest.mark.parametrize("name", ["model.layers.0.weight", "dspark.fc.weight", "language_model.model.fc.weight"])
def test_weights_outside_standalone_draft_are_rejected(helpers, name):
    with pytest.raises(ValueError, match="outside standalone DSpark"):
        helpers.weights.map_weight_name(name)


def test_complete_small_schema_and_matrix_orientation(helpers, tiny_config, tiny_header):
    mapped = helpers.weights.validate_header(tiny_header, tiny_config)
    assert len(mapped) == 25
    assert mapped["fc.weight"]["shape"] == [8, 16]
    assert mapped["layers.0.self_attn.k_proj.weight"]["shape"] == [4, 8]
    assert mapped["layers.0.mlp.down_proj.weight"]["shape"] == [8, 12]
    assert mapped["norm.weight"]["dtype"] == "BF16"
    # Safetensors metadata is not a tensor and must be ignored.
    tiny_header["__metadata__"] = {"format": "pt"}
    assert helpers.weights.validate_header(tiny_header, tiny_config) == mapped


@pytest.mark.parametrize(
    "malformation", ["missing", "extra", "duplicate", "wrong_shape", "scale_dtype", "weight_dtype"]
)
def test_malformed_checkpoint_fails_before_loading(helpers, tiny_config, tiny_header, malformation):
    key = "language_model.model.dspark.layers.0.decoder_layer.self_attn.k_proj.weight"
    if malformation == "missing":
        del tiny_header[key]
    elif malformation == "extra":
        tiny_header["language_model.model.dspark.unexpected.weight"] = {"shape": [1], "dtype": "BF16"}
    elif malformation == "duplicate":
        tiny_header[key.replace(".decoder_layer.", ".")] = copy.deepcopy(tiny_header[key])
    elif malformation == "wrong_shape":
        tiny_header[key]["shape"] = [8, 4]
    elif malformation == "scale_dtype":
        tiny_header[key.replace(".weight", ".weight_scale_inv")]["dtype"] = "F32"
    else:
        tiny_header[key]["dtype"] = "BF16"
    with pytest.raises(ValueError):
        helpers.weights.validate_header(tiny_header, tiny_config)


def test_scale_shape_uses_ceiling_for_partial_input_block(helpers, tiny_config):
    tiny_config["hidden_size"] = 33
    shapes = helpers.weights.expected_shapes(tiny_config)
    assert shapes["layers.0.self_attn.k_proj.weight"] == [4, 33]
    assert shapes["layers.0.self_attn.k_proj.weight_scale_inv"] == [4, 2]


def test_normalized_config_is_independent_of_source_and_other_views(helpers, source_config):
    original = copy.deepcopy(source_config)
    config = helpers.prepare.normalized_config(source_config, "gemma", "rms")
    other_view = helpers.prepare.normalized_config(source_config, "gemma", "gemma")
    assert source_config == original
    assert "quantization_config" not in config
    assert source_config["text_config"]["quantization_config"] == {"quant_method": "mxfp8"}
    assert config["architectures"] == ["MiniMaxM31FlashA3DSparkModel"]
    assert config["model_type"] == "minimax_m3_text"
    assert config["intermediate_size"] == 12
    assert config["n_predict"] == 7
    assert config["rope_parameters"]["partial_rotary_factor"] == 0.5
    assert config["dspark_hidden_norm_type"] == "gemma"
    assert config["dspark_final_norm_type"] == "rms"
    assert other_view["dspark_final_norm_type"] == "gemma"
    config["dspark_target_layer_ids"].append(99)
    config["dflash_config"]["target_layer_ids"][0] = 99
    config["dspark_config"]["block_size"] = 1
    assert source_config == original
    assert other_view["dspark_target_layer_ids"] == [3, 17, 31, 45, 59]
    assert other_view["dflash_config"]["target_layer_ids"][0] == 3
    assert other_view["dspark_config"]["block_size"] == 7


@pytest.mark.parametrize("malformation", ["architecture", "layer_count", "hybrid"])
def test_unsupported_config_is_rejected(helpers, source_config, malformation):
    if malformation == "architecture":
        source_config["architectures"] = ["Qwen3DSparkModel"]
    elif malformation == "layer_count":
        source_config["text_config"]["num_hidden_layers"] = 6
    else:
        source_config["dspark_hybrid_context_fusion"] = True
    with pytest.raises(ValueError):
        helpers.prepare.normalized_config(source_config, "gemma", "gemma")


def test_read_safetensors_header_without_tensor_payload(helpers, tmp_path):
    header = {"tiny.weight": {"shape": [1], "dtype": "BF16", "data_offsets": [0, 2]}}
    encoded = json.dumps(header).encode()
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded)
    assert helpers.prepare.read_header(path) == header


@pytest.mark.parametrize("size", [0, 10 * 1024 * 1024 + 1])
def test_invalid_header_size_fails_without_allocating_payload(helpers, tmp_path, size):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(size.to_bytes(8, "little"))
    with pytest.raises(ValueError, match="header size"):
        helpers.prepare.read_header(path)


def _prepare_args(source, destination):
    return [
        "prepare_draft_view.py",
        "--source",
        str(source),
        "--destination",
        str(destination),
        "--hidden-norm",
        "gemma",
        "--final-norm",
        "gemma",
    ]


def test_reserved_provenance_file_cannot_be_symlinked_then_overwritten(helpers, tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "draft-view"
    source.mkdir()
    provenance = source / "adapter-preparation.json"
    provenance.write_text("original source must survive", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _prepare_args(source, destination))
    with pytest.raises(ValueError, match="reserved provenance"):
        helpers.prepare.main()
    assert provenance.read_text(encoding="utf-8") == "original source must survive"
    assert not destination.exists()


def test_duplicate_tensor_in_different_shards_is_rejected(helpers, tmp_path, monkeypatch, source_config):
    source, destination = tmp_path / "source", tmp_path / "draft-view"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    key = "language_model.model.dspark.fc.weight"
    index = {"weight_map": {key: "first.safetensors", "another.tensor": "second.safetensors"}}
    (source / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    header = json.dumps({key: {"shape": [8, 40], "dtype": "BF16", "data_offsets": [0, 640]}}).encode()
    for name in ("first.safetensors", "second.safetensors"):
        (source / name).write_bytes(len(header).to_bytes(8, "little") + header)
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    monkeypatch.setattr(sys, "argv", _prepare_args(source, destination))
    with pytest.raises(ValueError, match="Duplicate tensor"):
        helpers.prepare.main()
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original
    assert not destination.exists()
