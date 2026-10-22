# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Check the installed plugin, real SpeculativeConfig and registry without a model.

Run inside the isolated, built Ascend container after installing the adapter and
preparing an independent draft view. This script reads config/header metadata;
it does not create a model, initialize a worker/process group, load tensor data,
or launch inference. Platform imports can query hardware capabilities.

Example:
  python preflight_runtime_config.py --target /weights/target \
      --draft /weights/draft-view --target-quantization ascend --tp 8
"""

import argparse
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path

ARCHITECTURE = "MiniMaxM31FlashA3DSparkModel"
PLUGIN_NAME = "minimax_dspark_flash"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not found"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target-quantization", default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--draft-tp", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom checkpoint code; only use with trusted checkpoints",
    )
    args = parser.parse_args()
    target_path, draft_path = args.target.resolve(), args.draft.resolve()
    for path in (target_path, draft_path):
        require((path / "config.json").is_file(), f"Missing local config: {path}")
    require(target_path != draft_path, "Target and draft must be separate paths")
    require(args.draft_tp in (None, 1, args.tp), "draft TP must be 1 or equal target TP")
    configured_plugins = os.environ.get("VLLM_PLUGINS")
    if configured_plugins is not None:
        allowed_plugins = {name.strip() for name in configured_plugins.split(",")}
        require(PLUGIN_NAME in allowed_plugins, f"VLLM_PLUGINS excludes {PLUGIN_NAME}")
    entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
    adapter_entry_points = [entry_point for entry_point in entry_points if entry_point.name == PLUGIN_NAME]
    require(len(adapter_entry_points) == 1, f"Expected one installed {PLUGIN_NAME} entry point")

    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm.platforms import current_platform

    require(
        current_platform.device_type == "npu",
        "Expected the installed Ascend platform, got "
        f"{current_platform.__class__.__module__}.{current_platform.__class__.__name__}",
    )
    # Same entry point/order as EngineArgs.create_engine_config(), before its
    # create_model_config(). Ascend uses this hook to register its quantization
    # configurations; loading general plugins alone is insufficient.
    current_platform.pre_register_and_update()
    from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    from vllm.model_executor.models import ModelRegistry
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import get_eagle3_aux_layers_from_config

    require(ARCHITECTURE in ModelRegistry.models, "General plugin did not register the draft architecture")
    if args.target_quantization is not None:
        require(
            args.target_quantization in QUANTIZATION_METHODS,
            f"Platform pre-registration did not provide quantization {args.target_quantization}",
        )
    target = ModelConfig(
        model=str(target_path),
        tokenizer=str(target_path),
        runner="generate",
        trust_remote_code=args.trust_remote_code,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        quantization=args.target_quantization,
    )
    parallel = ParallelConfig(
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="mp",
    )
    speculative = SpeculativeConfig(
        method="dspark",
        model=str(draft_path),
        num_speculative_tokens=7,
        quantization=None,
        draft_tensor_parallel_size=args.draft_tp,
        target_model_config=target,
        target_parallel_config=parallel,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )
    draft = speculative.draft_model_config
    config = draft.hf_config
    require(
        type(config) is MiniMaxM3TextConfig,
        f"Expected the real MiniMax text config, got {type(config).__module__}.{type(config).__name__}",
    )
    require(
        list(draft.architectures) == [ARCHITECTURE],
        f"Core rewrote the draft architecture to {draft.architectures}; check core guard patch",
    )
    require(config.architectures == [ARCHITECTURE], "HF draft architecture was rewritten")
    require(config.model_type == "minimax_m3_text", f"Unexpected parsed draft model_type: {config.model_type}")
    require(config.dspark_minimax_flash_adapter == 1, "Not an independently prepared Flash draft config")
    require(draft.quantization is None, f"Draft incorrectly inherited target quantization: {draft.quantization}")
    require(not getattr(config, "quantization_config", None), "Parsed draft still has quantization_config")
    require(
        speculative.method == "dspark" and speculative.parallel_drafting,
        "SpeculativeConfig did not enable DSpark parallel drafting",
    )
    require(speculative.num_speculative_tokens == 7, "Unexpected speculative token count")
    capture = get_eagle3_aux_layers_from_config(speculative)
    require(capture == (4, 18, 32, 46, 60), f"Unexpected auxiliary capture boundaries: {capture}")
    require(
        config.num_hidden_layers == 5 and config.dspark_ffn_hidden_size == 12288,
        "Parsed config lost Flash dense draft dimensions",
    )
    require(config.rope_parameters["partial_rotary_factor"] == 0.5, "Draft partial RoPE changed")
    require(config.vocab_size == target.get_vocab_size(), "Draft cannot share the target vocabulary")

    model_class, resolved_architecture = ModelRegistry.resolve_model_cls(draft.architectures, draft)
    from minimax_dspark_adapter.model import (
        DFlashQwen3Attention,
        MiniMaxFlashDraftLayer,
        MiniMaxFlashDSparkForCausalLM,
        MiniMaxFlashDSparkModel,
        MiniMaxM3MLP,
        _resolve_layer_attention,
    )

    require(model_class is MiniMaxFlashDSparkForCausalLM, "Registry resolved an unexpected model class")
    require(resolved_architecture == ARCHITECTURE, "Registry normalized the unique draft architecture")
    require(model_class.model_cls is MiniMaxFlashDSparkModel, "DFlash constructor would build the wrong backbone")
    require(
        MiniMaxFlashDSparkModel.decoder_layer_cls is MiniMaxFlashDraftLayer,
        "DFlash constructor would build the wrong decoder layers",
    )
    # Bind the actual runtime signatures without invoking constructors or allocating model tensors.
    inspect.signature(model_class.__init__).bind(None, vllm_config=None, prefix="")
    inspect.signature(MiniMaxFlashDraftLayer.__init__).bind(
        None, None, config=config, layer_idx=0, cache_config=None, quant_config=None, prefix="model.layers.60"
    )
    inspect.signature(DFlashQwen3Attention.__init__).bind(
        None,
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        rope_parameters=dict(config.rope_parameters),
        max_position=config.max_position_embeddings,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=False,
        sliding_window=config.sliding_window,
        causal=True,
        is_neox_style=True,
        cache_config=None,
        quant_config=None,
        prefix="model.layers.60.self_attn",
    )
    inspect.signature(MiniMaxM3MLP.__init__).bind(
        None, config, intermediate_size=config.dspark_ffn_hidden_size, quant_config=None, prefix="model.layers.60.mlp"
    )
    attention_modes = [_resolve_layer_attention(config, index) for index in range(config.num_hidden_layers)]
    require(
        attention_modes == [(4096, True)] * 5,
        f"Unexpected attention semantics resolved from the unmodified checkpoint: {attention_modes}",
    )

    from minimax_dspark_adapter.weights import validate_header
    from prepare_draft_view import read_header

    index = json.loads((draft_path / "model.safetensors.index.json").read_text())
    headers = {}
    for filename in sorted(set(index["weight_map"].values())):
        for name, metadata in read_header(draft_path / filename).items():
            if name == "__metadata__":
                continue
            require(name not in headers, f"Duplicate tensor across shards: {name}")
            headers[name] = metadata
    require(set(headers) == set(index["weight_map"]), "Draft header and index do not match")
    validate_header(headers, config.to_dict())
    report = {
        "status": "PASS",
        "scope": (
            "Real SpeculativeConfig, installed plugin and registry, signature and safetensors-header checks only; "
            "no model/worker/inference was created"
        ),
        "python": sys.version.split()[0],
        "versions": {name: version(name) for name in ("torch", "vllm", "vllm-ascend", "minimax-dspark-flash-adapter")},
        "plugin": {
            "name": PLUGIN_NAME,
            "entry_point": adapter_entry_points[0].value,
            "VLLM_PLUGINS": configured_plugins,
        },
        "platform": {
            "class": f"{current_platform.__class__.__module__}.{current_platform.__class__.__name__}",
            "device_type": current_platform.device_type,
            "pre_registration": (
                "current_platform.pre_register_and_update(), matching EngineArgs.create_engine_config()"
            ),
        },
        "target": {
            "path": str(target_path),
            "architectures": target.architectures,
            "quantization": target.quantization,
            "tp": args.tp,
        },
        "draft": {
            "path": str(draft_path),
            "architecture": resolved_architecture,
            "class": f"{model_class.__module__}.{model_class.__name__}",
            "model_type": config.model_type,
            "config_class": f"{type(config).__module__}.{type(config).__name__}",
            "quantization": draft.quantization,
            "tp": speculative.draft_tensor_parallel_size,
            "tensors": len(headers),
            "capture_boundaries": capture,
            "attention_modes": attention_modes,
            "auxiliary_norm_trial": {"hidden": config.dspark_hidden_norm_type, "final": config.dspark_final_norm_type},
        },
        "pending": (
            "Full tensor load, embedding/lm_head alias, NPU forward and speculative acceptance/output validation"
        ),
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
