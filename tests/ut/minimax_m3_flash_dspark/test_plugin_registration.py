# SPDX-License-Identifier: Apache-2.0
"""The optional A3 fallback must leave native architecture/config mappings intact."""

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

FALLBACK_ARCHITECTURE = "MiniMaxM31FlashA3DSparkModel"
NATIVE_ARCHITECTURE = "DSparkMiniMaxDraftModel"


@pytest.fixture
def plugin_environment(example_dir, monkeypatch):
    config_mapping = {}
    config_registrations = []
    native_model = object()
    model_mapping = {NATIVE_ARCHITECTURE: native_model}
    model_registrations = []

    class MiniMaxM3TextConfig:
        pass

    class AutoConfig:
        @staticmethod
        def register(model_type, config):
            if model_type in config_mapping:
                raise ValueError("Existing config mapping must not be replaced")
            config_mapping[model_type] = config
            config_registrations.append((model_type, config))

    class ModelRegistry:
        models = model_mapping

        @staticmethod
        def register_model(architecture, model):
            model_mapping[architecture] = model
            model_registrations.append((architecture, model))

    modules = {
        name: ModuleType(name)
        for name in (
            "transformers",
            "transformers.models",
            "transformers.models.auto",
            "transformers.models.auto.configuration_auto",
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.models",
            "vllm.transformers_utils",
            "vllm.transformers_utils.configs",
            "vllm.transformers_utils.configs.minimax_m3",
        )
    }
    modules["transformers"].AutoConfig = AutoConfig
    modules["transformers.models.auto.configuration_auto"].CONFIG_MAPPING = config_mapping
    modules["vllm.model_executor.models"].ModelRegistry = ModelRegistry
    modules["vllm.transformers_utils.configs.minimax_m3"].MiniMaxM3TextConfig = MiniMaxM3TextConfig
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "_flash_dspark_plugin_registration_test", example_dir / "minimax_dspark_adapter" / "__init__.py"
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    return SimpleNamespace(
        plugin=plugin,
        config_mapping=config_mapping,
        config_registrations=config_registrations,
        model_mapping=model_mapping,
        model_registrations=model_registrations,
        native_model=native_model,
        text_config=MiniMaxM3TextConfig,
    )


def test_plugin_registers_only_a3_alias_and_preserves_native_model(plugin_environment):
    env = plugin_environment
    env.plugin.register()
    assert env.model_mapping[NATIVE_ARCHITECTURE] is env.native_model
    assert env.model_registrations == [
        (FALLBACK_ARCHITECTURE, "minimax_dspark_adapter.model:MiniMaxFlashDSparkForCausalLM")
    ]
    assert set(env.model_mapping) == {NATIVE_ARCHITECTURE, FALLBACK_ARCHITECTURE}


def test_plugin_preserves_existing_minimax_text_config_mapping(plugin_environment):
    env = plugin_environment

    class NativeMiniMaxTextConfig:
        pass

    env.config_mapping["minimax_m3_text"] = NativeMiniMaxTextConfig
    env.plugin.register()
    assert env.config_mapping["minimax_m3_text"] is NativeMiniMaxTextConfig
    assert env.config_registrations == []


def test_plugin_registers_actual_text_config_when_mapping_is_missing(plugin_environment):
    env = plugin_environment
    env.plugin.register()
    assert env.config_mapping["minimax_m3_text"] is env.text_config
    assert env.config_registrations == [("minimax_m3_text", env.text_config)]
