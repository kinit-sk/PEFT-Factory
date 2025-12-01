from __future__ import annotations

from typing import Optional

import importlib.util
from pathlib import Path
import warnings

from peft import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists
from peft.utils import ModulesToSaveWrapper
from torch.nn.modules import Module

try:
    from .layer import SVFTLayer
except ImportError:  # pragma: no cover - handled during dynamic discovery
    _model_file = Path(__file__)
    _layer_file = _model_file.parent / "layer.py"
    if _layer_file.exists():
        spec = importlib.util.spec_from_file_location("svft_layer", _layer_file)
        if spec and spec.loader:
            layer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(layer_module)
            SVFTLayer = layer_module.SVFTLayer  # type: ignore
    else:  # pragma: no cover - defensive branch
        raise ImportError(f"Could not find layer.py in {_model_file.parent}")


TRANSFORMERS_MODELS_TO_SVFT_TARGET_MODULES_MAPPING = {
    "llama": ["q_proj", "k_proj"],
}


class SVFTModel(BaseTuner):
    prefix: str = "svft_"
    tuner_layer_cls = SVFTLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_SVFT_TARGET_MODULES_MAPPING

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)

    @staticmethod
    def _prepare_adapter_config(peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        if peft_config.target_modules is None:
            target_modules = TRANSFORMERS_MODELS_TO_SVFT_TARGET_MODULES_MAPPING.get(model_config.get("model_type"))
            if target_modules is None:
                raise ValueError("Please specify `target_modules` in `peft_config` for SVFT.")
            peft_config.target_modules = set(target_modules)
        return peft_config

    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: Module,
        target_name: str,
        parent: Module,
        current_key: str,
    ) -> None:
        new_module = self._create_new_module(peft_config, target, adapter_name)

        if adapter_name != self.active_adapter:
            new_module.requires_grad_(False)

        self._replace_module(parent, target_name, new_module, target)

    def _create_new_module(
        self,
        peft_config: PeftConfig,
        target: Module,
        adapter_name: str,
    ) -> Module:
        if not isinstance(target, SVFTLayer):
            new_module = SVFTLayer(target, adapter_name, peft_config)
        else:
            new_module = target
            new_module.update_layer(target.base_layer, adapter_name, peft_config)
        return new_module

    def _replace_module(self, parent: Module, child_name: str, new_module: Module, child: Module) -> None:
        setattr(parent, child_name, new_module)

        if hasattr(child, "base_layer"):
            child = child.base_layer

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        for name, module in new_module.named_modules():
            weight = child.qweight if hasattr(child, "qweight") else child.weight
            module.to(weight.device)

    def _check_target_module_exists(self, peft_config: PeftConfig, key: str):
        return check_target_module_exists(peft_config, key)

    def _mark_only_adapters_as_trainable(self, model: Module) -> None:
        for name, param in model.named_parameters():
            if "svft_layers" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _set_adapter_layers(self, enabled: bool) -> None:
        for module in self.model.modules():
            if isinstance(module, (SVFTLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        active_list = adapter_name if isinstance(adapter_name, list) else [adapter_name]
        for module in self.model.modules():
            if isinstance(module, SVFTLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                payload = active_list if len(active_list) > 1 else active_list[0]
                module.set_adapter(payload)
                if inference_mode:
                    module.set_requires_grad(active_list, requires_grad=False)
        self.active_adapter = adapter_name