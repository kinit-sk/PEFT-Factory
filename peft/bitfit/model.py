from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists
from peft import PeftConfig
from torch.nn.modules import Module
import importlib.util
from pathlib import Path
import warnings
from peft.utils import ModulesToSaveWrapper

TRANSFORMERS_MODELS_TO_BITFIT_TARGET_MODULES_MAPPING = {
    "llama": ["q_proj", "v_proj"],
}

# Handle both relative import (when used as a package) and dynamic loading (when loaded via importlib)
try:
    from .layer import BitFitLayer
except ImportError:
    # Fallback for dynamic loading: load layer module directly from the same directory
    _model_file = Path(__file__)
    _layer_file = _model_file.parent / "layer.py"
    if _layer_file.exists():
        spec = importlib.util.spec_from_file_location("bitfit_layer", _layer_file)
        if spec and spec.loader:
            layer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(layer_module)
            BitFitLayer = layer_module.BitFitLayer
    else:
        raise ImportError(f"Could not find layer.py in {_model_file.parent}")


class BitFitModel(BaseTuner):
    prefix: str = "bitfit_"

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
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_BITFIT_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_BITFIT_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
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
        # replace the original module with a same new module
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
        if not isinstance(target, BitFitLayer):
            new_module = BitFitLayer(target, adapter_name)
        else:
            new_module = target
            new_module.update_layer(target.base_layer, adapter_name)
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

    def _check_target_module_exists(self, peft_config: PeftConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)

    def _mark_only_adapters_as_trainable(self, model: Module):
        for n, p in model.named_parameters():
            if self.prefix not in n or "bias" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def _set_adapter_layers(self, enabled: bool) -> None:
        for module in self.model.modules():
            if isinstance(module, (BitFitLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)
        

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        self._set_adapter_layers(enabled=False)


    def set_adapter(self, adapter_name: str) -> None:
        for module in self.model.modules():
            if isinstance(module, BitFitLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name