from peft.tuners.tuners_utils import BaseTuner
from torch.nn.modules import Module
import importlib.util
from pathlib import Path

try:
    from .layer import SVDLayer
except ImportError:
    # Fallback for dynamic loading: load layer module directly from the same directory
    _model_file = Path(__file__)
    _layer_file = _model_file.parent / "layer.py"
    if _layer_file.exists():
        spec = importlib.util.spec_from_file_location("svd_layer", _layer_file)
        if spec and spec.loader:
            layer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(layer_module)
            BitFitLayer = layer_module.SVDLayer
    else:
        raise ImportError(f"Could not find layer.py in {_model_file.parent}")


class BitFitModel(BaseTuner):
    prefix: str = "svd_"

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":  
                raise
            return getattr(self.model, name)