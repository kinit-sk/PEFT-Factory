
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer

class SVFTLayer(nn.Module, BaseTunerLayer):
    adapter_layer_names = ("svft_layers",)

    def __init__(self, base_layer: nn.Module, adapter_name: str):
        super().__init__()
        self.base_layer = base_layer
        self.bitfit_layers = nn.ModuleDict({})
        self.update_layer(self.base_layer, adapter_name)
        self._active_adapter = adapter_name
        self.merged_adapters = []