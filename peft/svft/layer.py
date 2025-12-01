
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from peft import PeftConfig


class _SVFTAdapter(nn.Module):
    """Dense singular-vector adapter with sparse learnable corrections."""

    def __init__(self, base_layer: nn.Linear, config: PeftConfig) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("SVFT currently supports nn.Linear modules only.")

        self.out_features = base_layer.out_features
        self.in_features = base_layer.in_features
        self.param_dtype = base_layer.weight.dtype
        self.device = base_layer.weight.device

        weight_fp32 = base_layer.weight.detach().to(torch.float32)
        u, s, vh = torch.linalg.svd(weight_fp32, full_matrices=False)

        max_rank = s.shape[0]
        rank = config.rank if config.rank is not None else max_rank
        rank = max(1, min(rank, max_rank))
        diff_rank = max_rank - rank

        if config.fill_orthonormal and diff_rank > 0:
            u_extra = torch.randn((u.shape[0], diff_rank), device=u.device, dtype=u.dtype)
            torch.nn.init.orthogonal_(u_extra)
            v_extra = torch.randn((diff_rank, vh.shape[1]), device=vh.device, dtype=vh.dtype)
            torch.nn.init.orthogonal_(v_extra)
            u = torch.cat([u[:, :rank], u_extra], dim=1)
            vh = torch.cat([vh[:rank, :], v_extra], dim=0)
            s = torch.cat([s[:rank], torch.zeros(diff_rank, device=s.device)], dim=0)
            rank = u.shape[1]
        else:
            u = u[:, :rank]
            vh = vh[:rank, :]
            s = s[:rank]

        self.rank = rank
        self.register_buffer(
            "u",
            u.to(device=self.device, dtype=self.param_dtype).contiguous(),
        )
        self.register_buffer(
            "v",
            vh.to(device=self.device, dtype=self.param_dtype).contiguous(),
        )
        self.register_buffer(
            "s_base",
            s.to(device=self.device, dtype=self.param_dtype).contiguous(),
        )

        row_idx, col_idx = self._build_pattern(config.pattern, config.off_diag)
        self.register_buffer("row_idx", row_idx.contiguous())
        self.register_buffer("col_idx", col_idx.contiguous())

        num_slots = row_idx.numel()
        if num_slots == 0:
            raise ValueError("SVFT pattern produced zero trainable positions.")

        self.delta = nn.Parameter(torch.zeros(num_slots, device=self.device, dtype=self.param_dtype))
        self.gate = nn.Parameter(torch.tensor(0.0, device=self.device, dtype=self.param_dtype))

    def _build_pattern(self, pattern: str, off_diag: int) -> tuple[torch.Tensor, torch.Tensor]:
        pattern = pattern.lower()
        off_diag = max(0, int(off_diag))
        total = self.rank * self.rank

        if pattern == "banded":
            rows = []
            cols = []
            for i in range(self.rank):
                for shift in range(-off_diag, off_diag + 1):
                    j = i + shift
                    if 0 <= j < self.rank:
                        rows.append(i)
                        cols.append(j)
            if not rows:
                rows = list(range(self.rank))
                cols = list(range(self.rank))
            row_idx = torch.tensor(rows, device=self.device, dtype=torch.long)
            col_idx = torch.tensor(cols, device=self.device, dtype=torch.long)
        elif pattern == "random":
            target = self.rank * (2 * off_diag + 1) - off_diag * (off_diag + 1)
            k = max(1, min(target, total))
            flat = torch.randperm(total, device=self.device)[:k]
            row_idx = flat // self.rank
            col_idx = flat % self.rank
        elif pattern == "top_k":
            target = self.rank * (2 * off_diag + 1) - off_diag * (off_diag + 1)
            k = max(1, min(target, total))
            coeffs = torch.abs(torch.outer(self.s_base, self.s_base))
            flat = torch.topk(coeffs.flatten(), k=k).indices
            row_idx = flat // self.rank
            col_idx = flat % self.rank
        else:
            raise ValueError(f"Unsupported SVFT sparsity pattern: {pattern}")

        flat_idx = row_idx * self.rank + col_idx
        unique_flat = torch.unique(flat_idx)
        unique_row = (unique_flat // self.rank).to(torch.long)
        unique_col = (unique_flat % self.rank).to(torch.long)
        return unique_row, unique_col

    def full_weight(self) -> torch.Tensor:
        correction = torch.zeros((self.rank, self.rank), device=self.device, dtype=self.param_dtype)
        values = self.delta.to(dtype=correction.dtype)
        correction[self.row_idx, self.col_idx] = values
        correction = correction * torch.sigmoid(self.gate.to(dtype=correction.dtype))

        s_matrix = torch.diag(self.s_base) + correction
        weight = self.u @ s_matrix @ self.v
        return weight.to(dtype=self.param_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return F.linear(x, self.full_weight())

    def extra_repr(self) -> str:
        return f"rank={self.rank}, slots={self.delta.numel()}"


class SVFTLayer(nn.Module, BaseTunerLayer):
    adapter_layer_names = ("svft_layers",)

    def __init__(self, base_layer: nn.Module, adapter_name: str, config: PeftConfig) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("SVFTLayer expects nn.Linear modules as base layers.")

        self.base_layer = base_layer
        self.svft_layers = nn.ModuleDict({})
        self.layer_configs: Dict[str, PeftConfig] = {}
        self.merged_adapters: list[str] = []
        self._cached_base_weight = base_layer.weight.detach().clone()
        self._cached_base_bias = (
            base_layer.bias.detach().clone() if base_layer.bias is not None else None
        )

        self.update_layer(base_layer, adapter_name, config)
        self.set_adapter(adapter_name)
        self._disable_adapters = False

    def update_layer(self, layer: nn.Module, adapter_name: str, config: PeftConfig) -> None:
        self.layer_configs[adapter_name] = config
        adapter = _SVFTAdapter(layer, config)
        self.svft_layers[adapter_name] = adapter
        self._move_adapter_to_device_of_base_layer(adapter_name)

        current = self.active_adapter
        if isinstance(current, list):
            key = current if len(current) > 1 else current[0]
        else:
            key = current
        self.set_adapter(key)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        if len(self.active_adapters) != 1:
            raise ValueError("SVFT only supports a single active adapter per layer during forward.")

        active = self.active_adapters[0]
        adapter = self.svft_layers[active]
        if adapter is None:
            return self.base_layer(x, *args, **kwargs)

        weight = adapter.full_weight().to(self.base_layer.weight.dtype)
        x = self._cast_input_dtype(x, weight.dtype)
        bias = self.base_layer.bias
        return F.linear(x, weight, bias)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        if len(adapter_names) != 1:
            raise ValueError("SVFT currently supports merging one adapter at a time.")

        name = adapter_names[0]
        adapter = self.svft_layers.get(name)
        if adapter is None:
            return

        merged_weight = adapter.full_weight().to(self.base_layer.weight.dtype)
        if safe_merge and not torch.isfinite(merged_weight).all():
            raise ValueError("NaNs detected during SVFT merge; aborting.")

        if not self.merged:
            self._cached_base_weight = self.base_layer.weight.detach().clone()
            self._cached_base_bias = (
                self.base_layer.bias.detach().clone() if self.base_layer.bias is not None else None
            )

        self.base_layer.weight.data.copy_(merged_weight)
        self.merged_adapters.append(name)

    def unmerge(self) -> None:
        if not self.merged:
            return

        self.base_layer.weight.data.copy_(self._cached_base_weight)
        if self.base_layer.bias is not None and self._cached_base_bias is not None:
            self.base_layer.bias.data.copy_(self._cached_base_bias)
        self.merged_adapters.clear()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        rep = super().__repr__()
        return "svft." + rep