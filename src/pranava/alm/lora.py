"""LoRA adapters for the byte-level Mamba-2 Sanskrit core.

Injects low-rank adapters into the core's projection Linears (``in_proj``/``out_proj`` of the Mamba
mixers and attention blocks). The base weights stay frozen; only the rank-r ``A``/``B`` factors
train, jointly with the Sphoṭa Projector. A self-contained implementation (no peft model surgery on
the custom module) so behaviour is fully controlled.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable rank-r update: y = Wx + (B A x)·(α/r)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        # B stays zero at init → the adapter is a no-op until trained (stable).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = torch.nn.functional.linear(self.drop(x), self.A)
        lora = torch.nn.functional.linear(lora, self.B)
        return out + self.scale * lora.to(out.dtype)


def inject_lora(model: nn.Module, targets=("in_proj", "out_proj"), r: int = 8, alpha: int = 16
                ) -> list[nn.Parameter]:
    """Replace target Linears in ``model`` with :class:`LoRALinear`. Returns trainable LoRA params."""
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
                replaced += 1
    params = [p for n, p in model.named_parameters() if (".A" in n or ".B" in n) and p.requires_grad]
    if replaced == 0:
        raise RuntimeError(f"no target Linears {targets} found to LoRA-inject")
    return params


def lora_state_dict(model: nn.Module) -> dict:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if (".A" in n or ".B" in n)}
