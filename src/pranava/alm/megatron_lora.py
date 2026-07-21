"""LoRA for the Megatron-Core 1.13B (StructuredNemotronH) at TP=1.

The 200M LoRA (pranava.alm.lora) targets plain ``nn.Linear`` MLPs; Megatron's MambaModel instead uses
``ColumnParallelLinear`` / ``RowParallelLinear`` whose ``forward`` returns ``(output, bias)`` tuples.
At TP=1 (the only configuration prabhasa/pranava run) these are functionally dense linears with a
``.weight`` of shape (out, in) — so LoRA is a forward-wrap:  y = W x (+bias)  →  y + scale·B(A(x)).

Targets (per the prabhasa architecture digest; Mamba-2 mixers are state-space kernels and are NOT
LoRA-eligible):
  * every layer's   mlp.linear_fc1 / mlp.linear_fc2         (all 32 logical layers)
  * attention layers' self_attention.linear_qkv / linear_proj (the ~1/8 attention blocks)

``inject_megatron_lora(model, r, alpha)`` returns the list of new trainable params;
``megatron_lora_state_dict(model)`` serializes them; both mirror pranava.alm.lora's surface.
"""
from __future__ import annotations

import math

import torch
from torch import nn

TARGET_SUFFIXES = ("mlp.linear_fc1", "mlp.linear_fc2",
                   "self_attention.linear_qkv", "self_attention.linear_proj")


class _LoRAPair(nn.Module):
    """Holds A (r×in) and B (out×r); B starts at zero so injection is exactly identity at step 0."""

    def __init__(self, d_in: int, d_out: int, r: int, alpha: float, dtype, device):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, d_in, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(d_out, r, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.T) @ self.B.T * self.scale


def _wrap(mod: nn.Module, pair: _LoRAPair) -> None:
    orig = mod.forward

    def fwd(x, *args, **kwargs):
        out = orig(x, *args, **kwargs)
        if isinstance(out, tuple):  # Megatron linears return (output, bias[, ...])
            return (out[0] + pair.delta(x), *out[1:])
        return out + pair.delta(x)

    mod.forward = fwd


def inject_megatron_lora(model: nn.Module, r: int = 16, alpha: float | None = None) -> list[nn.Parameter]:
    """Attach LoRA to every TARGET_SUFFIXES linear under ``model``; returns the trainable params.

    Pairs are registered on the wrapped module as ``lora_pair`` so state_dict round-trips work.
    """
    alpha = alpha if alpha is not None else 2 * r
    params: list[nn.Parameter] = []
    for name, mod in model.named_modules():
        if not name.endswith(TARGET_SUFFIXES):
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 2:
            continue
        d_out, d_in = w.shape
        pair = _LoRAPair(d_in, d_out, r, alpha, w.dtype, w.device)
        mod.add_module("lora_pair", pair)
        _wrap(mod, pair)
        params += [pair.A, pair.B]
    if not params:
        raise RuntimeError("inject_megatron_lora: no target linears found — wrong model?")
    return params


def megatron_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if ".lora_pair." in n}


def load_megatron_lora(model: nn.Module, sd: dict[str, torch.Tensor]) -> int:
    """Load a saved LoRA state dict into an injected model; returns #tensors loaded.

    Rank expansion: if the injected adapter has a larger r than the checkpoint, the old
    weights land in the FIRST r_old ranks (A rows / B cols); the extra B cols stay zero,
    so the loaded function is exactly the checkpoint's (scale alpha/r is 2 under the
    default alpha=2r on both sides) while the extra ranks stay free to learn."""
    own = dict(model.named_parameters())
    n = 0
    for k, v in sd.items():
        if k not in own:
            continue
        o = own[k]
        v = v.to(o.device, o.dtype)
        if o.shape == v.shape:
            o.data.copy_(v)
        elif k.endswith(".A") and o.shape[0] > v.shape[0] and o.shape[1] == v.shape[1]:
            o.data[: v.shape[0]].copy_(v)
        elif k.endswith(".B") and o.shape[1] > v.shape[1] and o.shape[0] == v.shape[0]:
            o.data[:, : v.shape[1]].copy_(v)
        else:
            raise ValueError(f"load_megatron_lora: incompatible shape for {k}: {tuple(v.shape)} → {tuple(o.shape)}")
        n += 1
    return n
