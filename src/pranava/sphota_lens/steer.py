"""Steering the sphoṭa workspace — write a concept direction into the fused workspace band.

Adapts prabodha's steering idea (writer.concept_direction: a direction in residual space built from
unembedding rows of target concepts) to the ALM's byte core. We inject ``alpha * direction`` into
the hidden states *at the workspace band* during greedy decoding and observe the output shift — a
reproducible, seeded intervention on the paśyantī workspace.
"""
from __future__ import annotations

import torch


def concept_direction(core, target_bytes: list[int]) -> torch.Tensor:
    """Unit direction in the core's hidden space toward emitting `target_bytes`.

    Uses the tied head/embedding rows (the byte "unembeddings") — the residual-space analog of
    prabodha's ``concept_direction`` (weights @ U[ids]).
    """
    rows = core.model.head.weight[target_bytes]  # (k, d)
    g = rows.mean(dim=0)
    return (g / (g.norm() + 1e-8)).detach()


@torch.no_grad()
def greedy_steered(core, prefix_embeds: torch.Tensor, band: tuple[int, int],
                   direction: torch.Tensor | None, alpha: float, max_new: int = 24) -> list[int]:
    """Greedy decode while adding ``alpha*direction*||h||`` to hidden states inside the band.

    band = [b1, b2) over the block outputs (0 = input embeddings, i+1 = block i output). With
    ``direction=None`` or ``alpha=0`` this is plain greedy decoding — the control condition.
    """
    b1, b2 = band
    x = prefix_embeds.to(core.torch_device)
    out: list[int] = []
    for _ in range(max_new):
        h = x
        for li, block in enumerate(core.model.blocks):
            h = block(h)
            layer_idx = li + 1  # block i output is layer i+1
            if direction is not None and alpha != 0.0 and b1 <= layer_idx < b2:
                scale = alpha * h[:, -1:, :].norm(dim=-1, keepdim=True)
                h = h.clone()
                h[:, -1:, :] = h[:, -1:, :] + scale * direction
        logits = core.model.head(core.model.norm_f(h))[0, -1]
        nxt = int(torch.argmax(logits).item())
        out.append(nxt)
        nxt_emb = core.model.embed(
            torch.tensor([[nxt]], device=core.torch_device, dtype=torch.long)
        )
        x = torch.cat([x, nxt_emb], dim=1)
    return out
