"""Adapter over the prabhasa Nemotron-H Sanskrit core for multi-modal use.

The core (``scripts/m2/train_130m.py:NemotronH``) embeds byte tokens internally
(``features`` = ``embed(tokens)`` → blocks → ``norm_f``; ``head`` tied to ``embed``).
For an Audio LM we must instead feed *projected audio embeddings* into the block stack.
This adapter reuses the checkpoint loader (``NemotronHRunner``) and exposes:

  * ``embed_tokens`` — byte ids → (B,T,d) token embeddings (paśyantī input for text),
  * ``features_embeds`` — (B,T,d) embeddings → pre-head hidden states (the sphoṭa workspace),
  * ``forward_embeds`` — embeddings → next-token logits over the byte vocab,
  * ``greedy_from_embeds`` — decode bytes from a leading embedding prefix.

Nothing in the prabhasa repo is modified; we drive its own submodules (``embed``, ``blocks``,
``norm_f``, ``head``) directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from prabhasa.infrastructure.ml.inference import NemotronHRunner

def _projects_root() -> Path:
    """Resolve the projects tree on both host (/home/sharaths/projects) and container (/work)."""
    import os

    env = os.environ.get("PRANAVA_PROJECTS_ROOT")
    for cand in (env, "/work", "/home/sharaths/projects"):
        if cand and (Path(cand) / "prabhasa-samskrutam").exists():
            return Path(cand)
    return Path("/home/sharaths/projects")


_CKPT_REL = {
    "m2": "prabhasa-samskrutam/data/checkpoints/m2/treatment/final.pt",
    "m3": "prabhasa-samskrutam/data/checkpoints/m3/treatment/final.pt",
}
# Fully-trained small cores (see plan). m2 = 200M byte-level; m3 = ~350M + kāraka head.
CHECKPOINTS = {arm: str(_projects_root() / rel) for arm, rel in _CKPT_REL.items()}


def alm_runnable() -> bool:
    """True only where the ALM core can actually run: checkpoint + CUDA + mamba_ssm.

    Guards test skips so the suite passes on the host venv (no mamba_ssm) and runs in the
    prabhasa/nemo-gb10 container.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import mamba_ssm  # noqa: F401
    except Exception:
        return False
    return Path(CHECKPOINTS["m2"]).exists()


@dataclass(slots=True)
class SanskritCore:
    arm: str = "m2"
    device: str | None = None
    _runner: NemotronHRunner = None

    def load(self) -> "SanskritCore":
        ckpt = CHECKPOINTS[self.arm]
        if not Path(ckpt).exists():
            raise FileNotFoundError(ckpt)
        self._runner = NemotronHRunner(ckpt, device=self.device)
        return self

    # -- properties -----------------------------------------------------------
    @property
    def model(self):
        return self._runner.model

    @property
    def d_model(self) -> int:
        return int(self._runner.config["d_model"])

    @property
    def vocab_size(self) -> int:
        return self._runner.vocab_size

    @property
    def torch_device(self):
        return self._runner.device

    # -- embedding / forward paths -------------------------------------------
    @property
    def structural_bias(self) -> torch.Tensor:
        """The constant (d,) vector the core adds for structural channel-0.

        The trained core (treatment arm) is ``structured``: at inference it feeds boundary=role=0,
        so ``features`` adds ``boundary_embed(0) + role_embed(0)`` — *learned* vectors, not zeros.
        Any embedding fed into the block stack (text or projected audio) must live in this same
        space, so we expose the bias for callers to add. Zero for an unstructured (baseline) core.
        """
        if not getattr(self.model, "structured", False):
            return torch.zeros(self.d_model, device=self.torch_device)
        zero = torch.zeros(1, 1, dtype=torch.long, device=self.torch_device)
        with torch.no_grad():  # a fixed inference constant — never part of the training graph
            return (self.model.boundary_embed(zero) + self.model.role_embed(zero)).view(-1).detach()

    @torch.no_grad()
    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Byte ids (B,T) long → core-input embeddings (B,T,d), incl. the structural-zero bias."""
        return self.model.embed(ids.to(self.torch_device)) + self.structural_bias

    @torch.no_grad()
    def features_embeds(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """(B,T,d) embeddings → pre-head normalized hidden states (B,T,d).

        Mirrors ``NemotronH.features`` but starting from provided embeddings (no ``embed``
        lookup, no structured channels). This IS the sphoṭa workspace the lens will read.
        """
        x = inputs_embeds.to(self.torch_device)
        for block in self.model.blocks:
            x = block(x)
        return self.model.norm_f(x)

    @torch.no_grad()
    def features_per_layer(self, inputs_embeds: torch.Tensor) -> list[torch.Tensor]:
        """(B,T,d) embeddings → list of per-layer hidden states [input, block_0, …, block_{L-1}].

        The Sphoṭa-Lens reads these: each entry is (B,T,d). Entry 0 is the input embeddings;
        entry i+1 is the output of block i. The final entry, after ``norm_f``, is the workspace.
        """
        x = inputs_embeds.to(self.torch_device)
        layers = [x]
        for block in self.model.blocks:
            x = block(x)
            layers.append(x)
        return layers

    @torch.no_grad()
    def features_final_ablated(self, inputs_embeds: torch.Tensor, ablate_layer: int,
                               positions: slice) -> torch.Tensor:
        """Final hidden states (B,T,d) with layer ``ablate_layer``'s ``positions`` mean-collapsed.

        A true activation ablation: at the output of block ``ablate_layer-1`` (layer index
        ``ablate_layer``; 0 = input), the selected positions are replaced by their own mean over
        that span, destroying the meaning-carrying variation there while preserving gross level.
        Used to causally locate the sphoṭa layer (where ablating audio positions most hurts meaning).
        """
        x = inputs_embeds.to(self.torch_device)
        if ablate_layer == 0:
            x = x.clone()
            x[:, positions] = x[:, positions].mean(dim=1, keepdim=True)
        for li, block in enumerate(self.model.blocks):
            x = block(x)
            if li + 1 == ablate_layer:
                x = x.clone()
                x[:, positions] = x[:, positions].mean(dim=1, keepdim=True)
        return self.model.norm_f(x)

    @torch.no_grad()
    def forward_embeds(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """(B,T,d) embeddings → next-token logits (B,T,vocab)."""
        return self.model.head(self.features_embeds(inputs_embeds))

    @torch.no_grad()
    def greedy_from_embeds(self, prefix_embeds: torch.Tensor, max_new: int = 64) -> list[int]:
        """Greedy-decode byte ids conditioned on a leading embedding prefix (B=1)."""
        assert prefix_embeds.shape[0] == 1
        x = prefix_embeds.to(self.torch_device)
        out: list[int] = []
        for _ in range(max_new):
            logits = self.forward_embeds(x)[0, -1]
            nxt = int(torch.argmax(logits).item())
            out.append(nxt)
            nxt_emb = self.model.embed(
                torch.tensor([[nxt]], device=self.torch_device, dtype=torch.long)
            )
            x = torch.cat([x, nxt_emb], dim=1)
        return out
