"""Megatron1BCore — the ALM surface over the fully-trained 1B Nemotron-H (Megatron-Core).

Loads prabhasa's 1B `m4/final.pt` (a Megatron-Core `MambaModel` wrapped in `StructuredNemotronH`)
and exposes the same surface as `SanskritCore` so the Sphoṭa Projector + Sphoṭa-Lens port unchanged.

Injection uses Megatron's native ``decoder_input`` (precomputed ``[s,b,h]`` embeddings — MambaModel
skips its internal embedding when given it), the 1B analog of the small core's ``features_embeds``.

Runs ONLY in prabhasa/nemo-5090:26.02 on the RTX 5090 (Megatron + mamba). Reuses the exact model
builder from ``scripts/m4/megatron_pretrain.py`` so the architecture matches the checkpoint.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml


def _load_m4_module(prabhasa_root: Path):
    path = prabhasa_root / "scripts" / "m4" / "megatron_pretrain.py"
    spec = importlib.util.spec_from_file_location("prabhasa_m4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass(slots=True)
class Megatron1BCore:
    prabhasa_root: Path = Path("/work/prabhasa-samskrutam")
    checkpoint: Path = Path("/work/prabhasa-samskrutam/data/checkpoints/m4/final.pt")
    config_yaml: Path = Path("/work/prabhasa-samskrutam/configs/train/nemotron_h_1b.yaml")
    seq_len: int = 4096
    device: str = "cuda"
    _m4: object = None
    _model: object = None
    _mcfg: dict = field(default_factory=dict)

    def load(self) -> "Megatron1BCore":
        self._m4 = _load_m4_module(self.prabhasa_root)
        self._m4.init_distributed(seed=0)
        self._mcfg = yaml.safe_load(self.config_yaml.read_text())["model"]
        blob = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = {k.replace("_orig_mod.", ""): v for k, v in blob["model"].items()}
        # The trained m4/final.pt is the baseline arm (no boundary/role embeds); match it.
        self._mcfg = dict(self._mcfg)
        self._mcfg["structured_channels"] = "boundary_embed.weight" in state
        model = self._m4.StructuredNemotronH(self._mcfg, self.seq_len)
        model.load_state_dict(state, strict=True)
        self._model = model.to(self.device).eval()
        return self

    @property
    def d_model(self) -> int:
        return int(self._mcfg["d_model"])

    @property
    def vocab_size(self) -> int:
        return int(self._mcfg["vocab_size"])

    @property
    def torch_device(self):
        return torch.device(self.device)

    @property
    def structural_bias(self) -> torch.Tensor:
        m = self._model
        if not getattr(m, "structured", False):
            return torch.zeros(self.d_model, device=self.device)
        z = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            return (m.boundary_embed(z) + m.role_embed(z)).view(-1).detach()

    def _pos(self, b: int, t: int) -> torch.Tensor:
        return torch.arange(t, device=self.device).unsqueeze(0).expand(b, t)

    @torch.no_grad()
    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Byte ids (B,T) → core-input embeddings (B,T,d) incl. structural-zero bias."""
        ids = ids.to(self.device)
        b, t = ids.shape
        emb = self._model.core.embedding(input_ids=ids, position_ids=self._pos(b, t))  # (T,B,d)
        return emb.transpose(0, 1).contiguous() + self.structural_bias

    def _run_core(self, inputs_embeds: torch.Tensor):
        """(B,T,d) embeddings → (logits (B,T,vocab), hidden (B,T,d))."""
        x = inputs_embeds.to(self.device)
        b, t, _ = x.shape
        dec = x.transpose(0, 1).contiguous()  # Megatron wants (T,B,d)
        dummy_ids = torch.zeros(b, t, dtype=torch.long, device=self.device)
        captured = {}

        def hook(_m, _i, out):
            captured["h"] = out[0] if isinstance(out, tuple) else out  # (T,B,d)

        h = self._model.core.decoder.register_forward_hook(hook)
        try:
            logits = self._model.core(
                input_ids=dummy_ids, position_ids=self._pos(b, t),
                attention_mask=None, decoder_input=dec, labels=None,
            )  # (B,T,vocab) with post_process
        finally:
            h.remove()
        hidden = captured["h"].transpose(0, 1).contiguous()  # (B,T,d)
        return logits, hidden

    def logits_from_embeds(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Grad-enabled (B,T,d) → logits (B,T,vocab) for training the projector (core frozen)."""
        return self._run_core(inputs_embeds)[0]

    @torch.no_grad()
    def forward_embeds(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self._run_core(inputs_embeds)[0]

    @torch.no_grad()
    def features_embeds(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self._run_core(inputs_embeds)[1]

    @torch.no_grad()
    def greedy_from_embeds(self, prefix_embeds: torch.Tensor, max_new: int = 48,
                           stop_token: int | None = None) -> list[int]:
        """Greedy decode; stops early at ``stop_token`` (e.g. the instruct EOS sentinel) if given —
        same surface as SanskritCore.greedy_from_embeds so instruct/eval code ports unchanged."""
        x = prefix_embeds.to(self.device)
        out: list[int] = []
        for _ in range(max_new):
            logits = self.forward_embeds(x)[0, -1]
            nxt = int(torch.argmax(logits).item())
            if stop_token is not None and nxt == stop_token:
                break
            out.append(nxt)
            ids = torch.tensor([[nxt]], device=self.device, dtype=torch.long)
            nxt_emb = self.embed_tokens(ids)
            x = torch.cat([x, nxt_emb], dim=1)
        return out
