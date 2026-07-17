"""Śabda-ALM pipeline (Phase 0) — audio (vaikharī) → text, through the sphoṭa workspace.

Composes: ParakeetEncoder (dhvani frames) → SphotaProjector (madhyamā tokens) → SanskritCore
(paśyantī/sphoṭa workspace) → greedy byte decode. Phase 0 proves the wiring end-to-end and its
trainability (see tests/alm/test_projector_overfit.py); understanding-quality comes in Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.encoder import ParakeetEncoder
from pranava.alm.projector import SphotaProjector


@dataclass(slots=True)
class SabdaALM:
    core: SanskritCore
    encoder: ParakeetEncoder
    projector: SphotaProjector

    @classmethod
    def build(cls, arm: str = "m2", downsample: int = 4,
              encoder_model: str = "nvidia/parakeet-tdt-0.6b-v3") -> "SabdaALM":
        core = SanskritCore(arm=arm, device="cuda").load()
        enc = ParakeetEncoder(model_name=encoder_model).load()
        # probe encoder dim with a short dummy to size the projector
        d_enc = enc.encode(np.zeros(16000, dtype=np.float32)).shape[-1]
        proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=downsample).to(
            core.torch_device
        )
        return cls(core=core, encoder=enc, projector=proj)

    def audio_tokens(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        frames = self.encoder.encode(waveform, sr=sr)
        return self.projector(frames, structural_bias=self.core.structural_bias)

    @torch.no_grad()
    def transcribe(self, waveform: np.ndarray, sr: int, max_new: int = 48) -> bytes:
        tokens = self.audio_tokens(waveform, sr)
        out = self.core.greedy_from_embeds(tokens, max_new=max_new)
        return bytes(b for b in out if b != 0)
