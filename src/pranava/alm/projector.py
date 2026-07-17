"""Sphoṭa Projector — madhyamā: continuous acoustic frames → core-embedding tokens.

Maps a speech encoder's frame features ``(B, T_a, D_enc)`` into the Sanskrit core's embedding
space ``(B, T_a', d_model)``, optionally downsampling in time. A convolutional stack (temporal
context + stride downsample) followed by an MLP — the standard, effective audio-LLM connector.
The core's ``structural_bias`` is added so the projected tokens live in the space the frozen
block stack expects (see core_adapter).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SphotaProjector(nn.Module):
    def __init__(self, d_enc: int, d_model: int, downsample: int = 4, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_model
        # temporal conv with stride=downsample compresses the frame rate (dhvani → padas)
        self.conv = nn.Sequential(
            nn.Conv1d(d_enc, hidden, kernel_size=5, stride=downsample, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        self.d_model = d_model
        self.downsample = downsample

    def forward(self, frames: torch.Tensor, structural_bias: torch.Tensor | None = None) -> torch.Tensor:
        """frames (B, T_a, D_enc) → tokens (B, T_a', d_model)."""
        x = frames.transpose(1, 2)  # (B, D_enc, T_a)
        x = self.conv(x)  # (B, hidden, T_a')
        x = x.transpose(1, 2)  # (B, T_a', hidden)
        x = self.proj(x)  # (B, T_a', d_model)
        if structural_bias is not None:
            x = x + structural_bias
        return x

    def out_len(self, t_a: int) -> int:
        return (t_a + 2 * 2 - 5) // self.downsample + 1
