"""Per-position cumulative representations for speech and text models (E2).

For each utterance we produce a matrix ``[len(POSITIONS), dim]``: row i is the
mean-pool of all frames/tokens up to relative position POSITIONS[i]. Both modalities
share the relative-position grid so their trajectories are comparable (with the stated
granularity caveat in the pre-registration).
"""
from __future__ import annotations

import numpy as np
import torch

from pranava.experiments.holism import POSITIONS
from pranava.speech.harness import SpeechEncoder

TARGET_SR = 16000


def _cumulative_pool(frames: np.ndarray, positions: np.ndarray = POSITIONS) -> np.ndarray:
    """frames: [N, dim] → [len(positions), dim] cumulative mean up to each rel position."""
    n = frames.shape[0]
    out = np.empty((len(positions), frames.shape[1]), dtype=np.float32)
    for i, t in enumerate(positions):
        k = max(1, int(np.ceil(t * n)))
        out[i] = frames[:k].mean(axis=0)
    return out


class SpeechFeatureExtractor:
    def __init__(self, layer: int = 9, model_name: str = "microsoft/wavlm-base"):
        self.layer = layer
        self.encoder = SpeechEncoder(model_name=model_name).load()

    def positions_matrix(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        hs = self.encoder.encode(waveform, sr=sr)  # [n_layers, n_frames, dim]
        frames = hs.layers[self.layer].to("cpu").numpy().astype(np.float32)
        return _cumulative_pool(frames)


class TextFeatureExtractor:
    """Per-token hidden states from a text LM (GPT-2 small by default)."""

    def __init__(self, layer: int = 6, model_name: str = "gpt2", device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        self.layer = layer
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModel.from_pretrained(model_name, output_hidden_states=True)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def positions_matrix(self, text: str) -> np.ndarray:
        ids = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model(**ids)
        hidden = out.hidden_states[self.layer].squeeze(0).to("cpu").numpy().astype(np.float32)
        return _cumulative_pool(hidden)
