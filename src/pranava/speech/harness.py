"""Speech-model representation harness (Pillar II, E0).

Loads a self-supervised speech encoder (WavLM/HuBERT-class) on the GB10 GPU and
extracts per-frame, per-layer hidden states — the substrate the sphoṭa/holism
probes operate on. Deterministic, device-explicit, shape-checked.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_MODEL = "microsoft/wavlm-base"
TARGET_SR = 16000


def cuda_available() -> bool:
    return torch.cuda.is_available()


@dataclass(slots=True)
class HiddenStates:
    """Layer-wise frame representations for one utterance.

    ``layers`` has shape ``[n_layers, n_frames, dim]`` (layer 0 = embeddings).
    """

    layers: torch.Tensor
    frame_rate_hz: float
    model_name: str
    device: str
    seconds: float
    elapsed_s: float

    @property
    def n_layers(self) -> int:
        return int(self.layers.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.layers.shape[1])

    @property
    def dim(self) -> int:
        return int(self.layers.shape[2])


@dataclass(slots=True)
class SpeechEncoder:
    model_name: str = DEFAULT_MODEL
    device: str = "cuda"
    _model: object = None
    _fe: object = None

    def load(self) -> "SpeechEncoder":
        from transformers import AutoFeatureExtractor, AutoModel

        if self.device == "cuda" and not cuda_available():
            raise RuntimeError("CUDA requested but not available")
        self._fe = AutoFeatureExtractor.from_pretrained(self.model_name)
        self._model = (
            AutoModel.from_pretrained(self.model_name, output_hidden_states=True)
            .to(self.device)
            .eval()
        )
        return self

    @torch.no_grad()
    def encode(self, waveform: np.ndarray, sr: int = TARGET_SR) -> HiddenStates:
        if self._model is None:
            self.load()
        wav = _to_mono_16k(waveform, sr)
        inputs = self._fe(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        t0 = time.time()
        out = self._model(**inputs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        # stack layers → [n_layers, n_frames, dim]
        stacked = torch.stack([h.squeeze(0) for h in out.hidden_states], dim=0)
        n_frames = stacked.shape[1]
        seconds = len(wav) / TARGET_SR
        frame_rate = n_frames / seconds if seconds > 0 else 0.0
        return HiddenStates(
            layers=stacked,
            frame_rate_hz=frame_rate,
            model_name=self.model_name,
            device=str(stacked.device),
            seconds=seconds,
            elapsed_s=elapsed,
        )


def _to_mono_16k(waveform: np.ndarray, sr: int) -> np.ndarray:
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=tuple(range(wav.ndim - 1))) if wav.shape[-1] < wav.shape[0] else wav.mean(axis=0)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if sr != TARGET_SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav
