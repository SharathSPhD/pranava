"""Audio encoder (vaikharī → dhvani frames) — NVIDIA Parakeet/Canary via NeMo.

Loads a frozen FastConformer ASR encoder and returns per-frame acoustic features for a waveform.
Runs inside prabhasa/nemo-gb10 (NeMo present). The encoder is the continuous acoustic field; the
Sphoṭa Projector turns it into core-embedding tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
TARGET_SR = 16000


@dataclass(slots=True)
class ParakeetEncoder:
    model_name: str = DEFAULT_MODEL
    device: str = "cuda"
    _model: object = None
    _dim: int = 0

    def load(self) -> "ParakeetEncoder":
        from nemo.collections.asr.models import ASRModel

        m = ASRModel.from_pretrained(self.model_name, map_location=self.device)
        m.eval()
        self._model = m
        return self

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, waveform: np.ndarray, sr: int = TARGET_SR) -> torch.Tensor:
        """waveform (mono float32) → encoder frames (1, T_frames, D_enc)."""
        import torch as _t

        wav = np.asarray(waveform, dtype=np.float32)
        if sr != TARGET_SR:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sig = _t.tensor(wav, device=self.device).unsqueeze(0)
        length = _t.tensor([sig.shape[1]], device=self.device)
        # NeMo ASR: preprocessor → encoder. encoded is (B, D, T); transpose to (B, T, D).
        proc, proc_len = self._model.preprocessor(input_signal=sig, length=length)
        encoded, enc_len = self._model.encoder(audio_signal=proc, length=proc_len)
        frames = encoded.transpose(1, 2).contiguous()  # (B, T, D)
        self._dim = int(frames.shape[-1])
        return frames
