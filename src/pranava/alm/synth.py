"""Speech-out (vaikharī out) — NeMo TTS head so the ALM speaks its output.

Turns the core's text output back into audio (FastPitch + HiFiGAN), completing the speech-to-speech
loop: vaikharī(in) → dhvani → madhyamā → sphoṭa → text → vaikharī(out). Runs in prabhasa/nemo-gb10.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 22050


@dataclass(slots=True)
class TTSHead:
    _sp: object = None
    _voc: object = None

    def load(self) -> "TTSHead":
        from nemo.collections.tts.models import FastPitchModel, HifiGanModel

        self._sp = FastPitchModel.from_pretrained("tts_en_fastpitch").eval().cuda()
        self._voc = HifiGanModel.from_pretrained("tts_en_lj_hifigan_ft_mixertts").eval().cuda()
        return self

    def say(self, text: str) -> np.ndarray:
        """text → mono float32 waveform at 22.05 kHz."""
        import torch

        text = text.strip() or "a"
        with torch.no_grad():
            toks = self._sp.parse(text)
            spec = self._sp.generate_spectrogram(tokens=toks)
            audio = self._voc.convert_spectrogram_to_audio(spec=spec)
        return audio[0].cpu().numpy().astype(np.float32)
