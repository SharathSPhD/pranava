"""TDD Phase 0: the full ALM pipeline runs end-to-end on a real clip on the GB10.

Untrained projector → output is not yet meaningful (that is Phase 2); this asserts the WIRING:
real audio → Parakeet → projector → prabhasa core → bytes, correct shapes, on CUDA.
"""
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from pranava.alm.core_adapter import CHECKPOINTS, alm_runnable

pytestmark = pytest.mark.skipif(
    not alm_runnable(),
    reason="prabhasa m2 checkpoint or CUDA not present — run in prabhasa/nemo-gb10",
)

_WAV = Path("/work/pranava/data/stimuli/wav/s0000.wav")
if not _WAV.exists():
    _WAV = Path(__file__).resolve().parents[2] / "data/stimuli/wav/s0000.wav"


def _load(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr, raw = wf.getframerate(), wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


@pytest.mark.slow
def test_end_to_end_runs_on_real_clip():
    from pranava.alm.pipeline import SabdaALM

    alm = SabdaALM.build(arm="m2", downsample=4)
    wav, sr = _load(_WAV)

    tokens = alm.audio_tokens(wav, sr)
    assert tokens.shape[0] == 1 and tokens.shape[2] == alm.core.d_model
    assert tokens.is_cuda
    assert tokens.shape[1] >= 1  # at least one projected sphoṭa token

    out = alm.transcribe(wav, sr, max_new=16)
    assert isinstance(out, bytes)  # produces byte output (untrained → not yet meaningful)
