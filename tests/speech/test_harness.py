"""TDD: the speech-model representation harness runs on GPU and yields sane shapes.

Skipped if CUDA is unavailable. Requires network on first run to fetch the model
(cached under .hf_cache thereafter).
"""
import numpy as np
import pytest

from pranava.speech.harness import SpeechEncoder, cuda_available

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA not available")


def _chirp(seconds: float = 2.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    # deterministic sweep 200→2000 Hz — stands in for speech to exercise the harness
    f = np.linspace(200, 2000, t.size)
    return (0.1 * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.fixture(scope="module")
def encoder():
    return SpeechEncoder().load()


def test_runs_on_cuda(encoder):
    hs = encoder.encode(_chirp(), sr=16000)
    assert hs.device.startswith("cuda")
    assert hs.layers.is_cuda


def test_shapes_are_sane(encoder):
    hs = encoder.encode(_chirp(2.0), sr=16000)
    assert hs.n_layers == 13  # embeddings + 12 transformer layers for wavlm-base
    assert hs.dim == 768
    assert hs.n_frames > 50  # ~50 Hz frame rate over 2 s
    # WavLM-base runs at ~50 Hz
    assert 40 < hs.frame_rate_hz < 60


def test_determinism(encoder):
    a = encoder.encode(_chirp(1.0), sr=16000)
    b = encoder.encode(_chirp(1.0), sr=16000)
    assert a.layers.shape == b.layers.shape
    assert torch.allclose(a.layers, b.layers, atol=1e-4)


def test_longer_audio_more_frames(encoder):
    short = encoder.encode(_chirp(1.0), sr=16000)
    long = encoder.encode(_chirp(3.0), sr=16000)
    assert long.n_frames > short.n_frames


def test_resample_path(encoder):
    # feed 8 kHz; harness must resample to 16 kHz and still work
    t = np.linspace(0, 1.0, 8000, endpoint=False)
    wav = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    hs = encoder.encode(wav, sr=8000)
    assert hs.n_frames > 20


import torch  # noqa: E402  (imported late; only needed inside CUDA-gated tests)
