"""TDD Phase 1: the paired speech corpus loads, is split, and audio is real (host-runnable)."""
from pathlib import Path

import pytest

from pranava.alm.data import MANIFEST, load_manifest, read_wav, text_to_bytes

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="speech corpus not built")


def test_manifest_loads():
    exs = load_manifest()
    assert len(exs) >= 200


def test_split_present_and_disjoint():
    train = {e.id for e in load_manifest("train")}
    val = {e.id for e in load_manifest("val")}
    assert train and val
    assert not (train & val)


def test_wavs_exist_and_are_audio():
    for e in load_manifest()[:20]:
        assert e.wav_path.exists()
        wav, sr = read_wav(e.wav_path)
        assert sr == 22050
        assert len(wav) > sr * 0.2  # >0.2s of audio
        assert float(abs(wav).max()) > 0.0  # non-silent


def test_every_example_labelled():
    for e in load_manifest():
        assert e.text.strip()
        assert e.wav_path


def test_text_to_bytes_roundtrip():
    ids = text_to_bytes("balah")
    assert ids == [ord(c) for c in "balah"]
    assert all(0 <= b < 256 for b in ids)
