"""E0 evidence: extract WavLM hidden states for a real audio clip on the GB10 GPU.

Writes a small manifest proving the harness ran on CUDA with real shapes/timing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pranava.speech.harness import SpeechEncoder, cuda_available

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "speech"
CLIP = ROOT / "docs" / "topic-debate-1.mp3"


def load_clip_seconds(path: Path, seconds: float = 6.0, sr: int = 16000) -> np.ndarray:
    """Decode the first `seconds` of an audio file to mono 16k via ffmpeg (no torchaudio)."""
    cmd = [
        "ffmpeg", "-v", "error", "-t", str(seconds), "-i", str(path),
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def main() -> int:
    if not cuda_available():
        print("CUDA unavailable", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    wav = load_clip_seconds(CLIP)
    enc = SpeechEncoder().load()
    hs = enc.encode(wav, sr=16000)
    manifest = {
        "model": hs.model_name,
        "device": hs.device,
        "seconds": round(hs.seconds, 3),
        "n_layers": hs.n_layers,
        "n_frames": hs.n_frames,
        "dim": hs.dim,
        "frame_rate_hz": round(hs.frame_rate_hz, 2),
        "elapsed_s": round(hs.elapsed_s, 4),
        "source": str(CLIP.name),
    }
    # persist the last-layer representation for downstream probes (compact, float16)
    np.save(OUT / "e0_lastlayer.npy", hs.layers[-1].to("cpu").numpy().astype("float16"))
    (OUT / "e0_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
