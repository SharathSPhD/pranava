"""Phase 4 — full speech-to-speech demo: audio-in → ALM → text → TTS → audio-out.

Assembles the complete Śabda-ALM loop and measures round-trip intelligibility: synthesize the ALM's
output audio, re-transcribe it with Parakeet, and compare to the ALM's own text output (a
self-consistency WER — how faithfully the vaikharī-out preserves the model's meaning). Content
fidelity is capped by Phase-2 CER; this gate proves the *pipeline* is end-to-end speech-to-speech.
Run in prabhasa/nemo-gb10.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.data import CORPUS_DIR, load_manifest, read_wav, text_to_bytes
from pranava.alm.encoder import ParakeetEncoder
from pranava.alm.projector import SphotaProjector
from pranava.alm.synth import SR, TTSHead

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "alm" / "s2s"


def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def main(n: int = 5) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    enc = ParakeetEncoder().load()
    blob = torch.load(ROOT / "data/alm/projector.pt", map_location=dev, weights_only=True)
    proj = SphotaProjector(d_enc=blob["d_enc"], d_model=blob["d_model"], downsample=4).to(dev).eval()
    proj.load_state_dict(blob["state_dict"])
    tts = TTSHead().load()

    val = load_manifest("val")[:n]
    rows = []
    for ex in val:
        wav_in, sr = read_wav(ex.wav_path)
        # audio-in → text
        with torch.no_grad():
            frames = enc.encode(wav_in, sr=sr)
            audio_tok = proj(frames, structural_bias=bias)
            out_ids = core.greedy_from_embeds(audio_tok, max_new=len(text_to_bytes(ex.text)) + 4)
        text_out = bytes(b for b in out_ids if 32 <= b < 127).decode("latin-1").strip() or "a"
        # text → audio-out (vaikharī)
        audio_out = tts.say(text_out)
        out_wav = OUT / f"{ex.id}_out.wav"
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
            wf.writeframes((np.clip(audio_out, -1, 1) * 32767).astype(np.int16).tobytes())
        # round-trip: re-transcribe the synthesized audio with Parakeet ASR
        rt = enc._model.transcribe([out_wav.as_posix()], verbose=False)
        rt_text = (rt[0].text if hasattr(rt[0], "text") else str(rt[0])).strip()
        wer_chars = _levenshtein(text_out, rt_text) / max(1, len(text_out))
        rows.append({"id": ex.id, "text_out": text_out, "roundtrip_asr": rt_text,
                     "audio_out_s": round(len(audio_out) / SR, 2),
                     "roundtrip_cer": round(wer_chars, 3), "audio_nonsilent": bool(np.abs(audio_out).max() > 0.01)})

    summary = {
        "n": len(rows), "all_produced_audio": all(r["audio_out_s"] > 0.2 for r in rows),
        "all_nonsilent": all(r["audio_nonsilent"] for r in rows),
        "mean_roundtrip_cer": round(float(np.mean([r["roundtrip_cer"] for r in rows])), 3),
        "note": "end-to-end audio→text→audio; content fidelity capped by Phase-2 CER (0.77). "
                "Gate proves the speech-to-speech PIPELINE, not final quality.",
        "samples": rows,
    }
    (OUT / "s2s_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("n", "all_produced_audio", "all_nonsilent", "mean_roundtrip_cer")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
