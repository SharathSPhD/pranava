"""E7 — definitive, properly-powered holism test (matched vocabulary). Pre-reg: H-HOLISM-E7.

Pools 240 early + 240 late items (same 8 verb labels), trains ONE grouped-CV probe across both,
and settles P1 (late>early holism) and P2 (speech>text on late) with valid CV and full power.
"""
from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path

import numpy as np

from pranava.experiments.holism import bootstrap_diff, holm_correction
from pranava.experiments.stimuli_early import generate_early_scaled
from pranava.experiments.stimuli_verbfinal import generate_verbfinal_scaled

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "experiments"
FEAT = OUT / "features"
WAV_VF = ROOT / "data" / "stimuli" / "wav_vf"
WAV_EA = ROOT / "data" / "stimuli" / "wav_ea"

_spec = importlib.util.spec_from_file_location("e2_run", ROOT / "scripts" / "e2_run.py")
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)


def synth(stimuli, wav_dir: Path):
    from piper import PiperVoice

    wav_dir.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(str(ROOT / ".voices" / "en_US-lessac-medium.onnx"))
    for st in stimuli:
        p = wav_dir / f"{st.id}.wav"
        if not p.exists():
            with wave.open(str(p), "wb") as wf:
                voice.synthesize_wav(st.text, wf)


def load_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr, n = wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def speech_feats(stimuli, wav_dir, cache):
    if cache.exists():
        return np.load(cache)
    from pranava.experiments.features import SpeechFeatureExtractor

    ex = SpeechFeatureExtractor(layer=9)
    a = np.stack([ex.positions_matrix(*load_wav(wav_dir / f"{s.id}.wav")) for s in stimuli])
    np.save(cache, a)
    return a


def text_feats(stimuli, cache):
    if cache.exists():
        return np.load(cache)
    from pranava.experiments.features import TextFeatureExtractor

    tx = TextFeatureExtractor(layer=6)
    a = np.stack([tx.positions_matrix(s.text) for s in stimuli])
    np.save(cache, a)
    return a


def main() -> int:
    early = generate_early_scaled()
    late = generate_verbfinal_scaled()
    synth(early, WAV_EA)
    synth(late, WAV_VF)

    labels = np.array([s.meaning_label for s in early] + [s.meaning_label for s in late])
    # group must be unique per (pool, template) so early/late never share a fold slot
    groups = np.array([f"ea:{s.group}" for s in early] + [f"vf:{s.group}" for s in late])
    is_late = np.array([0] * len(early) + [1] * len(late), dtype=bool)

    sp = np.concatenate([speech_feats(early, WAV_EA, FEAT / "e7_sp_early.npy"),
                         speech_feats(late, WAV_VF, FEAT / "e7_sp_late.npy")])
    tx = np.concatenate([text_feats(early, FEAT / "e7_tx_early.npy"),
                         text_feats(late, FEAT / "e7_tx_late.npy")])

    hi_sp = e2.hi_per_item(sp, labels, groups)
    hi_tx = e2.hi_per_item(tx, labels, groups)

    late_m, early_m = is_late, ~is_late
    # P1: late>early within each modality
    p1_sp = bootstrap_diff(hi_sp[late_m], hi_sp[early_m], seed=0)
    p1_tx = bootstrap_diff(hi_tx[late_m], hi_tx[early_m], seed=0)
    # P2: speech>text on late
    p2 = bootstrap_diff(hi_sp[late_m], hi_tx[late_m], seed=0)

    def dp(r):
        return r.p_one_sided if r.effect >= 0 else 1 - r.p_one_sided
    rej = holm_correction([dp(p1_sp), dp(p1_tx), dp(p2)])

    def summ(r, rj):
        return {"effect": round(r.effect, 4), "ci95": [round(r.ci_low, 4), round(r.ci_high, 4)],
                "p": round(dp(r), 4), "supported_holm": bool(rj and r.supports_direction(True))}

    result = {
        "prereg": "research/prereg/H-HOLISM-E7.md",
        "n_early": len(early), "n_late": len(late),
        "means": {"HI_speech_late": round(float(np.nanmean(hi_sp[late_m])), 4),
                  "HI_speech_early": round(float(np.nanmean(hi_sp[early_m])), 4),
                  "HI_text_late": round(float(np.nanmean(hi_tx[late_m])), 4),
                  "HI_text_early": round(float(np.nanmean(hi_tx[early_m])), 4)},
        "P1_speech_late_gt_early": summ(p1_sp, rej[0]),
        "P1_text_late_gt_early": summ(p1_tx, rej[1]),
        "P2_speech_gt_text_late": summ(p2, rej[2]),
        "verdict": ("P2 supported — real speech>text holism (rescued)"
                    if (rej[2] and p2.supports_direction(True))
                    else "P2 null — no speech>text holism (confirms E6 correction)"),
    }
    (OUT / "e7_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
