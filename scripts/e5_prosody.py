"""E5 — quantify the prosody (pragmatic-information) gap: speech vs text.

Pre-reg: research/prereg/H-PROSODY-E5.md. Loop-proposed (X0). For a fixed sentence subset we make
two prosodic variants (neutral / urgent = rate×1.3 + pitch +2 st) of the SAME text, then ask a
grouped-CV probe to decode the prosody class from WavLM frames (speech) vs GPT-2 tokens (text).
Text is identical across variants → its probe is at chance by construction; the reported quantity
is the gap and the layer where prosody lives.
"""
from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path

import numpy as np

from pranava.experiments.holism import bootstrap_diff
from pranava.experiments.stimuli import generate_stimuli

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "experiments"
WAV = ROOT / "data" / "stimuli" / "wav"

_spec = importlib.util.spec_from_file_location("e2_run", ROOT / "scripts" / "e2_run.py")
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)

N_SENTENCES = 80  # subset; ×2 variants = 160 items


def load_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr, n = wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def make_urgent(wav: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    fast = librosa.effects.time_stretch(wav, rate=1.3)
    return librosa.effects.pitch_shift(fast, sr=sr, n_steps=2.0)


def speech_pooled(frames_layers: np.ndarray, layer: int) -> np.ndarray:
    return frames_layers[layer].mean(axis=0)  # mean-pool → [dim]


def probe_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seeds=(0, 1, 2)) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    accs = []
    for s in seeds:
        gkf = GroupKFold(n_splits=4)
        correct = total = 0
        for tr, te in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, random_state=s).fit(sc.transform(X[tr]), y[tr])
            pred = clf.predict(sc.transform(X[te]))
            correct += int((pred == y[te]).sum())
            total += len(te)
        accs.append(correct / total)
    return float(np.mean(accs))


def main() -> int:
    from pranava.speech.harness import SpeechEncoder
    from pranava.experiments.features import TextFeatureExtractor

    stimuli = generate_stimuli()[:N_SENTENCES]
    enc = SpeechEncoder().load()
    txt = TextFeatureExtractor(layer=6)

    # build features: for each sentence × {neutral, urgent}
    n_layers = 13
    sp_by_layer = {L: [] for L in range(n_layers)}
    tx_feats, y, groups = [], [], []
    for i, st in enumerate(stimuli):
        wav, sr = load_wav(WAV / f"{st.id}.wav")
        for variant, w in (("neutral", wav), ("urgent", make_urgent(wav, sr))):
            hs = enc.encode(w, sr=sr)  # [13, frames, 768]
            layers = hs.layers.to("cpu").numpy()
            for L in range(n_layers):
                sp_by_layer[L].append(speech_pooled(layers, L))
            # text: identical for both variants (prosody-blind by construction)
            tx_feats.append(txt.positions_matrix(st.text)[-1])
            y.append(variant)
            groups.append(st.id)

    y = np.array(y)
    groups = np.array(groups)

    # layer sweep for speech
    layer_acc = {L: probe_accuracy(np.array(sp_by_layer[L]), y, groups) for L in range(n_layers)}
    best_layer = max(layer_acc, key=layer_acc.get)
    acc_speech = layer_acc[best_layer]
    acc_text = probe_accuracy(np.array(tx_feats), y, groups)

    # bootstrap CI on the gap via per-item correctness at best layer
    def per_item_correct(X):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler

        out = np.zeros(len(y))
        gkf = GroupKFold(n_splits=4)
        for tr, te in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, random_state=0).fit(sc.transform(X[tr]), y[tr])
            out[te] = (clf.predict(sc.transform(X[te])) == y[te]).astype(float)
        return out

    sp_correct = per_item_correct(np.array(sp_by_layer[best_layer]))
    tx_correct = per_item_correct(np.array(tx_feats))
    gap = bootstrap_diff(sp_correct, tx_correct, seed=0)

    result = {
        "prereg": "research/prereg/H-PROSODY-E5.md",
        "n_sentences": N_SENTENCES,
        "n_items": int(len(y)),
        "chance": 0.5,
        "acc_speech_best_layer": round(acc_speech, 4),
        "best_layer": int(best_layer),
        "acc_text": round(acc_text, 4),
        "prosody_gap_speech_minus_text": {
            "effect": round(gap.effect, 4), "ci95": [round(gap.ci_low, 4), round(gap.ci_high, 4)],
            "p_one_sided": round(gap.p_one_sided, 4),
        },
        "layer_sweep": {str(L): round(a, 4) for L, a in layer_acc.items()},
        "PR1_speech_carries_prosody": acc_speech > 0.7,
        "PR2_text_near_chance": abs(acc_text - 0.5) < 0.1,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e5_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("acc_speech_best_layer", "best_layer", "acc_text",
                       "prosody_gap_speech_minus_text", "PR1_speech_carries_prosody",
                       "PR2_text_near_chance")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
