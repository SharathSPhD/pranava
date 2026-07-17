"""E6 — scale the verb-final contrast to tighten the CI on the key holism finding.

Loop-proposed (X0). Synthesizes 240 verb-final items (single fixed voice), extracts WavLM
(speech) and GPT-2 (text) features, computes Holism Index, and runs the speech>text bootstrap.
Compares the CI width against E2's verb_final CI [0.106, 0.361] (48 items). Records to ledger.
"""
from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path

import numpy as np

from pranava.autoresearch.loop import LEDGER, LedgerEntry, append_ledger, record_observation
from pranava.experiments.holism import bootstrap_diff
from pranava.experiments.stimuli_verbfinal import generate_verbfinal_scaled

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "experiments"
FEAT = OUT / "features"
WAV_DIR = ROOT / "data" / "stimuli" / "wav_vf"

_spec = importlib.util.spec_from_file_location("e2_run", ROOT / "scripts" / "e2_run.py")
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)


def synthesize(stimuli) -> None:
    from piper import PiperVoice

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(str(ROOT / ".voices" / "en_US-lessac-medium.onnx"))
    for st in stimuli:
        p = WAV_DIR / f"{st.id}.wav"
        if p.exists():
            continue
        with wave.open(str(p), "wb") as wf:
            voice.synthesize_wav(st.text, wf)


def load_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr, n = wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def main() -> int:
    stimuli = generate_verbfinal_scaled()
    synthesize(stimuli)
    labels = np.array([s.meaning_label for s in stimuli])
    groups = np.array([s.group for s in stimuli])

    sp_p, tx_p = FEAT / "vf_speech.npy", FEAT / "vf_text.npy"
    if sp_p.exists():
        speech = np.load(sp_p)
    else:
        from pranava.experiments.features import SpeechFeatureExtractor

        ex = SpeechFeatureExtractor(layer=9)
        speech = np.stack([ex.positions_matrix(*load_wav(WAV_DIR / f"{s.id}.wav")) for s in stimuli])
        np.save(sp_p, speech)
    if tx_p.exists():
        text = np.load(tx_p)
    else:
        from pranava.experiments.features import TextFeatureExtractor

        tx = TextFeatureExtractor(layer=6)
        text = np.stack([tx.positions_matrix(s.text) for s in stimuli])
        np.save(tx_p, text)

    hi_sp = e2.hi_per_item(speech, labels, groups)
    hi_tx = e2.hi_per_item(text, labels, groups)
    r = bootstrap_diff(hi_sp, hi_tx, seed=0)

    e2_ci_width = 0.361 - 0.106
    ci_width = r.ci_high - r.ci_low
    result = {
        "n_items": len(stimuli),
        "HI_speech": round(float(np.nanmean(hi_sp)), 4),
        "HI_text": round(float(np.nanmean(hi_tx)), 4),
        "speech_gt_text": {"effect": round(r.effect, 4),
                           "ci95": [round(r.ci_low, 4), round(r.ci_high, 4)],
                           "p_one_sided": round(r.p_one_sided, 4),
                           "supported": bool(r.supports_direction(True))},
        "ci_width": round(ci_width, 4),
        "e2_verbfinal_ci_width": round(e2_ci_width, 4),
        "ci_tightened": bool(ci_width < e2_ci_width),
    }
    (OUT / "e6_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    supported = result["speech_gt_text"]["supported"]
    tighter = result["ci_tightened"]
    gate = {"milestone": "E6",
            "code_gate": {"verdict": "pass", "evidence": "E2 pipeline reused; 240 vf items"},
            "domain_gate": {"verdict": "pass" if (supported and tighter) else "fail",
                            "evidence": f"speech>text CI {result['speech_gt_text']['ci95']} "
                                        f"width {ci_width:.3f} vs E2 {e2_ci_width:.3f}; "
                                        f"{'tighter' if tighter else 'not tighter'}"},
            "closed": supported and tighter}
    gp = ROOT / "gates" / "gate_E6.json"
    gp.write_text(json.dumps(gate, indent=2))
    record_observation("e6_scale_verbfinal", gp, ledger_path=LEDGER)
    append_ledger(LedgerEntry("disposition", "e6_scale_verbfinal",
                              {"note": f"scaled 48→240 vf items; CI {ci_width:.3f}"}), LEDGER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
