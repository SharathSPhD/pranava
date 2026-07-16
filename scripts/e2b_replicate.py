"""E2b — replicate the verb-final holism finding with a second speech encoder (HuBERT).

Pre-reg: research/prereg/H-HOLISM-2b.md. Reuses E2's stimuli, probe, and cached GPT-2 text
features; only the speech model changes to facebook/hubert-base-ls960.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pranava.experiments.holism import bootstrap_diff, holism_index, holm_correction
from pranava.experiments.stimuli import generate_stimuli

# reuse E2 machinery
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "experiments"
FEAT = OUT / "features"

_spec = importlib.util.spec_from_file_location("e2_run", ROOT / "scripts" / "e2_run.py")
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)


def extract_hubert(stimuli) -> np.ndarray:
    from pranava.experiments.features import SpeechFeatureExtractor

    ex = SpeechFeatureExtractor(layer=9, model_name="facebook/hubert-base-ls960")
    mats = []
    for st in stimuli:
        wav, sr = e2._load_wav(ROOT / "data" / "stimuli" / "wav" / f"{st.id}.wav")
        mats.append(ex.positions_matrix(wav, sr))
    return np.stack(mats)


def main() -> int:
    stimuli = generate_stimuli()
    labels = np.array([s.meaning_label for s in stimuli])
    groups = np.array([s.group for s in stimuli])
    resolution = np.array([s.resolution for s in stimuli])
    structure = np.array([s.structure for s in stimuli])

    hub_p = FEAT / "hubert.npy"
    hubert = np.load(hub_p) if hub_p.exists() else extract_hubert(stimuli)
    if not hub_p.exists():
        np.save(hub_p, hubert)
    text = np.load(FEAT / "text.npy")  # from E2

    hi_hub = e2.hi_per_item(hubert, labels, groups)
    hi_text = e2.hi_per_item(text, labels, groups)

    # probe validity
    def full_dec(feats):
        c = e2.per_item_curves(feats, labels, groups, seed=0)
        return float(np.nanmean(c[:, -1]))
    chance = 1.0 / len(np.unique(labels))

    vf = structure == "verb_final"
    late = resolution == "late"
    r1 = bootstrap_diff(hi_hub[vf], hi_text[vf], seed=0)   # R1: verb_final
    r2 = bootstrap_diff(hi_hub[late], hi_text[late], seed=0)  # R2: all late
    pvals = [r1.p_one_sided if r1.effect >= 0 else 1 - r1.p_one_sided,
             r2.p_one_sided if r2.effect >= 0 else 1 - r2.p_one_sided]
    rej = holm_correction(pvals)

    def summ(r, rj):
        return {"effect_HI_diff": round(r.effect, 4),
                "ci95": [round(r.ci_low, 4), round(r.ci_high, 4)],
                "p_one_sided": round(r.p_one_sided if r.effect >= 0 else 1 - r.p_one_sided, 4),
                "n": r.n, "supported_after_holm": bool(rj and r.supports_direction(True))}

    results = {
        "prereg": "research/prereg/H-HOLISM-2b.md",
        "speech_model": "facebook/hubert-base-ls960",
        "probe_validity": {"chance": round(chance, 4),
                           "hubert_full_utt_Ptrue": round(full_dec(hubert), 4)},
        "means": {
            "HI_hubert_verb_final": round(float(np.nanmean(hi_hub[vf])), 4),
            "HI_text_verb_final": round(float(np.nanmean(hi_text[vf])), 4),
            "HI_hubert_late": round(float(np.nanmean(hi_hub[late])), 4),
            "HI_text_late": round(float(np.nanmean(hi_text[late])), 4),
            "wavlm_verb_final_ref": 0.4055,  # from E2 for comparison
        },
        "tests": {
            "R1_hubert_gt_text_verb_final": summ(r1, rej[0]),
            "R2_hubert_gt_text_late": summ(r2, rej[1]),
        },
    }
    (OUT / "e2b_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("means", "tests", "probe_validity")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
