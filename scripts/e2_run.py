"""E2 — run the pre-registered H-HOLISM experiment. See research/prereg/H-HOLISM.md.

Pipeline (per modality ∈ {speech, text}):
  1. extract per-item cumulative representations [10 positions, dim] (GPU),
  2. GroupKFold(4) by template; train a linear probe on the full-utterance (t=1.0) rep,
  3. decodability(t) := probe P(true class) at each cumulative position (held-out item),
  4. Holism Index per item; average over 3 seeds.
Confirmatory (Holm across the 3 tests actually run — see report note on the pre-reg's
"2 primary tests" wording):
  P1-speech: HI_late > HI_early (speech)
  P1-text:   HI_late > HI_early (text)
  P2:        HI_speech(late) > HI_text(late)
Writes data/experiments/e2_results.json + a figure. Nulls are reported as findings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pranava.experiments.holism import POSITIONS, bootstrap_diff, holism_index, holm_correction
from pranava.experiments.stimuli import generate_stimuli

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "experiments"
FEAT = OUT / "features"
SEEDS = [0, 1, 2]


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return wav, sr


def extract_speech(stimuli) -> np.ndarray:
    from pranava.experiments.features import SpeechFeatureExtractor

    ex = SpeechFeatureExtractor(layer=9)
    mats = []
    for st in stimuli:
        wav, sr = _load_wav(ROOT / "data" / "stimuli" / "wav" / f"{st.id}.wav")
        mats.append(ex.positions_matrix(wav, sr))
    return np.stack(mats)  # [N, 10, dim]


def extract_text(stimuli) -> np.ndarray:
    from pranava.experiments.features import TextFeatureExtractor

    ex = TextFeatureExtractor(layer=6)
    return np.stack([ex.positions_matrix(st.text) for st in stimuli])


def per_item_curves(feats: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int):
    """Grouped-CV linear probe; return decodability curve [N, 10] = P(true class) per position."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    n, npos, dim = feats.shape
    curves = np.full((n, npos), np.nan, dtype=float)
    classes_all = np.unique(labels)
    gkf = GroupKFold(n_splits=4)
    full = feats[:, -1, :]  # t=1.0 representation for training
    for tr, te in gkf.split(full, labels, groups):
        scaler = StandardScaler().fit(full[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(scaler.transform(full[tr]), labels[tr])
        cls_index = {c: i for i, c in enumerate(clf.classes_)}
        for i in te:
            true_i = cls_index.get(labels[i])
            if true_i is None:  # class absent from this train fold
                continue
            X = scaler.transform(feats[i])  # [10, dim]
            proba = clf.predict_proba(X)  # [10, n_classes]
            curves[i] = proba[:, true_i]
    return curves


def hi_per_item(feats, labels, groups) -> np.ndarray:
    per_seed = []
    for s in SEEDS:
        curves = per_item_curves(feats, labels, groups, seed=s)
        per_seed.append(np.array([holism_index(c) for c in curves]))
    return np.nanmean(np.stack(per_seed), axis=0)  # [N]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FEAT.mkdir(parents=True, exist_ok=True)
    stimuli = generate_stimuli()
    labels = np.array([s.meaning_label for s in stimuli])
    groups = np.array([s.group for s in stimuli])
    resolution = np.array([s.resolution for s in stimuli])

    # features (cache)
    sp_p, tx_p = FEAT / "speech.npy", FEAT / "text.npy"
    speech = np.load(sp_p) if sp_p.exists() else extract_speech(stimuli)
    if not sp_p.exists():
        np.save(sp_p, speech)
    text = np.load(tx_p) if tx_p.exists() else extract_text(stimuli)
    if not tx_p.exists():
        np.save(tx_p, text)

    # Probe-validity: full-utterance decodability must beat the 11-class chance floor,
    # else HI is meaningless. Recorded, not hidden.
    def full_utt_decodability(feats):
        c = per_item_curves(feats, labels, groups, seed=0)
        return float(np.nanmean(c[:, -1]))
    chance = 1.0 / len(np.unique(labels))
    probe_validity = {
        "chance": round(chance, 4),
        "speech_full_utt_Ptrue": round(full_utt_decodability(speech), 4),
        "text_full_utt_Ptrue": round(full_utt_decodability(text), 4),
    }

    hi_speech = hi_per_item(speech, labels, groups)
    hi_text = hi_per_item(text, labels, groups)

    late = resolution == "late"
    early = resolution == "early"

    tests = {
        "P1_speech_late_gt_early": bootstrap_diff(hi_speech[late], hi_speech[early], seed=0),
        "P1_text_late_gt_early": bootstrap_diff(hi_text[late], hi_text[early], seed=0),
        "P2_speech_gt_text_on_late": bootstrap_diff(hi_speech[late], hi_text[late], seed=0),
    }
    # Real one-sided bootstrap p-values (direction = predicted positive). A test in the WRONG
    # direction gets p=1-p_pos so Holm cannot "reject" it in our favour.
    def directed_p(r) -> float:
        # r.p_one_sided is in the direction of the observed effect; convert to "predicted +"
        return r.p_one_sided if r.effect >= 0 else 1.0 - r.p_one_sided
    pvals = [directed_p(r) for r in tests.values()]
    rejects = holm_correction(pvals)

    def summarize(name, r, rej):
        return {
            "effect_HI_diff": round(r.effect, 4),
            "ci95": [round(r.ci_low, 4), round(r.ci_high, 4)],
            "p_one_sided_predicted_dir": round(directed_p(r), 4),
            "n": r.n,
            "supported_after_holm": bool(rej and r.supports_direction(True)),
        }

    results = {
        "prereg": "research/prereg/H-HOLISM.md",
        "note_on_prereg": "Pre-reg body specifies P1 within EACH modality; its 'two primary "
        "tests' line was inconsistent. We ran the 3 tests the body implies and applied Holm "
        "across all 3 (more conservative). Reported transparently.",
        "means": {
            "HI_speech_late": round(float(np.nanmean(hi_speech[late])), 4),
            "HI_speech_early": round(float(np.nanmean(hi_speech[early])), 4),
            "HI_text_late": round(float(np.nanmean(hi_text[late])), 4),
            "HI_text_early": round(float(np.nanmean(hi_text[early])), 4),
        },
        "probe_validity": probe_validity,
        "n_items": int(len(stimuli)),
        "n_valid_HI": {
            "speech": int(np.sum(~np.isnan(hi_speech))),
            "text": int(np.sum(~np.isnan(hi_text))),
        },
        "tests": {name: summarize(name, r, rej) for (name, r), rej in zip(tests.items(), rejects)},
        "seeds": SEEDS,
        "positions": POSITIONS.tolist(),
    }

    # mean curves for the figure
    def mean_curve(feats, mask):
        cs = [per_item_curves(feats, labels, groups, seed=0)[i] for i in np.where(mask)[0]]
        return np.nanmean(np.stack(cs), axis=0)

    curves = {
        "speech_late": mean_curve(speech, late).tolist(),
        "speech_early": mean_curve(speech, early).tolist(),
        "text_late": mean_curve(text, late).tolist(),
        "text_early": mean_curve(text, early).tolist(),
    }
    results["mean_curves"] = curves

    # Exploratory: HI by structure, to separate genuine late-resolution (verb_final, where the
    # label = the final verb) from structural cueing (garden_path items share one label). Reported
    # as exploratory per the pre-registration.
    structure = np.array([s.structure for s in stimuli])
    explore = {}
    for struct in ["canonical", "garden_path", "verb_final", "verb_first"]:
        m = structure == struct
        explore[struct] = {
            "n": int(m.sum()),
            "HI_speech": round(float(np.nanmean(hi_speech[m])), 4),
            "HI_text": round(float(np.nanmean(hi_text[m])), 4),
        }
    # focused: on verb_final only (genuinely late-resolving), speech vs text
    vf = structure == "verb_final"
    r_vf = bootstrap_diff(hi_speech[vf], hi_text[vf], seed=0)
    explore["verb_final_speech_gt_text"] = {
        "effect_HI_diff": round(r_vf.effect, 4),
        "ci95": [round(r_vf.ci_low, 4), round(r_vf.ci_high, 4)],
        "p_one_sided": round(r_vf.p_one_sided if r_vf.effect >= 0 else 1 - r_vf.p_one_sided, 4),
        "n": r_vf.n,
    }
    results["exploratory"] = explore
    (OUT / "e2_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot(curves, results, OUT / "e2_trajectories.png")
    print(json.dumps({k: results[k] for k in ("means", "tests", "n_valid_HI")}, indent=2))
    return 0


def _plot(curves, results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = POSITIONS
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ax[0].plot(x, curves["speech_late"], "-o", label="late", color="#c0392b")
    ax[0].plot(x, curves["speech_early"], "-o", label="early", color="#2980b9")
    ax[0].set_title(f"Speech (WavLM)  HI late={results['means']['HI_speech_late']} "
                    f"early={results['means']['HI_speech_early']}")
    ax[1].plot(x, curves["text_late"], "-o", label="late", color="#c0392b")
    ax[1].plot(x, curves["text_early"], "-o", label="early", color="#2980b9")
    ax[1].set_title(f"Text (GPT-2)  HI late={results['means']['HI_text_late']} "
                    f"early={results['means']['HI_text_early']}")
    for a in ax:
        a.set_xlabel("relative position in utterance")
        a.legend()
        a.grid(alpha=0.3)
    ax[0].set_ylabel("decodability  P(true meaning)")
    fig.suptitle("H-HOLISM: meaning-decodability trajectories (sphoṭa/pratibhā probe)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
