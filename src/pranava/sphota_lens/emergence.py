"""Meaning-emergence lens — the rigorous core of the Sphoṭa-Lens.

Bhartṛhari's *sphoṭa*: the indivisible meaning that *bursts* from continuous sound. We operationalise
it causally and measurably: at each layer of the ALM, how decodable is the sentence's meaning (its
kriyā/verb) from the **audio-token positions alone** — the positions carrying sound, not text? The
layer where this decodability peaks is where sound has become meaning: the **sphoṭa layer**.

This is a genuine instrument, not a similarity heuristic:
  * `meaning_emergence_curve` — grouped-CV probe accuracy per layer over audio-position reps
    (chance-referenced). A rise-to-peak is the emergence of meaning from sound.
  * `causal_layer_importance` — ablate (mean-replace) each layer's audio-position reps and measure
    the drop in meaning-decodability. The layer whose ablation hurts most is the causal sphoṭa layer.
A discovery is claimed only when the correlational peak and the causal peak agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class EmergenceReport:
    n_layers: int
    n_samples: int
    n_classes: int
    chance: float
    decodability_by_layer: list[float]  # audio-position meaning-decodability per layer
    peak_layer: int
    causal_importance_by_layer: list[float]  # decodability drop when layer ablated
    causal_peak_layer: int
    sphota_layer: int  # agreed layer if peaks coincide, else -1
    validated: bool

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers, "n_samples": self.n_samples, "n_classes": self.n_classes,
            "chance": round(self.chance, 4),
            "decodability_by_layer": [round(x, 4) for x in self.decodability_by_layer],
            "peak_layer": self.peak_layer,
            "causal_importance_by_layer": [round(x, 4) for x in self.causal_importance_by_layer],
            "causal_peak_layer": self.causal_peak_layer,
            "sphota_layer": self.sphota_layer, "validated": self.validated,
        }


def _grouped_cv_acc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> float:
    """Template-grouped CV accuracy of a linear probe (no lexical leakage)."""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=min(4, n_groups))
    correct = total = 0
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(sc.transform(X[tr]), y[tr])
        correct += int((clf.predict(sc.transform(X[te])) == y[te]).sum())
        total += len(te)
    return correct / total if total else float("nan")


def meaning_emergence(core, samples: list[tuple[torch.Tensor, int, str, str]]) -> EmergenceReport:
    """samples: (fused_embeds (1,T,d), n_audio_tokens, meaning_label, cv_group).

    Captures per-layer hidden states, pools the AUDIO positions, and probes meaning-decodability
    per layer (correlational) and under per-layer ablation (causal).
    """
    labels = np.array([s[2] for s in samples])
    groups = np.array([s[3] for s in samples])
    classes, y = np.unique(labels, return_inverse=True)
    chance = 1.0 / len(classes)

    # per-layer audio-position pooled reps: [layer][sample] -> (d,)  (correlational curve)
    pooled: list[list[np.ndarray]] = []
    n_layers = 0
    for emb, n_aud, _, _ in samples:
        layers = core.features_per_layer(emb)  # (L+1) x (1,T,d)
        if not pooled:
            n_layers = len(layers)
            pooled = [[] for _ in range(n_layers)]
        for li, h in enumerate(layers):
            pooled[li].append(h[0, :n_aud].mean(dim=0).detach().float().cpu().numpy())
    acts = [np.stack(p) for p in pooled]  # (L+1) x (N, d)
    decod = [_grouped_cv_acc(a, y, groups) for a in acts]
    peak = int(np.nanargmax(decod))

    # CAUSAL: for each layer, ablate (mean-collapse) the audio positions during the forward and
    # measure how much the FINAL meaning-decodability drops. True activation patching.
    base_final = float(_grouped_cv_acc(acts[-1], y, groups))
    causal = []
    for li in range(n_layers):
        abl_final = []
        for emb, n_aud, _, _ in samples:
            h = core.features_final_ablated(emb, ablate_layer=li, positions=slice(0, n_aud))
            abl_final.append(h[0, :n_aud].mean(dim=0).detach().float().cpu().numpy())
        acc = _grouped_cv_acc(np.stack(abl_final), y, groups)
        causal.append(max(0.0, base_final - (acc if not np.isnan(acc) else base_final)))
    causal_peak = int(np.argmax(causal))

    sphota = peak if abs(peak - causal_peak) <= 1 else -1  # correlational & causal agree (±1 layer)
    validated = bool(
        not np.isnan(decod[peak]) and decod[peak] > chance + 0.1
        and peak >= 1 and base_final > chance + 0.1 and sphota >= 0
    )
    return EmergenceReport(
        n_layers=len(acts), n_samples=len(samples), n_classes=len(classes), chance=chance,
        decodability_by_layer=decod, peak_layer=peak,
        causal_importance_by_layer=causal, causal_peak_layer=causal_peak,
        sphota_layer=sphota, validated=validated,
    )
