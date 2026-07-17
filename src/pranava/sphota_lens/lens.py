"""The Sphoṭa-Lens — locate, measure, and characterise the fused meaning workspace of the ALM.

Where prabodha's j-space locates a *text-only* global-workspace band via inter-layer CKA, the
Sphoṭa-Lens works on the ALM's *audio-fused* hidden states and adds two audio-native measures the
text-only lens cannot express:

  1. **Cross-modal fusion** (linear CKA between audio-position and text-position representations at
     each layer): where does the acoustic (dhvani) and the symbolic (pada) stream converge? The
     layer of maximal fusion is where the sphoṭa — the unified meaning — forms.
  2. **Articulation gradient** (top-k negentropy of the layer's readout via the tied head): a
     paśyantī→madhyamā→vaikharī axis — 0 = pre-articulate/held (paśyantī), 1 = uttered (vaikharī).

Reuses prabodha's pure-numpy primitives (``cka_matrix``, ``best_band_partition``, ``linear_cka``,
``topk_negentropy``) — see prabodha/src/prabodha/lens/e1_metrics.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from prabodha.lens.e1_metrics import (
    best_band_partition,
    cka_matrix,
    linear_cka,
    topk_negentropy,
)


@dataclass(frozen=True, slots=True)
class SphotaLensReport:
    n_layers: int
    n_samples: int
    band_contrast: float
    bands: tuple[int, int]  # (b1, b2): [0,b1) early, [b1,b2) workspace, [b2,L) late
    fusion_by_layer: list[float]  # cross-modal CKA per layer
    fusion_peak_layer: int
    articulation_by_layer: list[float]  # top-k negentropy per layer (paśyantī→vaikharī)

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "n_samples": self.n_samples,
            "band_contrast": round(self.band_contrast, 4),
            "bands": list(self.bands),
            "workspace_band": [self.bands[0], self.bands[1]],
            "fusion_by_layer": [round(x, 4) for x in self.fusion_by_layer],
            "fusion_peak_layer": self.fusion_peak_layer,
            "articulation_by_layer": [round(x, 4) for x in self.articulation_by_layer],
        }


def _pool(h: torch.Tensor, sl: slice) -> np.ndarray:
    """Mean-pool a (1,T,d) hidden state over positions `sl` → (d,) numpy."""
    return h[0, sl].mean(dim=0).detach().float().cpu().numpy()


def fit_sphota_lens(core, samples: list[tuple[torch.Tensor, int]], head_topk: int = 16
                    ) -> SphotaLensReport:
    """Fit the lens over ALM samples.

    Each sample is ``(inputs_embeds (1,T,d), n_audio_tokens)`` — the fused audio+text sequence and
    how many leading positions are audio (the rest are text). We capture per-layer hidden states and
    build: (a) an inter-layer CKA band partition (workspace), (b) cross-modal fusion per layer,
    (c) an articulation gradient per layer.
    """
    per_layer_pooled: list[list[np.ndarray]] | None = None  # [layer][sample] -> (d,)
    fusion_acc: list[list[float]] | None = None
    artic_acc: list[list[float]] | None = None

    for emb, n_audio in samples:
        layers = core.features_per_layer(emb)  # list of (1,T,d), len L+1
        L = len(layers)
        if per_layer_pooled is None:
            per_layer_pooled = [[] for _ in range(L)]
            fusion_acc = [[] for _ in range(L)]
            artic_acc = [[] for _ in range(L)]
        T = emb.shape[1]
        aud, txt = slice(0, n_audio), slice(n_audio, T)
        for li, h in enumerate(layers):
            per_layer_pooled[li].append(_pool(h, slice(0, T)))
            # cross-modal fusion: CKA between audio-token and text-token reps (need >=2 of each)
            if n_audio >= 2 and (T - n_audio) >= 2:
                a = h[0, aud].detach().float().cpu().numpy()
                t = h[0, txt].detach().float().cpu().numpy()
                m = min(len(a), len(t))
                fusion_acc[li].append(linear_cka(a[:m], t[:m]))
            # articulation: negentropy of this layer's readout at the last position
            logits = core.model.head(h)[0, -1].detach().float().cpu().numpy()
            artic_acc[li].append(topk_negentropy(logits, head_topk))

    acts = [np.stack(per_layer_pooled[li]) for li in range(len(per_layer_pooled))]
    c = cka_matrix(acts)
    contrast, bounds = best_band_partition(c)
    fusion = [float(np.mean(f)) if f else 0.0 for f in fusion_acc]
    artic = [float(np.mean(a)) if a else 0.0 for a in artic_acc]
    peak = int(np.argmax(fusion))
    return SphotaLensReport(
        n_layers=len(acts), n_samples=len(samples), band_contrast=contrast, bands=bounds,
        fusion_by_layer=fusion, fusion_peak_layer=peak, articulation_by_layer=artic,
    )
